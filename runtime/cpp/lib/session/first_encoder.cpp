#include "lib/session/first_encoder.h"

#include "lib/session/session.h"
#include "lib/scheduler/steady_batch_primitive.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>

using torch::inductor::AOTIModelPackageLoader;

namespace {

void require_tensor(const torch::Tensor& tensor, const char* name) {
  if (!tensor.defined()) {
    throw std::runtime_error(std::string("first_encoder input is undefined: ") + name);
  }
  if (!tensor.is_contiguous()) {
    throw std::runtime_error(std::string("first_encoder input is not contiguous: ") + name);
  }
}

void validate_first_encoder_inputs(const torch::Tensor& chunk,
                                   const torch::Tensor& L,
                                   const torch::Tensor& clc,
                                   const torch::Tensor& clt,
                                   const torch::Tensor& clcl) {
  require_tensor(chunk, "chunk");
  require_tensor(L, "L");
  require_tensor(clc, "clc");
  require_tensor(clt, "clt");
  require_tensor(clcl, "clcl");
  if (L.scalar_type() != at::kLong || L.numel() != 1) {
    throw std::runtime_error("first_encoder L must be a single int64 tensor");
  }
  // L == chunk.size(2) is guaranteed by construction: both run() impls build
  // L = torch::full({1}, chunk.size(2)). We deliberately do NOT read L.item()
  // here — that forces a device->host sync on every chunk-0 call, which matters
  // on the Step-5 serving path. The structural (numel/dtype/contiguous) checks
  // above are sync-free and sufficient.
}

void validate_first_encoder_outputs(const std::vector<at::Tensor>& out, const char* kind) {
  if (out.size() != 5) {
    throw std::runtime_error(std::string("first_encoder ") + kind +
                             " returned " + std::to_string(out.size()) +
                             " outputs, expected exactly 5");
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (!out[i].defined()) {
      throw std::runtime_error(std::string("first_encoder ") + kind +
                               " returned undefined output " + std::to_string(i));
    }
  }
  // Valid first-chunk output frames = SHIFT / subsampling(8): 2 for the en
  // profile, 4 for ml. The export slices enc_out to exactly this length.
  constexpr int64_t kFirstChunkValidOut = SHIFT / 8;
  if (out[0].dim() != 3 || out[0].size(2) != kFirstChunkValidOut) {
    std::ostringstream oss;
    oss << "first_encoder " << kind << " enc_out is not the sliced [:, :, :" << kFirstChunkValidOut
        << "] shape; sizes=";
    for (auto s : out[0].sizes()) oss << ' ' << s;
    throw std::runtime_error(oss.str());
  }
  if (out[1].scalar_type() != at::kLong || out[4].scalar_type() != at::kLong ||
      out[1].numel() != 1 || out[4].numel() != 1) {
    throw std::runtime_error(std::string("first_encoder ") + kind +
                             " integer outputs must be single-element int64 for enc_len/cache_ch_len");
  }
}

}  // namespace

TsFirstEncoder::TsFirstEncoder(torch::jit::Module& module) : module_(module) {}

std::vector<at::Tensor> TsFirstEncoder::run(const torch::Tensor& chunk,
                                            SessionState& state,
                                            c10::cuda::CUDAStream stream) {
  {
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto device = chunk.device();
    auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
    validate_first_encoder_inputs(chunk.contiguous(),
                                  L.contiguous(),
                                  state.clc.contiguous(),
                                  state.clt.contiguous(),
                                  state.clcl.contiguous());
  }
  auto out = run_first_encoder(module_, chunk, state, stream);
  validate_first_encoder_outputs(out, kind());
  return out;
}

const char* TsFirstEncoder::kind() const {
  return "ts";
}

AotiFirstEncoder::AotiFirstEncoder(
    const std::string& pkg_path,
    const std::unordered_map<std::string, at::Tensor>& shared_constants,
    torch::Device device,
    int num_runners)
    : pkg_path_(pkg_path), device_(device) {
  if (!device_.is_cuda()) throw std::runtime_error("AotiFirstEncoder requires a CUDA device");
  if (num_runners <= 0) throw std::runtime_error("AotiFirstEncoder num_runners must be positive");
  if (shared_constants.empty()) throw std::runtime_error("AotiFirstEncoder shared constants map is empty");
  c10::cuda::CUDAGuard device_guard(device_.index());
  loader_ = std::make_unique<AOTIModelPackageLoader>(
      pkg_path_, "model", false, num_runners, device_.index());
  auto constants = bsteady_detail::constants_for_bucket(shared_constants, *loader_, pkg_path_);
  loader_->load_constants(constants.values,
                          /*runtime_const_fold=*/false,
                          /*check=*/false,
                          /*user_managed=*/true);
  std::printf("first_encoder aoti constants bound: package=%s constants=%zu direct=%zu alias=%zu "
              "num_runners=%d\n",
              pkg_path_.c_str(),
              constants.values.size(),
              constants.direct_matches,
              constants.alias_fallbacks,
              num_runners);
  std::fflush(stdout);
}

AotiFirstEncoder::~AotiFirstEncoder() = default;

std::vector<at::Tensor> AotiFirstEncoder::run(const torch::Tensor& chunk,
                                              SessionState& state,
                                              c10::cuda::CUDAStream stream) {
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  std::vector<at::Tensor> inputs = {
      chunk.contiguous(),
      L.contiguous(),
      state.clc.contiguous(),
      state.clt.contiguous(),
      state.clcl.contiguous(),
  };
  validate_first_encoder_inputs(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
  auto out = loader_->run(inputs, reinterpret_cast<void*>(stream.stream()));
  validate_first_encoder_outputs(out, kind());
  return out;
}

const char* AotiFirstEncoder::kind() const {
  return "aoti";
}
