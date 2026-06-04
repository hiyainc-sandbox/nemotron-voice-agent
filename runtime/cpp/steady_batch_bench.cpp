// STEADY-BATCH-0 microbench.
//
// Loads fixed-B AOTI steady encoder packages (B=1/2/4), feeds packed
// independent streaming caches, checks that every batched row is close to the
// B=1 package for the same dummy input/cache, and reports per-row CUDA-event
// time plus the B=2/B=4 ratios.
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include "lib/scheduler/steady_batch_primitive.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;
namespace fs = std::filesystem;

static constexpr int SHIFT = 16;
static constexpr int PRE = 9;
static constexpr int INPUT_T = PRE + SHIFT;
static constexpr int MELS = 128;
static constexpr int LAYERS = 24;
static constexpr int ATT_LEFT = 70;
static constexpr int D_MODEL = 1024;
static constexpr int TIME_CACHE = 8;
static constexpr int CACHE_LEN_AFTER_FIRST = 3;

static const char* OUT_NAMES[5] = {"enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"};

static void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr
        << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

struct Args {
  std::string dir = "../artifacts";
  std::string shared_weights;
  int warmup = 10;
  int iters = 100;
  double atol = 5.0e-2;
  double rtol = 1.0e-3;
  bool fail_on_perf = false;
};

struct Summary {
  double mean = 0.0;
  double p50 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

struct BenchCase {
  int batch = 1;
  std::string package_path;
  std::unique_ptr<AOTIModelPackageLoader> loader;
  std::vector<at::Tensor> inputs;
  Summary total_ms;
  Summary per_row_ms;
};

static Args parse_args(int argc, char** argv) {
  Args args;
  bool dir_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--dir") {
      args.dir = need_value("--dir");
      dir_set = true;
    } else if (arg == "--shared-weights") {
      args.shared_weights = need_value("--shared-weights");
    } else if (arg == "--warmup") {
      args.warmup = std::stoi(need_value("--warmup"));
    } else if (arg == "--iters") {
      args.iters = std::stoi(need_value("--iters"));
    } else if (arg == "--atol") {
      args.atol = std::stod(need_value("--atol"));
    } else if (arg == "--rtol") {
      args.rtol = std::stod(need_value("--rtol"));
    } else if (arg == "--fail-on-perf") {
      args.fail_on_perf = true;
    } else if (!dir_set) {
      args.dir = arg;
      dir_set = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.warmup < 0) throw std::runtime_error("--warmup must be non-negative");
  if (args.iters <= 0) throw std::runtime_error("--iters must be positive");
  if (args.atol < 0.0 || args.rtol < 0.0) throw std::runtime_error("--atol/--rtol must be non-negative");
  return args;
}

static std::string resolve_shared_weights_path(const std::string& package_dir,
                                               const std::string& configured) {
  if (!configured.empty()) return configured;

  std::vector<fs::path> candidates;
  fs::path dir_path(package_dir);
  if (dir_path.has_parent_path()) {
    candidates.push_back(dir_path.parent_path() / "artifacts" / "finalize_shared_weights.ts");
  }
  candidates.push_back(dir_path / "finalize_shared_weights.ts");
  candidates.push_back("artifacts/finalize_shared_weights.ts");
  candidates.push_back("../artifacts/finalize_shared_weights.ts");
  candidates.push_back("runtime/artifacts/finalize_shared_weights.ts");

  for (const auto& candidate : candidates) {
    if (fs::exists(candidate)) return candidate.string();
  }
  throw std::runtime_error("could not resolve finalize_shared_weights.ts; pass --shared-weights");
}

static Summary summarize(std::vector<double> values) {
  if (values.empty()) return Summary{};
  std::sort(values.begin(), values.end());
  Summary s;
  s.min = values.front();
  s.max = values.back();
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  const size_t mid = values.size() / 2;
  s.p50 = (values.size() % 2) ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
  return s;
}

static std::string sizes_str(const at::Tensor& tensor) {
  std::ostringstream oss;
  oss << "(";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i) oss << ",";
    oss << tensor.size(i);
  }
  oss << ")";
  return oss.str();
}

static std::vector<at::Tensor> make_inputs(int batch, torch::Device device) {
  const int64_t b = static_cast<int64_t>(batch);
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto i64 = torch::TensorOptions().dtype(torch::kLong).device(device);

  auto base_chunk = torch::linspace(-0.75, 0.75, MELS * INPUT_T, f32)
                        .reshape({1, MELS, INPUT_T})
                        .contiguous();
  auto chunk = base_chunk.repeat({b, 1, 1}).contiguous();
  auto length = torch::full({b}, INPUT_T, i64).contiguous();

  auto base_clc = torch::arange(static_cast<int64_t>(LAYERS) * ATT_LEFT * D_MODEL, f32)
                      .reshape({LAYERS, 1, ATT_LEFT, D_MODEL})
                      .mul(1.0e-6)
                      .contiguous();
  auto clc = base_clc.repeat({1, b, 1, 1}).contiguous();

  auto base_clt = torch::arange(static_cast<int64_t>(LAYERS) * D_MODEL * TIME_CACHE, f32)
                      .reshape({LAYERS, 1, D_MODEL, TIME_CACHE})
                      .mul(1.0e-5)
                      .contiguous();
  auto clt = base_clt.repeat({1, b, 1, 1}).contiguous();

  auto clcl = torch::full({b}, CACHE_LEN_AFTER_FIRST, i64).contiguous();
  return {chunk, length, clc, clt, clcl};
}

static std::vector<at::Tensor> run_loader(AOTIModelPackageLoader& loader,
                                          const std::vector<at::Tensor>& inputs,
                                          c10::cuda::CUDAStream stream) {
  auto out = loader.run(inputs, reinterpret_cast<void*>(stream.stream()));
  if (out.size() < 5) throw std::runtime_error("steady AOTI encoder returned fewer than 5 outputs");
  return out;
}

static at::Tensor select_row(const std::vector<at::Tensor>& out, int output_index, int64_t row) {
  switch (output_index) {
    case 0:
      return out[0].select(0, row);
    case 1:
      return out[1].select(0, row);
    case 2:
      return out[2].select(1, row);
    case 3:
      return out[3].select(1, row);
    case 4:
      return out[4].select(0, row);
    default:
      throw std::runtime_error("invalid output index");
  }
}

static bool compare_rows_to_b1(int batch,
                               const std::vector<at::Tensor>& out,
                               const std::vector<at::Tensor>& ref,
                               double atol,
                               double rtol) {
  bool ok = true;
  double max_abs[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  bool per_output_ok[5] = {true, true, true, true, true};

  for (int output_index = 0; output_index < 5; ++output_index) {
    for (int64_t row = 0; row < batch; ++row) {
      auto got = select_row(out, output_index, row);
      auto expected = select_row(ref, output_index, 0);
      if (got.sizes() != expected.sizes()) {
        per_output_ok[output_index] = false;
        ok = false;
        std::printf("  B=%d row=%lld %s shape mismatch got=%s ref=%s\n",
                    batch,
                    static_cast<long long>(row),
                    OUT_NAMES[output_index],
                    sizes_str(got).c_str(),
                    sizes_str(expected).c_str());
        continue;
      }
      if (got.is_floating_point()) {
        auto diff = (got.to(torch::kFloat32) - expected.to(torch::kFloat32)).abs();
        double d = diff.numel() > 0 ? diff.max().item<double>() : 0.0;
        max_abs[output_index] = std::max(max_abs[output_index], d);
        bool close = at::allclose(got, expected, rtol, atol);
        per_output_ok[output_index] = per_output_ok[output_index] && close;
        ok = ok && close;
      } else {
        bool equal = at::equal(got, expected);
        per_output_ok[output_index] = per_output_ok[output_index] && equal;
        ok = ok && equal;
      }
    }
  }

  std::printf("CORRECTNESS B=%d vs B=1: ok=%d atol=%.1e rtol=%.1e "
              "enc_out_max=%.3e cache_ch_max=%.3e cache_t_max=%.3e "
              "enc_len_ok=%d cache_len_ok=%d\n",
              batch,
              static_cast<int>(ok),
              atol,
              rtol,
              max_abs[0],
              max_abs[2],
              max_abs[3],
              static_cast<int>(per_output_ok[1]),
              static_cast<int>(per_output_ok[4]));
  return ok;
}

static std::vector<double> measure_case(BenchCase& bench_case,
                                        c10::cuda::CUDAStream stream,
                                        int warmup,
                                        int iters) {
  for (int i = 0; i < warmup; ++i) {
    auto out = run_loader(*bench_case.loader, bench_case.inputs, stream);
    (void)out;
  }
  CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  std::vector<double> total_ms;
  total_ms.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
    auto out = run_loader(*bench_case.loader, bench_case.inputs, stream);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    total_ms.push_back(static_cast<double>(elapsed));
    (void)out;
  }

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  return total_ms;
}

int main(int argc, char** argv) {
  try {
    torch::NoGradGuard no_grad;
    Args args = parse_args(argc, argv);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) throw std::runtime_error("CUDA is not available");

    torch::Device device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(device.index());
    auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, /*device=*/device.index());
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    const std::string shared_weights_path = resolve_shared_weights_path(args.dir, args.shared_weights);
    auto shared_constants = bsteady_detail::load_shared_constants(shared_weights_path, device);

    std::printf("=== STEADY_BATCH_BENCH START dir=%s warmup=%d iters=%d geometry=B x %d x %d "
                "cache_ch=%d x B x %d x %d cache_t=%d x B x %d x %d shared_weights=%s ===\n",
                args.dir.c_str(),
                args.warmup,
                args.iters,
                MELS,
                INPUT_T,
                LAYERS,
                ATT_LEFT,
                D_MODEL,
                LAYERS,
                D_MODEL,
                TIME_CACHE,
                shared_weights_path.c_str());

    std::vector<BenchCase> cases;
    for (int batch : {1, 2, 4}) {
      fs::path pkg = fs::path(args.dir) / ("enc_steady_aoti_b" + std::to_string(batch) + ".pt2");
      if (!fs::exists(pkg)) throw std::runtime_error("missing package: " + pkg.string());
      BenchCase c;
      c.batch = batch;
      c.package_path = pkg.string();
      c.loader = std::make_unique<AOTIModelPackageLoader>(
          c.package_path, "model", /*run_single_threaded=*/false, /*num_runners=*/1, /*device_index=*/device.index());
      auto bucket_constants = bsteady_detail::constants_for_bucket(shared_constants, *c.loader, c.package_path);
      c.loader->load_constants(bucket_constants.values, false, false, /*user_managed=*/true);
      c.inputs = make_inputs(batch, device);
      cases.push_back(std::move(c));
      std::printf("loaded B=%d package=%s constants=%zu direct=%zu alias=%zu\n",
                  batch,
                  pkg.c_str(),
                  bucket_constants.values.size(),
                  bucket_constants.direct_matches,
                  bucket_constants.alias_fallbacks);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

    auto ref_out = run_loader(*cases[0].loader, cases[0].inputs, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    bool correctness_ok = true;
    for (auto& c : cases) {
      auto out = run_loader(*c.loader, c.inputs, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
      correctness_ok = compare_rows_to_b1(c.batch, out, ref_out, args.atol, args.rtol) && correctness_ok;
    }
    if (!correctness_ok) {
      std::printf("=== STEADY_BATCH_BENCH FAIL: correctness check failed ===\n");
      return 1;
    }

    for (auto& c : cases) {
      auto samples = measure_case(c, stream, args.warmup, args.iters);
      c.total_ms = summarize(samples);
      std::vector<double> per_row;
      per_row.reserve(samples.size());
      for (double v : samples) per_row.push_back(v / static_cast<double>(c.batch));
      c.per_row_ms = summarize(per_row);
      std::printf("TIMING B=%d total_ms mean=%.4f p50=%.4f min=%.4f max=%.4f "
                  "per_row_ms mean=%.4f p50=%.4f min=%.4f max=%.4f\n",
                  c.batch,
                  c.total_ms.mean,
                  c.total_ms.p50,
                  c.total_ms.min,
                  c.total_ms.max,
                  c.per_row_ms.mean,
                  c.per_row_ms.p50,
                  c.per_row_ms.min,
                  c.per_row_ms.max);
    }

    double b1 = cases[0].per_row_ms.p50;
    double ratio_b2 = cases[1].per_row_ms.p50 / b1;
    double ratio_b4 = cases[2].per_row_ms.p50 / b1;
    bool b2_pass = ratio_b2 <= 0.80;
    bool b4_pass = ratio_b4 <= 0.60;
    bool perf_go = b2_pass && b4_pass;

    std::printf("RATIO B=2 per_row_p50_ms=%.4f ratio_vs_B1=%.3f threshold=0.800 pass=%d\n",
                cases[1].per_row_ms.p50,
                ratio_b2,
                static_cast<int>(b2_pass));
    std::printf("RATIO B=4 per_row_p50_ms=%.4f ratio_vs_B1=%.3f threshold=0.600 pass=%d\n",
                cases[2].per_row_ms.p50,
                ratio_b4,
                static_cast<int>(b4_pass));
    std::printf("=== STEADY_BATCH_BENCH RESULT correctness=PASS perf_signal=%s "
                "B2_ratio=%.3f B4_ratio=%.3f ===\n",
                perf_go ? "GO" : "STOP",
                ratio_b2,
                ratio_b4);
    return (!perf_go && args.fail_on_perf) ? 2 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "steady_batch_bench error: %s\n", e.what());
    return 1;
  }
}
