#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct SessionState;

namespace torch::inductor {
class AOTIModelPackageLoader;
}

class FirstEncoder {
 public:
  virtual ~FirstEncoder() = default;

  // Runs chunk-0 of the streaming encoder under the caller-held enc_first lock + given stream.
  // Returns EXACTLY 5 tensors: {enc_out (sliced [:, :, :2]), enc_len-1, cache_ch, cache_t, cache_ch_len}.
  virtual std::vector<at::Tensor> run(const torch::Tensor& chunk,
                                      SessionState& state,
                                      c10::cuda::CUDAStream stream) = 0;
  virtual const char* kind() const = 0;
};

class TsFirstEncoder final : public FirstEncoder {
 public:
  explicit TsFirstEncoder(torch::jit::Module& module);

  std::vector<at::Tensor> run(const torch::Tensor& chunk,
                              SessionState& state,
                              c10::cuda::CUDAStream stream) override;
  const char* kind() const override;

 private:
  torch::jit::Module& module_;
};

class AotiFirstEncoder final : public FirstEncoder {
 public:
  AotiFirstEncoder(const std::string& pkg_path,
                   const std::unordered_map<std::string, at::Tensor>& shared_constants,
                   torch::Device device,
                   int num_runners);
  ~AotiFirstEncoder() override;

  AotiFirstEncoder(const AotiFirstEncoder&) = delete;
  AotiFirstEncoder& operator=(const AotiFirstEncoder&) = delete;

  std::vector<at::Tensor> run(const torch::Tensor& chunk,
                              SessionState& state,
                              c10::cuda::CUDAStream stream) override;
  const char* kind() const override;

 private:
  std::string pkg_path_;
  torch::Device device_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};
