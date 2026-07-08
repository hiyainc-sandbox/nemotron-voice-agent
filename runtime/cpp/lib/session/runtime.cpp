#include "lib/session/runtime.h"

#include "lib/runtime_io/jit_load.h"
#include "lib/runtime_io/prewarm.h"
#include "lib/runtime_io/steady_batch_dir.h"
#include "lib/runtime_io/tmp_hygiene.h"
#include "lib/scheduler/batched_steady_scheduler.h"
#include "lib/session/first_encoder.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;

namespace {

using FinalizeBucketKey = std::pair<int64_t, int64_t>;

class InferenceLane;

thread_local InferenceLane* current_inference_lane = nullptr;
std::atomic<int> g_active_shared_scheduler_owners{0};
std::atomic<int> g_active_shared_steady_loader_sets{0};

void runtime_cuda_check(cudaError_t err, const char* expr) {
  if (err == cudaSuccess) return;
  throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(err));
}

void runtime_cuda_warn(cudaError_t err, const char* expr) noexcept {
  if (err == cudaSuccess) return;
  std::fprintf(stderr, "%s failed during cleanup: %s\n", expr, cudaGetErrorString(err));
}

torch::jit::Module load_module_on_device(const std::string& path, torch::Device device);

int parse_positive_env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' ||
      value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(name) + " must be a positive integer: " + raw);
  }
  return static_cast<int>(value);
}

bool parse_enabled_env(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::strcmp(raw, "0") != 0;
}

bool parse_on_env(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) return false;
  std::string value(raw);
  size_t first = 0;
  while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
  size_t last = value.size();
  while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
  value = value.substr(first, last - first);
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value == "1" || value == "true" || value == "on";
}

size_t gpu_used_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  runtime_cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
}

double bytes_to_mib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

class InferenceLane {
 public:
  InferenceLane(int id, torch::Device device, const std::string& artifact_dir)
      : id_(id), device_(device) {
    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_check(cudaStreamCreateWithFlags(&raw_stream_, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags(inference_lane)");
    stream_.emplace(c10::cuda::getStreamFromExternal(raw_stream_, device_.index()));
    preproc_ = std::make_unique<torch::jit::Module>(
        load_module_on_device((fs::path(artifact_dir) / "preproc.ts").string(), device_));
    joint_ = std::make_unique<torch::jit::Module>(
        load_module_on_device((fs::path(artifact_dir) / "joint_step.ts").string(), device_));
    predict_ = std::make_unique<torch::jit::Module>(
        load_module_on_device((fs::path(artifact_dir) / "predict_step.ts").string(), device_));
    worker_ = std::thread([this]() { worker_loop(); });
  }

  ~InferenceLane() {
    close();
    destroy_stream();
  }

  InferenceLane(const InferenceLane&) = delete;
  InferenceLane& operator=(const InferenceLane&) = delete;

  int id() const noexcept { return id_; }
  bool warmed() const noexcept { return warmed_.load(std::memory_order_acquire); }

  bool mark_warmed() noexcept {
    bool expected = false;
    return warmed_.compare_exchange_strong(expected,
                                           true,
                                           std::memory_order_acq_rel,
                                           std::memory_order_acquire);
  }

  c10::cuda::CUDAStream stream() const {
    if (!stream_.has_value()) throw std::runtime_error("inference lane stream has not been initialized");
    return *stream_;
  }

  torch::jit::Module& joint() const { return *joint_; }
  torch::jit::Module& predict() const { return *predict_; }
  torch::jit::Module& preproc() const { return *preproc_; }

  ExecutionContext execution_context() const {
    return {stream(), joint(), predict(), preproc()};
  }

  void synchronize() const {
    runtime_cuda_check(cudaStreamSynchronize(stream().stream()), "cudaStreamSynchronize(inference_lane)");
  }

  template <class F>
  auto submit(F&& f, double* queue_wait_ms = nullptr)
      -> std::future<std::invoke_result_t<std::decay_t<F>&>> {
    if (current_inference_lane == this) {
      throw std::runtime_error("nested inference lane run is not allowed");
    }
    using Fn = std::decay_t<F>;
    using R = std::invoke_result_t<Fn&>;
    auto enqueued_at = std::chrono::steady_clock::now();
    auto task = std::make_shared<std::packaged_task<R()>>(
        [fn = Fn(std::forward<F>(f)), queue_wait_ms, enqueued_at]() mutable -> R {
          if (queue_wait_ms != nullptr) {
            *queue_wait_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - enqueued_at).count();
          }
          if constexpr (std::is_void_v<R>) {
            fn();
          } else {
            return fn();
          }
        });
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (closed_) throw std::runtime_error("inference lane is closed");
      tasks_.emplace_back([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

  template <class F>
  auto run(F&& f, double* queue_wait_ms = nullptr) -> std::invoke_result_t<std::decay_t<F>&> {
    using Fn = std::decay_t<F>;
    using R = std::invoke_result_t<Fn&>;
    auto future = submit(std::forward<F>(f), queue_wait_ms);
    if constexpr (std::is_void_v<R>) {
      future.get();
    } else {
      return future.get();
    }
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      closed_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) worker_.join();
  }

 private:
  void worker_loop() {
    current_inference_lane = this;
    torch::NoGradGuard no_grad;
    c10::cuda::CUDAGuard device_guard(device_.index());
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this]() { return closed_ || !tasks_.empty(); });
        if (closed_ && tasks_.empty()) break;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      c10::cuda::CUDAStreamGuard stream_guard(stream());
      task();
    }
    current_inference_lane = nullptr;
  }

  void destroy_stream() noexcept {
    if (raw_stream_ == nullptr) return;
    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_warn(cudaStreamSynchronize(raw_stream_), "cudaStreamSynchronize(inference_lane)");
    runtime_cuda_warn(cudaStreamDestroy(raw_stream_), "cudaStreamDestroy(inference_lane)");
    raw_stream_ = nullptr;
    stream_.reset();
  }

  int id_ = 0;
  torch::Device device_;
  cudaStream_t raw_stream_ = nullptr;
  std::optional<c10::cuda::CUDAStream> stream_;
  std::unique_ptr<torch::jit::Module> preproc_;
  std::unique_ptr<torch::jit::Module> joint_;
  std::unique_ptr<torch::jit::Module> predict_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool closed_ = false;
  std::thread worker_;
  std::atomic<bool> warmed_{false};
};

class SharedSchedulerOwnership {
 public:
  SharedSchedulerOwnership() = default;
  SharedSchedulerOwnership(const SharedSchedulerOwnership&) = delete;
  SharedSchedulerOwnership& operator=(const SharedSchedulerOwnership&) = delete;

  ~SharedSchedulerOwnership() {
    reset();
  }

  void register_loader_set() {
    int count = g_active_shared_steady_loader_sets.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (count != 1) {
      g_active_shared_steady_loader_sets.fetch_sub(1, std::memory_order_acq_rel);
      throw std::runtime_error("duplicate BatchedSteadyLoaderSet owner detected: active_loader_sets=" +
                               std::to_string(count));
    }
    loader_set_registered_ = true;
  }

  void register_scheduler() {
    int count = g_active_shared_scheduler_owners.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (count != 1) {
      g_active_shared_scheduler_owners.fetch_sub(1, std::memory_order_acq_rel);
      throw std::runtime_error("duplicate BatchedSteadyScheduler owner detected: active_schedulers=" +
                               std::to_string(count));
    }
    scheduler_registered_ = true;
  }

  int active_loader_sets() const {
    return g_active_shared_steady_loader_sets.load(std::memory_order_acquire);
  }

  int active_schedulers() const {
    return g_active_shared_scheduler_owners.load(std::memory_order_acquire);
  }

 private:
  void reset() noexcept {
    if (scheduler_registered_) {
      g_active_shared_scheduler_owners.fetch_sub(1, std::memory_order_acq_rel);
      scheduler_registered_ = false;
    }
    if (loader_set_registered_) {
      g_active_shared_steady_loader_sets.fetch_sub(1, std::memory_order_acq_rel);
      loader_set_registered_ = false;
    }
  }

  bool loader_set_registered_ = false;
  bool scheduler_registered_ = false;
};

struct WarmupInput {
  std::string label;
  std::vector<float> audio;
};

double unix_now_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

int validate_finalize_silence_ms(int value) {
  if (value < 0 || value >= 10000) {
    throw std::runtime_error("finalize_silence_ms must be in [0,10000)");
  }
  return value;
}

std::string parent_dir(const std::string& path) {
  fs::path p(path);
  if (p.has_parent_path()) return p.parent_path().string();
  return ".";
}

std::string artifact_dir_from_config(const SharedRuntimeConfig& cfg) {
  if (!cfg.steady_artifacts_dir.empty()) return cfg.steady_artifacts_dir;
  if (!cfg.bundle_path.empty()) return parent_dir(cfg.bundle_path);
  return "../artifacts";
}

std::string bundle_path_from_config(const SharedRuntimeConfig& cfg, const std::string& artifact_dir) {
  if (!cfg.bundle_path.empty()) return cfg.bundle_path;
  return (fs::path(artifact_dir) / "session_audio_bundle.ts").string();
}

std::string finalize_buckets_dir_from_config(const SharedRuntimeConfig& cfg, const std::string& artifact_dir) {
  if (!cfg.finalize_buckets_dir.empty()) return cfg.finalize_buckets_dir;
  std::string stripped = (fs::path(artifact_dir) / "stripped_finalize_buckets").string();
  if (directory_exists(stripped)) return stripped;
  return (fs::path(artifact_dir) / "finalize_buckets").string();
}

torch::jit::Module load_module_on_device(const std::string& path, torch::Device device) {
  auto module = load_jit_serialized(path);
  module.to(device);
  module.eval();
  return module;
}

bool steady_batch_dir_has_packages(const std::string& dir, std::string* error = nullptr) {
  return runtime_io::steady_batch_dir_has_declared_packages(dir, error);
}

std::string resolve_steady_batch_dir(const std::string& artifact_dir, const std::string& configured) {
  if (!configured.empty()) {
    std::string error;
    if (steady_batch_dir_has_packages(configured, &error)) return configured;
    throw std::runtime_error("scheduler_enabled requested but steady batch artifacts invalid: " + error);
  }
  std::vector<std::string> candidates;
  fs::path artifact_path(artifact_dir);
  if (artifact_path.has_parent_path()) {
    candidates.push_back((artifact_path.parent_path() / "steady_b_artifacts").string());
  }
  candidates.push_back("steady_b_artifacts");
  candidates.push_back("../steady_b_artifacts");
  candidates.push_back("runtime/steady_b_artifacts");
  for (const auto& candidate : candidates) {
    if (steady_batch_dir_has_packages(candidate)) return candidate;
  }
  throw std::runtime_error("scheduler_enabled requested but steady batch artifacts were not found");
}

struct FinalizeLoaderMemoryRecord {
  int64_t drop = 0;
  int64_t T = 0;
  int num_runners = 0;
  size_t used_before = 0;
  size_t used_after = 0;
  size_t delta = 0;
  size_t cumulative_delta = 0;
};

class SharedEncoderConstants {
 public:
  SharedEncoderConstants(const std::string& weights_ts_path,
                         torch::Device device,
                         std::function<void(const char*)> cold_phase = {}) {
    c10::cuda::CUDAGuard device_guard(device.index());
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before shared encoder constants)");
    size_t before = gpu_used_bytes();
    constants_ = load_shared_constants(weights_ts_path, device);
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after shared encoder constants)");
    size_t after = gpu_used_bytes();
    delta_bytes_ = after >= before ? after - before : 0;
    std::printf("shared encoder constants loaded: entries=%zu shared_delta_mib=%.3f source=%s\n",
                constants_.size(),
                bytes_to_mib(delta_bytes_),
                weights_ts_path.c_str());
    std::fflush(stdout);
    if (cold_phase) cold_phase("shared_encoder_constants_load");
  }

  const std::unordered_map<std::string, at::Tensor>& constants() const noexcept { return constants_; }
  size_t delta_bytes() const noexcept { return delta_bytes_; }

 private:
  std::unordered_map<std::string, at::Tensor> constants_;
  size_t delta_bytes_ = 0;
};

class FinalizeBucketLoaderPool final : public FinalizeBucketLoaderProvider {
 public:
  FinalizeBucketLoaderPool(const std::string& buckets_dir,
                           const std::string& shared_weights,
                           const std::string& shared_weights_pt,
                           torch::Device device,
                           int num_runners,
                           std::string policy,
                           const std::unordered_map<std::string, at::Tensor>* borrowed_shared_constants = nullptr,
                           std::function<void(const char*)> cold_phase = {})
      : buckets_dir_(buckets_dir),
        shared_weights_(shared_weights),
        device_(device),
        num_runners_(num_runners),
        policy_(std::move(policy)) {
    if (num_runners_ <= 0) throw std::runtime_error("finalize_num_runners must be positive");
    if (!directory_exists(buckets_dir_)) {
      throw std::runtime_error("finalize buckets directory missing: " + buckets_dir_);
    }
    if (!file_exists(shared_weights_)) {
      throw std::runtime_error("finalize shared weights missing: " + shared_weights_);
    }

    bucket_paths_ = discover_finalize_buckets(buckets_dir_);
    if (bucket_paths_.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir_);
    std::string manifest_path = (fs::path(buckets_dir_) / "manifest.json").string();
    if (!file_exists(manifest_path)) {
      throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
    }
    manifest_ = load_bucket_manifest(manifest_path);
    const ManifestShaVerifyMode sha_mode = manifest_sha_verify_mode_from_env();
    verify_bucket_manifest(manifest_, bucket_paths_, buckets_dir_, shared_weights_pt, sha_mode);
    std::printf("runtime finalize manifest verified: buckets=%zu weights_sha256=%s sha_mode=%s "
                "env=NEMOTRON_WS_VERIFY_MANIFEST_SHA num_runners=%d policy=%s\n",
                manifest_.buckets.size(),
                manifest_.contract.weights_sha256.c_str(),
                manifest_sha_verify_mode_name(sha_mode),
                num_runners_,
                policy_.c_str());
    if (cold_phase) cold_phase("finalize_manifest_verify");

    if (borrowed_shared_constants != nullptr) {
      if (borrowed_shared_constants->empty()) {
        throw std::runtime_error("finalize borrowed shared constants map is empty");
      }
      shared_constants_ptr_ = borrowed_shared_constants;
      borrowed_shared_constants_ = true;
      shared_delta_bytes_ = 0;
    } else {
      c10::cuda::CUDAGuard device_guard(device_.index());
      runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before finalize shared constants)");
      size_t before = gpu_used_bytes();
      shared_constants_ = load_shared_constants(shared_weights_, device_);
      runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after finalize shared constants)");
      size_t after = gpu_used_bytes();
      shared_delta_bytes_ = after >= before ? after - before : 0;
      shared_constants_ptr_ = &shared_constants_;
      if (cold_phase) cold_phase("finalize_shared_constants_load");
    }
    std::printf("runtime finalize shared constants ready: entries=%zu shared_delta_mib=%.3f source=%s policy=%s\n",
                shared_constants_ref().size(),
                bytes_to_mib(shared_delta_bytes_),
                borrowed_shared_constants_ ? "borrowed" : "owned",
                policy_.c_str());
    std::fflush(stdout);
  }

  AOTIModelPackageLoader& get(int64_t drop, int64_t T) override {
    std::lock_guard<std::mutex> lock(mu_);
    auto key = std::make_pair(drop, T);
    auto it = loaders_.find(key);
    if (it != loaders_.end()) return *it->second;
    auto loaded = load_bucket_locked(key);
    return *loaded->second;
  }

  void preload_all() {
    for (const auto& kv : bucket_paths_) {
      (void)get(kv.first.first, kv.first.second);
    }
  }

  std::vector<FinalizeBucketKey> bucket_keys() const {
    std::vector<FinalizeBucketKey> keys;
    keys.reserve(bucket_paths_.size());
    for (const auto& kv : bucket_paths_) keys.push_back(kv.first);
    return keys;
  }

  int num_runners() const noexcept { return num_runners_; }
  size_t total_bucket_count() const noexcept { return bucket_paths_.size(); }

  size_t loaded_bucket_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return loaders_.size();
  }

  size_t shared_delta_bytes() const noexcept { return shared_delta_bytes_; }

  const std::unordered_map<std::string, at::Tensor>& shared_constants() const noexcept {
    return shared_constants_ref();
  }

  size_t total_loader_delta_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_loader_delta_bytes_;
  }

  std::string memory_json() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t projected_all = 0;
    if (!records_.empty()) {
      long double mean = static_cast<long double>(total_loader_delta_bytes_) /
                         static_cast<long double>(records_.size());
      projected_all = static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()));
    }
    std::ostringstream oss;
    oss << "{\"policy\":\"" << policy_ << "\""
        << ",\"num_runners_per_loaded_bucket\":" << num_runners_
        << ",\"total_manifest_buckets\":" << bucket_paths_.size()
        << ",\"loaded_buckets\":" << loaders_.size()
        << ",\"shared_constants_delta_bytes\":" << shared_delta_bytes_
        << ",\"loader_delta_bytes\":" << total_loader_delta_bytes_
        << ",\"projected_all_buckets_same_runner_delta_bytes\":" << projected_all
        << ",\"records\":[";
    for (size_t i = 0; i < records_.size(); ++i) {
      const auto& r = records_[i];
      if (i > 0) oss << ",";
      oss << "{\"drop\":" << r.drop
          << ",\"T\":" << r.T
          << ",\"num_runners\":" << r.num_runners
          << ",\"used_before_bytes\":" << r.used_before
          << ",\"used_after_bytes\":" << r.used_after
          << ",\"delta_bytes\":" << r.delta
          << ",\"cumulative_delta_bytes\":" << r.cumulative_delta
          << "}";
    }
    oss << "]}";
    return oss.str();
  }

 private:
  const std::unordered_map<std::string, at::Tensor>& shared_constants_ref() const noexcept {
    return *shared_constants_ptr_;
  }

  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>>::iterator load_bucket_locked(
      const FinalizeBucketKey& key) {
    auto path_it = bucket_paths_.find(key);
    if (path_it == bucket_paths_.end()) {
      throw std::runtime_error("runtime finalize missing bucket drop=" +
                               std::to_string(key.first) +
                               " T=" + std::to_string(key.second));
    }

    c10::cuda::CUDAGuard device_guard(device_.index());
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before finalize bucket load)");
    size_t before = gpu_used_bytes();
    auto loader = std::make_unique<AOTIModelPackageLoader>(
        path_it->second, "model", false, num_runners_, device_.index());
    auto bucket_constants = constants_for_bucket(shared_constants_ref(), *loader, path_it->second);
    loader->load_constants(bucket_constants.values, false, false, true);
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after finalize bucket load)");
    size_t after = gpu_used_bytes();
    size_t delta = after >= before ? after - before : 0;
    total_loader_delta_bytes_ += delta;
    records_.push_back({
        key.first,
        key.second,
        num_runners_,
        before,
        after,
        delta,
        total_loader_delta_bytes_,
    });
    std::printf("runtime finalize bucket loaded: drop=%ld T=%ld constants=%zu direct=%zu alias=%zu "
                "num_runners=%d loader_delta_mib=%.3f cumulative_loader_mib=%.3f policy=%s\n",
                static_cast<long>(key.first),
                static_cast<long>(key.second),
                bucket_constants.values.size(),
                bucket_constants.direct_matches,
                bucket_constants.alias_fallbacks,
                num_runners_,
                bytes_to_mib(delta),
                bytes_to_mib(total_loader_delta_bytes_),
                policy_.c_str());
    std::fflush(stdout);
    auto inserted = loaders_.emplace(key, std::move(loader));
    return inserted.first;
  }

  std::string buckets_dir_;
  std::string shared_weights_;
  torch::Device device_;
  int num_runners_ = 1;
  std::string policy_;
  BucketManifest manifest_;
  std::map<FinalizeBucketKey, std::string> bucket_paths_;
  std::unordered_map<std::string, at::Tensor> shared_constants_;
  const std::unordered_map<std::string, at::Tensor>* shared_constants_ptr_ = nullptr;
  bool borrowed_shared_constants_ = false;
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> loaders_;
  mutable std::mutex mu_;
  std::vector<FinalizeLoaderMemoryRecord> records_;
  size_t shared_delta_bytes_ = 0;
  size_t total_loader_delta_bytes_ = 0;
};

std::vector<float> pcm_to_float(const PCMFrame& frame) {
  if (frame.count > 0 && frame.samples == nullptr) {
    throw std::runtime_error("PCMFrame samples is null with non-zero count");
  }
  std::vector<float> out;
  out.reserve(frame.count);
  for (size_t i = 0; i < frame.count; ++i) {
    out.push_back(static_cast<float>(frame.samples[i]) / 32768.0f);
  }
  return out;
}

std::vector<float> tensor_to_float_vector(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous().reshape({-1});
  std::vector<float> out(static_cast<size_t>(flat.numel()));
  if (!out.empty()) {
    std::memcpy(out.data(), flat.data_ptr<float>(), out.size() * sizeof(float));
  }
  return out;
}

std::optional<WarmupInput> make_bucket_warmup_input(const AudioGeometry& audio_geometry,
                                                    int64_t drop,
                                                    int64_t final_t) {
  int64_t audio_frames = -1;
  if (drop == 0) {
    audio_frames = final_t - FINAL_PADDING_FRAMES - 1;
    if (audio_frames <= 0 || audio_frames >= SHIFT + 1) return std::nullopt;
  } else if (drop == DROP) {
    constexpr int64_t kWarmupSteadyChunks = 2;
    const int64_t final_t_offset =
        PRE + FINAL_PADDING_FRAMES + 1 - kWarmupSteadyChunks * SHIFT;
    const int64_t min_audio_frames = kWarmupSteadyChunks * SHIFT + 1;
    const int64_t next_chunk_audio_frames = (kWarmupSteadyChunks + 1) * SHIFT + 1;
    audio_frames = final_t - final_t_offset;
    if (audio_frames < min_audio_frames || audio_frames >= next_chunk_audio_frames) {
      return std::nullopt;
    }
    const int64_t second_chunk_pending =
        (audio_frames - SHIFT) * audio_geometry.hop_samples;
    if (second_chunk_pending < audio_geometry.preprocess_new_audio_samples) {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }

  const int64_t audio_samples = audio_frames * audio_geometry.hop_samples;
  if (audio_samples <= 0) return std::nullopt;
  if (drop == DROP && audio_samples < audio_geometry.preprocess_new_audio_samples) {
    return std::nullopt;
  }
  WarmupInput input;
  input.label = "bucket.drop" + std::to_string(drop) + ".T" + std::to_string(final_t);
  input.audio.assign(static_cast<size_t>(audio_samples), 0.0f);
  return input;
}

std::vector<WarmupInput> make_bucket_warmup_inputs(
    const AudioGeometry& audio_geometry,
    const std::vector<FinalizeBucketKey>& finalize_bucket_keys) {
  std::vector<WarmupInput> inputs;
  inputs.reserve(finalize_bucket_keys.size());
  for (const auto& key : finalize_bucket_keys) {
    auto input = make_bucket_warmup_input(audio_geometry, key.first, key.second);
    if (input.has_value()) inputs.push_back(std::move(*input));
  }
  return inputs;
}

std::optional<WarmupInput> make_fixture_warmup_input(torch::jit::Module& bundle) {
  try {
    int64_t rows = scalar_i64(attr_tensor(bundle, "num_utts"));
    int best_utt = -1;
    int best_score = std::numeric_limits<int>::min();
    int64_t best_samples = std::numeric_limits<int64_t>::max();
    for (int64_t utt = 0; utt < rows; ++utt) {
      int64_t final_t = scalar_i64(utt_tensor(bundle, static_cast<int>(utt), "final_T"));
      if (final_t <= 0) continue;
      int64_t steady = scalar_i64(utt_tensor(bundle, static_cast<int>(utt), "num_steady"));
      auto audio = utt_tensor(bundle, static_cast<int>(utt), "audio");
      int64_t samples = audio.numel();
      int score = steady >= 2 ? 2 : (steady >= 1 ? 1 : 0);
      if (score > best_score || (score == best_score && samples < best_samples)) {
        best_utt = static_cast<int>(utt);
        best_score = score;
        best_samples = samples;
      }
    }
    if (best_utt < 0) return std::nullopt;
    WarmupInput input;
    input.label = "utt" + std::to_string(best_utt);
    input.audio = tensor_to_float_vector(utt_tensor(bundle, best_utt, "audio"));
    return input;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// Wire-layer language projection for the prompted (multilingual) profile:
// session-layer event text stays raw (byte-exact vs the Python oracle), the
// wire strips <xx-XX> tags and carries the resolved language, mirroring
// server.py::_transcript_payload. Identity for the en profile.
void project_language(WireEvent& wire, const std::string& raw_text, SessionState& state) {
  if (!MODEL_PROMPTED) {
    wire.text = raw_text;
    return;
  }
  std::string lang = last_lang_tag(raw_text);
  if (!lang.empty()) state.last_language = lang;
  wire.text = strip_lang_tags_text(raw_text);
  const std::string& tag = !lang.empty()
                               ? lang
                               : (!state.last_language.empty() ? state.last_language : state.language);
  if (!tag.empty()) wire.language = tag;
}

std::vector<WireEvent> project_events(const std::vector<EmittedEvent>& events,
                                      const std::optional<SessionTiming>& final_timing,
                                      SessionState& state,
                                      std::string& last_wire_interim_text) {
  std::vector<WireEvent> out;
  out.reserve(events.size());
  for (const auto& event : events) {
    if (event.kind == EVENT_SUPPRESSED) continue;
    WireEvent wire;
    wire.type = "transcript";
    project_language(wire, event.text, state);
    if (event.kind == EVENT_INTERIM) {
      wire.is_final = false;
      if (MODEL_PROMPTED) {
        // A tag-only hypothesis change must not surface as a duplicate interim.
        if (wire.text.value_or("") == last_wire_interim_text) continue;
        last_wire_interim_text = wire.text.value_or("");
      }
    } else if (event.kind == EVENT_FINAL) {
      wire.is_final = true;
      wire.finalize = true;
      if (final_timing.has_value()) wire.finalize_timing = final_timing->to_wire_json();
    } else {
      continue;
    }
    out.push_back(std::move(wire));
  }
  return out;
}

bool has_final_event(const std::vector<EmittedEvent>& events) {
  return std::any_of(events.begin(), events.end(), [](const EmittedEvent& event) {
    return event.kind == EVENT_FINAL;
  });
}

}  // namespace

struct SharedRuntime::Impl {
  explicit Impl(SharedRuntimeConfig config)
      : cfg(std::move(config)),
        artifact_dir(artifact_dir_from_config(cfg)),
        bundle_path(bundle_path_from_config(cfg, artifact_dir)),
        finalize_buckets_dir(finalize_buckets_dir_from_config(cfg, artifact_dir)),
        device(torch::kCUDA, cfg.device_index) {
    runtime_io::reclaim_stale_aoti_tmp_dirs();

    // Cold-start phase timing; aggregate lines intentionally overlap their sub-phases.
    using ColdStartClock = std::chrono::steady_clock;
    const auto cold_start_t0 = ColdStartClock::now();
    auto cold_phase_prev = cold_start_t0;
    auto cold_phase = [&](const char* name) {
      const auto now = ColdStartClock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - cold_phase_prev).count();
      const double cum = std::chrono::duration<double, std::milli>(now - cold_start_t0).count();
      std::printf("COLD_START_PHASE phase=%s elapsed_ms=%.1f cumulative_ms=%.1f\n", name, ms, cum);
      std::fflush(stdout);
      cold_phase_prev = now;
    };
    auto cold_phase_total = [&](const char* name, ColdStartClock::time_point start) {
      const auto now = ColdStartClock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - start).count();
      const double cum = std::chrono::duration<double, std::milli>(now - cold_start_t0).count();
      std::printf("COLD_START_PHASE phase=%s elapsed_ms=%.1f cumulative_ms=%.1f aggregate=1\n",
                  name,
                  ms,
                  cum);
      std::fflush(stdout);
      cold_phase_prev = now;
    };

    runtime_io::Prewarmer prewarmer;
    const bool enc_first_ts = parse_enabled_env("NEMOTRON_WS_ENC_FIRST_TS", true);
    if (parse_enabled_env("NEMOTRON_WS_PREWARM", true)) {
      std::vector<std::string> prewarm_paths{
          (fs::path(artifact_dir) / "finalize_shared_weights.ts").string()};
      if (enc_first_ts) {
        prewarm_paths.push_back((fs::path(artifact_dir) / "enc_first.ts").string());
      }
      prewarmer.start(prewarm_paths);
      const std::string queued_paths = prewarmer.queued_paths_csv();
      std::printf("PREWARM kicked files=%zu bytes=%llu paths=%s\n",
                  prewarmer.queued_file_count(),
                  static_cast<unsigned long long>(prewarmer.queued_bytes()),
                  queued_paths.c_str());
      std::fflush(stdout);
      cold_phase("prewarm_kickoff");
    }

    cfg.steady_dispatch_lanes =
        parse_positive_env_int("NEMOTRON_DENSITY_DISPATCH_LANES", cfg.steady_dispatch_lanes);
    if (cfg.steady_num_runners <= 0) throw std::runtime_error("steady_num_runners must be positive");
    if (cfg.steady_dispatch_lanes <= 0 || cfg.steady_dispatch_lanes > 2) {
      throw std::runtime_error("steady_dispatch_lanes must be in [1,2]");
    }
    if (cfg.scheduler_enabled && cfg.steady_dispatch_lanes > 1 &&
        parse_on_env("NEMOTRON_WS_STEADY_CUDAGRAPH")) {
      throw std::runtime_error(
          "NEMOTRON_DENSITY_DISPATCH_LANES > 1 is incompatible with NEMOTRON_WS_STEADY_CUDAGRAPH=1");
    }
    if (cfg.scheduler_enabled && cfg.steady_num_runners < cfg.steady_dispatch_lanes) {
      std::printf("shared runtime N1 auto-raise: steady_num_runners %d -> %d "
                  "for steady_dispatch_lanes=%d model=shared_loader_num_runners\n",
                  cfg.steady_num_runners,
                  cfg.steady_dispatch_lanes,
                  cfg.steady_dispatch_lanes);
      cfg.steady_num_runners = cfg.steady_dispatch_lanes;
    }
    background_warmup_enabled =
        parse_enabled_env("NEMOTRON_WS_BACKGROUND_WARMUP", cfg.background_warmup_enabled);
    if (background_warmup_enabled && cfg.warm_sync_lanes <= 0) {
      throw std::runtime_error("warm_sync_lanes must be positive when background warmup is enabled");
    }
    if (background_warmup_enabled) {
      warm_sync_lanes_requested =
          parse_positive_env_int("NEMOTRON_WS_WARM_SYNC_LANES", cfg.warm_sync_lanes);
    }
    torch::NoGradGuard ng;
    try {

    bundle = load_jit_serialized(bundle_path);
    verify_session_bundle_meta(bundle, false);
    tokenizer_value = tokenizer_from_bundle(bundle);
    if (cfg.verify_tokenizer) verify_tokenizer_selftest(bundle, tokenizer_value);
    initialize_prompt_runtime(artifact_dir, bundle, device);

    audio_geometry = session_runtime_audio_geometry_from_bundle(bundle);
    std::string preproc_path = (fs::path(artifact_dir) / "preproc.ts").string();
    session_runtime_verify_preproc_manifest(artifact_dir, preproc_path, audio_geometry);
    cold_phase("bundle_tokenizer_preproc");

    // Fail fast on the invalid shadow-without-scheduler config BEFORE doing any
    // expensive inline/finalize loads (shadow compares scheduler vs inline, so it
    // needs the scheduler). Mirrors the upstream ws_server.cpp:706 guard.
    if (cfg.steady_shadow_enabled && !cfg.scheduler_enabled) {
      throw std::runtime_error("NEMOTRON_WS_STEADY_SHADOW requires NEMOTRON_WS_SCHEDULER=1");
    }
    shared_encoder_constants_ = std::make_unique<SharedEncoderConstants>(
        (fs::path(artifact_dir) / "finalize_shared_weights.ts").string(),
        device,
        cold_phase);
    if (enc_first_ts) {
      enc_first_ts_module_ =
          load_module_on_device((fs::path(artifact_dir) / "enc_first.ts").string(), device);
      first_encoder_ = std::make_unique<TsFirstEncoder>(enc_first_ts_module_);
    } else {
      first_encoder_ = std::make_unique<AotiFirstEncoder>(
          (fs::path(artifact_dir) / "enc_first_aoti.pt2").string(),
          shared_encoder_constants_->constants(),
          device,
          cfg.steady_num_runners);
    }
    std::printf("shared runtime first encoder ready: adapter=%s env_NEMOTRON_WS_ENC_FIRST_TS=%s\n",
                first_encoder_->kind(),
                enc_first_ts ? "1" : "0");
    std::fflush(stdout);
    cold_phase("enc_first_load");
    const bool need_inline_at_startup = !cfg.scheduler_enabled || cfg.steady_shadow_enabled;
    if (need_inline_at_startup) {
      (void)ensure_inline_enc_steady();
      cold_phase("enc_steady_load");
    }

    const auto finalize_preload_t0 = ColdStartClock::now();
    finalize_loaders = std::make_unique<FinalizeBucketLoaderPool>(
        finalize_buckets_dir,
        (fs::path(artifact_dir) / "finalize_shared_weights.ts").string(),
        (fs::path(artifact_dir) / "finalize_shared_weights.pt").string(),
        device,
        cfg.finalize_num_runners,
        "ws_shared_finalize_pool",
        &shared_encoder_constants_->constants(),
        cold_phase);
    finalize_loaders->preload_all();
    cold_phase("finalize_bucket_bind_dlopen");
    std::printf("shared finalize loader pool ready: num_runners=%d loaded_buckets=%zu/%zu "
                "shared_constants_mib=%.3f loader_mib=%.3f memory=%s\n",
                finalize_loaders->num_runners(),
                finalize_loaders->loaded_bucket_count(),
                finalize_loaders->total_bucket_count(),
                bytes_to_mib(finalize_loaders->shared_delta_bytes()),
                bytes_to_mib(finalize_loaders->total_loader_delta_bytes()),
                finalize_loaders->memory_json().c_str());
    std::fflush(stdout);
    cold_phase_total("finalize_preload", finalize_preload_t0);

    build_inference_lanes();
    cold_phase("lane_build");

    if (cfg.scheduler_enabled) {
      const auto scheduler_preload_t0 = ColdStartClock::now();
      std::string batch_dir = resolve_steady_batch_dir(artifact_dir, cfg.steady_batch_dir);
      scheduler_ownership.register_loader_set();
      BatchedSteadySchedulerPolicy policy;
      policy.B_max = cfg.b_max;
      policy.window_ms = cfg.batch_window_ms;
      policy.lone_timeout_ms = cfg.batch_lone_timeout_ms;
      policy.max_queue_delay_ms = cfg.batch_max_queue_delay_ms;
      policy.queue_capacity = cfg.batch_queue_capacity;
      policy.min_fill_enabled = cfg.batch_min_fill_enabled;
      policy.disable_min_fill = cfg.batch_disable_min_fill;
      policy.force_bucket = cfg.batch_force_bucket;
      policy.dispatch_lanes = cfg.steady_dispatch_lanes;
      const auto required_scheduler_buckets = BatchedSteadyScheduler::required_buckets_for_policy(policy);
      cold_phase("scheduler_preload_setup");
      const auto& borrowed_shared_constants = shared_encoder_constants_->constants();
      batched_steady = std::make_unique<BatchedSteadyLoaderSet>(
          batch_dir,
          (fs::path(artifact_dir) / "finalize_shared_weights.ts").string(),
          device,
          cfg.steady_num_runners,
          "shared_runtime_scheduler",
          required_scheduler_buckets,
          &borrowed_shared_constants,
          cold_phase);
      batched_steady->preload_buckets(required_scheduler_buckets);
      cold_phase("scheduler_bucket_bind_dlopen");
      cold_phase_total("scheduler_preload", scheduler_preload_t0);
      scheduler_ownership.register_scheduler();
      scheduler = std::make_unique<BatchedSteadyScheduler>(*batched_steady, device, policy);
      scheduler->warmup_buckets();
      scheduler->start();
      cold_phase("scheduler_warmup_start");
      std::printf("shared runtime scheduler owner ready: owner=SharedRuntime scheduler_instances=%d "
                  "steady_loader_sets=%d warmup_complete=true dispatcher_started=true "
                  "steady_dispatch_lanes=%d steady_num_runners=%d runner_model=shared_loader_num_runners\n",
                  scheduler_ownership.active_schedulers(),
                  scheduler_ownership.active_loader_sets(),
                  cfg.steady_dispatch_lanes,
                  cfg.steady_num_runners);
      std::fflush(stdout);
      if (cfg.steady_shadow_enabled) {
        std::printf("steady shadow parity enabled: env=NEMOTRON_WS_STEADY_SHADOW commit=inline "
                    "compare=scheduler_vs_inline tolerance_name=B2_A1_PARITY tolerance=5.000000e-02 "
                    "timing=INVALID\n");
        std::fflush(stdout);
      }
    } else {
      std::printf("shared runtime scheduler disabled: owner=SharedRuntime scheduler_instances=0 "
                  "steady_loader_sets=0\n");
      std::fflush(stdout);
    }
    warm_inference_lanes();
    prewarmer.join();
    cold_phase("lane_warmup");
    if (background_warmup_enabled) {
      launch_pending_background_warmup(cold_start_t0);
      cold_phase("sync_warm_done");
    }
    } catch (...) {
      stop_background_warmup();
      throw;
    }
  }

  ~Impl() {
    session_runtime_print_steady_shadow_report();
    stop_background_warmup();
    if (scheduler) scheduler->close();
    {
      std::lock_guard<std::mutex> lock(lanes_mu);
      lanes_closing = true;
    }
    lanes_cv.notify_all();
    for (auto& lane : lanes) {
      if (lane) lane->close();
    }
  }

  AOTIModelPackageLoader& ensure_inline_enc_steady() {
    std::call_once(enc_steady_once, [this]() {
      const auto t0 = std::chrono::steady_clock::now();
      enc_steady = std::make_unique<AOTIModelPackageLoader>(
          (fs::path(artifact_dir) / "enc_steady_aoti.pt2").string(),
          "model",
          false,
          cfg.steady_num_runners,
          device.index());
      const auto now = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - t0).count();
      const char* reason = !cfg.scheduler_enabled
                               ? "scheduler_off"
                               : (cfg.steady_shadow_enabled ? "steady_shadow" : "lazy");
      std::printf("INLINE_ENC_STEADY_LAZY_LOAD elapsed_ms=%.1f reason=%s\n", ms, reason);
      std::fflush(stdout);
    });
    return *enc_steady;
  }

  AOTIModelPackageLoader* inline_enc_steady_ptr() {
    if (cfg.scheduler_enabled && !cfg.steady_shadow_enabled) return nullptr;
    return &ensure_inline_enc_steady();
  }

  void build_inference_lanes() {
    lane_count = parse_positive_env_int("NEMOTRON_WS_LANES", 1);
    c10::cuda::CUDAGuard device_guard(device.index());
    const size_t used_before_lanes = gpu_used_bytes();
    lanes.reserve(static_cast<size_t>(lane_count));
    for (int lane_id = 0; lane_id < lane_count; ++lane_id) {
      lanes.push_back(std::make_unique<InferenceLane>(lane_id, device, artifact_dir));
    }
    runtime_cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after inference lanes)");
    const size_t used_after_lanes = gpu_used_bytes();
    lane_delta_bytes = used_after_lanes >= used_before_lanes
                           ? used_after_lanes - used_before_lanes
                           : 0;
    lane_delta_per_lane_bytes = lane_count > 0
                                    ? lane_delta_bytes / static_cast<size_t>(lane_count)
                                    : 0;
    constexpr size_t kBigModuleDuplicationThresholdBytes = 512ull * 1024ull * 1024ull;
    if (lane_delta_per_lane_bytes > kBigModuleDuplicationThresholdBytes) {
      throw std::runtime_error("inference lane memory delta suggests big-module duplication: per_lane_mib=" +
                               std::to_string(bytes_to_mib(lane_delta_per_lane_bytes)));
    }
    const char* inline_status = enc_steady ? "loaded" : "skipped";
    const bool first_encoder_ts = first_encoder_ != nullptr &&
                                  std::string(first_encoder_->kind()) == "ts";
    std::string shared_big_modules;
    if (first_encoder_ts && enc_steady) {
      shared_big_modules = "enc_first_ts,enc_steady,finalize_loaders";
    } else if (first_encoder_ts) {
      shared_big_modules = "enc_first_ts,finalize_loaders";
    } else if (enc_steady) {
      shared_big_modules = "enc_steady,finalize_loaders";
    } else {
      shared_big_modules = "finalize_loaders";
    }
    std::printf("inference lane pool built: lanes=%d per_lane_mib=%.3f total_lane_mib=%.3f "
                "no_big_module_duplication=true shared_big_modules=%s inline_enc_steady=%s "
                "first_encoder=%s\n",
                lane_count,
                bytes_to_mib(lane_delta_per_lane_bytes),
                bytes_to_mib(lane_delta_bytes),
                shared_big_modules.c_str(),
                inline_status,
                first_encoder_ != nullptr ? first_encoder_->kind() : "unset");
    std::fflush(stdout);
  }

  InferenceLane& acquire_lane(const std::string& label) {
    std::unique_lock<std::mutex> lock(lanes_mu);
    lanes_cv.wait(lock, [this]() { return lanes_closing || !free_lanes.empty(); });
    if (lanes_closing) throw std::runtime_error("inference lane pool is closing");
    int lane_id = free_lanes.front();
    free_lanes.pop_front();
    InferenceLane& lane = *lanes.at(static_cast<size_t>(lane_id));
    if (!lane.warmed()) {
      throw std::runtime_error("attempted to acquire unwarmed inference lane: lane=" +
                               std::to_string(lane_id));
    }
    if (background_warmup_enabled) {
      std::printf("inference lane acquired: lane=%d label=%s warmed_lanes=%llu\n",
                  lane_id,
                  label.c_str(),
                  static_cast<unsigned long long>(
                      warmed_lanes.load(std::memory_order_acquire)));
    } else {
      std::printf("inference lane acquired: lane=%d label=%s\n", lane_id, label.c_str());
    }
    std::fflush(stdout);
    return lane;
  }

  void release_lane(InferenceLane* lane) noexcept {
    if (lane == nullptr) return;
    {
      std::lock_guard<std::mutex> lock(lanes_mu);
      if (!lanes_closing && lane->warmed()) free_lanes.push_back(lane->id());
    }
    lanes_cv.notify_one();
  }

  struct LaneWarmupContext {
    static constexpr int kFullWarmupItersPerInput = 5;

    std::shared_ptr<const std::vector<WarmupInput>> full_warmup_inputs;
    std::shared_ptr<const std::vector<WarmupInput>> tail_warmup_inputs;
    std::shared_ptr<const std::vector<FinalizeBucketKey>> tail_direct_finalize_warmup_keys;
    size_t bucket_warmup_inputs = 0;
    int full_warm_lanes = 0;
    int tail_warmup_iters = 1;
    int tail_warmup_input_env = 1;
  };

  struct LaneWarmupResult {
    int lane_id = 0;
    int completed = 0;
    int iters_per_input = 0;
    size_t input_count = 0;
    size_t direct_finalize_count = 0;
    bool full = false;
  };

  LaneWarmupContext build_lane_warmup_context() {
    const auto finalize_bucket_keys = finalize_loaders->bucket_keys();
    std::vector<WarmupInput> inputs;
    inputs.reserve(finalize_bucket_keys.size());
    std::vector<std::optional<FinalizeBucketKey>> input_bucket_keys;
    input_bucket_keys.reserve(finalize_bucket_keys.size());
    std::vector<size_t> steady_input_indices;
    steady_input_indices.reserve(finalize_bucket_keys.size());
    for (const auto& key : finalize_bucket_keys) {
      auto input = make_bucket_warmup_input(audio_geometry, key.first, key.second);
      if (!input.has_value()) continue;
      if (key.first == DROP) steady_input_indices.push_back(inputs.size());
      inputs.push_back(std::move(*input));
      input_bucket_keys.push_back(key);
    }
    const size_t bucket_warmup_inputs = inputs.size();
    if (inputs.empty()) {
      auto fixture = make_fixture_warmup_input(bundle);
      if (fixture.has_value()) {
        inputs.push_back(std::move(*fixture));
        input_bucket_keys.push_back(std::nullopt);
      }
    }
    if (inputs.empty()) {
      throw std::runtime_error("unable to build inference lane warmup input");
    }

    const int full_warm_lanes =
        std::min(lane_count, parse_positive_env_int("NEMOTRON_WS_WARM_FULL_LANES", 1));
    const int tail_warmup_iters = parse_positive_env_int("NEMOTRON_WS_WARM_TAIL_ITERS", 1);
    const int tail_warmup_input_env = parse_positive_env_int("NEMOTRON_WS_WARM_TAIL_INPUTS", 1);
    const size_t full_warmup_input_count = inputs.size();
    const size_t tail_warmup_input_count =
        std::min(full_warmup_input_count, static_cast<size_t>(tail_warmup_input_env));
    std::vector<WarmupInput> tail_inputs;
    tail_inputs.reserve(tail_warmup_input_count);
    std::vector<FinalizeBucketKey> tail_input_bucket_keys;
    tail_input_bucket_keys.reserve(tail_warmup_input_count);
    std::vector<bool> selected_tail_input(full_warmup_input_count, false);
    auto select_tail_input = [&](size_t input_index) {
      tail_inputs.push_back(inputs.at(input_index));
      if (input_bucket_keys.at(input_index).has_value()) {
        tail_input_bucket_keys.push_back(*input_bucket_keys.at(input_index));
      }
      selected_tail_input.at(input_index) = true;
    };
    if (tail_warmup_input_count > 0 && !steady_input_indices.empty()) {
      select_tail_input(steady_input_indices.front());
    }
    for (size_t input_index = 0;
         input_index < inputs.size() && tail_inputs.size() < tail_warmup_input_count;
         ++input_index) {
      if (selected_tail_input[input_index]) continue;
      select_tail_input(input_index);
    }
    std::vector<FinalizeBucketKey> tail_direct_finalize_keys;
    tail_direct_finalize_keys.reserve(finalize_bucket_keys.size());
    for (const auto& key : finalize_bucket_keys) {
      bool represented_drop = false;
      for (const auto& tail_key : tail_input_bucket_keys) {
        if (tail_key.first == key.first) {
          represented_drop = true;
          break;
        }
      }
      if (!represented_drop) continue;

      bool warmed_by_tail_input = false;
      for (const auto& tail_key : tail_input_bucket_keys) {
        if (tail_key == key) {
          warmed_by_tail_input = true;
          break;
        }
      }
      if (!warmed_by_tail_input) tail_direct_finalize_keys.push_back(key);
    }
    const auto full_warmup_inputs =
        std::make_shared<const std::vector<WarmupInput>>(std::move(inputs));
    const auto tail_warmup_inputs =
        std::make_shared<const std::vector<WarmupInput>>(std::move(tail_inputs));
    const auto tail_direct_finalize_warmup_keys =
        std::make_shared<const std::vector<FinalizeBucketKey>>(std::move(tail_direct_finalize_keys));

    std::printf("inference lane warmup dispatch: lanes=%d full_lanes=%d full_iters_per_input=%d "
                "tail_iters_per_input=%d full_warmup_inputs=%zu tail_warmup_inputs=%zu "
                "tail_requested_inputs=%d tail_direct_finalize_buckets=%zu finalize_buckets=%zu\n",
                lane_count,
                full_warm_lanes,
                LaneWarmupContext::kFullWarmupItersPerInput,
                tail_warmup_iters,
                full_warmup_inputs->size(),
                tail_warmup_inputs->size(),
                tail_warmup_input_env,
                tail_direct_finalize_warmup_keys->size(),
                finalize_loaders->total_bucket_count());
    std::fflush(stdout);

    LaneWarmupContext warmup;
    warmup.full_warmup_inputs = std::move(full_warmup_inputs);
    warmup.tail_warmup_inputs = std::move(tail_warmup_inputs);
    warmup.tail_direct_finalize_warmup_keys = std::move(tail_direct_finalize_warmup_keys);
    warmup.bucket_warmup_inputs = bucket_warmup_inputs;
    warmup.full_warm_lanes = full_warm_lanes;
    warmup.tail_warmup_iters = tail_warmup_iters;
    warmup.tail_warmup_input_env = tail_warmup_input_env;
    return warmup;
  }

  bool mark_lane_warmed(int lane_id) {
    InferenceLane& lane = *lanes.at(static_cast<size_t>(lane_id));
    if (!lane.mark_warmed()) return false;
    warmed_lanes.fetch_add(1, std::memory_order_acq_rel);
    return true;
  }

  void publish_warmed_lane(int lane_id) {
    InferenceLane& lane = *lanes.at(static_cast<size_t>(lane_id));
    if (!lane.warmed()) {
      throw std::runtime_error("attempted to publish unwarmed inference lane: lane=" +
                               std::to_string(lane_id));
    }
    {
      std::lock_guard<std::mutex> lock(lanes_mu);
      if (!lanes_closing) free_lanes.push_back(lane_id);
    }
    lanes_cv.notify_all();
  }

  std::future<LaneWarmupResult> submit_lane_warmup(size_t lane_index,
                                                   const LaneWarmupContext& warmup,
                                                   bool publish_when_done) {
    InferenceLane& lane = *lanes.at(lane_index);
    const bool full = static_cast<int>(lane_index) < warmup.full_warm_lanes;
    const int lane_warmup_iters =
        full ? LaneWarmupContext::kFullWarmupItersPerInput : warmup.tail_warmup_iters;
    const auto lane_warmup_inputs =
        full ? warmup.full_warmup_inputs : warmup.tail_warmup_inputs;
    const auto tail_direct_finalize_warmup_keys = warmup.tail_direct_finalize_warmup_keys;
    return lane.submit([this,
                        &lane,
                        warmup_inputs = lane_warmup_inputs,
                        tail_direct_finalize_warmup_keys,
                        warmup_iters = lane_warmup_iters,
                        full,
                        publish_when_done]() {
      int completed = 0;
      for (const auto& warmup_input : *warmup_inputs) {
        for (int iter = 0; iter < warmup_iters; ++iter) {
          SessionState warm_state;
          reset_session(warm_state, bundle, device);
          auto warm_audio = make_session_runtime_audio_frontend(bundle, lane.preproc(), device);
          reset_session_runtime_audio_front(warm_state, *warm_audio);

          std::vector<EmittedEvent> events;
          const std::string label = "inference_lane" + std::to_string(lane.id()) +
                                    ".warmup." + warmup_input.label +
                                    ".iter" + std::to_string(iter);
          auto ctx = lane.execution_context();
          {
            std::lock_guard<std::mutex> enc_first_lock(enc_first_mutex);
            BatchedSteadyScheduler* steady_scheduler = cfg.scheduler_enabled ? scheduler.get() : nullptr;
            (void)session_runtime_append_pcm_and_drain(warm_state,
                                                       warmup_input.audio,
                                                       *warm_audio,
                                                       *first_encoder_,
                                                       inline_enc_steady_ptr(),
                                                       ctx,
                                                       device,
                                                       tokenizer_value,
                                                       events,
                                                       label + ".append",
                                                       steady_scheduler,
                                                       cfg.steady_shadow_enabled);
          }
          vad_stop(warm_state);
          (void)session_runtime_finalize(warm_state,
                                         bundle,
                                         *warm_audio,
                                         *finalize_loaders,
                                         ctx,
                                         device,
                                         tokenizer_value,
                                         events,
                                         FinalizeFinish::SPECULATIVE_KEEP,
                                         label + ".finalize");
          ++completed;
        }
      }
      size_t direct_finalize_count = 0;
      if (!full && !tail_direct_finalize_warmup_keys->empty()) {
        c10::cuda::CUDAGuard device_guard(device.index());
        c10::cuda::CUDAStreamGuard stream_guard(lane.stream());
        SessionState direct_state;
        reset_session(direct_state, bundle, device);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        for (const auto& key : *tail_direct_finalize_warmup_keys) {
          auto final_chunk = torch::zeros({1, 128, key.second}, options);
          std::vector<at::Tensor> loader_inputs = {
              final_chunk.contiguous(),
              direct_state.clc.contiguous(),
              direct_state.clt.contiguous(),
              direct_state.clcl.contiguous(),
          };
          AOTIModelPackageLoader& loader = finalize_loaders->get(key.first, key.second);
          auto out = loader.run(loader_inputs, reinterpret_cast<void*>(lane.stream().stream()));
          if (out.size() < 2) {
            throw std::runtime_error("tail finalize warmup bucket returned fewer than 2 outputs");
          }
          ++direct_finalize_count;
        }
      }
      lane.synchronize();
      const bool first_mark = mark_lane_warmed(lane.id());
      if (publish_when_done && first_mark) publish_warmed_lane(lane.id());
      return LaneWarmupResult{
          lane.id(), completed, warmup_iters, warmup_inputs->size(), direct_finalize_count, full};
    });
  }

  void log_lane_warmed(const LaneWarmupResult& result) {
    if (background_warmup_enabled) {
      std::printf("inference lane warmed: lane=%d mode=%s iters=%d iters_per_input=%d "
                  "warmup_inputs=%zu direct_finalize_buckets=%zu finalize_buckets=%zu "
                  "warmed_lanes=%llu\n",
                  result.lane_id,
                  result.full ? "full" : "tail",
                  result.completed,
                  result.iters_per_input,
                  result.input_count,
                  result.direct_finalize_count,
                  finalize_loaders->total_bucket_count(),
                  static_cast<unsigned long long>(
                      warmed_lanes.load(std::memory_order_acquire)));
    } else {
      std::printf("inference lane warmed: lane=%d mode=%s iters=%d iters_per_input=%d "
                  "warmup_inputs=%zu direct_finalize_buckets=%zu finalize_buckets=%zu\n",
                  result.lane_id,
                  result.full ? "full" : "tail",
                  result.completed,
                  result.iters_per_input,
                  result.input_count,
                  result.direct_finalize_count,
                  finalize_loaders->total_bucket_count());
    }
    std::fflush(stdout);
  }

  int collect_lane_warmups(std::vector<std::future<LaneWarmupResult>>& futures) {
    int total_warmed = 0;
    std::exception_ptr first_exception;
    for (auto& future : futures) {
      try {
        LaneWarmupResult result = future.get();
        total_warmed += result.completed;
        log_lane_warmed(result);
      } catch (...) {
        if (!first_exception) first_exception = std::current_exception();
      }
    }
    if (first_exception) std::rethrow_exception(first_exception);
    return total_warmed;
  }

  int warm_lane_range(size_t begin_lane,
                      size_t end_lane,
                      const LaneWarmupContext& warmup,
                      bool publish_when_done) {
    std::vector<std::future<LaneWarmupResult>> futures;
    futures.reserve(end_lane - begin_lane);
    for (size_t lane_index = begin_lane; lane_index < end_lane; ++lane_index) {
      futures.push_back(submit_lane_warmup(lane_index, warmup, publish_when_done));
    }
    return collect_lane_warmups(futures);
  }

  void print_lane_pool_warmed(const LaneWarmupContext& warmup, int total_warmed) {
    std::printf("inference lane pool warmed: lanes=%d total_iters=%d full_warmup_inputs=%zu "
                "tail_warmup_inputs=%zu tail_requested_inputs=%d "
                "tail_direct_finalize_buckets=%zu "
                "finalize_bucket_coverage=%zu/%zu full_lanes=%d full_iters_per_input=%d "
                "tail_iters_per_input=%d per_lane_mib=%.3f\n",
                lane_count,
                total_warmed,
                warmup.full_warmup_inputs->size(),
                warmup.tail_warmup_inputs->size(),
                warmup.tail_warmup_input_env,
                warmup.tail_direct_finalize_warmup_keys->size(),
                warmup.bucket_warmup_inputs,
                finalize_loaders->total_bucket_count(),
                warmup.full_warm_lanes,
                LaneWarmupContext::kFullWarmupItersPerInput,
                warmup.tail_warmup_iters,
                bytes_to_mib(lane_delta_per_lane_bytes));
    std::fflush(stdout);
  }

  void print_background_warm_complete(std::chrono::steady_clock::time_point cold_start_t0,
                                      std::chrono::steady_clock::time_point background_t0) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(now - background_t0).count();
    const double cumulative_ms = std::chrono::duration<double, std::milli>(now - cold_start_t0).count();
    std::printf("COLD_START_PHASE phase=background_warm_complete elapsed_ms=%.1f "
                "cumulative_ms=%.1f background=1 warmed_lanes=%llu lanes=%d\n",
                elapsed_ms,
                cumulative_ms,
                static_cast<unsigned long long>(warmed_lanes.load(std::memory_order_acquire)),
                lane_count);
    std::fflush(stdout);
  }

  void start_background_warmup(LaneWarmupContext warmup,
                               size_t begin_lane,
                               int already_warmed_iters,
                               std::chrono::steady_clock::time_point cold_start_t0) {
    if (begin_lane >= lanes.size()) {
      print_lane_pool_warmed(warmup, already_warmed_iters);
      print_background_warm_complete(cold_start_t0, std::chrono::steady_clock::now());
      return;
    }
    background_warmup_stop.store(false, std::memory_order_release);
    const auto background_t0 = std::chrono::steady_clock::now();
    background_warmup_thread = std::thread([this, warmup = std::move(warmup), begin_lane,
                                            already_warmed_iters, cold_start_t0, background_t0]() {
      int total_warmed = already_warmed_iters;
      try {
        if (!background_warmup_stop.load(std::memory_order_acquire)) {
          total_warmed += warm_lane_range(begin_lane, lanes.size(), warmup, true);
        }
        if (!background_warmup_stop.load(std::memory_order_acquire)) {
          print_lane_pool_warmed(warmup, total_warmed);
          print_background_warm_complete(cold_start_t0, background_t0);
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "inference lane background warmup failed: %s\n", e.what());
        std::fflush(stderr);
      } catch (...) {
        std::fprintf(stderr, "inference lane background warmup failed: unknown exception\n");
        std::fflush(stderr);
      }
    });
  }

  void launch_pending_background_warmup(std::chrono::steady_clock::time_point cold_start_t0) {
    if (!pending_background_warmup.has_value()) return;
    start_background_warmup(std::move(*pending_background_warmup),
                            pending_background_begin_lane,
                            pending_background_warmup_iters,
                            cold_start_t0);
    pending_background_warmup.reset();
    pending_background_begin_lane = 0;
    pending_background_warmup_iters = 0;
  }

  void stop_background_warmup() {
    background_warmup_stop.store(true, std::memory_order_release);
    if (background_warmup_thread.joinable()) background_warmup_thread.join();
  }

  void warm_inference_lanes() {
    LaneWarmupContext warmup = build_lane_warmup_context();
    if (!background_warmup_enabled) {
      const int total_warmed = warm_lane_range(0, lanes.size(), warmup, false);
      for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
        publish_warmed_lane(static_cast<int>(lane_index));
      }
      print_lane_pool_warmed(warmup, total_warmed);
      return;
    }

    const int sync_lanes = std::min(lane_count, warm_sync_lanes_requested);
    std::printf("inference lane background warmup enabled: lanes=%d sync_lanes=%d "
                "remaining_lanes=%d warmed_lanes=%llu\n",
                lane_count,
                sync_lanes,
                lane_count - sync_lanes,
                static_cast<unsigned long long>(warmed_lanes.load(std::memory_order_acquire)));
    std::fflush(stdout);

    const int sync_total_warmed =
        warm_lane_range(0, static_cast<size_t>(sync_lanes), warmup, false);
    for (int lane_id = 0; lane_id < sync_lanes; ++lane_id) {
      publish_warmed_lane(lane_id);
    }
    std::printf("inference lane sync warmup done: sync_lanes=%d total_iters=%d warmed_lanes=%llu "
                "configured_lanes=%d\n",
                sync_lanes,
                sync_total_warmed,
                static_cast<unsigned long long>(warmed_lanes.load(std::memory_order_acquire)),
                lane_count);
    std::fflush(stdout);

    pending_background_begin_lane = static_cast<size_t>(sync_lanes);
    pending_background_warmup_iters = sync_total_warmed;
    pending_background_warmup = std::move(warmup);
  }

  SharedRuntimeConfig cfg;
  std::string artifact_dir;
  std::string bundle_path;
  std::string finalize_buckets_dir;
  torch::Device device;
  // Declared before every user-managed constants borrower so reverse destruction
  // keeps the shared encoder constants alive until all loaders are gone.
  std::unique_ptr<SharedEncoderConstants> shared_encoder_constants_;
  torch::jit::Module bundle;
  Tokenizer tokenizer_value;
  // TsFirstEncoder references enc_first_ts_module_; AotiFirstEncoder borrows
  // shared_encoder_constants_. Reverse destruction therefore drops
  // first_encoder_ before either referenced owner.
  torch::jit::Module enc_first_ts_module_;
  std::unique_ptr<FirstEncoder> first_encoder_;
  std::mutex enc_first_mutex;
  std::unique_ptr<AOTIModelPackageLoader> enc_steady;
  std::once_flag enc_steady_once;
  // Finalize and scheduler loaders both borrow shared_encoder_constants_; this
  // declaration order still destroys scheduler loaders before finalize loaders.
  std::unique_ptr<FinalizeBucketLoaderPool> finalize_loaders;
  AudioGeometry audio_geometry;
  int lane_count = 0;
  size_t lane_delta_bytes = 0;
  size_t lane_delta_per_lane_bytes = 0;
  std::vector<std::unique_ptr<InferenceLane>> lanes;
  std::mutex lanes_mu;
  std::condition_variable lanes_cv;
  std::deque<int> free_lanes;
  std::atomic<uint64_t> warmed_lanes{0};
  bool lanes_closing = false;
  bool background_warmup_enabled = true;
  int warm_sync_lanes_requested = 4;
  std::atomic<bool> background_warmup_stop{false};
  std::thread background_warmup_thread;
  std::optional<LaneWarmupContext> pending_background_warmup;
  size_t pending_background_begin_lane = 0;
  int pending_background_warmup_iters = 0;
  SharedSchedulerOwnership scheduler_ownership;
  std::unique_ptr<BatchedSteadyLoaderSet> batched_steady;
  std::unique_ptr<BatchedSteadyScheduler> scheduler;
};

SharedRuntime::SharedRuntime(SharedRuntimeConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}

SharedRuntime::~SharedRuntime() = default;

const Tokenizer& SharedRuntime::tokenizer() const {
  return impl_->tokenizer_value;
}

const SharedRuntimeConfig& SharedRuntime::config() const {
  return impl_->cfg;
}

bool SharedRuntime::has_scheduler() const noexcept {
  return impl_ && impl_->scheduler != nullptr;
}

uint64_t SharedRuntime::warmed_lane_count() const noexcept {
  return impl_ ? impl_->warmed_lanes.load(std::memory_order_acquire) : 0;
}

BatchedSteadySchedulerTelemetry SharedRuntime::scheduler_telemetry_snapshot() const {
  if (!impl_ || !impl_->scheduler) {
    throw std::runtime_error("SharedRuntime scheduler telemetry requested without scheduler");
  }
  return impl_->scheduler->telemetry_snapshot();
}

struct SessionRuntime::Impl {
  struct LaneLease {
    SharedRuntime::Impl* owner = nullptr;
    InferenceLane* lane = nullptr;

    LaneLease() = default;
    LaneLease(const LaneLease&) = delete;
    LaneLease& operator=(const LaneLease&) = delete;

    ~LaneLease() {
      reset();
    }

    void acquire(SharedRuntime::Impl& owner_in, const std::string& label) {
      reset();
      owner = &owner_in;
      lane = &owner->acquire_lane(label);
    }

    void reset() noexcept {
      if (owner != nullptr && lane != nullptr) {
        owner->release_lane(lane);
      }
      owner = nullptr;
      lane = nullptr;
    }

    InferenceLane& get() const {
      if (lane == nullptr) throw std::runtime_error("session is not bound to an inference lane");
      return *lane;
    }
  };

  struct SteadyTimingAccumulator {
    RuntimeSteadyTiming phases;
    double lane_queue_wait_ms = 0.0;
    uint64_t lane_queue_wait_count = 0;

    void add(const RuntimeSteadyTiming& sample, std::optional<double> lane_wait_ms) {
      if (!sample.has_any()) return;
      phases.merge(sample);
      if (lane_wait_ms.has_value()) {
        lane_queue_wait_ms += *lane_wait_ms;
        ++lane_queue_wait_count;
      }
    }

    void apply_to(SessionTiming& timing) const {
      if (lane_queue_wait_count > 0) timing.lane_queue_wait_ms = lane_queue_wait_ms;
      if (phases.preproc_count > 0) timing.preproc_ms = phases.preproc_ms;
      if (phases.scheduler_enqueue_wait_count > 0) {
        timing.scheduler_enqueue_wait_ms = phases.scheduler_enqueue_wait_ms;
      }
      if (phases.scheduler_future_wait_count > 0) {
        timing.scheduler_future_wait_ms = phases.scheduler_future_wait_ms;
      }
      if (phases.scheduler_completion_wait_count > 0) {
        timing.scheduler_completion_wait_ms = phases.scheduler_completion_wait_ms;
      }
      if (phases.decode_count > 0) timing.decode_ms = phases.decode_ms;
    }

    void reset() {
      phases = RuntimeSteadyTiming{};
      lane_queue_wait_ms = 0.0;
      lane_queue_wait_count = 0;
    }
  };

  Impl(const SharedRuntime& shared_in, SessionConfig config)
      : shared(shared_in),
        cfg(std::move(config)),
        finalize_silence_ms(validate_finalize_silence_ms(cfg.finalize_silence_ms)),
        audio(nullptr, nullptr) {
    auto& s = *shared.impl_;
    lane_lease.acquire(s, cfg.label);
    InferenceLane& lane = lane_lease.get();
    try {
      lane.run([&]() {
        torch::NoGradGuard ng;
        c10::cuda::CUDAGuard device_guard(s.device.index());
        c10::cuda::CUDAStreamGuard stream_guard(lane.stream());
        reset_session(state, s.bundle, s.device);
        if (MODEL_PROMPTED) {
          const PromptTable& table = prompt_runtime_table();
          int64_t index = table.default_index;
          std::string resolved = cfg.language;
          if (!resolved.empty()) {
            auto it = table.lang_to_index.find(resolved);
            if (it == table.lang_to_index.end()) {
              throw std::runtime_error("unsupported language for prompted model: " + resolved);
            }
            index = it->second;
          } else {
            for (const auto& kv : table.lang_to_index) {
              if (kv.second == index) { resolved = kv.first; break; }
            }
          }
          state.prompt = make_prompt_onehot(index, s.device);
          state.prompt_index = index;
          state.language = resolved;
        }
        audio = make_session_runtime_audio_frontend(s.bundle, lane.preproc(), s.device);
        reset_session_runtime_audio_front(state, *audio);
        lane.synchronize();
      });
    } catch (...) {
      lane_lease.reset();
      throw;
    }
  }

  ~Impl() {
    if (lane_lease.lane != nullptr) {
      try {
        lane().run([&]() { lane().synchronize(); });
      } catch (const std::exception& e) {
        std::fprintf(stderr, "session lane cleanup failed for %s: %s\n", cfg.label.c_str(), e.what());
      }
    }
  }

  InferenceLane& lane() const {
    return lane_lease.get();
  }

  ExecutionContext execution_context() const {
    return lane().execution_context();
  }

  void synchronize_lane_stream() const {
    lane().synchronize();
  }

  std::vector<WireEvent> append_pcm(const PCMFrame& frame) {
    std::vector<float> pcm = pcm_to_float(frame);
    std::vector<EmittedEvent> events;
    auto& s = *shared.impl_;
    RuntimeSteadyTiming local_steady_timing;
    RuntimeSteadyTiming* steady_timing_sink =
        s.cfg.steady_shadow_enabled ? nullptr : &local_steady_timing;
    double lane_queue_wait_ms = 0.0;
    lane().run([&]() {
      auto ctx = execution_context();
      BatchedSteadyScheduler* steady_scheduler = s.cfg.scheduler_enabled ? s.scheduler.get() : nullptr;
      if (state.emitted == 0) {
        auto enc_first_wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> enc_first_lock(s.enc_first_mutex);
        enc_first_lock_wait_since_finalize_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enc_first_wait_start).count();
        session_runtime_append_pcm_and_drain(state,
                                             pcm,
                                             *audio,
                                             *s.first_encoder_,
                                             s.inline_enc_steady_ptr(),
                                             ctx,
                                             s.device,
                                             s.tokenizer_value,
                                             events,
                                             cfg.label + ".append",
                                             steady_scheduler,
                                             s.cfg.steady_shadow_enabled,
                                             steady_timing_sink);
      } else {
        session_runtime_append_pcm_and_drain(state,
                                             pcm,
                                             *audio,
                                             *s.first_encoder_,
                                             s.inline_enc_steady_ptr(),
                                             ctx,
                                             s.device,
                                             s.tokenizer_value,
                                             events,
                                             cfg.label + ".append",
                                             steady_scheduler,
                                             s.cfg.steady_shadow_enabled,
                                             steady_timing_sink);
      }
      synchronize_lane_stream();
    }, &lane_queue_wait_ms);
    steady_timing.add(local_steady_timing, lane_queue_wait_ms);
    debug_events.insert(debug_events.end(), events.begin(), events.end());
    return project_events(events, std::nullopt, state, last_wire_interim_text);
  }

  void vad_start() {
    vad_state = VadState::SPEAKING;
    vad_deadline_ts.reset();
    pending_timing.reset();
    std::vector<EmittedEvent> events;
    auto& s = *shared.impl_;
    RuntimeSteadyTiming local_steady_timing;
    RuntimeSteadyTiming* steady_timing_sink =
        s.cfg.steady_shadow_enabled ? nullptr : &local_steady_timing;
    double lane_queue_wait_ms = 0.0;
    lane().run([&]() {
      auto ctx = execution_context();
      BatchedSteadyScheduler* steady_scheduler = s.cfg.scheduler_enabled ? s.scheduler.get() : nullptr;
      if (state.emitted == 0) {
        auto enc_first_wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> enc_first_lock(s.enc_first_mutex);
        enc_first_lock_wait_since_finalize_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enc_first_wait_start).count();
        session_runtime_vad_start(state,
                                  *audio,
                                  *s.first_encoder_,
                                  s.inline_enc_steady_ptr(),
                                  ctx,
                                  s.device,
                                  s.tokenizer_value,
                                  events,
                                  cfg.label + ".vad_start",
                                  steady_scheduler,
                                  s.cfg.steady_shadow_enabled,
                                  steady_timing_sink);
      } else {
        session_runtime_vad_start(state,
                                  *audio,
                                  *s.first_encoder_,
                                  s.inline_enc_steady_ptr(),
                                  ctx,
                                  s.device,
                                  s.tokenizer_value,
                                  events,
                                  cfg.label + ".vad_start",
                                  steady_scheduler,
                                  s.cfg.steady_shadow_enabled,
                                  steady_timing_sink);
      }
      synchronize_lane_stream();
    }, &lane_queue_wait_ms);
    steady_timing.add(local_steady_timing, lane_queue_wait_ms);
    debug_events.insert(debug_events.end(), events.begin(), events.end());
  }

  std::vector<WireEvent> vad_stop_and_maybe_finalize() {
    SessionTiming timing;
    double now = unix_now_seconds();
    timing.reason = "debounce_expired";
    timing.vad_stop_ts = now;
    timing.gil_attrib_enabled = cfg.gil_attrib_enabled || shared.impl_->cfg.gil_attrib_enabled;
    pending_timing = timing;
    vad_stop(state);
    if (finalize_silence_ms == 0) {
      return finalize_and_idle("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
    }
    vad_state = VadState::PENDING_FINALIZE;
    vad_deadline_ts = now + static_cast<double>(finalize_silence_ms) / 1000.0;
    return {};
  }

  std::vector<WireEvent> poll_timer(double now_unix_ts) {
    if (vad_state != VadState::PENDING_FINALIZE || !vad_deadline_ts.has_value()) return {};
    if (now_unix_ts < *vad_deadline_ts) return {};
    return finalize_and_idle("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
  }

  std::vector<WireEvent> soft_final(bool finalize_flag) {
    WireEvent wire;
    wire.type = "transcript";
    project_language(wire, shared.impl_->tokenizer_value.ids_to_text(state.hyp), state);
    wire.is_final = true;
    wire.finalize = finalize_flag;
    return {std::move(wire)};
  }

  std::vector<WireEvent> finalize_with(const std::string& reason, FinalizeFinish finish) {
    auto& s = *shared.impl_;
    SessionTiming timing = pending_timing.value_or(SessionTiming{});
    timing.reason = reason;
    timing.debounce_expiry_ts = unix_now_seconds();
    timing.finalize_seq = ++finalize_seq;
    timing.active_sessions_at_emit = cfg.active_sessions_at_emit;
    timing.gil_attrib_enabled = cfg.gil_attrib_enabled || s.cfg.gil_attrib_enabled;
    timing.enc_first_lock_wait_ms = enc_first_lock_wait_since_finalize_ms;
    steady_timing.apply_to(timing);

    if (finish == FinalizeFinish::SPECULATIVE_KEEP && state.mode == SessionMode::STREAMING) {
      vad_stop(state);
      if (!timing.vad_stop_ts.has_value()) timing.vad_stop_ts = timing.debounce_expiry_ts;
    }

    std::vector<EmittedEvent> events;
    FinalizeOutcome outcome;
    timing.fork_flush_start_ts = unix_now_seconds();
    auto lane_wait_start = std::chrono::steady_clock::now();
    outcome = lane().run([&]() {
      timing.inference_lock_acquire_wait_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - lane_wait_start).count();
      auto ctx = execution_context();
      auto result = session_runtime_finalize(state,
                                             s.bundle,
                                             *audio,
                                             *s.finalize_loaders,
                                             ctx,
                                             s.device,
                                             s.tokenizer_value,
                                             events,
                                             finish,
                                             cfg.label + ".finalize");
      synchronize_lane_stream();
      return result;
    });
    timing.fork_flush_done_ts = unix_now_seconds();
    timing.final_sent_ts = unix_now_seconds();
    timing.was_suppressed = !has_final_event(events);
    last_finalize_tokens = outcome.final_tokens;
    last_timing_value = timing;
    pending_timing.reset();
    enc_first_lock_wait_since_finalize_ms = 0.0;
    steady_timing.reset();
    debug_events.insert(debug_events.end(), events.begin(), events.end());

    // SessionRuntime leaves stats emission to the WS worker, which owns the stale-generation
    // send/drop decision and records last_timing() after that emit decision.
    auto wire = project_events(events, timing, state, last_wire_interim_text);
    // A finalize must always answer with a final on the wire (the Python
    // reference server does), even when the session suppressed the event
    // because the append-only delta was empty — e.g. a zero-token utterance.
    // Otherwise a client waiting on vad_stop -> is_final hangs.
    bool has_wire_final = std::any_of(wire.begin(), wire.end(), [](const WireEvent& event) {
      return event.is_final.value_or(false);
    });
    if (!has_wire_final) {
      WireEvent finale;
      finale.type = "transcript";
      project_language(finale, s.tokenizer_value.ids_to_text(state.hyp), state);
      finale.is_final = true;
      finale.finalize = true;
      finale.finalize_timing = timing.to_wire_json();
      wire.push_back(std::move(finale));
    }
    return wire;
  }

  std::vector<WireEvent> finalize_and_idle(const std::string& reason, FinalizeFinish finish) {
    auto events = finalize_with(reason, finish);
    clear_vad_state();
    return events;
  }

  void clear_vad_state() {
    vad_state = VadState::IDLE;
    vad_deadline_ts.reset();
  }

  const SharedRuntime& shared;
  SessionConfig cfg;
  int finalize_silence_ms = 0;
  LaneLease lane_lease;
  SessionState state;
  // Last interim text emitted on the wire post tag-stripping (prompted-profile
  // interim dedup; unused for en).
  std::string last_wire_interim_text;
  RuntimeAudioFrontendPtr audio;
  VadState vad_state = VadState::IDLE;
  std::optional<double> vad_deadline_ts;
  std::optional<SessionTiming> pending_timing;
  std::optional<SessionTiming> last_timing_value;
  uint64_t finalize_seq = 0;
  double enc_first_lock_wait_since_finalize_ms = 0.0;
  SteadyTimingAccumulator steady_timing;
  std::vector<EmittedEvent> debug_events;
  std::vector<int64_t> last_finalize_tokens;
};

SessionRuntime::SessionRuntime(const SharedRuntime& shared, SessionConfig cfg)
    : impl_(std::make_unique<Impl>(shared, std::move(cfg))) {}

SessionRuntime::~SessionRuntime() {
  if (impl_) bump_generation();
}

std::vector<WireEvent> SessionRuntime::append_pcm_and_drain(const PCMFrame& frame) {
  return impl_->append_pcm(frame);
}

void SessionRuntime::handle_vad_start() {
  impl_->vad_start();
}

std::vector<WireEvent> SessionRuntime::handle_vad_stop() {
  return impl_->vad_stop_and_maybe_finalize();
}

std::vector<WireEvent> SessionRuntime::poll_timer(double now_unix_ts) {
  return impl_->poll_timer(now_unix_ts);
}

VadState SessionRuntime::vad_state() const noexcept {
  return impl_->vad_state;
}

std::optional<double> SessionRuntime::vad_deadline_ts() const noexcept {
  return impl_->vad_deadline_ts;
}

std::vector<WireEvent> SessionRuntime::reset(bool finalize) {
  bump_generation();
  if (!finalize) {
    impl_->clear_vad_state();
    impl_->enc_first_lock_wait_since_finalize_ms = 0.0;
    impl_->steady_timing.reset();
    return impl_->soft_final(false);
  }
  return impl_->finalize_and_idle("reset", FinalizeFinish::SPECULATIVE_KEEP);
}

std::vector<WireEvent> SessionRuntime::end(bool finalize) {
  bump_generation();
  if (!finalize) {
    impl_->clear_vad_state();
    impl_->enc_first_lock_wait_since_finalize_ms = 0.0;
    impl_->steady_timing.reset();
    return impl_->soft_final(false);
  }
  return impl_->finalize_and_idle("end", FinalizeFinish::TRUE_BOUNDARY_COLD_RESET);
}

std::vector<WireEvent> SessionRuntime::finalize_now() {
  return impl_->finalize_and_idle("debounce_expired", FinalizeFinish::SPECULATIVE_KEEP);
}

uint64_t SessionRuntime::generation() const noexcept {
  return impl_->state.generation.load(std::memory_order_acquire);
}

void SessionRuntime::bump_generation() noexcept {
  impl_->state.generation.fetch_add(1, std::memory_order_acq_rel);
}

std::optional<SessionTiming> SessionRuntime::last_timing() const {
  return impl_->last_timing_value;
}

std::vector<EmittedEvent> session_runtime_debug_events(const SessionRuntime& runtime) {
  return runtime.impl_->debug_events;
}

std::vector<int64_t> session_runtime_debug_last_final_tokens(const SessionRuntime& runtime) {
  return runtime.impl_->last_finalize_tokens;
}
