// Step 0 probe: production steady B16 AOTI overlap model.
//
// This deliberately avoids the scheduler/runtime production sources.  The
// shared-loader arm calls BatchedSteadyLoaderSet::run_raw_prepacked(); the
// dedicated-loader arm obtains per-lane handles via
// create_dedicated_loader_for_bucket() and uses the same AOTI run call shape.
#include "lib/scheduler/steady_batch_primitive.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/script.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using torch::inductor::AOTIModelPackageLoader;
namespace fs = std::filesystem;

static constexpr int kBucket = 16;
static constexpr int kShift = 16;
static constexpr int kPreEncodeCache = 9;
static constexpr int kInputT = kPreEncodeCache + kShift;
static constexpr int kMels = 128;
static constexpr int kLayers = 24;
static constexpr int kAttLeft = 70;
static constexpr int kDModel = 1024;
static constexpr int kTimeCache = 8;
static constexpr int kCacheLenAfterFirst = 3;

static const char* kOutputNames[5] = {"enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"};

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
  std::string package_dir = "steady_b_artifacts_b16";
  std::string shared_weights;
  std::string model = "both";
  int warmup = 4;
  int correctness_iters = 16;
  int timing_iters = 48;
  double atol = 5.0e-2;
  double rtol = 1.0e-3;
  bool fail_on_no_go = false;
};

struct Summary {
  double mean = 0.0;
  double p50 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

struct CompareStats {
  bool ok = true;
  int row_output_checks = 0;
  int byte_equal_checks = 0;
  std::array<double, 5> max_abs{{0.0, 0.0, 0.0, 0.0, 0.0}};
};

struct RunSample {
  double event_start_ms = 0.0;
  double event_stop_ms = 0.0;
  double event_elapsed_ms = 0.0;
  double host_ms = 0.0;
};

struct TimingResult {
  bool ok = true;
  int errors = 0;
  std::string error_text;
  double wall_ms = 0.0;
  std::vector<RunSample> lane_samples[2];
};

struct ModelResult {
  std::string model;
  bool completed = false;
  bool correctness_ok = false;
  bool scratch_isolated = false;
  bool streams_unique = false;
  bool no_errors = false;
  bool event_overlap_go = false;
  bool wall_speedup_go = false;
  bool go = false;
  int correctness_checks = 0;
  int byte_equal_checks = 0;
  int overlap_pairs = 0;
  int overlap_total_pairs = 0;
  double overlap_fraction = 0.0;
  double mean_overlap_ms = 0.0;
  double min_overlap_ms = 0.0;
  double serial_wall_ms = 0.0;
  double concurrent_wall_ms = 0.0;
  double wall_speedup = 0.0;
  Summary serial_event_ms;
  Summary concurrent_event_ms;
  std::array<double, 5> max_abs{{0.0, 0.0, 0.0, 0.0, 0.0}};
  size_t used_before_load = 0;
  size_t used_after_base_loader = 0;
  size_t used_after_model_loaders = 0;
  size_t used_after_scratch = 0;
  size_t used_after_run = 0;
  size_t peak_used = 0;
  size_t base_loader_delta = 0;
  size_t model_loader_delta = 0;
  size_t dedicated_incremental_delta = 0;
  size_t scratch_delta = 0;
  std::string error_text;
};

static Args parse_args(int argc, char** argv) {
  Args args;
  bool positional_dir_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--dir" || arg == "--package-dir") {
      args.package_dir = need(arg.c_str());
      positional_dir_set = true;
    } else if (arg == "--shared-weights") {
      args.shared_weights = need("--shared-weights");
    } else if (arg == "--model") {
      args.model = need("--model");
    } else if (arg == "--warmup") {
      args.warmup = std::stoi(need("--warmup"));
    } else if (arg == "--correctness-iters") {
      args.correctness_iters = std::stoi(need("--correctness-iters"));
    } else if (arg == "--timing-iters") {
      args.timing_iters = std::stoi(need("--timing-iters"));
    } else if (arg == "--atol") {
      args.atol = std::stod(need("--atol"));
    } else if (arg == "--rtol") {
      args.rtol = std::stod(need("--rtol"));
    } else if (arg == "--fail-on-no-go") {
      args.fail_on_no_go = true;
    } else if (!positional_dir_set) {
      args.package_dir = arg;
      positional_dir_set = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.model != "both" && args.model != "shared" && args.model != "dedicated") {
    throw std::runtime_error("--model must be one of: both, shared, dedicated");
  }
  if (args.warmup < 0) throw std::runtime_error("--warmup must be non-negative");
  if (args.correctness_iters <= 0) throw std::runtime_error("--correctness-iters must be positive");
  if (args.timing_iters <= 0) throw std::runtime_error("--timing-iters must be positive");
  if (args.atol < 0.0 || args.rtol < 0.0) throw std::runtime_error("--atol/--rtol must be non-negative");
  return args;
}

static std::string resolve_shared_weights_path(const std::string& package_dir, const std::string& configured) {
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

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static size_t gpu_used_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes - free_bytes;
}

static void cleanup_cuda_cache() {
  CUDA_CHECK(cudaDeviceSynchronize());
  c10::cuda::CUDACachingAllocator::emptyCache();
  CUDA_CHECK(cudaDeviceSynchronize());
}

static size_t delta_bytes(size_t after, size_t before) {
  return after >= before ? after - before : 0;
}

static double gib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
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

struct MemorySampler {
  std::atomic<bool> stop{false};
  std::thread thread;
  std::atomic<size_t> peak{0};

  ~MemorySampler() {
    stop.store(true, std::memory_order_relaxed);
    if (thread.joinable()) thread.join();
  }

  void start() {
    peak.store(gpu_used_bytes(), std::memory_order_relaxed);
    stop.store(false, std::memory_order_relaxed);
    thread = std::thread([this] {
      while (!stop.load(std::memory_order_relaxed)) {
        try {
          const size_t used = gpu_used_bytes();
          size_t prev = peak.load(std::memory_order_relaxed);
          while (used > prev && !peak.compare_exchange_weak(prev, used)) {}
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
  }

  size_t finish() {
    stop.store(true, std::memory_order_relaxed);
    if (thread.joinable()) thread.join();
    const size_t used = gpu_used_bytes();
    size_t prev = peak.load(std::memory_order_relaxed);
    while (used > prev && !peak.compare_exchange_weak(prev, used)) {}
    return peak.load(std::memory_order_relaxed);
  }
};

class StartGate {
 public:
  explicit StartGate(int expected) : expected_(expected) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++ready_;
    if (ready_ == expected_) cv_.notify_all();
    cv_.wait(lock, [&] { return go_; });
  }

  void wait_until_ready_and_start() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return ready_ == expected_; });
    start_time_ = Clock::now();
    go_ = true;
    cv_.notify_all();
  }

  Clock::time_point start_time() const {
    return start_time_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int expected_ = 0;
  int ready_ = 0;
  bool go_ = false;
  Clock::time_point start_time_{};
};

class AbortableBarrier {
 public:
  explicit AbortableBarrier(int parties) : parties_(parties) {}

  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (aborted_) return false;
    const int generation = generation_;
    ++arrived_;
    if (arrived_ == parties_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return true;
    }
    cv_.wait(lock, [&] { return aborted_ || generation_ != generation; });
    return !aborted_;
  }

  void abort() {
    std::lock_guard<std::mutex> lock(mutex_);
    aborted_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int parties_ = 0;
  int arrived_ = 0;
  int generation_ = 0;
  bool aborted_ = false;
};

struct ScopedExternalStream {
  cudaStream_t raw = nullptr;
  int device_index = 0;

  explicit ScopedExternalStream(int device) : device_index(device) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking));
  }

  ScopedExternalStream(const ScopedExternalStream&) = delete;
  ScopedExternalStream& operator=(const ScopedExternalStream&) = delete;

  ~ScopedExternalStream() {
    if (raw != nullptr) cudaStreamDestroy(raw);
  }

  c10::cuda::CUDAStream stream() const {
    return c10::cuda::getStreamFromExternal(raw, device_index);
  }
};

static uintptr_t stream_handle_value(const c10::cuda::CUDAStream& stream) {
  return reinterpret_cast<uintptr_t>(stream.stream());
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

static std::vector<at::Tensor> make_lane_inputs(int lane, torch::Device device) {
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto i64 = torch::TensorOptions().dtype(torch::kLong).device(device);

  auto row_offsets = torch::arange(kBucket, f32).reshape({kBucket, 1, 1}).mul(1.0e-3);
  auto base_chunk = torch::linspace(-0.75, 0.75, kMels * kInputT, f32)
                        .reshape({1, kMels, kInputT})
                        .contiguous();
  auto chunk = base_chunk.repeat({kBucket, 1, 1})
                   .add(static_cast<double>(lane) * 1.0e-2)
                   .add(row_offsets)
                   .contiguous();
  auto length = torch::full({kBucket}, kInputT, i64).contiguous();

  auto cache_row_offsets = torch::arange(kBucket, f32).reshape({1, kBucket, 1, 1}).mul(1.0e-4);
  auto base_cache_ch = torch::arange(static_cast<int64_t>(kLayers) * kAttLeft * kDModel, f32)
                           .reshape({kLayers, 1, kAttLeft, kDModel})
                           .mul(1.0e-6)
                           .contiguous();
  auto cache_ch = base_cache_ch.repeat({1, kBucket, 1, 1})
                      .add(static_cast<double>(lane) * 2.0e-2)
                      .add(cache_row_offsets)
                      .contiguous();

  auto base_cache_t = torch::arange(static_cast<int64_t>(kLayers) * kDModel * kTimeCache, f32)
                          .reshape({kLayers, 1, kDModel, kTimeCache})
                          .mul(1.0e-5)
                          .contiguous();
  auto cache_t = base_cache_t.repeat({1, kBucket, 1, 1})
                     .add(static_cast<double>(lane) * 3.0e-2)
                     .add(cache_row_offsets)
                     .contiguous();

  auto cache_ch_len = torch::full({kBucket}, kCacheLenAfterFirst, i64).contiguous();
  return {chunk, length, cache_ch, cache_t, cache_ch_len};
}

static bool scratch_inputs_are_isolated(const std::vector<at::Tensor>& lane0,
                                        const std::vector<at::Tensor>& lane1,
                                        std::string* error) {
  if (lane0.size() != 5 || lane1.size() != 5) {
    if (error) *error = "expected five prepacked tensors per lane";
    return false;
  }
  for (size_t i = 0; i < lane0.size(); ++i) {
    if (lane0[i].data_ptr() == lane1[i].data_ptr()) {
      if (error) {
        *error = std::string("lane scratch aliases for input ") + std::to_string(i);
      }
      return false;
    }
  }
  return true;
}

static at::Tensor select_output_row(const std::vector<at::Tensor>& out, int output_index, int64_t row) {
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

static std::vector<at::Tensor> to_cpu_contiguous(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> out;
  out.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    out.push_back(tensor.detach().to(torch::kCPU).contiguous());
  }
  return out;
}

static size_t tensor_bytes(const at::Tensor& tensor) {
  return static_cast<size_t>(tensor.numel()) * static_cast<size_t>(tensor.element_size());
}

static void merge_compare_stats(CompareStats* dst, const CompareStats& src) {
  dst->ok = dst->ok && src.ok;
  dst->row_output_checks += src.row_output_checks;
  dst->byte_equal_checks += src.byte_equal_checks;
  for (size_t i = 0; i < dst->max_abs.size(); ++i) {
    dst->max_abs[i] = std::max(dst->max_abs[i], src.max_abs[i]);
  }
}

static CompareStats compare_outputs_to_ref(const std::vector<at::Tensor>& got_cpu,
                                           const std::vector<at::Tensor>& ref_cpu,
                                           double atol,
                                           double rtol,
                                           const std::string& label) {
  CompareStats stats;
  if (got_cpu.size() < 5 || ref_cpu.size() < 5) {
    throw std::runtime_error(label + ": fewer than five outputs");
  }
  for (int output_index = 0; output_index < 5; ++output_index) {
    for (int64_t row = 0; row < kBucket; ++row) {
      at::Tensor got = select_output_row(got_cpu, output_index, row).contiguous();
      at::Tensor ref = select_output_row(ref_cpu, output_index, row).contiguous();
      ++stats.row_output_checks;
      if (got.sizes() != ref.sizes() || got.scalar_type() != ref.scalar_type()) {
        std::printf("CORRECTNESS_MISMATCH label=%s output=%s row=%lld got_shape=%s ref_shape=%s "
                    "got_dtype=%s ref_dtype=%s\n",
                    label.c_str(),
                    kOutputNames[output_index],
                    static_cast<long long>(row),
                    sizes_str(got).c_str(),
                    sizes_str(ref).c_str(),
                    got.toString().c_str(),
                    ref.toString().c_str());
        stats.ok = false;
        continue;
      }
      const bool byte_equal =
          tensor_bytes(got) == tensor_bytes(ref) &&
          std::memcmp(got.data_ptr(), ref.data_ptr(), tensor_bytes(got)) == 0;
      if (byte_equal) ++stats.byte_equal_checks;
      if (got.is_floating_point()) {
        at::Tensor diff = (got.to(torch::kFloat32) - ref.to(torch::kFloat32)).abs();
        const double max_abs = diff.numel() > 0 ? diff.max().item<double>() : 0.0;
        stats.max_abs[static_cast<size_t>(output_index)] =
            std::max(stats.max_abs[static_cast<size_t>(output_index)], max_abs);
        if (!byte_equal && !at::allclose(got, ref, rtol, atol)) {
          std::printf("CORRECTNESS_MISMATCH label=%s output=%s row=%lld max_abs=%.6e "
                      "atol=%.6e rtol=%.6e\n",
                      label.c_str(),
                      kOutputNames[output_index],
                      static_cast<long long>(row),
                      max_abs,
                      atol,
                      rtol);
          stats.ok = false;
        }
      } else if (!byte_equal && !at::equal(got, ref)) {
        std::printf("CORRECTNESS_MISMATCH label=%s output=%s row=%lld nonfloat_not_equal\n",
                    label.c_str(),
                    kOutputNames[output_index],
                    static_cast<long long>(row));
        stats.ok = false;
      }
    }
  }
  return stats;
}

static std::vector<at::Tensor> run_raw_prepacked_dedicated(AOTIModelPackageLoader& loader,
                                                           const std::vector<at::Tensor>& inputs,
                                                           c10::cuda::CUDAStream stream) {
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto out = loader.run(inputs, reinterpret_cast<void*>(stream.stream()));
  if (out.size() < 5) throw std::runtime_error("dedicated steady AOTI returned fewer than 5 outputs");
  return out;
}

using Runner = std::function<std::vector<at::Tensor>(int, const std::vector<at::Tensor>&, c10::cuda::CUDAStream)>;

static std::vector<std::vector<at::Tensor>> build_serial_refs(const Runner& runner,
                                                              const std::vector<at::Tensor> lane_inputs[2],
                                                              c10::cuda::CUDAStream stream) {
  std::vector<std::vector<at::Tensor>> refs(2);
  for (int lane = 0; lane < 2; ++lane) {
    auto out = runner(lane, lane_inputs[lane], stream);
    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    refs[static_cast<size_t>(lane)] = to_cpu_contiguous(out);
  }
  return refs;
}

static CompareStats run_correctness_stress(const Args& args,
                                           torch::Device device,
                                           const Runner& runner,
                                           const std::vector<at::Tensor> lane_inputs[2],
                                           const std::vector<std::vector<at::Tensor>>& refs,
                                           const c10::cuda::CUDAStream streams[2],
                                           std::string* error_text) {
  CompareStats merged;
  StartGate start_gate(2);
  AbortableBarrier barrier(2);
  std::array<std::string, 2> errors;
  std::array<CompareStats, 2> lane_stats;
  std::vector<std::thread> threads;
  threads.reserve(2);
  for (int lane = 0; lane < 2; ++lane) {
    threads.emplace_back([&, lane] {
      try {
        c10::cuda::CUDAGuard guard(device.index());
        start_gate.arrive_and_wait();
        for (int iter = 0; iter < args.correctness_iters; ++iter) {
          if (!barrier.wait()) break;
          auto out = runner(lane, lane_inputs[lane], streams[lane]);
          CUDA_CHECK(cudaStreamSynchronize(streams[lane].stream()));
          auto got_cpu = to_cpu_contiguous(out);
          CompareStats stats = compare_outputs_to_ref(got_cpu,
                                                       refs[static_cast<size_t>(lane)],
                                                       args.atol,
                                                       args.rtol,
                                                       "lane" + std::to_string(lane) + ".iter" +
                                                           std::to_string(iter));
          merge_compare_stats(&lane_stats[static_cast<size_t>(lane)], stats);
        }
      } catch (const std::exception& e) {
        errors[static_cast<size_t>(lane)] = e.what();
        barrier.abort();
      }
    });
  }
  start_gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  for (int lane = 0; lane < 2; ++lane) {
    if (!errors[static_cast<size_t>(lane)].empty()) {
      if (error_text != nullptr) {
        if (!error_text->empty()) *error_text += "; ";
        *error_text += "correctness lane" + std::to_string(lane) + ": " + errors[static_cast<size_t>(lane)];
      }
      merged.ok = false;
    }
    merge_compare_stats(&merged, lane_stats[static_cast<size_t>(lane)]);
  }
  return merged;
}

static RunSample timed_run_once(const Runner& runner,
                                int lane,
                                const std::vector<at::Tensor>& inputs,
                                c10::cuda::CUDAStream stream,
                                cudaEvent_t zero_event) {
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  RunSample sample;
  auto host_start = Clock::now();
  CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
  auto out = runner(lane, inputs, stream);
  CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  sample.host_ms = elapsed_ms(host_start, Clock::now());
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
  sample.event_elapsed_ms = static_cast<double>(elapsed);
  if (zero_event != nullptr) {
    float start_offset = 0.0f;
    float stop_offset = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&start_offset, zero_event, ev_start));
    CUDA_CHECK(cudaEventElapsedTime(&stop_offset, zero_event, ev_stop));
    sample.event_start_ms = static_cast<double>(start_offset);
    sample.event_stop_ms = static_cast<double>(stop_offset);
  }
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  (void)out;
  return sample;
}

static TimingResult measure_serial_pairs(const Args& args,
                                         const Runner& runner,
                                         const std::vector<at::Tensor> lane_inputs[2],
                                         const c10::cuda::CUDAStream streams[2]) {
  TimingResult result;
  cudaStream_t coord_stream = nullptr;
  cudaEvent_t zero_event = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&coord_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&zero_event));
  CUDA_CHECK(cudaEventRecord(zero_event, coord_stream));
  CUDA_CHECK(cudaEventSynchronize(zero_event));
  const auto wall_start = Clock::now();
  try {
    for (int iter = 0; iter < args.timing_iters; ++iter) {
      for (int lane = 0; lane < 2; ++lane) {
        result.lane_samples[lane].push_back(timed_run_once(runner, lane, lane_inputs[lane], streams[lane], zero_event));
      }
    }
  } catch (const std::exception& e) {
    result.ok = false;
    ++result.errors;
    result.error_text = e.what();
  }
  result.wall_ms = elapsed_ms(wall_start, Clock::now());
  CUDA_CHECK(cudaEventDestroy(zero_event));
  CUDA_CHECK(cudaStreamDestroy(coord_stream));
  return result;
}

static TimingResult measure_concurrent_pairs(const Args& args,
                                             torch::Device device,
                                             const Runner& runner,
                                             const std::vector<at::Tensor> lane_inputs[2],
                                             const c10::cuda::CUDAStream streams[2]) {
  TimingResult result;
  result.lane_samples[0].resize(static_cast<size_t>(args.timing_iters));
  result.lane_samples[1].resize(static_cast<size_t>(args.timing_iters));

  cudaStream_t coord_stream = nullptr;
  cudaEvent_t zero_event = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&coord_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&zero_event));
  CUDA_CHECK(cudaEventRecord(zero_event, coord_stream));
  CUDA_CHECK(cudaEventSynchronize(zero_event));

  StartGate start_gate(2);
  AbortableBarrier barrier(2);
  std::array<std::string, 2> errors;
  std::vector<std::thread> threads;
  threads.reserve(2);
  for (int lane = 0; lane < 2; ++lane) {
    threads.emplace_back([&, lane] {
      try {
        c10::cuda::CUDAGuard guard(device.index());
        start_gate.arrive_and_wait();
        for (int iter = 0; iter < args.timing_iters; ++iter) {
          if (!barrier.wait()) break;
          result.lane_samples[lane][static_cast<size_t>(iter)] =
              timed_run_once(runner, lane, lane_inputs[lane], streams[lane], zero_event);
        }
      } catch (const std::exception& e) {
        errors[static_cast<size_t>(lane)] = e.what();
        barrier.abort();
      }
    });
  }
  start_gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  result.wall_ms = elapsed_ms(start_gate.start_time(), Clock::now());
  for (int lane = 0; lane < 2; ++lane) {
    if (!errors[static_cast<size_t>(lane)].empty()) {
      result.ok = false;
      ++result.errors;
      if (!result.error_text.empty()) result.error_text += "; ";
      result.error_text += "timing lane" + std::to_string(lane) + ": " + errors[static_cast<size_t>(lane)];
    }
  }
  CUDA_CHECK(cudaEventDestroy(zero_event));
  CUDA_CHECK(cudaStreamDestroy(coord_stream));
  return result;
}

static Summary summarize_event_ms(const TimingResult& timing) {
  std::vector<double> values;
  for (int lane = 0; lane < 2; ++lane) {
    for (const auto& sample : timing.lane_samples[lane]) {
      if (sample.event_elapsed_ms > 0.0) values.push_back(sample.event_elapsed_ms);
    }
  }
  return summarize(std::move(values));
}

static void compute_overlap(ModelResult* result, const TimingResult& concurrent) {
  const size_t pairs = std::min(concurrent.lane_samples[0].size(), concurrent.lane_samples[1].size());
  result->overlap_total_pairs = static_cast<int>(pairs);
  if (pairs == 0) return;
  std::vector<double> overlaps;
  overlaps.reserve(pairs);
  for (size_t i = 0; i < pairs; ++i) {
    const auto& a = concurrent.lane_samples[0][i];
    const auto& b = concurrent.lane_samples[1][i];
    const double overlap = std::min(a.event_stop_ms, b.event_stop_ms) -
                           std::max(a.event_start_ms, b.event_start_ms);
    overlaps.push_back(overlap);
    if (overlap > 0.05) ++result->overlap_pairs;
  }
  const Summary s = summarize(overlaps);
  result->mean_overlap_ms = s.mean;
  result->min_overlap_ms = s.min;
  result->overlap_fraction = static_cast<double>(result->overlap_pairs) / static_cast<double>(pairs);
  result->event_overlap_go = result->overlap_fraction >= 0.80;
}

static void warmup_runner(const Args& args,
                          const Runner& runner,
                          const std::vector<at::Tensor> lane_inputs[2],
                          const c10::cuda::CUDAStream streams[2]) {
  for (int i = 0; i < args.warmup; ++i) {
    for (int lane = 0; lane < 2; ++lane) {
      auto out = runner(lane, lane_inputs[lane], streams[lane]);
      (void)out;
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(streams[0].stream()));
  CUDA_CHECK(cudaStreamSynchronize(streams[1].stream()));
}

static void finish_common_model_result(const Args& args,
                                       torch::Device device,
                                       const Runner& runner,
                                       ModelResult* result) {
  ScopedExternalStream raw_stream0(device.index());
  ScopedExternalStream raw_stream1(device.index());
  c10::cuda::CUDAStream streams[2] = {raw_stream0.stream(), raw_stream1.stream()};
  result->streams_unique = stream_handle_value(streams[0]) != stream_handle_value(streams[1]);

  std::vector<at::Tensor> lane_inputs[2] = {
      make_lane_inputs(0, device),
      make_lane_inputs(1, device),
  };
  CUDA_CHECK(cudaDeviceSynchronize());
  std::string scratch_error;
  result->scratch_isolated = scratch_inputs_are_isolated(lane_inputs[0], lane_inputs[1], &scratch_error);
  result->used_after_scratch = gpu_used_bytes();
  result->scratch_delta = delta_bytes(result->used_after_scratch, result->used_after_model_loaders);
  if (!scratch_error.empty()) result->error_text = scratch_error;

  MemorySampler mem;
  mem.start();
  const auto refs = build_serial_refs(runner, lane_inputs, streams[0]);
  warmup_runner(args, runner, lane_inputs, streams);
  std::string correctness_error;
  CompareStats correctness = run_correctness_stress(args,
                                                    device,
                                                    runner,
                                                    lane_inputs,
                                                    refs,
                                                    streams,
                                                    &correctness_error);
  if (!correctness_error.empty()) {
    if (!result->error_text.empty()) result->error_text += "; ";
    result->error_text += correctness_error;
  }
  result->correctness_ok = correctness.ok;
  result->correctness_checks = correctness.row_output_checks;
  result->byte_equal_checks = correctness.byte_equal_checks;
  result->max_abs = correctness.max_abs;

  TimingResult serial = measure_serial_pairs(args, runner, lane_inputs, streams);
  TimingResult concurrent = measure_concurrent_pairs(args, device, runner, lane_inputs, streams);
  result->peak_used = mem.finish();
  result->used_after_run = gpu_used_bytes();

  result->serial_wall_ms = serial.wall_ms;
  result->concurrent_wall_ms = concurrent.wall_ms;
  result->wall_speedup = concurrent.wall_ms > 0.0 ? serial.wall_ms / concurrent.wall_ms : 0.0;
  result->wall_speedup_go = result->wall_speedup >= 1.10;
  result->serial_event_ms = summarize_event_ms(serial);
  result->concurrent_event_ms = summarize_event_ms(concurrent);
  compute_overlap(result, concurrent);
  result->no_errors = serial.ok && concurrent.ok && result->error_text.empty();
  if (!serial.error_text.empty()) {
    if (!result->error_text.empty()) result->error_text += "; ";
    result->error_text += "serial timing: " + serial.error_text;
  }
  if (!concurrent.error_text.empty()) {
    if (!result->error_text.empty()) result->error_text += "; ";
    result->error_text += "concurrent timing: " + concurrent.error_text;
  }
  result->completed = true;
  result->go = result->correctness_ok &&
               result->scratch_isolated &&
               result->streams_unique &&
               result->no_errors &&
               (result->event_overlap_go || result->wall_speedup_go);
}

static ModelResult run_shared_model(const Args& args, torch::Device device, const std::string& shared_weights_path) {
  ModelResult result;
  result.model = "shared_num_runners_2";
  cleanup_cuda_cache();
  result.used_before_load = gpu_used_bytes();
  BatchedSteadyLoaderSet loader(args.package_dir,
                                shared_weights_path,
                                device,
                                /*num_runners=*/2,
                                "step0_shared_num_runners_2_probe",
                                {kBucket});
  loader.preload_buckets({kBucket});
  CUDA_CHECK(cudaDeviceSynchronize());
  result.used_after_base_loader = gpu_used_bytes();
  result.used_after_model_loaders = result.used_after_base_loader;
  result.base_loader_delta = delta_bytes(result.used_after_base_loader, result.used_before_load);
  result.model_loader_delta = result.base_loader_delta;

  Runner runner = [&loader](int,
                            const std::vector<at::Tensor>& inputs,
                            c10::cuda::CUDAStream stream) {
    return loader.run_raw_prepacked(inputs, kBucket, stream);
  };
  finish_common_model_result(args, device, runner, &result);
  return result;
}

static ModelResult run_dedicated_model(const Args& args,
                                       torch::Device device,
                                       const std::string& shared_weights_path) {
  ModelResult result;
  result.model = "dedicated_per_lane_num_runners_1";
  cleanup_cuda_cache();
  result.used_before_load = gpu_used_bytes();
  BatchedSteadyLoaderSet base(args.package_dir,
                              shared_weights_path,
                              device,
                              /*num_runners=*/1,
                              "step0_dedicated_base_probe",
                              {kBucket});
  base.preload_buckets({kBucket});
  CUDA_CHECK(cudaDeviceSynchronize());
  result.used_after_base_loader = gpu_used_bytes();
  auto lane0 = base.create_dedicated_loader_for_bucket(kBucket, /*num_runners=*/1, "step0_dedicated_lane0");
  auto lane1 = base.create_dedicated_loader_for_bucket(kBucket, /*num_runners=*/1, "step0_dedicated_lane1");
  CUDA_CHECK(cudaDeviceSynchronize());
  result.used_after_model_loaders = gpu_used_bytes();
  result.base_loader_delta = delta_bytes(result.used_after_base_loader, result.used_before_load);
  result.model_loader_delta = delta_bytes(result.used_after_model_loaders, result.used_before_load);
  result.dedicated_incremental_delta = delta_bytes(result.used_after_model_loaders, result.used_after_base_loader);

  std::array<AOTIModelPackageLoader*, 2> lane_loaders{{lane0.get(), lane1.get()}};
  Runner runner = [&lane_loaders](int lane,
                                  const std::vector<at::Tensor>& inputs,
                                  c10::cuda::CUDAStream stream) {
    return run_raw_prepacked_dedicated(*lane_loaders[static_cast<size_t>(lane)], inputs, stream);
  };
  finish_common_model_result(args, device, runner, &result);
  return result;
}

static std::string json_bool(bool value) {
  return value ? "true" : "false";
}

static std::string json_escape(const std::string& text) {
  std::ostringstream oss;
  oss << '"';
  for (char ch : text) {
    switch (ch) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << ch;
    }
  }
  oss << '"';
  return oss.str();
}

static std::string summary_json(const Summary& s) {
  std::ostringstream oss;
  oss << "{\"mean_ms\":" << s.mean
      << ",\"p50_ms\":" << s.p50
      << ",\"min_ms\":" << s.min
      << ",\"max_ms\":" << s.max
      << "}";
  return oss.str();
}

static std::string max_abs_json(const std::array<double, 5>& max_abs) {
  std::ostringstream oss;
  oss << "{";
  for (int i = 0; i < 5; ++i) {
    if (i) oss << ",";
    oss << "\"" << kOutputNames[i] << "\":" << max_abs[static_cast<size_t>(i)];
  }
  oss << "}";
  return oss.str();
}

static void print_model_result(const ModelResult& r) {
  std::printf("STEP0_MODEL_SUMMARY model=%s correctness=%s checks=%d byte_equal=%d "
              "streams_unique=%s scratch_isolated=%s serial_wall_ms=%.3f concurrent_wall_ms=%.3f "
              "wall_speedup=%.3f overlap_pairs=%d/%d overlap_fraction=%.3f mean_overlap_ms=%.3f "
              "loader_delta_gib=%.3f dedicated_incremental_gib=%.3f scratch_delta_gib=%.3f "
              "peak_gib=%.3f go=%s error=%s\n",
              r.model.c_str(),
              r.correctness_ok ? "PASS" : "FAIL",
              r.correctness_checks,
              r.byte_equal_checks,
              r.streams_unique ? "true" : "false",
              r.scratch_isolated ? "true" : "false",
              r.serial_wall_ms,
              r.concurrent_wall_ms,
              r.wall_speedup,
              r.overlap_pairs,
              r.overlap_total_pairs,
              r.overlap_fraction,
              r.mean_overlap_ms,
              gib(r.model_loader_delta),
              gib(r.dedicated_incremental_delta),
              gib(r.scratch_delta),
              gib(r.peak_used),
              r.go ? "GO" : "NO-GO",
              r.error_text.empty() ? "none" : r.error_text.c_str());

  std::ostringstream json;
  json << "{\"model\":" << json_escape(r.model)
       << ",\"completed\":" << json_bool(r.completed)
       << ",\"go\":" << json_bool(r.go)
       << ",\"correctness_ok\":" << json_bool(r.correctness_ok)
       << ",\"correctness_checks\":" << r.correctness_checks
       << ",\"byte_equal_checks\":" << r.byte_equal_checks
       << ",\"max_abs\":" << max_abs_json(r.max_abs)
       << ",\"streams_unique\":" << json_bool(r.streams_unique)
       << ",\"scratch_isolated\":" << json_bool(r.scratch_isolated)
       << ",\"no_errors\":" << json_bool(r.no_errors)
       << ",\"serial_wall_ms\":" << r.serial_wall_ms
       << ",\"concurrent_wall_ms\":" << r.concurrent_wall_ms
       << ",\"wall_speedup\":" << r.wall_speedup
       << ",\"wall_speedup_go\":" << json_bool(r.wall_speedup_go)
       << ",\"event_overlap_go\":" << json_bool(r.event_overlap_go)
       << ",\"overlap_pairs\":" << r.overlap_pairs
       << ",\"overlap_total_pairs\":" << r.overlap_total_pairs
       << ",\"overlap_fraction\":" << r.overlap_fraction
       << ",\"mean_overlap_ms\":" << r.mean_overlap_ms
       << ",\"min_overlap_ms\":" << r.min_overlap_ms
       << ",\"serial_event\":" << summary_json(r.serial_event_ms)
       << ",\"concurrent_event\":" << summary_json(r.concurrent_event_ms)
       << ",\"used_before_load_bytes\":" << r.used_before_load
       << ",\"used_after_base_loader_bytes\":" << r.used_after_base_loader
       << ",\"used_after_model_loaders_bytes\":" << r.used_after_model_loaders
       << ",\"used_after_scratch_bytes\":" << r.used_after_scratch
       << ",\"used_after_run_bytes\":" << r.used_after_run
       << ",\"peak_used_bytes\":" << r.peak_used
       << ",\"base_loader_delta_bytes\":" << r.base_loader_delta
       << ",\"model_loader_delta_bytes\":" << r.model_loader_delta
       << ",\"dedicated_incremental_delta_bytes\":" << r.dedicated_incremental_delta
       << ",\"scratch_delta_bytes\":" << r.scratch_delta
       << ",\"error\":" << json_escape(r.error_text)
       << "}";
  std::printf("STEP0_PROBE_RESULT %s\n", json.str().c_str());
}

int main(int argc, char** argv) {
  try {
    torch::NoGradGuard no_grad;
    Args args = parse_args(argc, argv);
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) throw std::runtime_error("CUDA is not available");

    torch::Device device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard guard(device.index());
    torch::manual_seed(20260601);
    const std::string shared_weights_path = resolve_shared_weights_path(args.package_dir, args.shared_weights);
    std::printf("STEP0_AOTI_OVERLAP_PROBE_START package_dir=%s shared_weights=%s model=%s "
                "bucket=%d warmup=%d correctness_iters=%d timing_iters=%d atol=%.6e rtol=%.6e\n",
                args.package_dir.c_str(),
                shared_weights_path.c_str(),
                args.model.c_str(),
                kBucket,
                args.warmup,
                args.correctness_iters,
                args.timing_iters,
                args.atol,
                args.rtol);

    std::vector<ModelResult> results;
    if (args.model == "both" || args.model == "shared") {
      results.push_back(run_shared_model(args, device, shared_weights_path));
      print_model_result(results.back());
    }
    if (args.model == "both" || args.model == "dedicated") {
      results.push_back(run_dedicated_model(args, device, shared_weights_path));
      print_model_result(results.back());
    }

    bool shared_go = false;
    bool dedicated_go = false;
    for (const auto& result : results) {
      if (result.model == "shared_num_runners_2" && result.go) shared_go = true;
      if (result.model == "dedicated_per_lane_num_runners_1" && result.go) dedicated_go = true;
    }
    const char* chosen = shared_go ? "shared_num_runners_2"
                         : (dedicated_go ? "dedicated_per_lane_num_runners_1" : "none");
    const char* n1 = shared_go ? "steady_num_runners>=dispatch_lanes"
                    : (dedicated_go ? "runner_count_independent" : "no_viable_runner_count");
    std::printf("STEP0_PROBE_VERDICT chosen=%s shared_go=%s dedicated_go=%s n1=%s\n",
                chosen,
                shared_go ? "true" : "false",
                dedicated_go ? "true" : "false",
                n1);
    if (args.fail_on_no_go && !shared_go && !dedicated_go) return 2;
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "step0_aoti_overlap_probe error: %s\n", e.what());
    return 1;
  }
}
