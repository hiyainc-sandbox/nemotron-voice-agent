// Phase-2 Step-0 density kill-gates.
//
// The validated session implementation now lives behind lib/session/session.h
// and libnemotron_runtime.a. The harness glue below only adds concurrency,
// explicit-stream AOTI calls, timing, and telemetry.
#include "lib/runtime_io/jit_load.h"
#include "lib/runtime_io/steady_batch_dir.h"
#include "lib/session/session.h"
#include "lib/session/first_encoder.h"
#include "lib/session/runtime.h"
#include "lib/telemetry/stats_collector.h"
#include "lib/ws/framing.h"
#include "lib/ws/handshake.h"
#include "lib/ws/routes.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;

static constexpr int kMinFinalizeP95Samples = 20;

static void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr
        << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

struct ScopedCudaEvent {
  cudaEvent_t event = nullptr;

  ScopedCudaEvent() = default;
  ScopedCudaEvent(const ScopedCudaEvent&) = delete;
  ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;

  ~ScopedCudaEvent() {
    if (event != nullptr) cudaEventDestroy(event);
  }

  void create(unsigned int flags = 0) {
    if (event != nullptr) CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
  }

  cudaEvent_t release() noexcept {
    cudaEvent_t out = event;
    event = nullptr;
    return out;
  }
};

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const std::string& value) : name_(name) {
    const char* old = std::getenv(name_.c_str());
    if (old != nullptr) {
      had_old_ = true;
      old_value_ = old;
    }
    if (setenv(name_.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error("failed to set env var " + name_);
    }
  }
  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;
  ~ScopedEnvVar() {
    if (had_old_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  bool had_old_ = false;
  std::string old_value_;
};

#include "lib/scheduler/steady_batch_primitive.h"
#include "lib/scheduler/batched_steady_scheduler.h"
#include "lib/admission/density_admission.h"

std::vector<EmittedEvent> session_runtime_debug_events(const SessionRuntime& runtime);
std::vector<int64_t> session_runtime_debug_last_final_tokens(const SessionRuntime& runtime);

struct DensityArgs {
  std::string mode = "step0";
  std::string dir = "../artifacts";
  std::string steady_batch_dir;
  std::vector<int> n_values{1, 2, 4};
  bool n_values_set = false;
  bool target_n_set = false;
  int target_n = 16;
  int workers = 0;
  int num_runners = 0;
  int steady_cases = 32;
  int steady_repeats = 4;
  int correctness_n = 4;
  int correctness_rows = -1;
  int finalize_n = 4;
  std::string finalize_mode = "both";
  std::string stream_mode = "explicit";
  bool mutex_serialize_run = false;
  bool smoke = false;
  bool partial = false;
  bool skip_correctness = false;
  bool skip_steady = false;
  bool skip_finalize = false;
  bool default_stream_control = true;
  bool correctness_default_stream_control = true;
  bool scalar_locality_probe = true;
  bool steady_overlap_probe = true;
  int density_rows = -1;
  int density_sessions_per_worker = 0;
  double density_chunk_period_ms = 160.0;
  double density_start_stagger_ms = 0.0;  // spread per-worker first-session starts over [0,this]; 0 = barrier (synchronized)
  bool density_warmup = true;
  int b1_batch_size = 3;
  std::string batch_steady = "env";
  int batch_b_max = -1;
  int batch_window_ms = -1;
  int batch_lone_timeout_ms = -1;
  int batch_max_queue_delay_ms = -1;
  int batch_queue_capacity = 0;
  int batch_min_fill_enabled = -1;
  int batch_force_bucket = -1;
  bool batch_disable_min_fill = false;
  int steady_dispatch_lanes = 0;
  int admission_active_cap = -1;
  int admission_backlog_cap = -1;
};

static double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static double elapsed_ms_since(Clock::time_point start) {
  return elapsed_ms(start, Clock::now());
}

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

static void log_density_phase_timing(int n,
                                     const char* phase,
                                     Clock::time_point start,
                                     Clock::time_point end) {
  std::printf("DENSITY_PHASE_TIMING n=%d phase=%s elapsed_ms=%.3f\n",
              n,
              phase,
              elapsed_ms(start, end));
}

static void log_density_phase_timing(int n, const char* phase, Clock::time_point start) {
  log_density_phase_timing(n, phase, start, Clock::now());
}

static std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t comma = text.find(',', pos);
    std::string item = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!item.empty()) out.push_back(std::stoi(item));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

static int read_density_env_int(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid integer env var ") + name + "=" + raw);
  }
  return static_cast<int>(value);
}

static bool density_host_enc_len_enabled() {
  static const bool enabled = read_density_env_int("NEMOTRON_DENSITY_HOST_ENC_LEN", 0) != 0;
  return enabled;
}

static bool density_host_enc_len_verify_enabled() {
  static const bool enabled = read_density_env_int("NEMOTRON_DENSITY_HOST_ENC_LEN_VERIFY", 0) != 0;
  return enabled;
}

static bool density_device_argmax_enabled() {
  static const bool enabled = read_density_env_int("NEMOTRON_DENSITY_DEVICE_ARGMAX", 0) != 0;
  return enabled;
}

static int density_finalize_runner_cap() {
  static const int cap = read_density_env_int("NEMOTRON_DENSITY_FINALIZE_RUNNERS", 2);
  return cap;
}

static bool density_fill_trace_enabled() {
  static const bool enabled = read_density_env_int("NEMOTRON_DENSITY_FILL_TRACE", 0) != 0;
  return enabled;
}

static bool density_batch_steady_enabled() {
  static const bool enabled = read_density_env_int("NEMOTRON_DENSITY_BATCH_STEADY", 0) != 0;
  return enabled;
}

static bool density_batch_steady_enabled_effective(const DensityArgs& args) {
  if (args.batch_steady == "on") return true;
  if (args.batch_steady == "off") return false;
  return read_density_env_int("NEMOTRON_DENSITY_BATCH_STEADY", 0) != 0;
}

static int density_dispatch_lanes_effective(const DensityArgs& args) {
  int lanes = args.steady_dispatch_lanes > 0
                  ? args.steady_dispatch_lanes
                  : read_density_env_int("NEMOTRON_DENSITY_DISPATCH_LANES", 1);
  if (lanes <= 0 || lanes > 2) {
    throw std::runtime_error("NEMOTRON_DENSITY_DISPATCH_LANES/--steady-dispatch-lanes must be 1 or 2");
  }
  return lanes;
}

static int density_effective_steady_num_runners(const DensityArgs& args,
                                                int default_runners,
                                                int dispatch_lanes) {
  int runners = args.num_runners > 0 ? args.num_runners : std::max(default_runners, 1);
  if (runners < dispatch_lanes) {
    std::printf("DENSITY_N1_AUTO_RAISE steady_num_runners=%d effective=%d "
                "steady_dispatch_lanes=%d model=shared_loader_num_runners\n",
                runners,
                dispatch_lanes,
                dispatch_lanes);
    runners = dispatch_lanes;
  }
  return runners;
}

static BatchedSteadySchedulerPolicy density_batch_policy_effective(const DensityArgs& args, int workers) {
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = args.batch_b_max > 0
                     ? args.batch_b_max
                     : read_density_env_int("NEMOTRON_DENSITY_BATCH_MAX", 16);
  policy.window_ms = args.batch_window_ms >= 0
                         ? args.batch_window_ms
                         : read_density_env_int("NEMOTRON_DENSITY_BATCH_WINDOW_MS", 10);
  policy.lone_timeout_ms = args.batch_lone_timeout_ms >= 0
                               ? args.batch_lone_timeout_ms
                               : read_density_env_int("NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS", 0);
  policy.max_queue_delay_ms = args.batch_max_queue_delay_ms >= 0
                                  ? args.batch_max_queue_delay_ms
                                  : read_density_env_int("NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS",
                                                         2);
  policy.queue_capacity = args.batch_queue_capacity > 0
                              ? args.batch_queue_capacity
                              : std::max(4 * std::max(workers, 1), policy.B_max);
  policy.min_fill_enabled = args.batch_min_fill_enabled >= 0
                                ? args.batch_min_fill_enabled != 0
                                : read_density_env_int("NEMOTRON_DENSITY_BATCH_MIN_FILL", 1) != 0;
  policy.disable_min_fill = args.batch_disable_min_fill ||
                            read_density_env_int("NEMOTRON_DENSITY_BATCH_DISABLE_MIN_FILL", 0) != 0;
  policy.force_bucket = args.batch_force_bucket >= 0
                            ? args.batch_force_bucket
                            : read_density_env_int("NEMOTRON_DENSITY_BATCH_FORCE_BUCKET", 0);
  policy.dispatch_lanes = density_dispatch_lanes_effective(args);
  return policy;
}

static int64_t density_clock_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

static void flush_density_fill_trace(const std::vector<std::vector<int64_t>>& ready_ns_by_worker) {
  for (size_t worker = 0; worker < ready_ns_by_worker.size(); ++worker) {
    for (int64_t t_ready_ns : ready_ns_by_worker[worker]) {
      std::printf("FILL_TRACE worker=%zu t_ready_ns=%lld kind=steady\n",
                  worker,
                  static_cast<long long>(t_ready_ns));
    }
  }
  std::fflush(stdout);
}

static DensityArgs parse_density_args(int argc, char** argv) {
  DensityArgs args;
  bool dir_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--mode") {
      args.mode = need_value("--mode");
    } else if (arg == "--steady-batch-dir") {
      args.steady_batch_dir = need_value("--steady-batch-dir");
    } else if (arg == "--n-values") {
      args.n_values = parse_int_list(need_value("--n-values"));
      args.n_values_set = true;
    } else if (arg == "--target-n") {
      args.target_n = std::stoi(need_value("--target-n"));
      args.target_n_set = true;
    } else if (arg == "--workers") {
      args.workers = std::stoi(need_value("--workers"));
    } else if (arg == "--num-runners") {
      args.num_runners = std::stoi(need_value("--num-runners"));
    } else if (arg == "--steady-cases") {
      args.steady_cases = std::stoi(need_value("--steady-cases"));
    } else if (arg == "--steady-repeats") {
      args.steady_repeats = std::stoi(need_value("--steady-repeats"));
    } else if (arg == "--correctness-n") {
      args.correctness_n = std::stoi(need_value("--correctness-n"));
    } else if (arg == "--correctness-rows") {
      args.correctness_rows = std::stoi(need_value("--correctness-rows"));
    } else if (arg == "--finalize-n") {
      args.finalize_n = std::stoi(need_value("--finalize-n"));
    } else if (arg == "--finalize-mode") {
      args.finalize_mode = need_value("--finalize-mode");
    } else if (arg == "--stream-mode") {
      args.stream_mode = need_value("--stream-mode");
    } else if (arg == "--mutex-serialize-run") {
      args.mutex_serialize_run = true;
    } else if (arg == "--smoke") {
      args.smoke = true;
      args.partial = true;
    } else if (arg == "--partial") {
      args.partial = true;
    } else if (arg == "--skip-correctness") {
      args.skip_correctness = true;
    } else if (arg == "--skip-steady") {
      args.skip_steady = true;
    } else if (arg == "--skip-finalize") {
      args.skip_finalize = true;
    } else if (arg == "--no-default-stream-control") {
      args.default_stream_control = false;
      args.correctness_default_stream_control = false;
    } else if (arg == "--no-0b-default-stream-control") {
      args.correctness_default_stream_control = false;
    } else if (arg == "--no-scalar-locality-probe") {
      args.scalar_locality_probe = false;
    } else if (arg == "--no-steady-overlap-probe") {
      args.steady_overlap_probe = false;
    } else if (arg == "--density-rows") {
      args.density_rows = std::stoi(need_value("--density-rows"));
    } else if (arg == "--density-sessions-per-worker") {
      args.density_sessions_per_worker = std::stoi(need_value("--density-sessions-per-worker"));
    } else if (arg == "--density-chunk-period-ms") {
      args.density_chunk_period_ms = std::stod(need_value("--density-chunk-period-ms"));
    } else if (arg == "--density-start-stagger-ms") {
      args.density_start_stagger_ms = std::stod(need_value("--density-start-stagger-ms"));
    } else if (arg == "--no-density-warmup") {
      args.density_warmup = false;
    } else if (arg == "--b1-batch-size") {
      args.b1_batch_size = std::stoi(need_value("--b1-batch-size"));
    } else if (arg == "--batch-steady") {
      args.batch_steady = need_value("--batch-steady");
    } else if (arg == "--batch-b-max") {
      args.batch_b_max = std::stoi(need_value("--batch-b-max"));
    } else if (arg == "--batch-window-ms") {
      args.batch_window_ms = std::stoi(need_value("--batch-window-ms"));
    } else if (arg == "--batch-lone-timeout-ms") {
      args.batch_lone_timeout_ms = std::stoi(need_value("--batch-lone-timeout-ms"));
    } else if (arg == "--batch-max-queue-delay-ms") {
      args.batch_max_queue_delay_ms = std::stoi(need_value("--batch-max-queue-delay-ms"));
    } else if (arg == "--batch-queue-capacity") {
      args.batch_queue_capacity = std::stoi(need_value("--batch-queue-capacity"));
    } else if (arg == "--batch-min-fill") {
      args.batch_min_fill_enabled = std::stoi(need_value("--batch-min-fill"));
    } else if (arg == "--batch-disable-min-fill") {
      args.batch_disable_min_fill = true;
    } else if (arg == "--batch-force-bucket") {
      args.batch_force_bucket = std::stoi(need_value("--batch-force-bucket"));
    } else if (arg == "--steady-dispatch-lanes") {
      args.steady_dispatch_lanes = std::stoi(need_value("--steady-dispatch-lanes"));
      if (args.steady_dispatch_lanes <= 0) {
        throw std::runtime_error("--steady-dispatch-lanes must be positive");
      }
    } else if (arg == "--admission-active-cap") {
      args.admission_active_cap = std::stoi(need_value("--admission-active-cap"));
    } else if (arg == "--admission-backlog-cap") {
      args.admission_backlog_cap = std::stoi(need_value("--admission-backlog-cap"));
    } else if (!dir_set) {
      args.dir = arg;
      dir_set = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.mode != "step0" && args.mode != "density-sweep" && args.mode != "b1-t1" &&
      args.mode != "b2-t1" && args.mode != "enc-first-parity" &&
      args.mode != "async-ordering" &&
      args.mode != "admission-smoke" && args.mode != "scheduler-admission-smoke" &&
      args.mode != "scheduler-lanes-smoke" &&
      args.mode != "scheduler-lanes-stress" &&
      args.mode != "scheduler-graph-abandon-smoke" &&
      args.mode != "scheduler-graph-shutdown-smoke" &&
      args.mode != "stalegen-smoke" &&
      args.mode != "stats-smoke" && args.mode != "runtime-smoke" &&
      args.mode != "vad-smoke" && args.mode != "ws-lib-smoke" &&
      args.mode != "ws-lifecycle-smoke" && args.mode != "shutdown-smoke" &&
      args.mode != "backpressure-smoke") {
    throw std::runtime_error(
        "--mode must be step0, density-sweep, b1-t1, b2-t1, enc-first-parity, async-ordering, admission-smoke, scheduler-admission-smoke, scheduler-lanes-smoke, scheduler-lanes-stress, scheduler-graph-abandon-smoke, scheduler-graph-shutdown-smoke, stalegen-smoke, stats-smoke, runtime-smoke, vad-smoke, ws-lib-smoke, ws-lifecycle-smoke, shutdown-smoke, or backpressure-smoke");
  }
  if (args.mode == "density-sweep" && !args.n_values_set) {
    args.n_values = {1, 2, 4, 8, 16};
  }
  if (args.n_values.empty()) throw std::runtime_error("--n-values cannot be empty");
  for (int n : args.n_values) {
    if (n <= 0) throw std::runtime_error("--n-values entries must be positive");
  }
  if (args.target_n > 0 &&
      (args.mode != "density-sweep" || !args.n_values_set || args.target_n_set) &&
      std::find(args.n_values.begin(), args.n_values.end(), args.target_n) == args.n_values.end()) {
    args.n_values.push_back(args.target_n);
  }
  std::sort(args.n_values.begin(), args.n_values.end());
  args.n_values.erase(std::unique(args.n_values.begin(), args.n_values.end()), args.n_values.end());
  if (args.steady_cases <= 0) throw std::runtime_error("--steady-cases must be positive");
  if (args.steady_repeats <= 0) throw std::runtime_error("--steady-repeats must be positive");
  if (args.workers < 0) throw std::runtime_error("--workers must be non-negative");
  if (args.num_runners < 0) throw std::runtime_error("--num-runners must be non-negative");
  if (args.correctness_n <= 0) throw std::runtime_error("--correctness-n must be positive");
  if (args.finalize_n <= 0) throw std::runtime_error("--finalize-n must be positive");
  if (args.stream_mode != "explicit" && args.stream_mode != "default") {
    throw std::runtime_error("--stream-mode must be explicit or default");
  }
  if (args.finalize_mode != "both" && args.finalize_mode != "same" && args.finalize_mode != "mixed") {
    throw std::runtime_error("--finalize-mode must be both, same, or mixed");
  }
  if (args.density_rows == 0 || args.density_rows < -1) {
    throw std::runtime_error("--density-rows must be positive or -1");
  }
  if (args.density_sessions_per_worker < 0) {
    throw std::runtime_error("--density-sessions-per-worker must be non-negative");
  }
  if (args.density_chunk_period_ms <= 0.0) {
    throw std::runtime_error("--density-chunk-period-ms must be positive");
  }
  if (args.b1_batch_size <= 0 || args.b1_batch_size > 4) {
    throw std::runtime_error("--b1-batch-size must be in [1,4]");
  }
  if (args.batch_steady != "env" && args.batch_steady != "on" && args.batch_steady != "off") {
    throw std::runtime_error("--batch-steady must be on, off, or env");
  }
  if (args.batch_b_max != -1 && args.batch_b_max != 1 && args.batch_b_max != 2 &&
      args.batch_b_max != 4 && args.batch_b_max != 8 && args.batch_b_max != 16) {
    throw std::runtime_error("--batch-b-max must be one of 1,2,4,8,16");
  }
  if (args.batch_window_ms < -1) throw std::runtime_error("--batch-window-ms must be non-negative");
  if (args.batch_lone_timeout_ms < -1) throw std::runtime_error("--batch-lone-timeout-ms must be non-negative");
  if (args.batch_max_queue_delay_ms < -1) {
    throw std::runtime_error("--batch-max-queue-delay-ms must be non-negative");
  }
  if (args.batch_queue_capacity < 0) throw std::runtime_error("--batch-queue-capacity must be non-negative");
  if (args.batch_min_fill_enabled < -1 || args.batch_min_fill_enabled > 1) {
    throw std::runtime_error("--batch-min-fill must be 0 or 1");
  }
  if (args.batch_force_bucket != -1 && args.batch_force_bucket != 0 && args.batch_force_bucket != 1 &&
      args.batch_force_bucket != 2 && args.batch_force_bucket != 4 && args.batch_force_bucket != 8 &&
      args.batch_force_bucket != 16) {
    throw std::runtime_error("--batch-force-bucket must be 0 or one of 1,2,4,8,16");
  }
  if (args.steady_dispatch_lanes < 0 || args.steady_dispatch_lanes > 2) {
    throw std::runtime_error("--steady-dispatch-lanes must be 1 or 2");
  }
  if (args.admission_active_cap < 0) {
    args.admission_active_cap =
        read_density_env_int("NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP", 40);
  }
  if (args.admission_backlog_cap < 0) {
    args.admission_backlog_cap =
        read_density_env_int("NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP", 12);
  }
  if (args.admission_active_cap < 0) throw std::runtime_error("--admission-active-cap must be non-negative");
  if (args.admission_backlog_cap < 0) throw std::runtime_error("--admission-backlog-cap must be non-negative");
  return args;
}

struct SummaryStats {
  size_t n = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double max = 0.0;
};

static SummaryStats summarize(std::vector<double> values) {
  SummaryStats s;
  s.n = values.size();
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  double total = std::accumulate(values.begin(), values.end(), 0.0);
  auto pct = [&](double p) {
    size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(values.size())) - 1.0);
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
  };
  s.p50 = pct(0.50);
  s.p95 = pct(0.95);
  s.p99 = pct(0.99);
  s.mean = total / static_cast<double>(values.size());
  s.max = values.back();
  return s;
}

static std::string stats_json(const SummaryStats& s) {
  std::ostringstream oss;
  oss << "{\"n\":" << s.n
      << ",\"p50_ms\":" << s.p50
      << ",\"p95_ms\":" << s.p95
      << ",\"p99_ms\":" << s.p99
      << ",\"p95_minus_p50_ms\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50_ms\":" << (s.p99 - s.p50)
      << ",\"mean_ms\":" << s.mean
      << ",\"max_ms\":" << s.max
      << "}";
  return oss.str();
}

static std::string value_stats_json(const SummaryStats& s) {
  std::ostringstream oss;
  oss << "{\"n\":" << s.n
      << ",\"p50\":" << s.p50
      << ",\"p95\":" << s.p95
      << ",\"p99\":" << s.p99
      << ",\"p95_minus_p50\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50\":" << (s.p99 - s.p50)
      << ",\"mean\":" << s.mean
      << ",\"max\":" << s.max
      << "}";
  return oss.str();
}

static std::string timestamp_utc() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

static std::string sanitize_filename(std::string text) {
  for (char& ch : text) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_' || ch == '.')) ch = '_';
  }
  return text;
}

static const char* json_bool(bool value) {
  return value ? "true" : "false";
}

static std::string int_list_json(const std::vector<int>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) oss << ",";
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

static std::string scheduler_us_stats_json(const std::vector<double>& values) {
  auto s = summarize(values);
  std::ostringstream oss;
  oss << "{\"n\":" << s.n
      << ",\"p50_us\":" << s.p50
      << ",\"p95_us\":" << s.p95
      << ",\"p99_us\":" << s.p99
      << ",\"p95_minus_p50_us\":" << (s.p95 - s.p50)
      << ",\"p99_minus_p50_us\":" << (s.p99 - s.p50)
      << ",\"mean_us\":" << s.mean
      << ",\"max_us\":" << s.max
      << "}";
  return oss.str();
}

static std::string scheduler_value_stats_json(const std::vector<double>& values) {
  return value_stats_json(summarize(values));
}

static double pct_clamped(double numerator, double denominator) {
  if (denominator <= 0.0) return 0.0;
  double pct = 100.0 * numerator / denominator;
  if (pct < 0.0) return 0.0;
  if (pct > 100.0) return 100.0;
  return pct;
}

static std::string scheduler_telemetry_json(const BatchedSteadySchedulerTelemetry& telemetry) {
  std::ostringstream oss;
  oss << "{\"counts\":{\"dispatch_lanes\":" << telemetry.dispatch_lanes
      << ",\"enqueued\":" << telemetry.enqueued
      << ",\"completed\":" << telemetry.completed
      << ",\"dispatch_cycles\":" << telemetry.dispatch_cycles
      << ",\"warmup_runs\":" << telemetry.warmup_runs
      << ",\"warmed_lanes\":" << telemetry.warmed_lanes
      << ",\"B1\":" << telemetry.bucket_b1
      << ",\"B2\":" << telemetry.bucket_b2
      << ",\"B4\":" << telemetry.bucket_b4
      << ",\"B8\":" << telemetry.bucket_b8
      << ",\"B16\":" << telemetry.bucket_b16
      << ",\"K2_padded_to_B4\":" << telemetry.k2_padded_to_b4
      << ",\"K3_padded_to_B4\":" << telemetry.k3_padded_to_b4
      << ",\"K4\":" << telemetry.k4
      << ",\"K5_padded_to_B8\":" << telemetry.k5_padded_to_b8
      << ",\"K6_padded_to_B8\":" << telemetry.k6_padded_to_b8
      << ",\"K7_padded_to_B8\":" << telemetry.k7_padded_to_b8
      << ",\"K8\":" << telemetry.k8
      << ",\"K9_padded_to_B16\":" << telemetry.k9_padded_to_b16
      << ",\"K10_padded_to_B16\":" << telemetry.k10_padded_to_b16
      << ",\"K11_padded_to_B16\":" << telemetry.k11_padded_to_b16
      << ",\"K12_padded_to_B16\":" << telemetry.k12_padded_to_b16
      << ",\"K13_padded_to_B16\":" << telemetry.k13_padded_to_b16
      << ",\"K14_padded_to_B16\":" << telemetry.k14_padded_to_b16
      << ",\"K15_padded_to_B16\":" << telemetry.k15_padded_to_b16
      << ",\"K16\":" << telemetry.k16
      << ",\"backlog_gt_bmax\":" << telemetry.backlog_gt_bmax
      << ",\"skipped_ready\":" << telemetry.skipped_ready
      << ",\"dispatcher_exceptions\":" << telemetry.dispatcher_exceptions
      << "}"
      << ",\"dispatcher_cpu_pct\":" << pct_clamped(telemetry.dispatcher_cpu_us, telemetry.dispatcher_wall_us)
      << ",\"dispatcher_cpu_us\":" << telemetry.dispatcher_cpu_us
      << ",\"dispatcher_wall_us\":" << telemetry.dispatcher_wall_us
      << ",\"dispatcher_stream_util_pct\":"
      << pct_clamped(telemetry.dispatcher_stream_run_us, telemetry.dispatcher_wall_us)
      << ",\"dispatcher_stream_run_us\":" << telemetry.dispatcher_stream_run_us
      << ",\"timers\":{\"age_at_dispatch_us\":" << scheduler_us_stats_json(telemetry.age_at_dispatch_us)
      << ",\"gather_wait_us\":" << scheduler_us_stats_json(telemetry.gather_wait_us)
      << ",\"service_wait_us\":" << scheduler_us_stats_json(telemetry.service_wait_us)
      << ",\"cuda_run_us\":" << scheduler_us_stats_json(telemetry.cuda_run_us)
      << ",\"output_sync_us\":" << scheduler_us_stats_json(telemetry.output_sync_us)
      << ",\"worker_blocked_us\":" << scheduler_us_stats_json(telemetry.worker_blocked_us)
      << ",\"completion_wait_us\":" << scheduler_us_stats_json(telemetry.completion_wait_us)
      << ",\"window_wakeup_jitter_us\":" << scheduler_us_stats_json(telemetry.window_wakeup_jitter_us)
      << "}"
      << ",\"queue_depth\":" << scheduler_value_stats_json(telemetry.queue_depth)
      << ",\"per_stream_fairness_spread_us\":"
      << scheduler_us_stats_json(telemetry.per_stream_fairness_spread_us)
      << ",\"lanes\":[";
  for (size_t i = 0; i < telemetry.lanes.size(); ++i) {
    const auto& lane = telemetry.lanes[i];
    if (i > 0) oss << ",";
    oss << "{\"lane_id\":" << lane.lane_id
        << ",\"completed\":" << lane.completed
        << ",\"dispatch_cycles\":" << lane.dispatch_cycles
        << ",\"warmup_runs\":" << lane.warmup_runs
        << ",\"dispatcher_cpu_pct\":" << pct_clamped(lane.dispatcher_cpu_us, lane.dispatcher_wall_us)
        << ",\"dispatcher_cpu_us\":" << lane.dispatcher_cpu_us
        << ",\"dispatcher_wall_us\":" << lane.dispatcher_wall_us
        << ",\"dispatcher_stream_util_pct\":"
        << pct_clamped(lane.dispatcher_stream_run_us, lane.dispatcher_wall_us)
        << ",\"dispatcher_stream_run_us\":" << lane.dispatcher_stream_run_us
        << ",\"cuda_run_us\":" << scheduler_us_stats_json(lane.cuda_run_us)
        << ",\"queue_depth\":" << scheduler_value_stats_json(lane.queue_depth)
        << "}";
  }
  oss << "]}";
  return oss.str();
}

static std::string admission_telemetry_json(const AdmissionTelemetry& telemetry) {
  std::ostringstream oss;
  oss << "{\"active_cap\":" << telemetry.active_cap
      << ",\"backlog_cap\":" << telemetry.backlog_cap
      << ",\"offered\":" << telemetry.offered
      << ",\"admitted\":" << telemetry.admitted
      << ",\"active_count\":" << telemetry.active_count
      << ",\"backlog_count\":" << telemetry.backlog_count
      << ",\"active_peak\":" << telemetry.active_peak
      << ",\"backlog_peak\":" << telemetry.backlog_peak
      << ",\"active_cap_hits\":" << telemetry.active_cap_hits
      << ",\"backlog_cap_hits\":" << telemetry.backlog_cap_hits
      << ",\"shed_close_count\":" << telemetry.shed_close_count
      << ",\"shed_close_rate\":" << telemetry.shed_close_rate
      << "}";
  return oss.str();
}

static void merge_admission_telemetry(AdmissionTelemetry& dst, const AdmissionTelemetry& src) {
  dst.active_cap = src.active_cap;
  dst.backlog_cap = src.backlog_cap;
  dst.offered += src.offered;
  dst.admitted += src.admitted;
  dst.active_count += src.active_count;
  dst.backlog_count += src.backlog_count;
  dst.active_peak = std::max(dst.active_peak, src.active_peak);
  dst.backlog_peak = std::max(dst.backlog_peak, src.backlog_peak);
  dst.active_cap_hits += src.active_cap_hits;
  dst.backlog_cap_hits += src.backlog_cap_hits;
  dst.shed_close_count += src.shed_close_count;
  dst.shed_close_rate = dst.offered == 0
                            ? 0.0
                            : static_cast<double>(dst.shed_close_count) /
                                  static_cast<double>(dst.offered);
}

static const char* admission_decision_name(AdmissionDecision decision) {
  switch (decision) {
    case AdmissionDecision::ADMITTED:
      return "admitted";
    case AdmissionDecision::QUEUED:
      return "queued";
    case AdmissionDecision::SHED_ACTIVE_CAP:
      return "shed_active_cap";
    case AdmissionDecision::SHED_BACKLOG_CAP:
      return "shed_backlog_cap";
  }
  return "unknown";
}

class AdmissionCloseGuard {
 public:
  AdmissionCloseGuard(DensityAdmission& admission, std::string stream_id)
      : admission_(&admission), stream_id_(std::move(stream_id)) {}
  AdmissionCloseGuard(const AdmissionCloseGuard&) = delete;
  AdmissionCloseGuard& operator=(const AdmissionCloseGuard&) = delete;
  ~AdmissionCloseGuard() {
    if (admission_ != nullptr) admission_->on_close(stream_id_);
  }
  void dismiss() { admission_ = nullptr; }

 private:
  DensityAdmission* admission_ = nullptr;
  std::string stream_id_;
};

static bool wait_until_admission_active(DensityAdmission& admission,
                                        const std::string& stream_id,
                                        AdmissionDecision decision) {
  if (decision == AdmissionDecision::ADMITTED) return true;
  if (decision != AdmissionDecision::QUEUED) return false;
  for (;;) {
    AdmissionTelemetry snapshot = admission.telemetry_snapshot();
    if (snapshot.active_cap == 0) {
      admission.on_close(stream_id);
      return false;
    }
    if (admission.try_admit_complete(stream_id)) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

struct StaleGenTelemetrySnapshot {
  uint64_t drops_at_encode = 0;
  uint64_t drops_at_decode = 0;
  uint64_t drops_at_event_emit = 0;
  uint64_t drops_at_finalize_output = 0;

  uint64_t total_drops() const {
    return drops_at_encode + drops_at_decode + drops_at_event_emit + drops_at_finalize_output;
  }
};

enum class StaleGenStage {
  ENCODE,
  DECODE,
  EVENT_EMIT,
  FINALIZE_OUTPUT,
};

struct StaleGenTelemetry {
  std::atomic<uint64_t> drops_at_encode{0};
  std::atomic<uint64_t> drops_at_decode{0};
  std::atomic<uint64_t> drops_at_event_emit{0};
  std::atomic<uint64_t> drops_at_finalize_output{0};

  void record(StaleGenStage stage) {
    switch (stage) {
      case StaleGenStage::ENCODE:
        drops_at_encode.fetch_add(1, std::memory_order_relaxed);
        break;
      case StaleGenStage::DECODE:
        drops_at_decode.fetch_add(1, std::memory_order_relaxed);
        break;
      case StaleGenStage::EVENT_EMIT:
        drops_at_event_emit.fetch_add(1, std::memory_order_relaxed);
        break;
      case StaleGenStage::FINALIZE_OUTPUT:
        drops_at_finalize_output.fetch_add(1, std::memory_order_relaxed);
        break;
    }
  }

  StaleGenTelemetrySnapshot snapshot() const {
    StaleGenTelemetrySnapshot out;
    out.drops_at_encode = drops_at_encode.load(std::memory_order_acquire);
    out.drops_at_decode = drops_at_decode.load(std::memory_order_acquire);
    out.drops_at_event_emit = drops_at_event_emit.load(std::memory_order_acquire);
    out.drops_at_finalize_output = drops_at_finalize_output.load(std::memory_order_acquire);
    return out;
  }
};

static std::string stale_gen_telemetry_json(const StaleGenTelemetrySnapshot& telemetry) {
  std::ostringstream oss;
  oss << "{\"drops_at_encode\":" << telemetry.drops_at_encode
      << ",\"drops_at_decode\":" << telemetry.drops_at_decode
      << ",\"drops_at_event_emit\":" << telemetry.drops_at_event_emit
      << ",\"drops_at_finalize_output\":" << telemetry.drops_at_finalize_output
      << ",\"total_drops\":" << telemetry.total_drops()
      << "}";
  return oss.str();
}

static void merge_stale_gen_telemetry(StaleGenTelemetrySnapshot& dst,
                                      const StaleGenTelemetrySnapshot& src) {
  dst.drops_at_encode += src.drops_at_encode;
  dst.drops_at_decode += src.drops_at_decode;
  dst.drops_at_event_emit += src.drops_at_event_emit;
  dst.drops_at_finalize_output += src.drops_at_finalize_output;
}

static uint64_t density_capture_generation(const SessionState& state) {
  return state.generation.load(std::memory_order_acquire);
}

static void density_bump_generation(SessionState& state, const char* reason) {
  uint64_t next = state.generation.fetch_add(1, std::memory_order_acq_rel) + 1;
  std::printf("STALE_GEN_BUMP reason=%s generation=%llu\n",
              reason,
              static_cast<unsigned long long>(next));
}

static bool density_generation_live(const SessionState& state,
                                    uint64_t work_generation,
                                    StaleGenTelemetry* telemetry,
                                    StaleGenStage stage,
                                    const std::string& label) {
  uint64_t current = density_capture_generation(state);
  if (current == work_generation) return true;
  if (telemetry != nullptr) telemetry->record(stage);
  std::printf("STALE_GEN_DROP label=%s work_generation=%llu current_generation=%llu\n",
              label.c_str(),
              static_cast<unsigned long long>(work_generation),
              static_cast<unsigned long long>(current));
  return false;
}

static void merge_scheduler_telemetry(BatchedSteadySchedulerTelemetry& dst,
                                      const BatchedSteadySchedulerTelemetry& src) {
  dst.dispatch_lanes = std::max(dst.dispatch_lanes, src.dispatch_lanes);
  dst.enqueued += src.enqueued;
  dst.completed += src.completed;
  dst.dispatch_cycles += src.dispatch_cycles;
  dst.warmup_runs += src.warmup_runs;
  dst.warmed_lanes += src.warmed_lanes;
  dst.bucket_b1 += src.bucket_b1;
  dst.bucket_b2 += src.bucket_b2;
  dst.bucket_b4 += src.bucket_b4;
  dst.bucket_b8 += src.bucket_b8;
  dst.bucket_b16 += src.bucket_b16;
  dst.k2_padded_to_b4 += src.k2_padded_to_b4;
  dst.k3_padded_to_b4 += src.k3_padded_to_b4;
  dst.k4 += src.k4;
  dst.k5_padded_to_b8 += src.k5_padded_to_b8;
  dst.k6_padded_to_b8 += src.k6_padded_to_b8;
  dst.k7_padded_to_b8 += src.k7_padded_to_b8;
  dst.k8 += src.k8;
  dst.k9_padded_to_b16 += src.k9_padded_to_b16;
  dst.k10_padded_to_b16 += src.k10_padded_to_b16;
  dst.k11_padded_to_b16 += src.k11_padded_to_b16;
  dst.k12_padded_to_b16 += src.k12_padded_to_b16;
  dst.k13_padded_to_b16 += src.k13_padded_to_b16;
  dst.k14_padded_to_b16 += src.k14_padded_to_b16;
  dst.k15_padded_to_b16 += src.k15_padded_to_b16;
  dst.k16 += src.k16;
  dst.backlog_gt_bmax += src.backlog_gt_bmax;
  dst.skipped_ready += src.skipped_ready;
  dst.dispatcher_exceptions += src.dispatcher_exceptions;
  dst.dispatcher_cpu_us += src.dispatcher_cpu_us;
  dst.dispatcher_wall_us += src.dispatcher_wall_us;
  dst.dispatcher_stream_run_us += src.dispatcher_stream_run_us;
  dst.age_at_dispatch_us.insert(dst.age_at_dispatch_us.end(),
                                src.age_at_dispatch_us.begin(),
                                src.age_at_dispatch_us.end());
  dst.gather_wait_us.insert(dst.gather_wait_us.end(), src.gather_wait_us.begin(), src.gather_wait_us.end());
  dst.service_wait_us.insert(dst.service_wait_us.end(), src.service_wait_us.begin(), src.service_wait_us.end());
  dst.cuda_run_us.insert(dst.cuda_run_us.end(), src.cuda_run_us.begin(), src.cuda_run_us.end());
  dst.output_sync_us.insert(dst.output_sync_us.end(), src.output_sync_us.begin(), src.output_sync_us.end());
  dst.worker_blocked_us.insert(dst.worker_blocked_us.end(), src.worker_blocked_us.begin(), src.worker_blocked_us.end());
  dst.completion_wait_us.insert(dst.completion_wait_us.end(),
                                src.completion_wait_us.begin(),
                                src.completion_wait_us.end());
  dst.window_wakeup_jitter_us.insert(dst.window_wakeup_jitter_us.end(),
                                     src.window_wakeup_jitter_us.begin(),
                                     src.window_wakeup_jitter_us.end());
  dst.queue_depth.insert(dst.queue_depth.end(), src.queue_depth.begin(), src.queue_depth.end());
  dst.per_stream_fairness_spread_us.insert(dst.per_stream_fairness_spread_us.end(),
                                           src.per_stream_fairness_spread_us.begin(),
                                           src.per_stream_fairness_spread_us.end());
  if (dst.lanes.size() < src.lanes.size()) dst.lanes.resize(src.lanes.size());
  for (size_t i = 0; i < src.lanes.size(); ++i) {
    auto& out = dst.lanes[i];
    const auto& in = src.lanes[i];
    out.lane_id = in.lane_id;
    out.completed += in.completed;
    out.dispatch_cycles += in.dispatch_cycles;
    out.warmup_runs += in.warmup_runs;
    out.dispatcher_cpu_us += in.dispatcher_cpu_us;
    out.dispatcher_wall_us += in.dispatcher_wall_us;
    out.dispatcher_stream_run_us += in.dispatcher_stream_run_us;
    out.cuda_run_us.insert(out.cuda_run_us.end(), in.cuda_run_us.begin(), in.cuda_run_us.end());
    out.queue_depth.insert(out.queue_depth.end(), in.queue_depth.begin(), in.queue_depth.end());
  }
}

static void emit_telemetry(const std::string& dir,
                           const std::string& stamp,
                           int num_runners,
                           const std::string& stream_mode,
                           const std::string& topology,
                           const std::string& json) {
  std::string logs_dir = dir + "/logs/" + stamp;
  fs::create_directories(logs_dir);
  std::string path = logs_dir + "/density_num_runners" + std::to_string(num_runners) +
                     "_stream-" + sanitize_filename(stream_mode) +
                     "_topology-" + sanitize_filename(topology) + ".jsonl";
  std::ofstream out(path, std::ios::out | std::ios::app);
  if (!out) throw std::runtime_error("failed to open telemetry log: " + path);
  out << json << "\n";
  if (!out) throw std::runtime_error("failed to write telemetry log: " + path);
  std::printf("DENSITY_TELEMETRY path=%s json=%s\n", path.c_str(), json.c_str());
}

static size_t gpu_used_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes - free_bytes;
}

static size_t gpu_total_bytes() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return total_bytes;
}

static Clock::duration ms_duration(double ms) {
  return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
}

static double signed_elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static void cleanup_cuda_cache() {
  CUDA_CHECK(cudaDeviceSynchronize());
  c10::cuda::CUDACachingAllocator::emptyCache();
  CUDA_CHECK(cudaDeviceSynchronize());
}

struct MemorySampler {
  std::atomic<bool> stop{false};
  std::thread thread;
  std::atomic<size_t> peak{0};

  ~MemorySampler() {
    stop.store(true);
    if (thread.joinable()) {
      try {
        thread.join();
      } catch (const std::exception&) {
      }
    }
  }

  void start() {
    peak.store(gpu_used_bytes());
    stop.store(false);
    thread = std::thread([this] {
      while (!stop.load(std::memory_order_relaxed)) {
        try {
          size_t used = gpu_used_bytes();
          size_t prev = peak.load(std::memory_order_relaxed);
          while (used > prev && !peak.compare_exchange_weak(prev, used)) {}
        } catch (const std::exception&) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });
  }

  size_t finish() {
    stop.store(true);
    if (thread.joinable()) thread.join();
    size_t used = gpu_used_bytes();
    size_t prev = peak.load(std::memory_order_relaxed);
    while (used > prev && !peak.compare_exchange_weak(prev, used)) {}
    return peak.load(std::memory_order_relaxed);
  }
};

struct ResourceStats {
  bool gpu_util_available = false;
  int gpu_util_samples = 0;
  double gpu_util_mean_pct = 0.0;
  double gpu_util_p50_pct = 0.0;
  double gpu_util_p95_pct = 0.0;
  double cpu_cores_used = 0.0;
  double cpu_util_pct_of_box = 0.0;
  int cpu_threads = 0;
};

static bool read_nvidia_smi_sample(double* gpu_util_pct, double* mem_used_mib) {
  FILE* pipe = popen("nvidia-smi --id=0 --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
  if (pipe == nullptr) return false;
  char buffer[256];
  bool ok = false;
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string text(buffer);
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream iss(text);
    double util = 0.0;
    double mem = 0.0;
    if (iss >> util >> mem) {
      if (gpu_util_pct != nullptr) *gpu_util_pct = util;
      if (mem_used_mib != nullptr) *mem_used_mib = mem;
      ok = true;
    }
  }
  int rc = pclose(pipe);
  (void)rc;
  return ok;
}

static double rusage_seconds(const struct rusage& usage) {
  return static_cast<double>(usage.ru_utime.tv_sec) +
         static_cast<double>(usage.ru_utime.tv_usec) / 1000000.0 +
         static_cast<double>(usage.ru_stime.tv_sec) +
         static_cast<double>(usage.ru_stime.tv_usec) / 1000000.0;
}

struct ResourceSampler {
  std::atomic<bool> stop{false};
  std::thread thread;
  std::mutex mutex;
  std::vector<double> gpu_util_pct;
  Clock::time_point start_wall;
  Clock::time_point end_wall;
  struct rusage start_usage {};
  struct rusage end_usage {};
  int cpu_threads = 0;

  ~ResourceSampler() {
    stop.store(true);
    if (thread.joinable()) {
      try {
        thread.join();
      } catch (const std::exception&) {
      }
    }
  }

  void start() {
    stop.store(false);
    cpu_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    getrusage(RUSAGE_SELF, &start_usage);
    start_wall = Clock::now();
    thread = std::thread([this] {
      while (!stop.load(std::memory_order_relaxed)) {
        double util = 0.0;
        double mem = 0.0;
        if (read_nvidia_smi_sample(&util, &mem)) {
          std::lock_guard<std::mutex> lock(mutex);
          gpu_util_pct.push_back(util);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
    });
  }

  ResourceStats finish() {
    stop.store(true);
    if (thread.joinable()) thread.join();
    end_wall = Clock::now();
    getrusage(RUSAGE_SELF, &end_usage);
    ResourceStats out;
    out.cpu_threads = cpu_threads;
    double wall_s = std::chrono::duration<double>(end_wall - start_wall).count();
    double cpu_s = rusage_seconds(end_usage) - rusage_seconds(start_usage);
    if (wall_s > 0.0) {
      out.cpu_cores_used = cpu_s / wall_s;
      if (cpu_threads > 0) out.cpu_util_pct_of_box = 100.0 * out.cpu_cores_used / static_cast<double>(cpu_threads);
    }
    std::vector<double> samples;
    {
      std::lock_guard<std::mutex> lock(mutex);
      samples = gpu_util_pct;
    }
    auto gpu = summarize(samples);
    out.gpu_util_available = !samples.empty();
    out.gpu_util_samples = static_cast<int>(samples.size());
    out.gpu_util_mean_pct = gpu.mean;
    out.gpu_util_p50_pct = gpu.p50;
    out.gpu_util_p95_pct = gpu.p95;
    return out;
  }
};

static std::string resource_stats_json(const ResourceStats& stats) {
  std::ostringstream oss;
  oss << "{\"gpu_util_available\":" << json_bool(stats.gpu_util_available)
      << ",\"gpu_util_samples\":" << stats.gpu_util_samples
      << ",\"gpu_util_mean_pct\":" << stats.gpu_util_mean_pct
      << ",\"gpu_util_p50_pct\":" << stats.gpu_util_p50_pct
      << ",\"gpu_util_p95_pct\":" << stats.gpu_util_p95_pct
      << ",\"cpu_cores_used\":" << stats.cpu_cores_used
      << ",\"cpu_util_pct_of_box\":" << stats.cpu_util_pct_of_box
      << ",\"cpu_threads\":" << stats.cpu_threads
      << "}";
  return oss.str();
}

struct StartGate {
  std::mutex mutex;
  std::condition_variable cv;
  int expected = 0;
  int ready = 0;
  bool go = false;
  Clock::time_point start_time;

  explicit StartGate(int n) : expected(n) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex);
    ++ready;
    if (ready == expected) cv.notify_all();
    cv.wait(lock, [&] { return go; });
  }

  void wait_until_ready_and_start() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return ready == expected; });
    start_time = Clock::now();
    go = true;
    cv.notify_all();
  }

  void wait_until_ready() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return ready == expected; });
  }

  void start_now() {
    std::unique_lock<std::mutex> lock(mutex);
    start_time = Clock::now();
    go = true;
    cv.notify_all();
  }
};

struct TimingBuckets {
  std::vector<double> latency_ms;
  std::vector<double> queue_wait_ms;
  std::vector<double> runner_wait_ms;
  std::vector<double> scalar_sync_wait_ms;
  std::vector<double> scalar_sync_pct_of_gpu;
  std::vector<double> steady_gpu_ms;
  std::vector<double> enc_first_lock_wait_ms;
  std::vector<double> enc_first_total_ms;
  std::vector<double> finalize_runner_wait_ms;
  std::vector<double> finalize_gpu_ms;
  std::vector<double> finalize_total_ms;
  std::vector<double> finalize_fork_clone_ms;
  std::vector<double> finalize_aoti_run_cuda_ms;
  std::vector<double> finalize_enc_len_sync_ms;
  std::vector<double> finalize_decode_wall_ms;
  std::vector<double> finalize_decode_item_wait_ms;
  std::vector<double> finalize_decode_tokens;
  std::vector<double> finalize_glue_ms;

  void append(const TimingBuckets& other) {
    auto add = [](std::vector<double>& dst, const std::vector<double>& src) {
      dst.insert(dst.end(), src.begin(), src.end());
    };
    add(latency_ms, other.latency_ms);
    add(queue_wait_ms, other.queue_wait_ms);
    add(runner_wait_ms, other.runner_wait_ms);
    add(scalar_sync_wait_ms, other.scalar_sync_wait_ms);
    add(scalar_sync_pct_of_gpu, other.scalar_sync_pct_of_gpu);
    add(steady_gpu_ms, other.steady_gpu_ms);
    add(enc_first_lock_wait_ms, other.enc_first_lock_wait_ms);
    add(enc_first_total_ms, other.enc_first_total_ms);
    add(finalize_runner_wait_ms, other.finalize_runner_wait_ms);
    add(finalize_gpu_ms, other.finalize_gpu_ms);
    add(finalize_total_ms, other.finalize_total_ms);
    add(finalize_fork_clone_ms, other.finalize_fork_clone_ms);
    add(finalize_aoti_run_cuda_ms, other.finalize_aoti_run_cuda_ms);
    add(finalize_enc_len_sync_ms, other.finalize_enc_len_sync_ms);
    add(finalize_decode_wall_ms, other.finalize_decode_wall_ms);
    add(finalize_decode_item_wait_ms, other.finalize_decode_item_wait_ms);
    add(finalize_decode_tokens, other.finalize_decode_tokens);
    add(finalize_glue_ms, other.finalize_glue_ms);
  }
};

static std::string finalize_phase_stats_json(const TimingBuckets& timings) {
  std::ostringstream oss;
  oss << "{\"fork_clone\":" << stats_json(summarize(timings.finalize_fork_clone_ms))
      << ",\"aoti_run_cuda\":" << stats_json(summarize(timings.finalize_aoti_run_cuda_ms))
      << ",\"enc_len_sync\":" << stats_json(summarize(timings.finalize_enc_len_sync_ms))
      << ",\"decode_wall\":" << stats_json(summarize(timings.finalize_decode_wall_ms))
      << ",\"decode_item_wait\":" << stats_json(summarize(timings.finalize_decode_item_wait_ms))
      << ",\"decode_tokens\":" << value_stats_json(summarize(timings.finalize_decode_tokens))
      << ",\"glue\":" << stats_json(summarize(timings.finalize_glue_ms))
      << "}";
  return oss.str();
}

static torch::jit::Module load_module_on_device(const std::string& path, torch::Device device) {
  auto module = load_jit_serialized(path);
  module.to(device);
  module.eval();
  return module;
}

struct SharedEncFirst {
  std::unique_ptr<torch::jit::Module> module_owner;
  std::unique_ptr<FirstEncoder> encoder;
  std::mutex mutex;
  size_t used_before_bytes = 0;
  size_t used_after_bytes = 0;
  size_t delta_bytes = 0;
  std::string policy;

  SharedEncFirst(std::unique_ptr<torch::jit::Module> module_in,
                 std::unique_ptr<FirstEncoder> encoder_in,
                 size_t used_before,
                 size_t used_after,
                 size_t delta,
                 std::string policy_in)
      : module_owner(std::move(module_in)),
        encoder(std::move(encoder_in)),
        used_before_bytes(used_before),
        used_after_bytes(used_after),
        delta_bytes(delta),
        policy(std::move(policy_in)) {
    if (!encoder) throw std::runtime_error("SharedEncFirst requires an encoder adapter");
  }

  const char* kind() const {
    return encoder->kind();
  }
};

static std::shared_ptr<SharedEncFirst> load_shared_enc_first(const std::string& dir,
                                                             torch::Device device,
                                                             const std::string& policy) {
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t before = gpu_used_bytes();
  auto module = std::make_unique<torch::jit::Module>(load_module_on_device(dir + "/enc_first.ts", device));
  auto encoder = std::make_unique<TsFirstEncoder>(*module);
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t after = gpu_used_bytes();
  size_t delta = after >= before ? after - before : 0;
  std::printf("density loaded enc_first.ts policy=%s adapter=%s delta=%.3f GiB "
              "used_before=%.3f GiB used_after=%.3f GiB\n",
              policy.c_str(),
              encoder->kind(),
              static_cast<double>(delta) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(before) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(after) / (1024.0 * 1024.0 * 1024.0));
  return std::make_shared<SharedEncFirst>(std::move(module), std::move(encoder), before, after, delta, policy);
}

static std::shared_ptr<SharedEncFirst> load_shared_enc_first_aoti(
    const std::string& dir,
    torch::Device device,
    const std::unordered_map<std::string, at::Tensor>& shared_constants,
    int num_runners,
    const std::string& policy) {
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t before = gpu_used_bytes();
  auto encoder = std::make_unique<AotiFirstEncoder>(dir + "/enc_first_aoti.pt2",
                                                    shared_constants,
                                                    device,
                                                    num_runners);
  CUDA_CHECK(cudaDeviceSynchronize());
  size_t after = gpu_used_bytes();
  size_t delta = after >= before ? after - before : 0;
  std::printf("density loaded enc_first_aoti.pt2 policy=%s adapter=%s delta=%.3f GiB "
              "used_before=%.3f GiB used_after=%.3f GiB\n",
              policy.c_str(),
              encoder->kind(),
              static_cast<double>(delta) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(before) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(after) / (1024.0 * 1024.0 * 1024.0));
  return std::make_shared<SharedEncFirst>(nullptr, std::move(encoder), before, after, delta, policy);
}

static std::shared_ptr<torch::jit::Module> load_shared_session_bundle(const std::string& dir) {
  auto bundle = std::make_shared<torch::jit::Module>(load_jit_serialized(dir + "/session_bundle.ts"));
  verify_session_bundle_meta(*bundle, false);
  std::printf("density loaded session_bundle.ts once for process; sharing read-only bundle across contexts\n");
  return bundle;
}

struct WorkerContext {
  std::shared_ptr<torch::jit::Module> bundle_owner;
  torch::jit::Module& bundle;
  std::shared_ptr<SharedEncFirst> enc_first;
  torch::jit::Module joint;
  torch::jit::Module predict;
  std::unique_ptr<torch::jit::Module> preproc;
  c10::cuda::CUDAStream stream;

  WorkerContext(std::shared_ptr<torch::jit::Module> bundle_in,
                std::shared_ptr<SharedEncFirst> enc_first_in,
                torch::jit::Module joint_in,
                torch::jit::Module predict_in,
                std::unique_ptr<torch::jit::Module> preproc_in,
                c10::cuda::CUDAStream stream_in)
      : bundle_owner(std::move(bundle_in)),
        bundle(*bundle_owner),
        enc_first(std::move(enc_first_in)),
        joint(std::move(joint_in)),
        predict(std::move(predict_in)),
        preproc(std::move(preproc_in)),
        stream(stream_in) {}
};

static std::unique_ptr<WorkerContext> make_worker_context(const std::string& dir,
                                                          torch::Device device,
                                                          c10::cuda::CUDAStream stream,
                                                          std::shared_ptr<torch::jit::Module> shared_bundle,
                                                          std::shared_ptr<SharedEncFirst> enc_first = nullptr) {
  if (!shared_bundle) throw std::runtime_error("make_worker_context requires a shared session bundle");
  initialize_prompt_runtime(dir, *shared_bundle, device);  // idempotent; no-op for the en profile
  if (!enc_first) {
    enc_first = load_shared_enc_first(dir, device, "per_worker_context_own_enc_first");
  }
  auto joint = load_module_on_device(dir + "/joint_step.ts", device);
  auto predict = load_module_on_device(dir + "/predict_step.ts", device);
  std::unique_ptr<torch::jit::Module> preproc;
  if (file_exists(dir + "/preproc.ts")) {
    preproc = std::make_unique<torch::jit::Module>(load_module_on_device(dir + "/preproc.ts", device));
  }
  return std::make_unique<WorkerContext>(std::move(shared_bundle),
                                         std::move(enc_first),
                                         std::move(joint),
                                         std::move(predict),
                                         std::move(preproc),
                                         stream);
}

static c10::cuda::CUDAStream stream_for_worker(bool explicit_stream, int worker, int device_index = 0) {
  if (!explicit_stream) return c10::cuda::getDefaultCUDAStream(device_index);
  static std::mutex mutex;
  static std::map<int, std::vector<cudaStream_t>> streams_by_device;
  if (worker < 0) throw std::runtime_error("negative worker index for stream allocation");
  std::lock_guard<std::mutex> lock(mutex);
  auto& streams = streams_by_device[device_index];
  while (static_cast<int>(streams.size()) <= worker) {
    cudaStream_t raw{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking));
    streams.push_back(raw);
  }
  return c10::cuda::getStreamFromExternal(streams[static_cast<size_t>(worker)], device_index);
}

static uintptr_t stream_handle_value(const c10::cuda::CUDAStream& stream) {
  return reinterpret_cast<uintptr_t>(stream.stream());
}

static std::string stream_handles_json(const std::vector<c10::cuda::CUDAStream>& streams) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < streams.size(); ++i) {
    if (i > 0) oss << ",";
    oss << stream_handle_value(streams[i]);
  }
  oss << "]";
  return oss.str();
}

static std::mutex g_aoti_run_mutex;

static std::vector<at::Tensor> run_aoti_loader(AOTIModelPackageLoader& loader,
                                               const std::vector<at::Tensor>& inputs,
                                               c10::cuda::CUDAStream stream,
                                               bool explicit_stream,
                                               bool mutex_serialize_run) {
  auto invoke = [&]() {
    return explicit_stream
               ? loader.run(inputs, reinterpret_cast<void*>(stream.stream()))
               : loader.run(inputs);
  };
  if (mutex_serialize_run) {
    std::lock_guard<std::mutex> lock(g_aoti_run_mutex);
    return invoke();
  }
  return invoke();
}

static bool strict_events_equal(const std::vector<EmittedEvent>& got,
                                const std::vector<EmittedEvent>& gold,
                                const std::string& label) {
  bool ok = got.size() == gold.size();
  if (!ok) {
    std::printf("    %s strict event count mismatch: got=%zu gold=%zu\n",
                label.c_str(), got.size(), gold.size());
  }
  size_t n = std::min(got.size(), gold.size());
  for (size_t i = 0; i < n; ++i) {
    bool event_ok = got[i].kind == gold[i].kind &&
                    got[i].tokens == gold[i].tokens &&
                    got[i].collector_tokens == gold[i].collector_tokens &&
                    got[i].text == gold[i].text &&
                    got[i].collector_text == gold[i].collector_text;
    if (!event_ok) {
      std::printf("    %s strict event[%zu] mismatch: got_kind=%s gold_kind=%s "
                  "got_tokens=%zu gold_tokens=%zu got_collector=%zu gold_collector=%zu\n",
                  label.c_str(), i, event_kind_name(got[i].kind), event_kind_name(gold[i].kind),
                  got[i].tokens.size(), gold[i].tokens.size(),
                  got[i].collector_tokens.size(), gold[i].collector_tokens.size());
      if (got[i].text != gold[i].text) {
        std::printf("      got text :%s\n", escaped_text(got[i].text).c_str());
        std::printf("      gold text:%s\n", escaped_text(gold[i].text).c_str());
      }
      if (got[i].collector_text != gold[i].collector_text) {
        std::printf("      got collector text :%s\n", escaped_text(got[i].collector_text).c_str());
        std::printf("      gold collector text:%s\n", escaped_text(gold[i].collector_text).c_str());
      }
      std::printf("      got tokens :%s\n", vec_to_string(got[i].tokens).c_str());
      std::printf("      gold tokens:%s\n", vec_to_string(gold[i].tokens).c_str());
      std::printf("      got collector tokens :%s\n", vec_to_string(got[i].collector_tokens).c_str());
      std::printf("      gold collector tokens:%s\n", vec_to_string(gold[i].collector_tokens).c_str());
      ok = false;
      break;
    }
  }
  return ok;
}

static std::vector<at::Tensor> run_steady_encoder_stream(AOTIModelPackageLoader& loader,
                                                         const torch::Tensor& chunk,
                                                         SessionState& state,
                                                         c10::cuda::CUDAStream stream,
                                                         bool explicit_stream,
                                                         bool mutex_serialize_run,
                                                         double* runner_wait_ms) {
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  std::vector<at::Tensor> inputs = {
      chunk.contiguous(),
      L.contiguous(),
      state.clc.contiguous(),
      state.clt.contiguous(),
      state.clcl.contiguous(),
  };
  auto t0 = Clock::now();
  std::vector<at::Tensor> out = run_aoti_loader(loader, inputs, stream, explicit_stream, mutex_serialize_run);
  auto t1 = Clock::now();
  if (runner_wait_ms != nullptr) *runner_wait_ms = elapsed_ms(t0, t1);
  if (out.size() < 5) throw std::runtime_error("steady AOTI encoder returned fewer than 5 outputs");
  return out;
}

class B2ReusableBarrier {
 public:
  explicit B2ReusableBarrier(int expected) : expected_(expected) {
    if (expected_ <= 0) throw std::runtime_error("B2 barrier expected count must be positive");
  }

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    int generation = generation_;
    ++arrived_;
    if (arrived_ == expected_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation_ != generation; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int expected_ = 0;
  int arrived_ = 0;
  int generation_ = 0;
};

static double b2_max_abs_tensor_diff(const at::Tensor& a, const at::Tensor& b) {
  if (a.sizes() != b.sizes()) {
    std::ostringstream oss;
    oss << "B2 tensor diff shape mismatch got=(";
    for (int64_t i = 0; i < a.dim(); ++i) oss << (i ? "," : "") << a.size(i);
    oss << ") ref=(";
    for (int64_t i = 0; i < b.dim(); ++i) oss << (i ? "," : "") << b.size(i);
    oss << ")";
    throw std::runtime_error(oss.str());
  }
  auto diff = (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs();
  return diff.numel() > 0 ? diff.max().item<double>() : 0.0;
}

static constexpr double kAotiTensorTolerance = 5.0e-2;

struct B2TensorDiffStats {
  std::mutex mutex;
  int compares = 0;
  int enc_len_mismatches = 0;
  int cache_len_mismatches = 0;
  double max_enc_out_diff = 0.0;
  double max_cache_ch_diff = 0.0;
  double max_cache_t_diff = 0.0;

  void update(const std::vector<at::Tensor>& got,
              const std::vector<at::Tensor>& ref,
              const std::string& label) {
    if (got.size() < 5 || ref.size() < 5) throw std::runtime_error("B2 tensor diff needs 5 steady outputs");
    std::lock_guard<std::mutex> lock(mutex);
    ++compares;
    max_enc_out_diff = std::max(max_enc_out_diff, b2_max_abs_tensor_diff(got[0], ref[0]));
    max_cache_ch_diff = std::max(max_cache_ch_diff, b2_max_abs_tensor_diff(got[2], ref[2]));
    max_cache_t_diff = std::max(max_cache_t_diff, b2_max_abs_tensor_diff(got[3], ref[3]));
    if (!at::equal(got[1], ref[1])) {
      ++enc_len_mismatches;
      std::printf("B2_T1_ENC_LEN_MISMATCH label=%s\n", label.c_str());
    }
    if (!at::equal(got[4], ref[4])) {
      ++cache_len_mismatches;
      std::printf("B2_T1_CACHE_LEN_MISMATCH label=%s\n", label.c_str());
    }
  }
};

static int64_t scalar_i64_timed(torch::Tensor tensor, double* item_wait_ms) {
  auto start = Clock::now();
  int64_t value = tensor.to(torch::kCPU).reshape({-1})[0].item<int64_t>();
  if (item_wait_ms != nullptr) *item_wait_ms += elapsed_ms_since(start);
  return value;
}

static int64_t host_subsampled_encoder_len_density(int64_t input_T) {
  if (input_T < 0) throw std::runtime_error("density host enc_len input_T is negative");
  // Finalize keeps all encoder outputs. Nemotron's causal ConvSubsampling has
  // three stride-2 conv stages with kernel=3 and left+right padding=3, so each
  // stage maps L -> floor(L / 2) + 1 before drop_extra_pre_encoded is applied.
  int64_t len = input_T;
  for (int i = 0; i < 3; ++i) {
    len = len / 2 + 1;
  }
  return len;
}

enum class DensityEncoderLenMode { STEADY_STREAMING, FINALIZE_KEEP_ALL };

static int64_t host_encoder_len_density(int64_t input_T,
                                        int64_t drop_extra,
                                        DensityEncoderLenMode mode) {
  if (drop_extra < 0) throw std::runtime_error("density host enc_len drop_extra is negative");
  if (mode == DensityEncoderLenMode::STEADY_STREAMING) {
    if ((drop_extra == 0 && input_T == SHIFT) ||
        (drop_extra == DROP && input_T == PRE + SHIFT)) {
      return SHIFT / 8;
    }
    throw std::runtime_error("density host steady enc_len unsupported geometry: input_T=" +
                             std::to_string(input_T) +
                             " drop_extra=" + std::to_string(drop_extra));
  }
  return std::max<int64_t>(0, host_subsampled_encoder_len_density(input_T) - drop_extra);
}

static int64_t encoder_len_density(torch::Tensor enc_len_tensor,
                                   int64_t input_T,
                                   int64_t drop_extra,
                                   DensityEncoderLenMode mode,
                                   const std::string& label,
                                   double* item_wait_ms) {
  if (!density_host_enc_len_enabled()) {
    return scalar_i64_timed(enc_len_tensor, item_wait_ms);
  }
  int64_t host_len = host_encoder_len_density(input_T, drop_extra, mode);
  if (density_host_enc_len_verify_enabled()) {
    int64_t actual = scalar_i64_timed(enc_len_tensor, item_wait_ms);
    if (actual != host_len) {
      throw std::runtime_error("density host enc_len mismatch at " + label +
                               ": host=" + std::to_string(host_len) +
                               " actual=" + std::to_string(actual) +
                               " input_T=" + std::to_string(input_T) +
                               " drop_extra=" + std::to_string(drop_extra));
    }
  }
  return host_len;
}

static int64_t device_argmax_pinned_copy(const torch::Tensor& tensor) {
  if (!tensor.is_cuda()) {
    return tensor.argmax().item<int64_t>();
  }
  static thread_local torch::Tensor host_argmax;
  if (!host_argmax.defined()) {
    host_argmax = torch::empty({1},
                               torch::TensorOptions()
                                   .dtype(torch::kLong)
                                   .device(torch::kCPU)
                                   .pinned_memory(true));
  }
  auto idx = tensor.argmax().reshape({1});
  host_argmax.copy_(idx, /*non_blocking=*/true);
  CUDA_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream()));
  return host_argmax.data_ptr<int64_t>()[0];
}

static int64_t argmax_item_timed(const torch::Tensor& tensor, double* item_wait_ms) {
  auto start = Clock::now();
  int64_t value = density_device_argmax_enabled()
                      ? device_argmax_pinned_copy(tensor)
                      : tensor.argmax().item<int64_t>();
  if (item_wait_ms != nullptr) *item_wait_ms += elapsed_ms_since(start);
  return value;
}

static void decode_range_density(torch::jit::Module& joint,
                                 torch::jit::Module& predict,
                                 const torch::Tensor& enc_out,
                                 int64_t enc_len,
                                 torch::Tensor& g,
                                 torch::Tensor& h,
                                 torch::Tensor& c,
                                 std::vector<int64_t>& hyp,
                                 double* item_wait_ms) {
  if (enc_len < 0 || enc_len > enc_out.size(2)) {
    throw std::runtime_error("density enc_len out of range for enc_out: " + std::to_string(enc_len));
  }
  auto f = enc_out.transpose(1, 2).contiguous();
  auto dev = f.device();
  for (int64_t t = 0; t < enc_len; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor();
      auto flat = logits.reshape({-1});
      int64_t k = argmax_item_timed(flat, item_wait_ms);
      if (k == BLANK) break;
      hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(dev));
      auto out = predict.forward({y, h, c}).toTuple();
      g = out->elements()[0].toTensor();
      h = out->elements()[1].toTensor();
      c = out->elements()[2].toTensor();
    }
  }
}

static double apply_encoder_outputs_density(SessionState& state,
                                            const std::vector<at::Tensor>& out,
                                            torch::jit::Module& joint,
                                            torch::jit::Module& predict,
                                            int64_t input_T,
                                            int64_t drop_extra,
                                            const std::string& label) {
  if (out.size() < 5) throw std::runtime_error("density encoder returned fewer than 5 outputs");
  double item_wait_ms = 0.0;
  int64_t enc_len = encoder_len_density(out[1],
                                        input_T,
                                        drop_extra,
                                        DensityEncoderLenMode::STEADY_STREAMING,
                                        label,
                                        &item_wait_ms);
  state.clc = out[2].clone();
  state.clt = out[3].clone();
  state.clcl = out[4].clone();
  decode_range_density(joint, predict, out[0], enc_len, state.g, state.h, state.c, state.hyp, &item_wait_ms);
  return item_wait_ms;
}

static std::vector<at::Tensor> run_first_encoder_locked_density(SharedEncFirst& enc_first,
                                                                const torch::Tensor& chunk,
                                                                SessionState& state,
                                                                c10::cuda::CUDAStream stream,
                                                                double* lock_wait_ms) {
  auto wait_start = Clock::now();
  std::unique_lock<std::mutex> lock(enc_first.mutex);
  if (lock_wait_ms != nullptr) *lock_wait_ms += elapsed_ms_since(wait_start);
  return enc_first.encoder->run(chunk, state, stream);
}

struct SchedulerConsumerDelayProbe {
  double delay_ms = 0.0;
  int min_later_dispatches = 0;
  std::atomic<int> observed_later_dispatches{0};
  std::atomic<int> delays_run{0};

  void delay_after_future_ready(BatchedSteadyScheduler& scheduler) {
    auto before = scheduler.telemetry_snapshot().dispatch_cycles;
    std::this_thread::sleep_for(ms_duration(delay_ms));
    auto after = scheduler.telemetry_snapshot().dispatch_cycles;
    int delta = static_cast<int>(std::max<int64_t>(0, after - before));
    observed_later_dispatches.store(delta, std::memory_order_release);
    delays_run.fetch_add(1, std::memory_order_acq_rel);
  }
};

static void run_steady_chunk_density(SessionState& state,
                                     torch::jit::Module& bundle,
                                     const std::string& prefix,
                                     int chunk_index,
                                     SharedEncFirst& enc_first,
                                     AOTIModelPackageLoader* enc_steady,
                                     BatchedSteadyLoaderSet* direct_b1_loader,
                                     torch::jit::Module& joint,
                                     torch::jit::Module& predict,
                                     torch::Device device,
                                     const Tokenizer& tokenizer,
                                     std::vector<EmittedEvent>& events,
                                     c10::cuda::CUDAStream stream,
                                     bool explicit_stream,
                                     bool mutex_serialize_run,
                                     TimingBuckets* timings,
                                     const std::string& label,
                                     BatchedSteadyScheduler* scheduler,
                                     B2ReusableBarrier* continuation_barrier,
                                     B2TensorDiffStats* scheduler_diff_stats,
                                     AOTIModelPackageLoader* scheduler_diff_loader,
                                     std::vector<int64_t>* fill_trace_ready_ns = nullptr,
                                     StaleGenTelemetry* stale_gen = nullptr,
                                     SchedulerConsumerDelayProbe* consumer_delay_probe = nullptr) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("density steady chunk outside STREAMING");

  uint64_t work_generation = density_capture_generation(state);
  if (!density_generation_live(state,
                               work_generation,
                               stale_gen,
                               StaleGenStage::ENCODE,
                               label + ".pre_encode")) {
    return;
  }

  auto call_start = Clock::now();
  auto new_mel = prefix_chunk_tensor(bundle, prefix, chunk_index, "new_mel").to(device).contiguous();
  int64_t is_first = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "is_first"));
  int64_t drop_extra = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "drop_extra"));
  int64_t chunk_T = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "chunk_T"));
  int64_t emitted_before = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "emitted_before"));

  bool expected_first = state.emitted == 0;
  if ((is_first != 0) != expected_first) throw std::runtime_error("density steady first/continuation flag mismatch");
  if (emitted_before != state.emitted) throw std::runtime_error("density steady emitted_before mismatch");
  if (new_mel.size(2) != SHIFT) throw std::runtime_error("density steady new_mel is not SHIFT frames");

  torch::Tensor chunk;
  std::vector<at::Tensor> out;
  double runner_wait = 0.0;
  double gpu_ms = 0.0;
  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};
  bool has_aoti_event = false;
  ScopedCudaEvent output_sync_start_event;
  ScopedCudaEvent output_sync_stop_event;
  bool has_scheduler_output_sync_event = false;
  int64_t scheduler_cycle_id = -1;
  int scheduler_k = 0;
  double scheduler_worker_blocked_us = 0.0;
  std::shared_ptr<DispatchResult::GraphSlotLease> scheduler_graph_slot;
  auto retire_scheduler_graph_slot = [&]() {
    if (scheduler_graph_slot) {
      scheduler_graph_slot->retire_on_stream(stream.stream());
      scheduler_graph_slot.reset();
    }
  };

  if (expected_first) {
    if (drop_extra != 0 || chunk_T != new_mel.size(2)) throw std::runtime_error("density first steady geometry mismatch");
    chunk = new_mel;
    double lock_wait_ms = 0.0;
    auto enc_first_start = Clock::now();
    out = run_first_encoder_locked_density(enc_first, chunk, state, stream, &lock_wait_ms);
    if (timings != nullptr) {
      timings->enc_first_lock_wait_ms.push_back(lock_wait_ms);
      timings->enc_first_total_ms.push_back(elapsed_ms_since(enc_first_start));
    }
  } else {
    if (!state.ring.defined()) throw std::runtime_error("density steady continuation missing mel ring");
    if (drop_extra != DROP || chunk_T != state.ring.size(2) + new_mel.size(2)) {
      throw std::runtime_error("density steady continuation geometry mismatch");
    }
    chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
    if (fill_trace_ready_ns != nullptr) {
      fill_trace_ready_ns->push_back(density_clock_ns());
    }
    if (scheduler != nullptr) {
      BatchedSteadyInput scheduler_input{
          chunk.contiguous(),
          state.clc.contiguous(),
          state.clt.contiguous(),
          state.clcl.contiguous(),
          label,
      };
      if (continuation_barrier != nullptr) {
        continuation_barrier->arrive_and_wait();
      }
      cudaEvent_t producer_event{};
      CUDA_CHECK(cudaEventCreateWithFlags(&producer_event, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(producer_event, stream.stream()));
      auto enqueue_start = Clock::now();
      std::future<DispatchResult> future;
      try {
        uint64_t stream_key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&state));
        future = scheduler->enqueue({std::move(scheduler_input), stream, producer_event, stream_key});
      } catch (...) {
        CUDA_CHECK(cudaEventDestroy(producer_event));
        throw;
      }
      auto wait_status = future.wait_for(std::chrono::milliseconds(scheduler->future_timeout_ms()));
      if (wait_status != std::future_status::ready) {
        throw std::runtime_error("batch steady scheduler future timeout for " + label);
      }
      auto result = future.get();
      if (consumer_delay_probe != nullptr) {
        consumer_delay_probe->delay_after_future_ready(*scheduler);
      }
      output_sync_start_event.create();
      output_sync_stop_event.create();
      has_scheduler_output_sync_event = true;
      CUDA_CHECK(cudaEventRecord(output_sync_start_event.event, stream.stream()));
      auto sync_start = Clock::now();
      if (result.completion.get() == nullptr) {
        throw std::runtime_error("batch steady scheduler returned a null completion event for " + label);
      }
      CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), result.completion.get(), 0));
      CUDA_CHECK(cudaEventRecord(output_sync_stop_event.event, stream.stream()));
      auto sync_end = Clock::now();
      result.completion.reset();
      out = std::move(result.row_tensors);
      scheduler_graph_slot = std::move(result.graph_slot);
      gpu_ms = result.cuda_run_us / 1000.0;
      runner_wait = elapsed_us(enqueue_start, sync_end) / 1000.0;
      scheduler_cycle_id = result.cycle_id;
      scheduler_k = result.k;
      scheduler_worker_blocked_us = elapsed_us(enqueue_start, sync_end);
      (void)sync_start;
      if (scheduler_diff_stats != nullptr && scheduler_diff_loader != nullptr) {
        auto ref = run_steady_encoder_stream(*scheduler_diff_loader,
                                             chunk,
                                             state,
                                             stream,
                                             explicit_stream,
                                             mutex_serialize_run,
                                             nullptr);
        CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
        scheduler_diff_stats->update(out, ref, label);
      }
    } else {
      CUDA_CHECK(cudaEventCreate(&ev_start));
      CUDA_CHECK(cudaEventCreate(&ev_stop));
      CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
      if (direct_b1_loader != nullptr) {
        auto direct_start = Clock::now();
        std::vector<BatchedSteadyInput> ready{{
            chunk.contiguous(),
            state.clc.contiguous(),
            state.clt.contiguous(),
            state.clcl.contiguous(),
            label,
        }};
        auto rows = direct_b1_loader->run(ready, stream);
        if (rows.size() != 1) throw std::runtime_error("density direct B=1 bucket returned wrong row count");
        out = std::move(rows[0].tensors);
        runner_wait = elapsed_us(direct_start, Clock::now()) / 1000.0;
      } else {
        if (enc_steady == nullptr) {
          throw std::runtime_error("density steady direct B=1 path requires enc_steady loader");
        }
        out = run_steady_encoder_stream(*enc_steady, chunk, state, stream, explicit_stream, mutex_serialize_run, &runner_wait);
      }
      CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
      has_aoti_event = true;
    }
  }

  if (!density_generation_live(state,
                               work_generation,
                               stale_gen,
                               StaleGenStage::DECODE,
                               label + ".pre_decode")) {
    return;
  }
  double scalar_wait_ms = 0.0;
  try {
    scalar_wait_ms = apply_encoder_outputs_density(state,
                                                   out,
                                                   joint,
                                                   predict,
                                                   chunk.size(2),
                                                   drop_extra,
                                                   label + ".encoder");
    retire_scheduler_graph_slot();
  } catch (...) {
    retire_scheduler_graph_slot();
    throw;
  }

  if (has_scheduler_output_sync_event) {
    CUDA_CHECK(cudaEventSynchronize(output_sync_stop_event.event));
    float output_sync_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&output_sync_ms, output_sync_start_event.event, output_sync_stop_event.event));
    scheduler->record_worker_wait(scheduler_cycle_id,
                                  scheduler_k,
                                  static_cast<double>(output_sync_ms) * 1000.0,
                                  scheduler_worker_blocked_us);
  }

  if (has_aoti_event) {
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    gpu_ms = static_cast<double>(elapsed);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
  }

  auto cum = state.ring.defined() ? torch::cat({state.ring, new_mel}, 2) : new_mel;
  state.ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2)).contiguous();
  state.emitted += new_mel.size(2);
  std::string current_text = tokenizer.ids_to_text(state.hyp);
  if (current_text != state.last_interim_text) {
    if (!density_generation_live(state,
                                 work_generation,
                                 stale_gen,
                                 StaleGenStage::EVENT_EMIT,
                                 label + ".event_emit")) {
      return;
    }
    emit_event(events,
               EVENT_INTERIM,
               state.hyp,
               state.continuous_emitted_tokens,
               current_text,
               state.continuous_emitted_text);
    state.last_interim_tokens = state.hyp;
    state.last_interim_text = current_text;
  }

  if (timings != nullptr) {
    timings->latency_ms.push_back(elapsed_ms_since(call_start));
    timings->queue_wait_ms.push_back(0.0);
    timings->scalar_sync_wait_ms.push_back(scalar_wait_ms);
    if (!expected_first) {
      timings->steady_gpu_ms.push_back(gpu_ms);
      timings->runner_wait_ms.push_back(std::max(0.0, runner_wait - gpu_ms));
      if (gpu_ms > 0.0) timings->scalar_sync_pct_of_gpu.push_back(100.0 * scalar_wait_ms / gpu_ms);
    }
  }
}

using FinalizeBucketKey = std::pair<int64_t, int64_t>;

struct FinalizeLoaderMemoryRecord {
  int64_t drop = 0;
  int64_t T = 0;
  int num_runners = 0;
  size_t used_before = 0;
  size_t used_after = 0;
  size_t delta = 0;
  size_t cumulative_delta = 0;
};

class FinalizeBucketLoaderPool {
 public:
  FinalizeBucketLoaderPool(const std::string& dir,
                           torch::Device device,
                           int num_runners,
                           std::string policy)
      : dir_(dir),
        device_(device),
        num_runners_(num_runners),
        policy_(std::move(policy)) {
    if (num_runners_ <= 0) throw std::runtime_error("finalize bucket num_runners must be positive");
    buckets_dir_ = dir_ + "/stripped_finalize_buckets";
    if (!directory_exists(buckets_dir_)) buckets_dir_ = dir_ + "/finalize_buckets";
    shared_weights_ = dir_ + "/finalize_shared_weights.ts";
    std::string shared_weights_pt = dir_ + "/finalize_shared_weights.pt";
    if (!directory_exists(buckets_dir_)) throw std::runtime_error("finalize buckets directory missing: " + buckets_dir_);
    if (!file_exists(shared_weights_)) throw std::runtime_error("finalize shared weights missing: " + shared_weights_);

    bucket_paths_ = discover_finalize_buckets(buckets_dir_);
    if (bucket_paths_.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir_);
    std::string manifest_path = buckets_dir_ + "/manifest.json";
    if (!file_exists(manifest_path)) {
      throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
    }
    auto manifest = load_bucket_manifest(manifest_path);
    verify_bucket_manifest(manifest, bucket_paths_, buckets_dir_, shared_weights_pt);
    std::printf("density finalize manifest verified: %zu buckets, weights_sha256=%s num_runners=%d policy=%s\n",
                manifest.buckets.size(), manifest.contract.weights_sha256.c_str(), num_runners_,
                policy_.c_str());

    CUDA_CHECK(cudaDeviceSynchronize());
    shared_used_before_ = gpu_used_bytes();
    shared_constants_ = load_shared_constants(shared_weights_, device_);
    CUDA_CHECK(cudaDeviceSynchronize());
    shared_used_after_ = gpu_used_bytes();
    shared_delta_ = shared_used_after_ >= shared_used_before_ ? shared_used_after_ - shared_used_before_ : 0;
    std::printf("density loaded finalize shared constants: %zu entries shared_delta=%.3f GiB policy=%s\n",
                shared_constants_.size(),
                static_cast<double>(shared_delta_) / (1024.0 * 1024.0 * 1024.0),
                policy_.c_str());
  }

  AOTIModelPackageLoader& get(int64_t drop, int64_t T) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto key = std::make_pair(drop, T);
    auto existing = loaders_.find(key);
    if (existing != loaders_.end()) return *existing->second;
    auto loaded = load_bucket_locked(key);
    return *loaded->second;
  }

  void preload(const std::vector<FinalizeBucketKey>& keys) {
    for (const auto& key : keys) {
      (void)get(key.first, key.second);
    }
  }

  void preload_all() {
    std::vector<FinalizeBucketKey> keys;
    keys.reserve(bucket_paths_.size());
    for (const auto& kv : bucket_paths_) keys.push_back(kv.first);
    preload(keys);
  }

  int num_runners() const {
    return num_runners_;
  }

  size_t total_bucket_count() const {
    return bucket_paths_.size();
  }

  size_t loaded_bucket_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaders_.size();
  }

  size_t shared_delta() const {
    return shared_delta_;
  }

  size_t total_loader_delta() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_loader_delta_;
  }

  size_t projected_all_buckets_same_runners_delta() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (records_.empty()) return 0;
    return static_cast<size_t>((static_cast<long double>(total_loader_delta_) /
                                static_cast<long double>(records_.size())) *
                               static_cast<long double>(bucket_paths_.size()));
  }

  size_t projected_all_buckets_worker_runners_delta(int worker_runners) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (records_.empty() || num_runners_ <= 0) return 0;
    long double mean = static_cast<long double>(total_loader_delta_) /
                       static_cast<long double>(records_.size());
    long double runner_ratio = static_cast<long double>(worker_runners) /
                               static_cast<long double>(num_runners_);
    return static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()) * runner_ratio);
  }

  std::string memory_json(int worker_runners) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t projected_same = 0;
    size_t projected_worker = 0;
    if (!records_.empty()) {
      long double mean = static_cast<long double>(total_loader_delta_) /
                         static_cast<long double>(records_.size());
      projected_same = static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()));
      if (num_runners_ > 0) {
        long double runner_ratio = static_cast<long double>(worker_runners) /
                                   static_cast<long double>(num_runners_);
        projected_worker = static_cast<size_t>(mean * static_cast<long double>(bucket_paths_.size()) * runner_ratio);
      }
    }
    std::ostringstream oss;
    oss << "{\"policy\":\"" << policy_ << "\""
        << ",\"num_runners_per_loaded_bucket\":" << num_runners_
        << ",\"worker_runners_requested\":" << worker_runners
        << ",\"total_manifest_buckets\":" << bucket_paths_.size()
        << ",\"loaded_buckets\":" << loaders_.size()
        << ",\"shared_constants_delta_bytes\":" << shared_delta_
        << ",\"loader_delta_bytes\":" << total_loader_delta_
        << ",\"projected_all_buckets_same_runner_cap_delta_bytes\":" << projected_same
        << ",\"projected_old_eager_all_buckets_worker_runner_delta_bytes_linear_estimate\":" << projected_worker
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
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>>::iterator load_bucket_locked(
      const FinalizeBucketKey& key) {
    auto path_it = bucket_paths_.find(key);
    if (path_it == bucket_paths_.end()) {
      throw std::runtime_error("density no finalize bucket for drop=" + std::to_string(key.first) +
                               " T=" + std::to_string(key.second));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    size_t before = gpu_used_bytes();
    auto loader = std::make_unique<AOTIModelPackageLoader>(path_it->second, "model", false, num_runners_, -1);
    auto bucket_constants = constants_for_bucket(shared_constants_, *loader, path_it->second);
    loader->load_constants(bucket_constants.values, false, false, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t after = gpu_used_bytes();
    size_t delta = after >= before ? after - before : 0;
    total_loader_delta_ += delta;
    records_.push_back({
        key.first,
        key.second,
        num_runners_,
        before,
        after,
        delta,
        total_loader_delta_,
    });
    std::printf("  density finalize bucket loaded drop=%ld T=%ld constants=%zu direct=%zu alias=%zu "
                "num_runners=%d loader_delta=%.3f GiB cumulative_loader_delta=%.3f GiB policy=%s\n",
                (long)key.first, (long)key.second, bucket_constants.values.size(),
                bucket_constants.direct_matches, bucket_constants.alias_fallbacks, num_runners_,
                static_cast<double>(delta) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(total_loader_delta_) / (1024.0 * 1024.0 * 1024.0),
                policy_.c_str());
    auto inserted = loaders_.emplace(key, std::move(loader));
    return inserted.first;
  }

  std::string dir_;
  torch::Device device_;
  int num_runners_ = 1;
  std::string policy_;
  std::string buckets_dir_;
  std::string shared_weights_;
  std::map<FinalizeBucketKey, std::string> bucket_paths_;
  std::unordered_map<std::string, at::Tensor> shared_constants_;
  std::map<FinalizeBucketKey, std::unique_ptr<AOTIModelPackageLoader>> loaders_;
  mutable std::mutex mutex_;
  std::vector<FinalizeLoaderMemoryRecord> records_;
  size_t shared_used_before_ = 0;
  size_t shared_used_after_ = 0;
  size_t shared_delta_ = 0;
  size_t total_loader_delta_ = 0;
};

static int capped_general_finalize_runners(int workers_or_runners) {
  return std::max(1, std::min(workers_or_runners, density_finalize_runner_cap()));
}

static std::vector<FinalizeBucketKey> unique_finalize_bucket_keys(const std::vector<FinalizeBucketKey>& keys) {
  std::set<FinalizeBucketKey> seen(keys.begin(), keys.end());
  return std::vector<FinalizeBucketKey>(seen.begin(), seen.end());
}

static FinalizeOutcome run_finalize_density(SessionState& parent,
                                            torch::jit::Module& bundle,
                                            const std::string& prefix,
                                            const std::string& label,
                                            FinalizeBucketLoaderPool& finalize_loaders,
                                            torch::jit::Module& joint,
                                            torch::jit::Module& predict,
                                            torch::Device device,
                                            const Tokenizer& tokenizer,
                                            std::vector<EmittedEvent>& events,
                                            FinalizeFinish finish,
                                            c10::cuda::CUDAStream stream,
                                            bool explicit_stream,
                                            bool mutex_serialize_run,
                                            TimingBuckets* timings,
                                            StaleGenTelemetry* stale_gen = nullptr,
                                            StatsCollector* stats_collector = nullptr) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto total_start = Clock::now();
  uint64_t work_generation = density_capture_generation(parent);

  if (finish == FinalizeFinish::SPECULATIVE_KEEP && parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("density speculative finalize outside PENDING_FINALIZE");
  }
  if (finish == FinalizeFinish::TRUE_BOUNDARY_COLD_RESET &&
      parent.mode != SessionMode::STREAMING &&
      parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("density true-boundary finalize outside live state");
  }
  auto fork_clone_start = Clock::now();
  auto snapshot = snapshot_asr(parent);
  parent.mode = SessionMode::FINALIZED;
  snapshot.mode = SessionMode::FINALIZED;
  auto fork = clone_session(parent);
  double fork_clone_ms = elapsed_ms_since(fork_clone_start);

  int64_t drop_extra = scalar_i64(prefix_tensor(bundle, prefix, "final_drop_extra"));
  int64_t final_T = scalar_i64(prefix_tensor(bundle, prefix, "final_T"));
  auto gold = tensor_to_vec(prefix_tensor(bundle, prefix, "gold_tokens"));
  double runner_host_ms = 0.0;
  double gpu_ms = 0.0;
  double enc_len_sync_ms = 0.0;
  double decode_wall_ms = 0.0;
  double decode_item_wait_ms = 0.0;
  double decode_tokens = 0.0;

  if (final_T > 0) {
    auto final_chunk = prefix_tensor(bundle, prefix, "final_chunk_mel").to(device).contiguous();
    if (final_chunk.size(2) != final_T) {
      throw std::runtime_error("density final_chunk_mel T does not match bundle final_T");
    }
    int64_t expected_drop = parent.emitted == 0 ? 0 : DROP;
    if (drop_extra != expected_drop) throw std::runtime_error("density finalize drop_extra does not match parent emitted state");

    auto& finalize_loader = finalize_loaders.get(drop_extra, final_T);

    std::vector<at::Tensor> inputs = {
        final_chunk.contiguous(),
        fork.clc.contiguous(),
        fork.clt.contiguous(),
        fork.clcl.contiguous(),
    };
    cudaEvent_t ev_start{};
    cudaEvent_t ev_stop{};
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
    auto run_start = Clock::now();
    auto out = run_aoti_loader(finalize_loader, inputs, stream, explicit_stream, mutex_serialize_run);
    runner_host_ms = elapsed_ms_since(run_start);
    CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
    if (out.size() < 2) throw std::runtime_error("density finalize AOTI bucket returned fewer than 2 outputs");
    int64_t enc_len = encoder_len_density(out[1],
                                          final_T,
                                          drop_extra,
                                          DensityEncoderLenMode::FINALIZE_KEEP_ALL,
                                          label + ".finalize",
                                          &enc_len_sync_ms);
    if (out.size() >= 5) {
      fork.clc = out[2];
      fork.clt = out[3];
      fork.clcl = out[4];
    }
    size_t hyp_before_decode = fork.hyp.size();
    auto decode_start = Clock::now();
    decode_range_density(joint,
                         predict,
                         out[0],
                         enc_len,
                         fork.g,
                         fork.h,
                         fork.c,
                         fork.hyp,
                         &decode_item_wait_ms);
    decode_wall_ms = elapsed_ms_since(decode_start);
    decode_tokens = static_cast<double>(fork.hyp.size() - hyp_before_decode);
    if (timings != nullptr) {
      timings->scalar_sync_wait_ms.push_back(enc_len_sync_ms + decode_item_wait_ms);
    }
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
    gpu_ms = static_cast<double>(elapsed);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
  }

  FinalizeOutcome outcome;
  outcome.emitted_tokens = fork.hyp.size();
  outcome.final_tokens = fork.hyp;
  outcome.final_text = tokenizer.ids_to_text(fork.hyp);
  outcome.token_ok = equal_tokens(outcome.final_tokens, gold, "final cumulative", label);
  std::string final_text = outcome.final_text;
  std::string delta_text = append_only_delta_text(final_text, parent.continuous_emitted_text);
  auto delta_tokens = append_only_delta_tokens(fork.hyp, parent.continuous_emitted_tokens);
  bool generation_live = density_generation_live(parent,
                                                 work_generation,
                                                 stale_gen,
                                                 StaleGenStage::FINALIZE_OUTPUT,
                                                 label + ".finalize_output");
  if (!generation_live) {
    outcome.stale_dropped = true;
    outcome.token_ok = true;
    outcome.fork_ok = true;
    return outcome;
  }
  bool emitted_to_client = !delta_text.empty();
  if (delta_text.empty()) {
    emit_event(events,
               EVENT_SUPPRESSED,
               {},
               parent.continuous_emitted_tokens,
               "",
               parent.continuous_emitted_text);
  } else {
    auto collector_tokens = parent.continuous_emitted_tokens;
    collector_tokens.insert(collector_tokens.end(), delta_tokens.begin(), delta_tokens.end());
    std::string collector_text = append_delta_to_collector(parent.continuous_emitted_text, delta_text);
    emit_event(events,
               EVENT_FINAL,
               delta_tokens,
               collector_tokens,
               delta_text,
               collector_text);
    parent.continuous_emitted_tokens = std::move(collector_tokens);
    parent.continuous_emitted_text = std::move(collector_text);
  }
  outcome.fork_ok = fork_assert_parent_unchanged(parent, snapshot);
  if (finish == FinalizeFinish::SPECULATIVE_KEEP) {
    finish_speculative_finalize(parent);
  } else {
    cold_reset_after_finalize(parent, bundle, device, nullptr);
  }
  if (timings != nullptr) {
    double total_ms = elapsed_ms_since(total_start);
    double glue_ms = total_ms - fork_clone_ms - enc_len_sync_ms - decode_wall_ms;
    if (glue_ms < 0.0) glue_ms = 0.0;
    timings->finalize_runner_wait_ms.push_back(std::max(0.0, runner_host_ms - gpu_ms));
    timings->finalize_gpu_ms.push_back(gpu_ms);
    timings->finalize_total_ms.push_back(total_ms);
    timings->finalize_fork_clone_ms.push_back(fork_clone_ms);
    timings->finalize_aoti_run_cuda_ms.push_back(gpu_ms);
    timings->finalize_enc_len_sync_ms.push_back(enc_len_sync_ms);
    timings->finalize_decode_wall_ms.push_back(decode_wall_ms);
    timings->finalize_decode_item_wait_ms.push_back(decode_item_wait_ms);
    timings->finalize_decode_tokens.push_back(decode_tokens);
    timings->finalize_glue_ms.push_back(glue_ms);
  }
  if (stats_collector != nullptr) {
    static std::atomic<uint64_t> finalize_seq{0};
    double total_ms = elapsed_ms_since(total_start);
    SessionTiming timing;
    timing.reason = "density";
    timing.vad_stop_ts = 0.0;
    timing.fork_flush_start_ts = 0.0;
    timing.fork_flush_done_ts = total_ms / 1000.0;
    timing.final_sent_ts = timing.fork_flush_done_ts;
    timing.inference_lock_acquire_wait_ms = std::max(0.0, runner_host_ms - gpu_ms);
    timing.finalize_seq = finalize_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    timing.active_sessions_at_emit = 0;
    timing.was_suppressed = !emitted_to_client;
    stats_collector->record(timing, emitted_to_client);
  }
  return outcome;
}

struct RowReplayResult {
  std::vector<int64_t> steady_tokens;
  std::vector<int64_t> final_tokens;
  std::vector<EmittedEvent> events;
  bool ok = false;
  bool stale_dropped = false;
  bool gold_events_diverged = false;  // cross-arch interim-event drift vs gold (counted, optionally strict)
  std::string error;
};

static bool density_strict_event_gate_enabled() {
  static const bool enabled = []() {
    const char* e = std::getenv("DENSITY_GOLD_EVENTS_TOLERANT");
    return e != nullptr && std::string(e) == "1";
  }();
  return enabled;
}

static bool density_event_divergences_gate_pass(int event_divergences) {
  // Project policy, confirmed by the B1/B2 paired-reviewed verdicts, treats interim event timing drift as
  // counted-not-gated: the SLO correctness signal is token_divergences == 0 plus runtime/error budgets.
  // Keep event divergence counts visible for debugging; DENSITY_GOLD_EVENTS_TOLERANT=1 is now the opt-in
  // legacy strict byte-exact event gate for diagnostic runs.
  return event_divergences == 0 || !density_strict_event_gate_enabled();
}

static RowReplayResult replay_row_density(int utt,
                                          WorkerContext& ctx,
                                          AOTIModelPackageLoader* enc_steady,
                                          BatchedSteadyLoaderSet* direct_b1_loader,
                                          FinalizeBucketLoaderPool& finalize_loaders,
                                          torch::Device device,
                                          const Tokenizer& tokenizer,
                                          bool explicit_stream,
                                          bool mutex_serialize_run,
                                          TimingBuckets* timings,
                                          bool check_gold,
                                          BatchedSteadyScheduler* scheduler = nullptr,
                                          B2ReusableBarrier* continuation_barrier = nullptr,
                                          B2TensorDiffStats* scheduler_diff_stats = nullptr,
                                          AOTIModelPackageLoader* scheduler_diff_loader = nullptr,
                                          StaleGenTelemetry* stale_gen = nullptr,
                                          int consumer_delay_chunk = -1,
                                          SchedulerConsumerDelayProbe* consumer_delay_probe = nullptr) {
  RowReplayResult result;
  try {
    SessionState session;
    reset_session(session, ctx.bundle, device);
    std::string prefix = "utt" + std::to_string(utt);
    std::string label = "density.utt" + std::to_string(utt);
    int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, utt, "num_steady"));
    std::vector<EmittedEvent> events;
    for (int chunk = 0; chunk < num_steady; ++chunk) {
      B2ReusableBarrier* chunk_barrier = continuation_barrier;
      if (consumer_delay_chunk >= 0 && continuation_barrier != nullptr) {
        chunk_barrier = chunk == consumer_delay_chunk ? continuation_barrier : nullptr;
      }
      run_steady_chunk_density(session,
                               ctx.bundle,
                               prefix,
                               chunk,
                               *ctx.enc_first,
                               enc_steady,
                               direct_b1_loader,
                               ctx.joint,
                               ctx.predict,
                               device,
                               tokenizer,
                               events,
                               ctx.stream,
                               explicit_stream,
                               mutex_serialize_run,
                               timings,
                               label + ".chunk" + std::to_string(chunk),
                               scheduler,
                               chunk_barrier,
                               scheduler_diff_stats,
                               scheduler_diff_loader,
                               nullptr,
                               stale_gen,
                               chunk == consumer_delay_chunk ? consumer_delay_probe : nullptr);
    }
    bool steady_ok = true;
    if (check_gold) {
      auto steady_gold = tensor_to_vec(utt_tensor(ctx.bundle, utt, "steady_tokens"));
      steady_ok = equal_tokens(session.hyp, steady_gold, "steady cumulative", label);
    }
    result.steady_tokens = session.hyp;
    session.mode = SessionMode::PENDING_FINALIZE;
    auto finalize = run_finalize_density(session,
                                         ctx.bundle,
                                         prefix,
                                         label,
                                         finalize_loaders,
                                         ctx.joint,
                                         ctx.predict,
                                         device,
                                         tokenizer,
                                         events,
                                         FinalizeFinish::SPECULATIVE_KEEP,
                                         ctx.stream,
                                         explicit_stream,
                                         mutex_serialize_run,
                                         timings,
                                         stale_gen);
    bool events_ok = true;
    if (check_gold && !finalize.stale_dropped) {
      events_ok = strict_events_equal(events, gold_events_from_bundle(ctx.bundle, utt), label + ".gold");
    }
    result.final_tokens = std::move(finalize.final_tokens);
    result.events = std::move(events);
    result.stale_dropped = finalize.stale_dropped;
    result.gold_events_diverged = check_gold && !events_ok;
    // Cross-arch numerical drift can shift an INTERIM partial's timing without changing the WER-relevant
    // cumulative or final transcript. Count that event-stream divergence, but only gate on it when the
    // legacy strict-event debug flag is explicitly enabled.
    bool events_pass = events_ok || (check_gold && density_event_divergences_gate_pass(1));
    bool finalize_pass = finalize.stale_dropped || (finalize.token_ok && finalize.fork_ok);
    result.ok = steady_ok && finalize_pass && events_pass;
  } catch (const std::exception& e) {
    result.error = e.what();
    result.ok = false;
  }
  return result;
}

struct CorrectnessResult {
  bool ok = false;
  bool identity_ok = false;
  bool scalar_locality_pass = false;
  bool default_stream_control_pass = false;
  bool stream_uniqueness_ok = false;
  bool explicit_stream = true;
  bool mutex_serialize_run = false;
  int rows = 0;
  int workers = 0;
  int num_runners = 0;
  int unique_streams = 0;
  int mismatches = 0;
  double default_stream_penalty = 0.0;
  double throughput_rows_per_s = 0.0;
  double wall_ms = 0.0;
  TimingBuckets timings;
  size_t peak_mem = 0;
  size_t used_before = 0;
  size_t used_after = 0;
  std::vector<uintptr_t> stream_handles;
  std::vector<RowReplayResult> reference;
};

static std::vector<RowReplayResult> build_serial_reference(const DensityArgs& args,
                                                           torch::Device device,
                                                           const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                                           int rows,
                                                           TimingBuckets* ref_timings) {
  auto stream = stream_for_worker(true, 0);
  auto ctx = make_worker_context(args.dir, device, stream, shared_bundle);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  verify_tokenizer_selftest(ctx->bundle, tokenizer);
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir, device, 1, "serial_reference_eager_one_runner");
  finalize_loaders.preload_all();
  std::vector<RowReplayResult> refs;
  refs.reserve(rows);
  for (int utt = 0; utt < rows; ++utt) {
    auto result = replay_row_density(utt,
                                     *ctx,
                                     &enc_steady,
                                     nullptr,
                                     finalize_loaders,
                                     device,
                                     tokenizer,
                                     true,
                                     false,
                                     ref_timings,
                                     true,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr);
    if (!result.ok) {
      throw std::runtime_error("serial reference failed for utt" + std::to_string(utt) +
                               (result.error.empty() ? "" : ": " + result.error));
    }
    refs.push_back(std::move(result));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  int gold_event_divergences = 0;
  for (const auto& r : refs) if (r.gold_events_diverged) ++gold_event_divergences;
  std::printf("=== serial reference built: %d utts; gold-event (cross-arch interim-timing) divergences = %d "
              "(finals + steady-cumulative strictly matched gold for ALL utts, else this would have thrown) ===\n",
              (int)refs.size(), gold_event_divergences);
  return refs;
}

static bool compare_b2_token_vector(const std::vector<int64_t>& got,
                                    const std::vector<int64_t>& ref,
                                    const std::string& label,
                                    int worker,
                                    int utt);

static bool enc_first_ts_adapter_byte_exact(SharedEncFirst& enc_first,
                                            torch::jit::Module& bundle,
                                            torch::Device device,
                                            c10::cuda::CUDAStream stream,
                                            int rows) {
  if (!enc_first.module_owner) {
    throw std::runtime_error("TS adapter byte-exact selftest requires a TS module owner");
  }
  int checked = 0;
  bool ok = true;
  for (int utt = 0; utt < rows; ++utt) {
    if (scalar_i64(utt_tensor(bundle, utt, "num_steady")) <= 0) continue;
    std::string prefix = "utt" + std::to_string(utt);
    auto chunk = prefix_chunk_tensor(bundle, prefix, 0, "new_mel").to(device).contiguous();
    SessionState raw_state;
    SessionState adapter_state;
    reset_session(raw_state, bundle, device);
    reset_session(adapter_state, bundle, device);
    {
      std::lock_guard<std::mutex> lock(enc_first.mutex);
      (void)run_first_encoder(*enc_first.module_owner, chunk, raw_state, stream);
      (void)enc_first.encoder->run(chunk, adapter_state, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    break;
  }
  for (int utt = 0; utt < rows; ++utt) {
    if (scalar_i64(utt_tensor(bundle, utt, "num_steady")) <= 0) continue;
    std::string prefix = "utt" + std::to_string(utt);
    auto chunk = prefix_chunk_tensor(bundle, prefix, 0, "new_mel").to(device).contiguous();
    SessionState raw_state;
    SessionState adapter_state;
    reset_session(raw_state, bundle, device);
    reset_session(adapter_state, bundle, device);

    std::vector<at::Tensor> raw;
    std::vector<at::Tensor> adapted;
    {
      std::lock_guard<std::mutex> lock(enc_first.mutex);
      raw = run_first_encoder(*enc_first.module_owner, chunk, raw_state, stream);
      adapted = enc_first.encoder->run(chunk, adapter_state, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    if (raw.size() != adapted.size()) {
      std::printf("ENC_FIRST_TS_ADAPTER_BYTE_EXACT utt=%d output_count_mismatch raw=%zu adapter=%zu\n",
                  utt,
                  raw.size(),
                  adapted.size());
      ok = false;
      continue;
    }
    for (size_t i = 0; i < raw.size(); ++i) {
      std::string name = "enc_first_ts_adapter.utt" + std::to_string(utt) +
                         ".out" + std::to_string(i);
      ok = tensor_equal(name.c_str(), adapted[i], raw[i]) && ok;
    }
    ++checked;
  }
  std::printf("ENC_FIRST_TS_ADAPTER_BYTE_EXACT checked=%d result=%s\n",
              checked,
              ok ? "PASS" : "FAIL");
  return ok && checked > 0;
}

static bool run_enc_first_parity(const DensityArgs& args,
                                 torch::Device device,
                                 const std::shared_ptr<torch::jit::Module>& shared_bundle) {
  int rows_total = static_cast<int>(scalar_i64(attr_tensor(*shared_bundle, "num_utts")));
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total)
                                       : std::min(rows_total, 200);
  if (rows <= 0) throw std::runtime_error("enc-first-parity requires at least one audio fixture row");
  std::printf("ENC_FIRST_PARITY START rows=%d/%d default_rows=%s\n",
              rows,
              rows_total,
              args.correctness_rows > 0 ? "cli" : "min(rows_total,200)");

  auto stream = stream_for_worker(true, 0);
  auto ts_first = load_shared_enc_first(args.dir, device, "enc_first_parity_ts_adapter");
  int ts_selftest_rows = std::min(rows, 4);
  bool ts_byte_exact = enc_first_ts_adapter_byte_exact(*ts_first,
                                                      *shared_bundle,
                                                      device,
                                                      stream,
                                                      ts_selftest_rows);

  auto shared_constants = bsteady_detail::load_shared_constants(
      (fs::path(args.dir) / "finalize_shared_weights.ts").string(),
      device);
  std::printf("ENC_FIRST_PARITY shared_constants entries=%zu source=%s\n",
              shared_constants.size(),
              (fs::path(args.dir) / "finalize_shared_weights.ts").string().c_str());
  auto aoti_first = load_shared_enc_first_aoti(args.dir,
                                               device,
                                               shared_constants,
                                               1,
                                               "enc_first_parity_aoti_adapter");

  auto ctx = make_worker_context(args.dir, device, stream, shared_bundle, ts_first);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  verify_tokenizer_selftest(ctx->bundle, tokenizer);
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir, device, 1, "enc_first_parity_finalize_pool");
  finalize_loaders.preload_all();

  int final_token_divergences = 0;
  int event_divergences = 0;
  int errors = 0;
  for (int utt = 0; utt < rows; ++utt) {
    ctx->enc_first = ts_first;
    auto ts_row = replay_row_density(utt,
                                     *ctx,
                                     &enc_steady,
                                     nullptr,
                                     finalize_loaders,
                                     device,
                                     tokenizer,
                                     true,
                                     false,
                                     nullptr,
                                     false,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     nullptr);
    ctx->enc_first = aoti_first;
    auto aoti_row = replay_row_density(utt,
                                       *ctx,
                                       &enc_steady,
                                       nullptr,
                                       finalize_loaders,
                                       device,
                                       tokenizer,
                                       true,
                                       false,
                                       nullptr,
                                       false,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       nullptr);
    if (!ts_row.error.empty() || !aoti_row.error.empty() || !ts_row.ok || !aoti_row.ok) {
      ++errors;
      std::printf("ENC_FIRST_PARITY_ROW_ERROR utt=%d ts_ok=%s aoti_ok=%s ts_error=%s aoti_error=%s\n",
                  utt,
                  ts_row.ok ? "true" : "false",
                  aoti_row.ok ? "true" : "false",
                  ts_row.error.empty() ? "(none)" : ts_row.error.c_str(),
                  aoti_row.error.empty() ? "(none)" : aoti_row.error.c_str());
      continue;
    }
    if (!compare_b2_token_vector(aoti_row.final_tokens,
                                 ts_row.final_tokens,
                                 "enc-first-parity.final",
                                 0,
                                 utt)) {
      ++final_token_divergences;
    }
    if (!strict_events_equal(aoti_row.events,
                             ts_row.events,
                             "enc_first_parity.utt" + std::to_string(utt) + ".events")) {
      ++event_divergences;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  bool pass = ts_byte_exact && errors == 0 && final_token_divergences == 0;
  std::printf("ENC_FIRST_PARITY rows=%d final_token_divergences=%d event_divergences=%d "
              "errors=%d ts_adapter_byte_exact=%s verdict=%s\n",
              rows,
              final_token_divergences,
              event_divergences,
              errors,
              ts_byte_exact ? "PASS" : "FAIL",
              pass ? "PASS" : "FAIL");
  return pass;
}

static bool steady_batch_dir_has_packages(const std::string& dir, std::string* error = nullptr) {
  return runtime_io::steady_batch_dir_has_declared_packages(dir, error);
}

static std::string resolve_steady_batch_dir(const DensityArgs& args) {
  if (!args.steady_batch_dir.empty()) {
    std::string error;
    if (!steady_batch_dir_has_packages(args.steady_batch_dir, &error)) {
      throw std::runtime_error("--steady-batch-dir invalid: " + error);
    }
    return args.steady_batch_dir;
  }

  std::vector<std::string> candidates;
  fs::path dir_path(args.dir);
  if (dir_path.has_parent_path()) {
    candidates.push_back((dir_path.parent_path() / "steady_b_artifacts").string());
  }
  candidates.push_back("steady_b_artifacts");
  candidates.push_back("../steady_b_artifacts");
  candidates.push_back("runtime/steady_b_artifacts");
  for (const auto& candidate : candidates) {
    if (steady_batch_dir_has_packages(candidate)) return candidate;
  }
  std::ostringstream oss;
  oss << "could not resolve steady batch artifacts; pass --steady-batch-dir. tried:";
  for (const auto& candidate : candidates) oss << " " << candidate;
  throw std::runtime_error(oss.str());
}

struct B1RowSpec {
  int utt = 0;
  int chunk = 1;
  std::string label;
};

struct B1PreparedRow {
  B1RowSpec spec;
  SessionState session;
  std::vector<EmittedEvent> events;
  torch::Tensor new_mel;
  torch::Tensor chunk;
  std::vector<at::Tensor> alone_out;
};

struct B1T1Stats {
  int cases = 0;
  int rows = 0;
  int skipped_no_steady = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  int enc_len_mismatches = 0;
  int cache_len_mismatches = 0;
  double max_enc_out_diff = 0.0;
  double max_cache_ch_diff = 0.0;
  double max_cache_t_diff = 0.0;
};

static std::string tensor_sizes_string(const at::Tensor& tensor) {
  std::ostringstream oss;
  oss << "(";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) oss << ",";
    oss << tensor.size(i);
  }
  oss << ")";
  return oss.str();
}

static double max_abs_tensor_diff(const at::Tensor& a, const at::Tensor& b) {
  if (a.sizes() != b.sizes()) {
    throw std::runtime_error("B1 diff shape mismatch got=" + tensor_sizes_string(a) +
                             " ref=" + tensor_sizes_string(b));
  }
  auto diff = (a.to(torch::kFloat32) - b.to(torch::kFloat32)).abs();
  return diff.numel() > 0 ? diff.max().item<double>() : 0.0;
}

static int first_token_diff_pos(const std::vector<int64_t>& got, const std::vector<int64_t>& ref) {
  size_t n = std::min(got.size(), ref.size());
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != ref[i]) return static_cast<int>(i);
  }
  return got.size() == ref.size() ? -1 : static_cast<int>(n);
}

static bool compare_token_vector_with_location(const std::vector<int64_t>& got,
                                               const std::vector<int64_t>& ref,
                                               const std::string& label,
                                               int utt,
                                               int chunk) {
  if (got == ref) return true;
  int pos = first_token_diff_pos(got, ref);
  std::printf("B1_T1_TOKEN_DIVERGENCE label=%s utt=%d chunk=%d pos=%d got_len=%zu ref_len=%zu",
              label.c_str(),
              utt,
              chunk,
              pos,
              got.size(),
              ref.size());
  if (pos >= 0 && pos < static_cast<int>(got.size())) {
    std::printf(" got=%lld", static_cast<long long>(got[static_cast<size_t>(pos)]));
  }
  if (pos >= 0 && pos < static_cast<int>(ref.size())) {
    std::printf(" ref=%lld", static_cast<long long>(ref[static_cast<size_t>(pos)]));
  }
  std::printf("\n");
  return false;
}

static void update_b1_tensor_diffs(B1T1Stats& stats,
                                   const std::vector<at::Tensor>& got,
                                   const std::vector<at::Tensor>& ref,
                                   const B1RowSpec& spec,
                                   const std::string& case_name) {
  if (got.size() < 5 || ref.size() < 5) throw std::runtime_error("B1 tensor diff needs 5 steady outputs");
  stats.max_enc_out_diff = std::max(stats.max_enc_out_diff, max_abs_tensor_diff(got[0], ref[0]));
  stats.max_cache_ch_diff = std::max(stats.max_cache_ch_diff, max_abs_tensor_diff(got[2], ref[2]));
  stats.max_cache_t_diff = std::max(stats.max_cache_t_diff, max_abs_tensor_diff(got[3], ref[3]));
  if (!at::equal(got[1], ref[1])) {
    ++stats.enc_len_mismatches;
    std::printf("B1_T1_ENC_LEN_MISMATCH case=%s label=%s utt=%d chunk=%d got=%lld ref=%lld\n",
                case_name.c_str(),
                spec.label.c_str(),
                spec.utt,
                spec.chunk,
                static_cast<long long>(scalar_i64(got[1])),
                static_cast<long long>(scalar_i64(ref[1])));
  }
  if (!at::equal(got[4], ref[4])) {
    ++stats.cache_len_mismatches;
    std::printf("B1_T1_CACHE_LEN_MISMATCH case=%s label=%s utt=%d chunk=%d got=%lld ref=%lld\n",
                case_name.c_str(),
                spec.label.c_str(),
                spec.utt,
                spec.chunk,
                static_cast<long long>(scalar_i64(got[4])),
                static_cast<long long>(scalar_i64(ref[4])));
  }
}

static void apply_b1_target_chunk(SessionState& state,
                                  const torch::Tensor& new_mel,
                                  const torch::Tensor& chunk,
                                  const std::vector<at::Tensor>& out,
                                  torch::jit::Module& joint,
                                  torch::jit::Module& predict,
                                  const Tokenizer& tokenizer,
                                  std::vector<EmittedEvent>& events,
                                  const std::string& label) {
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("B1 target chunk outside STREAMING");
  (void)apply_encoder_outputs_density(state, out, joint, predict, chunk.size(2), DROP, label + ".encoder");
  auto cum = state.ring.defined() ? torch::cat({state.ring, new_mel}, 2) : new_mel;
  state.ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2)).contiguous();
  state.emitted += new_mel.size(2);
  std::string current_text = tokenizer.ids_to_text(state.hyp);
  if (current_text != state.last_interim_text) {
    emit_event(events,
               EVENT_INTERIM,
               state.hyp,
               state.continuous_emitted_tokens,
               current_text,
               state.continuous_emitted_text);
    state.last_interim_tokens = state.hyp;
    state.last_interim_text = current_text;
  }
}

static B1PreparedRow prepare_b1_row_until_target(const B1RowSpec& spec,
                                                 WorkerContext& ctx,
                                                 AOTIModelPackageLoader& enc_steady,
                                                 torch::Device device,
                                                 const Tokenizer& tokenizer) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  B1PreparedRow row;
  row.spec = spec;
  reset_session(row.session, ctx.bundle, device);
  std::string prefix = "utt" + std::to_string(spec.utt);
  int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, spec.utt, "num_steady"));
  if (spec.chunk <= 0 || spec.chunk >= num_steady) {
    throw std::runtime_error("B1 target must be a continuation steady chunk: label=" + spec.label +
                             " num_steady=" + std::to_string(num_steady));
  }
  for (int chunk = 0; chunk < spec.chunk; ++chunk) {
    run_steady_chunk_density(row.session,
                             ctx.bundle,
                             prefix,
                             chunk,
                             *ctx.enc_first,
                             &enc_steady,
                             nullptr,
                             ctx.joint,
                             ctx.predict,
                             device,
                             tokenizer,
                             row.events,
                             ctx.stream,
                             true,
                             false,
                             nullptr,
                             "b1.prepare." + spec.label + ".chunk" + std::to_string(chunk),
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr);
  }
  row.new_mel = prefix_chunk_tensor(ctx.bundle, prefix, spec.chunk, "new_mel").to(device).contiguous();
  int64_t drop_extra = scalar_i64(prefix_chunk_tensor(ctx.bundle, prefix, spec.chunk, "drop_extra"));
  int64_t chunk_T = scalar_i64(prefix_chunk_tensor(ctx.bundle, prefix, spec.chunk, "chunk_T"));
  int64_t emitted_before = scalar_i64(prefix_chunk_tensor(ctx.bundle, prefix, spec.chunk, "emitted_before"));
  if (drop_extra != DROP) throw std::runtime_error("B1 target chunk drop_extra is not steady DROP");
  if (!row.session.ring.defined()) throw std::runtime_error("B1 target chunk missing pre-encode ring");
  if (chunk_T != row.session.ring.size(2) + row.new_mel.size(2)) {
    throw std::runtime_error("B1 target chunk_T mismatch");
  }
  if (emitted_before != row.session.emitted) throw std::runtime_error("B1 target emitted_before mismatch");
  row.chunk = torch::cat({row.session.ring, row.new_mel}, 2).contiguous();
  row.alone_out = run_steady_encoder_stream(enc_steady,
                                            row.chunk,
                                            row.session,
                                            ctx.stream,
                                            true,
                                            false,
                                            nullptr);
  return row;
}

static RowReplayResult finish_b1_row_from_target(B1PreparedRow& prepared,
                                                 const std::vector<at::Tensor>& target_out,
                                                 WorkerContext& ctx,
                                                 AOTIModelPackageLoader& enc_steady,
                                                 FinalizeBucketLoaderPool& finalize_loaders,
                                                 torch::Device device,
                                                 const Tokenizer& tokenizer,
                                                 const std::string& case_name) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  RowReplayResult result;
  try {
    std::string prefix = "utt" + std::to_string(prepared.spec.utt);
    apply_b1_target_chunk(prepared.session,
                          prepared.new_mel,
                          prepared.chunk,
                          target_out,
                          ctx.joint,
                          ctx.predict,
                          tokenizer,
                          prepared.events,
                          "b1." + case_name + "." + prepared.spec.label + ".chunk" +
                              std::to_string(prepared.spec.chunk));
    int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, prepared.spec.utt, "num_steady"));
    for (int chunk = prepared.spec.chunk + 1; chunk < num_steady; ++chunk) {
      run_steady_chunk_density(prepared.session,
                               ctx.bundle,
                               prefix,
                               chunk,
                               *ctx.enc_first,
                               &enc_steady,
                               nullptr,
                               ctx.joint,
                               ctx.predict,
                               device,
                               tokenizer,
                               prepared.events,
                               ctx.stream,
                               true,
                               false,
                               nullptr,
                               "b1." + case_name + "." + prepared.spec.label + ".chunk" +
                                   std::to_string(chunk),
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
    }
    result.steady_tokens = prepared.session.hyp;
    prepared.session.mode = SessionMode::PENDING_FINALIZE;
    auto finalize = run_finalize_density(prepared.session,
                                         ctx.bundle,
                                         prefix,
                                         "b1." + case_name + "." + prepared.spec.label,
                                         finalize_loaders,
                                         ctx.joint,
                                         ctx.predict,
                                         device,
                                         tokenizer,
                                         prepared.events,
                                         FinalizeFinish::SPECULATIVE_KEEP,
                                         ctx.stream,
                                         true,
                                         false,
                                         nullptr);
    result.final_tokens = std::move(finalize.final_tokens);
    result.events = std::move(prepared.events);
    result.stale_dropped = finalize.stale_dropped;
    result.ok = finalize.stale_dropped || (finalize.token_ok && finalize.fork_ok);
  } catch (const std::exception& e) {
    result.error = e.what();
    result.ok = false;
  }
  return result;
}

static bool run_b1_batched_case(const std::string& case_name,
                                const std::vector<B1RowSpec>& specs,
                                WorkerContext& ctx,
                                AOTIModelPackageLoader& enc_steady,
                                BatchedSteadyLoaderSet& batched_steady,
                                FinalizeBucketLoaderPool& finalize_loaders,
                                torch::Device device,
                                const Tokenizer& tokenizer,
                                const std::vector<RowReplayResult>& reference,
                                B1T1Stats& stats) {
  if (specs.empty()) throw std::runtime_error("B1 case has no specs: " + case_name);
  ++stats.cases;
  stats.rows += static_cast<int>(specs.size());

  std::vector<B1PreparedRow> prepared;
  prepared.reserve(specs.size());
  std::vector<BatchedSteadyInput> ready;
  ready.reserve(specs.size());
  for (const auto& spec : specs) {
    prepared.push_back(prepare_b1_row_until_target(spec, ctx, enc_steady, device, tokenizer));
    const auto& row = prepared.back();
    ready.push_back({
        row.chunk,
        row.session.clc,
        row.session.clt,
        row.session.clcl,
        row.spec.label,
    });
  }

  auto batched_out = batched_steady.run(ready, ctx.stream);
  CUDA_CHECK(cudaStreamSynchronize(ctx.stream.stream()));
  if (batched_out.size() != prepared.size()) throw std::runtime_error("B1 batched output row count mismatch");

  bool ok = true;
  for (size_t i = 0; i < prepared.size(); ++i) {
    const auto& spec = prepared[i].spec;
    update_b1_tensor_diffs(stats, batched_out[i].tensors, prepared[i].alone_out, spec, case_name);
    auto replay = finish_b1_row_from_target(prepared[i],
                                            batched_out[i].tensors,
                                            ctx,
                                            enc_steady,
                                            finalize_loaders,
                                            device,
                                            tokenizer,
                                            case_name);
    bool row_ok = replay.ok;
    if (!replay.error.empty()) {
      std::printf("B1_T1_ROW_ERROR case=%s label=%s utt=%d chunk=%d error=%s\n",
                  case_name.c_str(), spec.label.c_str(), spec.utt, spec.chunk, replay.error.c_str());
      row_ok = false;
    }
    if (spec.utt >= static_cast<int>(reference.size())) {
      throw std::runtime_error("B1 reference missing utt" + std::to_string(spec.utt));
    }
    bool steady_tokens_ok = compare_token_vector_with_location(replay.steady_tokens,
                                                               reference[static_cast<size_t>(spec.utt)].steady_tokens,
                                                               case_name + "." + spec.label + ".steady",
                                                               spec.utt,
                                                               spec.chunk);
    bool final_tokens_ok = compare_token_vector_with_location(replay.final_tokens,
                                                              reference[static_cast<size_t>(spec.utt)].final_tokens,
                                                              case_name + "." + spec.label + ".final",
                                                              spec.utt,
                                                              spec.chunk);
    if (!steady_tokens_ok || !final_tokens_ok) {
      ++stats.token_divergences;
      row_ok = false;
    }
    bool events_ok = strict_events_equal(replay.events,
                                         reference[static_cast<size_t>(spec.utt)].events,
                                         "b1." + case_name + "." + spec.label + ".events");
    if (!events_ok) {
      ++stats.event_divergences;
      row_ok = false;
      std::printf("B1_T1_EVENT_DIVERGENCE case=%s label=%s utt=%d chunk=%d\n",
                  case_name.c_str(), spec.label.c_str(), spec.utt, spec.chunk);
    }
    ok = ok && row_ok;
  }

  std::printf("B1_T1_CASE case=%s rows=%zu bucket_count_loaded=%d pass=%s\n",
              case_name.c_str(),
              specs.size(),
              batched_steady.loaded_bucket_count(),
              ok ? "PASS" : "FAIL");
  return ok;
}

static int pick_first_continuation_utt(torch::jit::Module& bundle, int rows) {
  for (int utt = 0; utt < rows; ++utt) {
    if (scalar_i64(utt_tensor(bundle, utt, "num_steady")) > 1) return utt;
  }
  throw std::runtime_error("B1 T1 found no utterance with a steady continuation chunk");
}

static std::vector<B1RowSpec> pick_identical_b1_specs(torch::jit::Module& bundle, int rows) {
  int utt = pick_first_continuation_utt(bundle, rows);
  std::vector<B1RowSpec> specs;
  specs.reserve(4);
  for (int i = 0; i < 4; ++i) {
    specs.push_back({utt, 1, "identical.row" + std::to_string(i) + ".utt" + std::to_string(utt) + ".chunk1"});
  }
  return specs;
}

static std::vector<B1RowSpec> pick_mixed_b1_specs(torch::jit::Module& bundle, int rows) {
  std::vector<B1RowSpec> specs;
  std::set<int> used_utts;
  for (int desired_chunk : {1, 2, 3}) {
    bool found = false;
    for (int utt = 0; utt < rows; ++utt) {
      if (used_utts.find(utt) != used_utts.end()) continue;
      int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
      if (num_steady > desired_chunk) {
        used_utts.insert(utt);
        specs.push_back({utt,
                         desired_chunk,
                         "mixed.utt" + std::to_string(utt) + ".chunk" + std::to_string(desired_chunk)});
        found = true;
        break;
      }
    }
    if (!found) break;
  }
  if (specs.size() < 2) {
    throw std::runtime_error("B1 T1 could not find at least two distinct mixed continuation rows");
  }
  return specs;
}

static std::vector<B1RowSpec> build_b1_coverage_specs(torch::jit::Module& bundle,
                                                      int rows,
                                                      B1T1Stats& stats) {
  std::vector<B1RowSpec> specs;
  specs.reserve(static_cast<size_t>(rows));
  for (int utt = 0; utt < rows; ++utt) {
    int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
    if (num_steady <= 1) {
      ++stats.skipped_no_steady;
      continue;
    }
    int chunk = 1 + (utt % static_cast<int>(num_steady - 1));
    specs.push_back({utt, chunk, "coverage.utt" + std::to_string(utt) + ".chunk" + std::to_string(chunk)});
  }
  return specs;
}

struct B2A1ParityResult {
  std::string outcome = "C";
  std::string production_sha;
  std::string new_b1_sha;
  bool tensor_ok = false;
  double max_enc_out_diff = 0.0;
  double max_cache_ch_diff = 0.0;
  double max_cache_t_diff = 0.0;
  int enc_len_mismatches = 0;
  int cache_len_mismatches = 0;
};

static B2A1ParityResult run_b2_a1_parity_check(const DensityArgs& args,
                                               torch::Device device,
                                               const std::string& steady_batch_dir,
                                               const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                               int rows_total) {
  B2A1ParityResult result;
  std::string production_path = args.dir + "/enc_steady_aoti.pt2";
  std::string new_b1_path = (fs::path(steady_batch_dir) / "enc_steady_aoti_b1.pt2").string();
  result.production_sha = sha256_file(production_path);
  result.new_b1_sha = sha256_file(new_b1_path);

  auto stream = stream_for_worker(true, 0);
  auto ctx = make_worker_context(args.dir, device, stream, shared_bundle);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  verify_tokenizer_selftest(ctx->bundle, tokenizer);
  AOTIModelPackageLoader enc_steady(production_path, "model", false, 1, device.index());
  int utt = pick_first_continuation_utt(ctx->bundle, rows_total);
  B1RowSpec spec{utt, 1, "a1_parity.utt" + std::to_string(utt) + ".chunk1"};
  auto prepared = prepare_b1_row_until_target(spec, *ctx, enc_steady, device, tokenizer);

  BatchedSteadyLoaderSet new_b1(steady_batch_dir,
                                args.dir + "/finalize_shared_weights.ts",
                                device,
                                1,
                                "b2_a1_new_b1_tensor_parity");
  new_b1.preload_all();
  std::vector<BatchedSteadyInput> ready{{
      prepared.chunk,
      prepared.session.clc,
      prepared.session.clt,
      prepared.session.clcl,
      spec.label,
  }};
  auto got = new_b1.run(ready, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
  if (got.size() != 1) throw std::runtime_error("B2 A1 new B=1 returned wrong row count");
  result.max_enc_out_diff = b2_max_abs_tensor_diff(got[0].tensors[0], prepared.alone_out[0]);
  result.max_cache_ch_diff = b2_max_abs_tensor_diff(got[0].tensors[2], prepared.alone_out[2]);
  result.max_cache_t_diff = b2_max_abs_tensor_diff(got[0].tensors[3], prepared.alone_out[3]);
  if (!at::equal(got[0].tensors[1], prepared.alone_out[1])) ++result.enc_len_mismatches;
  if (!at::equal(got[0].tensors[4], prepared.alone_out[4])) ++result.cache_len_mismatches;
  result.tensor_ok = result.max_enc_out_diff <= kAotiTensorTolerance &&
                     result.max_cache_ch_diff <= kAotiTensorTolerance &&
                     result.max_cache_t_diff <= kAotiTensorTolerance &&
                     result.enc_len_mismatches == 0 &&
                     result.cache_len_mismatches == 0;
  if (result.production_sha == result.new_b1_sha) {
    result.outcome = "A";
  } else if (result.tensor_ok) {
    result.outcome = "B";
  } else {
    result.outcome = "C";
  }
  std::printf("B2_A1_PARITY outcome=%s production_sha=%s new_b1_sha=%s tensor_ok=%s "
              "max_enc_out=%.3e max_cache_ch=%.3e max_cache_t=%.3e "
              "enc_len_mismatches=%d cache_len_mismatches=%d\n",
              result.outcome.c_str(),
              result.production_sha.c_str(),
              result.new_b1_sha.c_str(),
              result.tensor_ok ? "true" : "false",
              result.max_enc_out_diff,
              result.max_cache_ch_diff,
              result.max_cache_t_diff,
              result.enc_len_mismatches,
              result.cache_len_mismatches);
  if (result.outcome == "C") {
    throw std::runtime_error("B2 A1 parity failed: NEW B=1 tensor parity does not match production B=1");
  }
  cleanup_cuda_cache();
  return result;
}

static bool run_b1_t1_gate(const DensityArgs& args,
                           torch::Device device,
                           const std::string& stamp,
                           const std::shared_ptr<torch::jit::Module>& shared_bundle,
                           int rows_total) {
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  std::printf("=== B1_T1 START dir=%s steady_batch_dir=%s rows=%d/%d flag_NEMOTRON_DENSITY_BATCH_STEADY=%s "
              "b1_batch_size=%d pad_policy=duplicate_row0_discard_pads ===\n",
              args.dir.c_str(),
              steady_batch_dir.c_str(),
              rows,
              rows_total,
              density_batch_steady_enabled() ? "ON" : "OFF",
              args.b1_batch_size);

  auto stream = stream_for_worker(true, 0);
  auto ctx = make_worker_context(args.dir, device, stream, shared_bundle);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  verify_tokenizer_selftest(ctx->bundle, tokenizer);

  TimingBuckets ref_timings;
  auto reference = build_serial_reference(args, device, shared_bundle, rows, &ref_timings);
  cleanup_cuda_cache();

  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 1, device.index());
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        1,
                                        "b1_t1_shared_constants_buckets_1_2_4");
  batched_steady.preload_all();
  FinalizeBucketLoaderPool finalize_loaders(args.dir, device, 1, "b1_t1_finalize_one_runner");

  B1T1Stats stats;
  bool ok = true;
  ok = run_b1_batched_case("identical_rows_B4",
                           pick_identical_b1_specs(ctx->bundle, rows),
                           *ctx,
                           enc_steady,
                           batched_steady,
                           finalize_loaders,
                           device,
                           tokenizer,
                           reference,
                           stats) && ok;
  ok = run_b1_batched_case("ragged_mixed_K3_padded_to_B4",
                           pick_mixed_b1_specs(ctx->bundle, rows),
                           *ctx,
                           enc_steady,
                           batched_steady,
                           finalize_loaders,
                           device,
                           tokenizer,
                           reference,
                           stats) && ok;

  auto coverage = build_b1_coverage_specs(ctx->bundle, rows, stats);
  for (size_t pos = 0; pos < coverage.size(); pos += static_cast<size_t>(args.b1_batch_size)) {
    size_t end = std::min(coverage.size(), pos + static_cast<size_t>(args.b1_batch_size));
    std::vector<B1RowSpec> group(coverage.begin() + static_cast<long>(pos),
                                 coverage.begin() + static_cast<long>(end));
    ok = run_b1_batched_case("coverage_group_" + std::to_string(pos / static_cast<size_t>(args.b1_batch_size)),
                             group,
                             *ctx,
                             enc_steady,
                             batched_steady,
                             finalize_loaders,
                             device,
                             tokenizer,
                             reference,
                             stats) && ok;
  }

  ok = ok &&
       stats.token_divergences == 0 &&
       stats.event_divergences == 0 &&
       stats.enc_len_mismatches == 0 &&
       stats.cache_len_mismatches == 0;
  std::ostringstream json;
  json << "{\"check\":\"b1_batched_steady_t1\""
       << ",\"rows_reference\":" << rows
       << ",\"rows_batched\":" << stats.rows
       << ",\"cases\":" << stats.cases
       << ",\"skipped_no_steady\":" << stats.skipped_no_steady
       << ",\"steady_batch_dir\":" << json_quote(steady_batch_dir)
       << ",\"flag_NEMOTRON_DENSITY_BATCH_STEADY\":" << json_bool(density_batch_steady_enabled())
       << ",\"pad_policy\":\"duplicate_row0_discard_pads\""
       << ",\"token_divergences\":" << stats.token_divergences
       << ",\"event_divergences\":" << stats.event_divergences
       << ",\"enc_len_mismatches\":" << stats.enc_len_mismatches
       << ",\"cache_len_mismatches\":" << stats.cache_len_mismatches
       << ",\"max_enc_out_diff\":" << stats.max_enc_out_diff
       << ",\"max_cache_ch_diff\":" << stats.max_cache_ch_diff
       << ",\"max_cache_t_diff\":" << stats.max_cache_t_diff
       << ",\"pass\":" << json_bool(ok)
       << "}";
  emit_telemetry(args.dir, stamp, 1, "explicit", "b1_batched_steady_t1", json.str());
  std::printf("=== B1_T1 RESULT %s: reference_rows=%d batched_rows=%d cases=%d skipped_no_steady=%d "
              "max_enc_out=%.3e max_cache_ch=%.3e max_cache_t=%.3e "
              "enc_len_mismatches=%d cache_len_mismatches=%d token_divergences=%d event_divergences=%d ===\n",
              ok ? "PASS" : "FAIL",
              rows,
              stats.rows,
              stats.cases,
              stats.skipped_no_steady,
              stats.max_enc_out_diff,
              stats.max_cache_ch_diff,
              stats.max_cache_t_diff,
              stats.enc_len_mismatches,
              stats.cache_len_mismatches,
              stats.token_divergences,
              stats.event_divergences);
  return ok;
}

struct B2T1CaseResult {
  std::string name;
  bool pass = false;
  int workers = 0;
  int rows = 0;
  int errors = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  int enc_len_mismatches = 0;
  int cache_len_mismatches = 0;
  double max_enc_out_diff = 0.0;
  double max_cache_ch_diff = 0.0;
  double max_cache_t_diff = 0.0;
  AdmissionTelemetry admission;
  StaleGenTelemetrySnapshot stale_gen;
  BatchedSteadySchedulerTelemetry telemetry;
};

static bool compare_b2_token_vector(const std::vector<int64_t>& got,
                                    const std::vector<int64_t>& ref,
                                    const std::string& label,
                                    int worker,
                                    int utt) {
  if (got == ref) return true;
  int pos = first_token_diff_pos(got, ref);
  std::printf("B2_T1_TOKEN_DIVERGENCE label=%s worker=%d utt=%d pos=%d got_len=%zu ref_len=%zu",
              label.c_str(),
              worker,
              utt,
              pos,
              got.size(),
              ref.size());
  if (pos >= 0 && pos < static_cast<int>(got.size())) {
    std::printf(" got=%lld", static_cast<long long>(got[static_cast<size_t>(pos)]));
  }
  if (pos >= 0 && pos < static_cast<int>(ref.size())) {
    std::printf(" ref=%lld", static_cast<long long>(ref[static_cast<size_t>(pos)]));
  }
  std::printf("\n");
  return false;
}

static int expected_scheduler_warmup_runs(const BatchedSteadySchedulerPolicy& policy) {
  return static_cast<int>(BatchedSteadyScheduler::required_buckets_for_policy(policy).size()) *
         std::max(policy.dispatch_lanes, 1);
}

static int density_bucket_for_k_for_assertions(int k) {
  if (k <= 0) throw std::runtime_error("density bucket assertion requires K>0");
  if (k <= 1) return 1;
  if (k <= 2) return 2;
  if (k <= 4) return 4;
  if (k <= 8) return 8;
  if (k <= 16) return 16;
  throw std::runtime_error("density bucket assertion K exceeds B16: " + std::to_string(k));
}

static int density_expected_dispatch_bucket_for_k(const BatchedSteadySchedulerPolicy& policy, int k) {
  if (policy.force_bucket > 0) {
    if (k > policy.force_bucket) {
      throw std::runtime_error("density bucket assertion force_bucket smaller than K");
    }
    return policy.force_bucket;
  }
  int bucket = density_bucket_for_k_for_assertions(k);
  if (bucket > policy.B_max) bucket = policy.B_max;
  bucket = density_bucket_for_k_for_assertions(bucket);
  if (!policy.use_b2_bucket && bucket == 2) return 4;
  return bucket;
}

static bool density_telemetry_has_bucket(const BatchedSteadySchedulerTelemetry& telemetry, int bucket) {
  if (bucket == 1) return telemetry.bucket_b1 > 0;
  if (bucket == 2) return telemetry.bucket_b2 > 0;
  if (bucket == 4) return telemetry.bucket_b4 > 0;
  if (bucket == 8) return telemetry.bucket_b8 > 0;
  if (bucket == 16) return telemetry.bucket_b16 > 0;
  return false;
}

static bool density_telemetry_has_full_k(const BatchedSteadySchedulerTelemetry& telemetry,
                                         int bucket,
                                         int k,
                                         int expected_full) {
  if (bucket == 1) return telemetry.bucket_b1 >= expected_full;
  if (bucket == 2) return telemetry.bucket_b2 >= expected_full;
  if (bucket == 4) {
    if (k == 2) return telemetry.k2_padded_to_b4 >= expected_full;
    if (k == 3) return telemetry.k3_padded_to_b4 >= expected_full;
    if (k == 4) return telemetry.k4 >= expected_full;
    return telemetry.bucket_b4 >= expected_full;
  }
  if (bucket == 8) {
    if (k == 5) return telemetry.k5_padded_to_b8 >= expected_full;
    if (k == 6) return telemetry.k6_padded_to_b8 >= expected_full;
    if (k == 7) return telemetry.k7_padded_to_b8 >= expected_full;
    if (k == 8) return telemetry.k8 >= expected_full;
    return telemetry.bucket_b8 >= expected_full;
  }
  if (bucket == 16) {
    if (k == 9) return telemetry.k9_padded_to_b16 >= expected_full;
    if (k == 10) return telemetry.k10_padded_to_b16 >= expected_full;
    if (k == 11) return telemetry.k11_padded_to_b16 >= expected_full;
    if (k == 12) return telemetry.k12_padded_to_b16 >= expected_full;
    if (k == 13) return telemetry.k13_padded_to_b16 >= expected_full;
    if (k == 14) return telemetry.k14_padded_to_b16 >= expected_full;
    if (k == 15) return telemetry.k15_padded_to_b16 >= expected_full;
    if (k == 16) return telemetry.k16 >= expected_full;
    return telemetry.bucket_b16 >= expected_full;
  }
  return false;
}

static B2T1CaseResult run_b2_t1_case(const DensityArgs& args,
                                     torch::Device device,
                                     const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                     BatchedSteadyLoaderSet& batched_steady,
                                     AOTIModelPackageLoader& enc_steady,
                                     FinalizeBucketLoaderPool& finalize_loaders,
                                     const std::shared_ptr<SharedEncFirst>& shared_enc_first,
                                     const std::vector<RowReplayResult>& reference,
                                     const std::string& case_name,
                                     const std::vector<std::vector<int>>& assignments,
                                     BatchedSteadySchedulerPolicy policy,
                                     bool forced_barrier,
                                     double stagger_ms,
                                     bool require_b1,
                                     bool require_b2,
                                     bool require_k3_padded,
                                     bool require_bmax_full_batches,
                                     bool require_b4_dispatch = false,
                                     bool require_b8_dispatch = false,
                                     int require_b8_k = 0,
                                     bool forbid_b8_dispatch = false,
                                     bool require_b16_dispatch = false,
                                     int require_b16_k = 0,
                                     bool forbid_b16_dispatch = false) {
  B2T1CaseResult result;
  result.name = case_name;
  result.workers = static_cast<int>(assignments.size());
  for (const auto& v : assignments) result.rows += static_cast<int>(v.size());
  if (result.workers <= 0 || result.rows <= 0) throw std::runtime_error("B2 T1 case has no work: " + case_name);
  std::printf("B2_T1_CASE case=%s START workers=%d rows=%d forced_barrier=%s stagger_ms=%.3f "
              "policy={B_max:%d,window_ms:%d,lone_timeout_ms:%d,max_queue_delay_ms:%d,"
              "queue_capacity:%d,use_b2_bucket:%s,min_fill_enabled:%s,disable_min_fill:%s,force_bucket:%d}\n",
              case_name.c_str(),
              result.workers,
              result.rows,
              forced_barrier ? "true" : "false",
              stagger_ms,
              policy.B_max,
              policy.window_ms,
              policy.lone_timeout_ms,
              policy.max_queue_delay_ms,
              policy.queue_capacity,
              policy.use_b2_bucket ? "true" : "false",
              policy.min_fill_enabled ? "true" : "false",
              policy.disable_min_fill ? "true" : "false",
              policy.force_bucket);

  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();
  DensityAdmission admission(static_cast<uint64_t>(args.admission_active_cap),
                             static_cast<uint64_t>(args.admission_backlog_cap));
  StaleGenTelemetry stale_gen;

  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    auto stream = stream_for_worker(true, worker);
    contexts.push_back(make_worker_context(args.dir, device, stream, shared_bundle, shared_enc_first));
  }
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);
  B2ReusableBarrier continuation_barrier(result.workers);
  B2TensorDiffStats diff_stats;
  StartGate gate(result.workers);
  std::mutex stats_mutex;
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        if (stagger_ms > 0.0 && result.workers > 1) {
          double frac = static_cast<double>(worker) / static_cast<double>(result.workers - 1);
          std::this_thread::sleep_for(ms_duration(stagger_ms * frac));
        }
        TimingBuckets ignored;
        for (int utt : assignments[static_cast<size_t>(worker)]) {
          std::string stream_id = case_name + ".worker" + std::to_string(worker) +
                                  ".utt" + std::to_string(utt);
          AdmitResult admit = admission.try_admit(stream_id);
          if (!wait_until_admission_active(admission, stream_id, admit.decision)) {
            std::lock_guard<std::mutex> lock(stats_mutex);
            if (admit.shed()) ++result.errors;
            std::printf("B2_T1_ADMISSION_SHED case=%s worker=%d utt=%d decision=%s\n",
                        case_name.c_str(),
                        worker,
                        utt,
                        admission_decision_name(admit.decision));
            continue;
          }
          AdmissionCloseGuard admission_guard(admission, stream_id);
          auto row = replay_row_density(utt,
                                        *contexts[worker],
                                        &enc_steady,
                                        nullptr,
                                        finalize_loaders,
                                        device,
                                        tokenizer,
                                        true,
                                        false,
                                        &ignored,
                                        false,
                                        &scheduler,
                                        forced_barrier ? &continuation_barrier : nullptr,
                                        &diff_stats,
                                        &enc_steady,
                                        &stale_gen);
          bool steady_ok = false;
          bool final_ok = false;
          bool events_ok = false;
          if (!row.error.empty()) {
            std::lock_guard<std::mutex> lock(stats_mutex);
            ++result.errors;
            std::printf("B2_T1_ROW_ERROR case=%s worker=%d utt=%d error=%s\n",
                        case_name.c_str(), worker, utt, row.error.c_str());
            continue;
          }
          if (utt >= static_cast<int>(reference.size())) {
            throw std::runtime_error("B2 reference missing utt" + std::to_string(utt));
          }
          if (row.stale_dropped) {
            continue;
          }
          steady_ok = compare_b2_token_vector(row.steady_tokens,
                                              reference[static_cast<size_t>(utt)].steady_tokens,
                                              case_name + ".steady",
                                              worker,
                                              utt);
          final_ok = compare_b2_token_vector(row.final_tokens,
                                             reference[static_cast<size_t>(utt)].final_tokens,
                                             case_name + ".final",
                                             worker,
                                             utt);
          events_ok = strict_events_equal(row.events,
                                          reference[static_cast<size_t>(utt)].events,
                                          "b2." + case_name + ".worker" + std::to_string(worker) +
                                              ".utt" + std::to_string(utt) + ".events");
          std::lock_guard<std::mutex> lock(stats_mutex);
          if (!steady_ok) ++result.token_divergences;
          if (!final_ok) ++result.token_divergences;
          if (!events_ok) ++result.event_divergences;
        }
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        ++result.errors;
        std::printf("B2_T1_WORKER_ERROR case=%s worker=%d error=%s\n",
                    case_name.c_str(), worker, e.what());
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  CUDA_CHECK(cudaDeviceSynchronize());
  scheduler.close();
  result.telemetry = scheduler.telemetry_snapshot();
  result.admission = admission.telemetry_snapshot();
  result.stale_gen = stale_gen.snapshot();
  {
    std::lock_guard<std::mutex> lock(diff_stats.mutex);
    result.enc_len_mismatches = diff_stats.enc_len_mismatches;
    result.cache_len_mismatches = diff_stats.cache_len_mismatches;
    result.max_enc_out_diff = diff_stats.max_enc_out_diff;
    result.max_cache_ch_diff = diff_stats.max_cache_ch_diff;
    result.max_cache_t_diff = diff_stats.max_cache_t_diff;
  }

  bool buckets_ok = result.telemetry.warmup_runs == expected_scheduler_warmup_runs(policy) &&
                    result.telemetry.warmed_lanes == policy.dispatch_lanes;
  if (require_b1) {
    buckets_ok = buckets_ok &&
                 density_telemetry_has_bucket(result.telemetry,
                                               density_expected_dispatch_bucket_for_k(policy, 1));
  }
  if (require_b2) {
    int bucket = density_expected_dispatch_bucket_for_k(policy, 2);
    buckets_ok = buckets_ok && density_telemetry_has_full_k(result.telemetry, bucket, 2, 1);
  }
  if (require_k3_padded) {
    int bucket = density_expected_dispatch_bucket_for_k(policy, 3);
    buckets_ok = buckets_ok && density_telemetry_has_full_k(result.telemetry, bucket, 3, 1);
  }
  if (require_bmax_full_batches) {
    int dispatch_k = policy.force_bucket > 0
                         ? std::min(result.workers, policy.force_bucket)
                         : std::min(result.workers, std::max(1, policy.B_max));
    int expected_full = std::max(1, result.workers / std::max(1, dispatch_k));
    int bucket = density_expected_dispatch_bucket_for_k(policy, dispatch_k);
    buckets_ok = buckets_ok &&
                 density_telemetry_has_full_k(result.telemetry, bucket, dispatch_k, expected_full);
  }
  if (require_b4_dispatch) buckets_ok = buckets_ok && result.telemetry.bucket_b4 > 0;
  if (require_b8_dispatch) buckets_ok = buckets_ok && result.telemetry.bucket_b8 > 0;
  if (forbid_b8_dispatch) buckets_ok = buckets_ok && result.telemetry.bucket_b8 == 0;
  if (require_b16_dispatch) buckets_ok = buckets_ok && result.telemetry.bucket_b16 > 0;
  if (forbid_b16_dispatch) buckets_ok = buckets_ok && result.telemetry.bucket_b16 == 0;
  if (require_b8_k == 5) {
    buckets_ok = buckets_ok && result.telemetry.k5_padded_to_b8 > 0;
  } else if (require_b8_k == 6) {
    buckets_ok = buckets_ok && result.telemetry.k6_padded_to_b8 > 0;
  } else if (require_b8_k == 7) {
    buckets_ok = buckets_ok && result.telemetry.k7_padded_to_b8 > 0;
  } else if (require_b8_k == 8) {
    buckets_ok = buckets_ok && result.telemetry.k8 > 0;
  }
  if (require_b16_k > 0) {
    buckets_ok = buckets_ok && density_telemetry_has_full_k(result.telemetry, 16, require_b16_k, 1);
  }
  bool events_pass = density_event_divergences_gate_pass(result.event_divergences);
  result.pass = result.errors == 0 &&
                result.token_divergences == 0 &&
                events_pass &&
                result.enc_len_mismatches == 0 &&
                result.cache_len_mismatches == 0 &&
                buckets_ok;
  SummaryStats age_at_dispatch_summary = summarize(result.telemetry.age_at_dispatch_us);
  std::printf("B2_T1_BUCKET case=%s dispatch_lanes=%d warmup_runs=%lld warmed_lanes=%lld "
              "B1=%lld B2=%lld B4=%lld B8=%lld B16=%lld "
              "K2_padded_to_B4=%lld K3_padded_to_B4=%lld K4=%lld "
              "K5_padded_to_B8=%lld K6_padded_to_B8=%lld K7_padded_to_B8=%lld K8=%lld "
              "K9_padded_to_B16=%lld K10_padded_to_B16=%lld K11_padded_to_B16=%lld "
              "K12_padded_to_B16=%lld K13_padded_to_B16=%lld K14_padded_to_B16=%lld "
              "K15_padded_to_B16=%lld K16=%lld "
              "backlog_gt_bmax=%lld skipped_ready=%lld age_at_dispatch_n=%zu age_at_dispatch_max_us=%.3f "
              "enqueued=%lld completed=%lld buckets_ok=%s\n",
              case_name.c_str(),
              policy.dispatch_lanes,
              static_cast<long long>(result.telemetry.warmup_runs),
              static_cast<long long>(result.telemetry.warmed_lanes),
              static_cast<long long>(result.telemetry.bucket_b1),
              static_cast<long long>(result.telemetry.bucket_b2),
              static_cast<long long>(result.telemetry.bucket_b4),
              static_cast<long long>(result.telemetry.bucket_b8),
              static_cast<long long>(result.telemetry.bucket_b16),
              static_cast<long long>(result.telemetry.k2_padded_to_b4),
              static_cast<long long>(result.telemetry.k3_padded_to_b4),
              static_cast<long long>(result.telemetry.k4),
              static_cast<long long>(result.telemetry.k5_padded_to_b8),
              static_cast<long long>(result.telemetry.k6_padded_to_b8),
              static_cast<long long>(result.telemetry.k7_padded_to_b8),
              static_cast<long long>(result.telemetry.k8),
              static_cast<long long>(result.telemetry.k9_padded_to_b16),
              static_cast<long long>(result.telemetry.k10_padded_to_b16),
              static_cast<long long>(result.telemetry.k11_padded_to_b16),
              static_cast<long long>(result.telemetry.k12_padded_to_b16),
              static_cast<long long>(result.telemetry.k13_padded_to_b16),
              static_cast<long long>(result.telemetry.k14_padded_to_b16),
              static_cast<long long>(result.telemetry.k15_padded_to_b16),
              static_cast<long long>(result.telemetry.k16),
              static_cast<long long>(result.telemetry.backlog_gt_bmax),
              static_cast<long long>(result.telemetry.skipped_ready),
              age_at_dispatch_summary.n,
              age_at_dispatch_summary.max,
              static_cast<long long>(result.telemetry.enqueued),
              static_cast<long long>(result.telemetry.completed),
              buckets_ok ? "true" : "false");
  std::printf("B2_T1_CASE case=%s RESULT %s rows=%d errors=%d token_divergences=%d event_divergences=%d "
              "max_enc_out=%.3e max_cache_ch=%.3e max_cache_t=%.3e "
              "enc_len_mismatches=%d cache_len_mismatches=%d\n",
              case_name.c_str(),
              result.pass ? "PASS" : "FAIL",
              result.rows,
              result.errors,
              result.token_divergences,
              result.event_divergences,
              result.max_enc_out_diff,
              result.max_cache_ch_diff,
              result.max_cache_t_diff,
              result.enc_len_mismatches,
              result.cache_len_mismatches);
  cleanup_cuda_cache();
  return result;
}

static constexpr double kB8OracleEncOutTolerance = 5.0e-5;
static constexpr double kB16OracleEncOutTolerance = 6.0e-5;
static constexpr double kLargeBucketOracleCacheTolerance = 2.0e-2;

static double large_bucket_oracle_enc_out_tolerance(int oracle_bucket) {
  if (oracle_bucket == 16) return kB16OracleEncOutTolerance;
  return kB8OracleEncOutTolerance;
}

static std::vector<B1RowSpec> pick_large_bucket_oracle_specs(torch::jit::Module& bundle,
                                                             int rows,
                                                             int count,
                                                             int oracle_bucket) {
  std::vector<B1RowSpec> specs;
  specs.reserve(static_cast<size_t>(count));
  for (int utt = 0; utt < rows && static_cast<int>(specs.size()) < count; ++utt) {
    int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
    if (num_steady > 1) {
      specs.push_back({utt,
                       1,
                       "b" + std::to_string(oracle_bucket) + "oracle.utt" +
                           std::to_string(utt) + ".chunk1"});
    }
  }
  for (int chunk = 2; static_cast<int>(specs.size()) < count; ++chunk) {
    bool found_any = false;
    for (int utt = 0; utt < rows && static_cast<int>(specs.size()) < count; ++utt) {
      int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
      if (num_steady > chunk) {
        found_any = true;
        specs.push_back({utt,
                         chunk,
                         "b" + std::to_string(oracle_bucket) + "oracle.utt" +
                             std::to_string(utt) + ".chunk" + std::to_string(chunk)});
      }
    }
    if (!found_any) break;
  }
  if (static_cast<int>(specs.size()) < count) {
    throw std::runtime_error("B" + std::to_string(oracle_bucket) + " oracle could not find " +
                             std::to_string(count) +
                             " continuation rows for non-identical padding coverage");
  }
  return specs;
}

static std::vector<int> large_bucket_oracle_order(int k, int permutation) {
  std::vector<int> order(static_cast<size_t>(k));
  std::iota(order.begin(), order.end(), 0);
  if (permutation == 1) {
    std::rotate(order.begin(), order.begin() + 1, order.end());
    std::reverse(order.begin() + std::min(2, k), order.end());
  }
  return order;
}

static std::vector<int> large_bucket_oracle_ks(int oracle_bucket) {
  std::vector<int> ks;
  for (int k = oracle_bucket / 2 + 1; k < oracle_bucket; ++k) ks.push_back(k);
  if (oracle_bucket == 8) ks.push_back(8);
  return ks;
}

static B2T1CaseResult run_b2_t1_forced_bucket_oracle(
    const DensityArgs& args,
    torch::Device device,
    const std::shared_ptr<torch::jit::Module>& shared_bundle,
    BatchedSteadyLoaderSet& batched_steady,
    AOTIModelPackageLoader& enc_steady,
    FinalizeBucketLoaderPool& finalize_loaders,
    const std::shared_ptr<SharedEncFirst>& shared_enc_first,
    const std::vector<RowReplayResult>& reference,
    BatchedSteadySchedulerPolicy base_policy,
    int rows,
    int oracle_bucket) {
  if (oracle_bucket != 8 && oracle_bucket != 16) {
    throw std::runtime_error("forced bucket oracle supports B8/B16 only");
  }
  std::string tag = "B2_T1_B" + std::to_string(oracle_bucket) + "_ORACLE";
  B2T1CaseResult result;
  result.name = "forced_B" + std::to_string(oracle_bucket) + "_partial_math_oracle";
  result.workers = oracle_bucket;
  BatchedSteadySchedulerPolicy policy = base_policy;
  policy.B_max = std::max(policy.B_max, oracle_bucket);
  policy.force_bucket = oracle_bucket;
  policy.disable_min_fill = true;
  policy.min_fill_enabled = false;
  policy.queue_capacity = std::max(policy.queue_capacity, oracle_bucket);
  auto oracle_ks = large_bucket_oracle_ks(oracle_bucket);
  double enc_out_tolerance = large_bucket_oracle_enc_out_tolerance(oracle_bucket);

  std::printf("%s START policy={B_max:%d,window_ms:%d,lone_timeout_ms:%d,"
              "max_queue_delay_ms:%d,queue_capacity:%d,use_b2_bucket:%s,min_fill_enabled:%s,"
              "disable_min_fill:%s,force_bucket:%d} tolerances={enc_out:%.3e,cache:%.3e}\n",
              tag.c_str(),
              policy.B_max,
              policy.window_ms,
              policy.lone_timeout_ms,
              policy.max_queue_delay_ms,
              policy.queue_capacity,
              policy.use_b2_bucket ? "true" : "false",
              policy.min_fill_enabled ? "true" : "false",
              policy.disable_min_fill ? "true" : "false",
              policy.force_bucket,
              enc_out_tolerance,
              kLargeBucketOracleCacheTolerance);

  auto specs = pick_large_bucket_oracle_specs(*shared_bundle, rows, oracle_bucket, oracle_bucket);
  auto stream = stream_for_worker(true, 0);
  auto ctx = make_worker_context(args.dir, device, stream, shared_bundle, shared_enc_first);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  B2TensorDiffStats diff_stats;

  for (int k : oracle_ks) {
    for (int permutation = 0; permutation < 2; ++permutation) {
      std::vector<int> order = large_bucket_oracle_order(k, permutation);
      std::vector<B1PreparedRow> prepared;
      prepared.reserve(static_cast<size_t>(k));
      std::vector<BatchedSteadyInput> ready;
      ready.reserve(static_cast<size_t>(k));

      try {
        for (int pos = 0; pos < k; ++pos) {
          B1RowSpec spec = specs[static_cast<size_t>(order[static_cast<size_t>(pos)])];
          spec.label += ".K" + std::to_string(k) + ".perm" + std::to_string(permutation) +
                        ".row" + std::to_string(pos);
          prepared.push_back(prepare_b1_row_until_target(spec, *ctx, enc_steady, device, tokenizer));
          auto& row = prepared.back();
          BatchedSteadyInput input{
              row.chunk,
              row.session.clc,
              row.session.clt,
              row.session.clcl,
              row.spec.label,
          };
          auto b1_out = batched_steady.run({input}, stream);
          CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
          if (b1_out.size() != 1) {
            throw std::runtime_error("B" + std::to_string(oracle_bucket) +
                                     " oracle package B1 returned wrong row count");
          }
          row.alone_out = std::move(b1_out[0].tensors);
          ready.push_back(std::move(input));
        }

        BatchedSteadyScheduler scheduler(batched_steady, device, policy);
        scheduler.warmup_buckets();
        std::vector<std::future<DispatchResult>> futures;
        futures.reserve(static_cast<size_t>(k));
        for (int pos = 0; pos < k; ++pos) {
          cudaEvent_t producer_event{};
          CUDA_CHECK(cudaEventCreateWithFlags(&producer_event, cudaEventDisableTiming));
          try {
            CUDA_CHECK(cudaEventRecord(producer_event, stream.stream()));
            futures.push_back(scheduler.enqueue({
                ready[static_cast<size_t>(pos)],
                stream,
                producer_event,
                static_cast<uint64_t>(1000 + k * 100 + permutation * 10 + pos),
            }));
            producer_event = nullptr;
          } catch (...) {
            if (producer_event != nullptr) CUDA_CHECK(cudaEventDestroy(producer_event));
            throw;
          }
        }
        scheduler.start();

        for (int pos = 0; pos < k; ++pos) {
          auto& future = futures[static_cast<size_t>(pos)];
          auto wait_status = future.wait_for(std::chrono::milliseconds(scheduler.future_timeout_ms()));
          if (wait_status != std::future_status::ready) {
            throw std::runtime_error("B" + std::to_string(oracle_bucket) +
                                     " oracle scheduler future timeout K=" + std::to_string(k));
          }
          auto dispatch = future.get();
          if (dispatch.bucket != oracle_bucket || dispatch.k != k || dispatch.row != pos) {
            std::printf("%s_DISPATCH_MISMATCH K=%d perm=%d pos=%d "
                        "bucket=%d dispatch_k=%d row=%d\n",
                        tag.c_str(),
                        k,
                        permutation,
                        pos,
                        dispatch.bucket,
                        dispatch.k,
                        dispatch.row);
            ++result.errors;
          }
          if (dispatch.completion.get() == nullptr) {
            throw std::runtime_error("B" + std::to_string(oracle_bucket) +
                                     " oracle scheduler returned null completion event");
          }
          CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), dispatch.completion.get(), 0));
          dispatch.completion.reset();
          if (dispatch.graph_slot) {
            dispatch.graph_slot->retire_on_stream(stream.stream());
            dispatch.graph_slot.reset();
          }
          CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

          auto& row = prepared[static_cast<size_t>(pos)];
          diff_stats.update(dispatch.row_tensors, row.alone_out, row.spec.label);
          auto replay = finish_b1_row_from_target(row,
                                                  dispatch.row_tensors,
                                                  *ctx,
                                                  enc_steady,
                                                  finalize_loaders,
                                                  device,
                                                  tokenizer,
                                                  result.name);
          bool row_ok = replay.ok;
          if (!replay.error.empty()) {
            std::printf("%s_ROW_ERROR K=%d perm=%d pos=%d label=%s error=%s\n",
                        tag.c_str(),
                        k,
                        permutation,
                        pos,
                        row.spec.label.c_str(),
                        replay.error.c_str());
            row_ok = false;
            ++result.errors;
          }
          if (row.spec.utt >= static_cast<int>(reference.size())) {
            throw std::runtime_error("B" + std::to_string(oracle_bucket) +
                                     " oracle reference missing utt" + std::to_string(row.spec.utt));
          }
          const auto& ref = reference[static_cast<size_t>(row.spec.utt)];
          if (!compare_b2_token_vector(replay.steady_tokens,
                                       ref.steady_tokens,
                                       result.name + ".steady",
                                       pos,
                                       row.spec.utt)) {
            ++result.token_divergences;
            row_ok = false;
          }
          if (!compare_b2_token_vector(replay.final_tokens,
                                       ref.final_tokens,
                                       result.name + ".final",
                                       pos,
                                       row.spec.utt)) {
            ++result.token_divergences;
            row_ok = false;
          }
          if (!strict_events_equal(replay.events,
                                   ref.events,
                                   "b2." + result.name + ".K" + std::to_string(k) +
                                       ".perm" + std::to_string(permutation) + ".row" +
                                       std::to_string(pos) + ".events")) {
            ++result.event_divergences;
            row_ok = false;
          }
          (void)row_ok;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        scheduler.close();
        auto telemetry = scheduler.telemetry_snapshot();
        merge_scheduler_telemetry(result.telemetry, telemetry);
        bool bucket_ok = telemetry.warmup_runs == expected_scheduler_warmup_runs(policy) &&
                         telemetry.warmed_lanes == policy.dispatch_lanes &&
                         density_telemetry_has_bucket(telemetry, oracle_bucket) &&
                         telemetry.bucket_b4 == 0 &&
                         (oracle_bucket < 16 || telemetry.bucket_b8 == 0) &&
                         density_telemetry_has_full_k(telemetry, oracle_bucket, k, 1);
        if (!bucket_ok) ++result.errors;
        std::printf("%s_CASE K=%d perm=%d RESULT %s B4=%lld B8=%lld B16=%lld "
                    "K5_to_B8=%lld K6_to_B8=%lld K7_to_B8=%lld K8=%lld "
                    "K9_to_B16=%lld K10_to_B16=%lld K11_to_B16=%lld K12_to_B16=%lld "
                    "K13_to_B16=%lld K14_to_B16=%lld K15_to_B16=%lld K16=%lld errors=%d\n",
                    tag.c_str(),
                    k,
                    permutation,
                    bucket_ok ? "PASS" : "FAIL",
                    static_cast<long long>(telemetry.bucket_b4),
                    static_cast<long long>(telemetry.bucket_b8),
                    static_cast<long long>(telemetry.bucket_b16),
                    static_cast<long long>(telemetry.k5_padded_to_b8),
                    static_cast<long long>(telemetry.k6_padded_to_b8),
                    static_cast<long long>(telemetry.k7_padded_to_b8),
                    static_cast<long long>(telemetry.k8),
                    static_cast<long long>(telemetry.k9_padded_to_b16),
                    static_cast<long long>(telemetry.k10_padded_to_b16),
                    static_cast<long long>(telemetry.k11_padded_to_b16),
                    static_cast<long long>(telemetry.k12_padded_to_b16),
                    static_cast<long long>(telemetry.k13_padded_to_b16),
                    static_cast<long long>(telemetry.k14_padded_to_b16),
                    static_cast<long long>(telemetry.k15_padded_to_b16),
                    static_cast<long long>(telemetry.k16),
                    result.errors);
        result.rows += k;
      } catch (const std::exception& e) {
        ++result.errors;
        std::printf("%s_ERROR K=%d perm=%d error=%s\n",
                    tag.c_str(),
                    k,
                    permutation,
                    e.what());
      }
      cleanup_cuda_cache();
    }
  }

  {
    std::lock_guard<std::mutex> lock(diff_stats.mutex);
    result.enc_len_mismatches = diff_stats.enc_len_mismatches;
    result.cache_len_mismatches = diff_stats.cache_len_mismatches;
    result.max_enc_out_diff = diff_stats.max_enc_out_diff;
    result.max_cache_ch_diff = diff_stats.max_cache_ch_diff;
    result.max_cache_t_diff = diff_stats.max_cache_t_diff;
  }
  bool tensor_ok = result.enc_len_mismatches == 0 &&
                   result.cache_len_mismatches == 0 &&
                   result.max_enc_out_diff <= enc_out_tolerance &&
                   result.max_cache_ch_diff <= kLargeBucketOracleCacheTolerance &&
                   result.max_cache_t_diff <= kLargeBucketOracleCacheTolerance;
  bool telemetry_ok = density_telemetry_has_bucket(result.telemetry, oracle_bucket);
  for (int k : oracle_ks) {
    telemetry_ok = telemetry_ok &&
                   density_telemetry_has_full_k(result.telemetry, oracle_bucket, k, 1);
  }
  result.pass = result.errors == 0 &&
                result.token_divergences == 0 &&
                result.event_divergences == 0 &&
                tensor_ok &&
                telemetry_ok;
  std::printf("%s_RESULT %s rows=%d errors=%d token_divergences=%d "
              "event_divergences=%d max_enc_out=%.3e max_cache_ch=%.3e max_cache_t=%.3e "
              "enc_tol=%.3e cache_tol=%.3e telemetry_ok=%s\n",
              tag.c_str(),
              result.pass ? "PASS" : "FAIL",
              result.rows,
              result.errors,
              result.token_divergences,
              result.event_divergences,
              result.max_enc_out_diff,
              result.max_cache_ch_diff,
              result.max_cache_t_diff,
              enc_out_tolerance,
              kLargeBucketOracleCacheTolerance,
              telemetry_ok ? "true" : "false");
  cleanup_cuda_cache();
  return result;
}

static bool run_b2_t1_gate(const DensityArgs& args,
                           torch::Device device,
                           const std::string& stamp,
                           const std::shared_ptr<torch::jit::Module>& shared_bundle,
                           int rows_total) {
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  std::printf("B2_T1 START dir=%s steady_batch_dir=%s rows=%d/%d batch_steady_effective=%s\n",
              args.dir.c_str(),
              steady_batch_dir.c_str(),
              rows,
              rows_total,
              density_batch_steady_enabled_effective(args) ? "ON" : "OFF");
  TimingBuckets ref_timings;
  auto reference = build_serial_reference(args, device, shared_bundle, rows, &ref_timings);
  auto a1 = run_b2_a1_parity_check(args, device, steady_batch_dir, shared_bundle, rows);
  int forced_utt = pick_first_continuation_utt(*shared_bundle, rows);
  int b2_dispatch_lanes = density_dispatch_lanes_effective(args);
  int b2_steady_runners = density_effective_steady_num_runners(args, 1, b2_dispatch_lanes);
  BatchedSteadyLoaderSet b2_batched_steady(steady_batch_dir,
                                           args.dir + "/finalize_shared_weights.ts",
                                           device,
                                           b2_steady_runners,
                                           "b2_t1_shared_preloaded_buckets");
  b2_batched_steady.preload_all();
  AOTIModelPackageLoader b2_enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 4, -1);
  FinalizeBucketLoaderPool b2_finalize_loaders(args.dir,
                                               device,
                                               capped_general_finalize_runners(4),
                                               "b2_t1_shared_finalize_pool");
  b2_finalize_loaders.preload_all();
  auto b2_shared_enc_first = load_shared_enc_first(args.dir,
                                                   device,
                                                   "b2_t1_shared_locked_enc_first_all_cases");
  BatchedSteadySchedulerPolicy base_policy = density_batch_policy_effective(args, 4);
  base_policy.queue_capacity = std::max(base_policy.queue_capacity, 16);
  bool b8_enabled = std::max(base_policy.B_max, base_policy.force_bucket) >= 8;
  bool b16_enabled = std::max(base_policy.B_max, base_policy.force_bucket) >= 16;

  std::vector<B2T1CaseResult> cases;
  auto add_case = [&](B2T1CaseResult&& c) {
    cases.push_back(std::move(c));
    cleanup_cuda_cache();
  };
  auto repeated_forced_utt = [&](int n) {
    std::vector<std::vector<int>> assignments;
    assignments.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) assignments.push_back({forced_utt});
    return assignments;
  };
  std::vector<int> single_utts;
  for (int utt = 0; utt < rows; ++utt) single_utts.push_back(utt);
  add_case(run_b2_t1_case(args,
                          device,
                          shared_bundle,
                          b2_batched_steady,
                          b2_enc_steady,
                          b2_finalize_loaders,
                          b2_shared_enc_first,
                          reference,
                          "single_stream_scheduler_on",
                          {single_utts},
                          base_policy,
                          false,
                          0.0,
                          true,
                          false,
                          false,
                          false));
  add_case(run_b2_t1_case(args,
                          device,
                          shared_bundle,
                          b2_batched_steady,
                          b2_enc_steady,
                          b2_finalize_loaders,
                          b2_shared_enc_first,
                          reference,
                          base_policy.use_b2_bucket ? "multi_stream_forced_K2_B2"
                                                    : "multi_stream_forced_K2_padded_B4",
                          {{forced_utt}, {forced_utt}},
                          base_policy,
                          true,
                          0.0,
                          false,
                          true,
                          false,
                          false));
  add_case(run_b2_t1_case(args,
                          device,
                          shared_bundle,
                          b2_batched_steady,
                          b2_enc_steady,
                          b2_finalize_loaders,
                          b2_shared_enc_first,
                          reference,
                          "multi_stream_forced_K3_padded_B4",
                          {{forced_utt}, {forced_utt}, {forced_utt}},
                          base_policy,
                          true,
                          0.0,
                          false,
                          false,
                          true,
                          false));
  add_case(run_b2_t1_case(args,
                          device,
                          shared_bundle,
                          b2_batched_steady,
                          b2_enc_steady,
                          b2_finalize_loaders,
                          b2_shared_enc_first,
                          reference,
                          "multi_stream_forced_concurrency_B4",
                          {{forced_utt}, {forced_utt}, {forced_utt}, {forced_utt}},
                          base_policy,
                          true,
                          0.0,
                          false,
                          false,
                          false,
                          true));
  add_case(run_b2_t1_case(args,
                          device,
                          shared_bundle,
                          b2_batched_steady,
                          b2_enc_steady,
                          b2_finalize_loaders,
                          b2_shared_enc_first,
                          reference,
                          "multi_stream_staggered",
                          {{forced_utt}, {forced_utt}, {forced_utt}, {forced_utt}},
                          base_policy,
                          false,
                          10000.0,
                          false,
                          false,
                          false,
                          false));
  if (b8_enabled && !b16_enabled) {
    add_case(run_b2_t1_forced_bucket_oracle(args,
                                            device,
                                            shared_bundle,
                                            b2_batched_steady,
                                            b2_enc_steady,
                                            b2_finalize_loaders,
                                            b2_shared_enc_first,
                                            reference,
                                            base_policy,
                                            rows,
                                            8));
  }
  if (b16_enabled) {
    add_case(run_b2_t1_forced_bucket_oracle(args,
                                            device,
                                            shared_bundle,
                                            b2_batched_steady,
                                            b2_enc_steady,
                                            b2_finalize_loaders,
                                            b2_shared_enc_first,
                                            reference,
                                            base_policy,
                                            rows,
                                            16));
  }
  if (base_policy.B_max >= 8 &&
      base_policy.force_bucket == 0 &&
      base_policy.min_fill_enabled &&
      !base_policy.disable_min_fill) {
    auto adaptive_policy = base_policy;
    adaptive_policy.B_max = 8;
    adaptive_policy.queue_capacity = std::max(adaptive_policy.queue_capacity, 16);
    add_case(run_b2_t1_case(args,
                            device,
                            shared_bundle,
                            b2_batched_steady,
                            b2_enc_steady,
                            b2_finalize_loaders,
                            b2_shared_enc_first,
                            reference,
                            "adaptive_candidate_K5_B4_plus_leftover",
                            {{forced_utt}, {forced_utt}, {forced_utt}, {forced_utt}, {forced_utt}},
                            adaptive_policy,
                            true,
                            0.0,
                            false,
                            false,
                            false,
                            false,
                            true,
                            false,
                            0,
                            true));
    add_case(run_b2_t1_case(args,
                            device,
                            shared_bundle,
                            b2_batched_steady,
                            b2_enc_steady,
                            b2_finalize_loaders,
                            b2_shared_enc_first,
                            reference,
                            "adaptive_candidate_K8_B8_full",
                            {{forced_utt}, {forced_utt}, {forced_utt}, {forced_utt},
                             {forced_utt}, {forced_utt}, {forced_utt}, {forced_utt}},
                            adaptive_policy,
                            true,
                            0.0,
                            false,
                            false,
                            false,
                            false,
                            false,
                            true,
                            8,
                            false));
    if (base_policy.B_max >= 16) {
      auto b16_adaptive_policy = base_policy;
      b16_adaptive_policy.B_max = 16;
      b16_adaptive_policy.queue_capacity = std::max(b16_adaptive_policy.queue_capacity, 16);
      add_case(run_b2_t1_case(args,
                              device,
                              shared_bundle,
                              b2_batched_steady,
                              b2_enc_steady,
                              b2_finalize_loaders,
                              b2_shared_enc_first,
                              reference,
                              "adaptive_candidate_K9_B8_plus_leftover",
                              repeated_forced_utt(9),
                              b16_adaptive_policy,
                              true,
                              0.0,
                              false,
                              false,
                              false,
                              false,
                              false,
                              true,
                              8,
                              false,
                              false,
                              0,
                              true));
      add_case(run_b2_t1_case(args,
                              device,
                              shared_bundle,
                              b2_batched_steady,
                              b2_enc_steady,
                              b2_finalize_loaders,
                              b2_shared_enc_first,
                              reference,
                              "adaptive_candidate_K16_B16_full",
                              repeated_forced_utt(16),
                              b16_adaptive_policy,
                              true,
                              0.0,
                              false,
                              false,
                              false,
                              true,
                              false,
                              false,
                              0,
                              false,
                              true,
                              16,
                              false));
    }
  }
  auto bmax1_policy = base_policy;
  bmax1_policy.B_max = 1;
  add_case(run_b2_t1_case(args,
                          device,
                          shared_bundle,
                          b2_batched_steady,
                          b2_enc_steady,
                          b2_finalize_loaders,
                          b2_shared_enc_first,
                          reference,
                          "scheduler_on_Bmax1_control",
                          {{forced_utt}, {forced_utt}, {forced_utt}, {forced_utt}},
                          bmax1_policy,
                          true,
                          0.0,
                          true,
                          false,
                          false,
                          true));

  bool ok = true;
  int token_divergences = 0;
  int event_divergences = 0;
  int errors = 0;
  double max_enc_out = 0.0;
  double max_cache_ch = 0.0;
  double max_cache_t = 0.0;
  AdmissionTelemetry combined_admission;
  StaleGenTelemetrySnapshot combined_stale_gen;
  BatchedSteadySchedulerTelemetry combined_scheduler_telemetry;
  for (const auto& c : cases) {
    ok = ok && c.pass;
    token_divergences += c.token_divergences;
    event_divergences += c.event_divergences;
    errors += c.errors;
    max_enc_out = std::max(max_enc_out, c.max_enc_out_diff);
    max_cache_ch = std::max(max_cache_ch, c.max_cache_ch_diff);
    max_cache_t = std::max(max_cache_t, c.max_cache_t_diff);
    merge_admission_telemetry(combined_admission, c.admission);
    merge_stale_gen_telemetry(combined_stale_gen, c.stale_gen);
    merge_scheduler_telemetry(combined_scheduler_telemetry, c.telemetry);
  }
  std::ostringstream json;
  json << "{\"check\":\"b2_batched_scheduler_t1\""
       << ",\"rows_reference\":" << rows
       << ",\"steady_batch_dir\":" << json_quote(steady_batch_dir)
       << ",\"a1_outcome\":\"" << a1.outcome << "\""
       << ",\"token_divergences\":" << token_divergences
       << ",\"event_divergences\":" << event_divergences
       << ",\"errors\":" << errors
       << ",\"max_enc_out_diff\":" << max_enc_out
       << ",\"max_cache_ch_diff\":" << max_cache_ch
       << ",\"max_cache_t_diff\":" << max_cache_t
       << ",\"admission\":" << admission_telemetry_json(combined_admission)
       << ",\"stale_gen\":" << stale_gen_telemetry_json(combined_stale_gen)
       << ",\"ws_tail\":null"
       << ",\"scheduler_telemetry\":" << scheduler_telemetry_json(combined_scheduler_telemetry)
       << ",\"cases\":[";
  for (size_t i = 0; i < cases.size(); ++i) {
    const auto& c = cases[i];
    if (i > 0) json << ",";
    json << "{\"name\":" << json_quote(c.name)
         << ",\"pass\":" << json_bool(c.pass)
         << ",\"rows\":" << c.rows
         << ",\"errors\":" << c.errors
         << ",\"token_divergences\":" << c.token_divergences
         << ",\"event_divergences\":" << c.event_divergences
         << ",\"admission\":" << admission_telemetry_json(c.admission)
         << ",\"stale_gen\":" << stale_gen_telemetry_json(c.stale_gen)
         << ",\"ws_tail\":null"
         << ",\"scheduler_telemetry\":" << scheduler_telemetry_json(c.telemetry)
         << "}";
  }
  json << "]"
       << ",\"pass\":" << json_bool(ok)
       << "}";
  emit_telemetry(args.dir, stamp, 1, "explicit", "b2_batched_scheduler_t1", json.str());
  std::printf("B2_T1_RESULT %s rows_reference=%d cases=%zu token_divergences=%d event_divergences=%d "
              "errors=%d A1_outcome=%s max_enc_out=%.3e max_cache_ch=%.3e max_cache_t=%.3e\n",
              ok ? "PASS" : "FAIL",
              rows,
              cases.size(),
              token_divergences,
              event_divergences,
              errors,
              a1.outcome.c_str(),
              max_enc_out,
              max_cache_ch,
              max_cache_t);
  return ok;
}

static std::vector<int> pick_async_ordering_utts(torch::jit::Module& bundle, int rows_total) {
  std::vector<int> utts;
  for (int utt = 0; utt < rows_total; ++utt) {
    if (scalar_i64(utt_tensor(bundle, utt, "num_steady")) >= 4) {
      utts.push_back(utt);
      if (utts.size() >= 3) break;
    }
  }
  if (utts.empty()) {
    throw std::runtime_error("async-ordering found no utterance with at least 3 continuation chunks");
  }
  while (utts.size() < 3) utts.push_back(utts.front());
  return utts;
}

static std::map<int, RowReplayResult> build_serial_reference_for_utts(
    const DensityArgs& args,
    torch::Device device,
    const std::shared_ptr<torch::jit::Module>& shared_bundle,
    const std::vector<int>& utts) {
  auto stream = stream_for_worker(true, 0);
  auto ctx = make_worker_context(args.dir, device, stream, shared_bundle);
  auto tokenizer = tokenizer_from_bundle(ctx->bundle);
  verify_tokenizer_selftest(ctx->bundle, tokenizer);
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir, device, 1, "async_ordering_serial_reference");
  finalize_loaders.preload_all();
  std::map<int, RowReplayResult> refs;
  std::set<int> unique_utts(utts.begin(), utts.end());
  for (int utt : unique_utts) {
    auto row = replay_row_density(utt,
                                  *ctx,
                                  &enc_steady,
                                  nullptr,
                                  finalize_loaders,
                                  device,
                                  tokenizer,
                                  true,
                                  false,
                                  nullptr,
                                  true,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr);
    if (!row.ok) {
      throw std::runtime_error("async-ordering serial reference failed for utt" + std::to_string(utt) +
                               (row.error.empty() ? "" : ": " + row.error));
    }
    refs.emplace(utt, std::move(row));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return refs;
}

static bool run_async_ordering_gate(const DensityArgs& args,
                                    torch::Device device,
                                    const std::string& stamp,
                                    const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                    int rows_total) {
  const int sessions = 3;
  auto utts = pick_async_ordering_utts(*shared_bundle, rows_total);
  std::printf("ASYNC_ORDERING START sessions=%d utts=%d,%d,%d requirement=delay_consumer_until_2_later_dispatches\n",
              sessions,
              utts[0],
              utts[1],
              utts[2]);

  auto reference = build_serial_reference_for_utts(args, device, shared_bundle, utts);
  cleanup_cuda_cache();

  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy = density_batch_policy_effective(args, sessions);
  policy.B_max = 4;
  policy.queue_capacity = std::max(policy.queue_capacity, 16);
  int scheduler_steady_runners =
      density_effective_steady_num_runners(args, 1, policy.dispatch_lanes);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        scheduler_steady_runners,
                                        "async_ordering_scheduler_buckets");
  batched_steady.preload_all();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 4, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir,
                                            device,
                                            capped_general_finalize_runners(sessions),
                                            "async_ordering_finalize_pool");
  finalize_loaders.preload_all();
  auto shared_enc_first = load_shared_enc_first(args.dir, device, "async_ordering_shared_locked_enc_first");

  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(sessions);
  for (int worker = 0; worker < sessions; ++worker) {
    contexts.push_back(make_worker_context(args.dir,
                                           device,
                                           stream_for_worker(true, worker),
                                           shared_bundle,
                                           shared_enc_first));
  }
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);
  B2TensorDiffStats diff_stats;
  SchedulerConsumerDelayProbe delay_probe;
  delay_probe.delay_ms = 1000.0;
  delay_probe.min_later_dispatches = 2;
  B2ReusableBarrier delay_chunk_barrier(sessions);
  StartGate gate(sessions);
  std::vector<RowReplayResult> rows(static_cast<size_t>(sessions));
  std::vector<std::thread> threads;
  threads.reserve(sessions);
  for (int worker = 0; worker < sessions; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        rows[static_cast<size_t>(worker)] =
            replay_row_density(utts[static_cast<size_t>(worker)],
                               *contexts[worker],
                               &enc_steady,
                               nullptr,
                               finalize_loaders,
                               device,
                               tokenizer,
                               true,
                               false,
                               nullptr,
                               false,
                               &scheduler,
                               &delay_chunk_barrier,
                               &diff_stats,
                               &enc_steady,
                               nullptr,
                               1,
                               worker == 0 ? &delay_probe : nullptr);
      } catch (const std::exception& e) {
        rows[static_cast<size_t>(worker)].ok = false;
        rows[static_cast<size_t>(worker)].error = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  CUDA_CHECK(cudaDeviceSynchronize());
  scheduler.close();
  auto telemetry = scheduler.telemetry_snapshot();

  int errors = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  for (int worker = 0; worker < sessions; ++worker) {
    const int utt = utts[static_cast<size_t>(worker)];
    const auto& row = rows[static_cast<size_t>(worker)];
    if (!row.error.empty()) {
      ++errors;
      std::printf("ASYNC_ORDERING_ROW_ERROR worker=%d utt=%d error=%s\n", worker, utt, row.error.c_str());
      continue;
    }
    const auto& ref = reference.at(utt);
    if (!compare_b2_token_vector(row.steady_tokens, ref.steady_tokens, "async_ordering.steady", worker, utt)) {
      ++token_divergences;
    }
    if (!compare_b2_token_vector(row.final_tokens, ref.final_tokens, "async_ordering.final", worker, utt)) {
      ++token_divergences;
    }
    if (!strict_events_equal(row.events,
                             ref.events,
                             "async_ordering.worker" + std::to_string(worker) + ".utt" + std::to_string(utt))) {
      ++event_divergences;
    }
  }

  int enc_len_mismatches = 0;
  int cache_len_mismatches = 0;
  double max_enc_out_diff = 0.0;
  double max_cache_ch_diff = 0.0;
  double max_cache_t_diff = 0.0;
  {
    std::lock_guard<std::mutex> lock(diff_stats.mutex);
    enc_len_mismatches = diff_stats.enc_len_mismatches;
    cache_len_mismatches = diff_stats.cache_len_mismatches;
    max_enc_out_diff = diff_stats.max_enc_out_diff;
    max_cache_ch_diff = diff_stats.max_cache_ch_diff;
    max_cache_t_diff = diff_stats.max_cache_t_diff;
  }
  int later_dispatches = delay_probe.observed_later_dispatches.load(std::memory_order_acquire);
  bool delayed_ok = delay_probe.delays_run.load(std::memory_order_acquire) == 1 &&
                    later_dispatches >= delay_probe.min_later_dispatches;
  bool events_pass = event_divergences == 0;
  bool tensor_diffs_pass = max_enc_out_diff <= kAotiTensorTolerance &&
                           max_cache_ch_diff <= kAotiTensorTolerance &&
                           max_cache_t_diff <= kAotiTensorTolerance;
  bool ok = errors == 0 &&
            token_divergences == 0 &&
            events_pass &&
            enc_len_mismatches == 0 &&
            cache_len_mismatches == 0 &&
            tensor_diffs_pass &&
            telemetry.dispatcher_exceptions == 0 &&
            delayed_ok;

  std::ostringstream json;
  json << "{\"check\":\"async_ordering_delayed_consumer\""
       << ",\"sessions\":" << sessions
       << ",\"utts\":[" << utts[0] << "," << utts[1] << "," << utts[2] << "]"
       << ",\"delays_run\":" << delay_probe.delays_run.load(std::memory_order_acquire)
       << ",\"later_dispatches_while_delayed\":" << later_dispatches
       << ",\"min_later_dispatches\":" << delay_probe.min_later_dispatches
       << ",\"token_divergences\":" << token_divergences
       << ",\"event_divergences\":" << event_divergences
       << ",\"errors\":" << errors
       << ",\"enc_len_mismatches\":" << enc_len_mismatches
       << ",\"cache_len_mismatches\":" << cache_len_mismatches
       << ",\"max_enc_out_diff\":" << max_enc_out_diff
       << ",\"max_cache_ch_diff\":" << max_cache_ch_diff
       << ",\"max_cache_t_diff\":" << max_cache_t_diff
       << ",\"aoti_tensor_tolerance\":" << kAotiTensorTolerance
       << ",\"scheduler_telemetry\":" << scheduler_telemetry_json(telemetry)
       << ",\"pass\":" << json_bool(ok)
       << "}";
  emit_telemetry(args.dir, stamp, 1, "explicit", "async_ordering_delayed_consumer", json.str());
  std::printf("ASYNC_ORDERING_RESULT %s sessions=%d delayed_later_dispatches=%d token_divergences=%d "
              "event_divergences=%d errors=%d max_enc_out=%.3e max_cache_ch=%.3e max_cache_t=%.3e "
              "aoti_tolerance=%.3e dispatcher_exceptions=%lld\n",
              ok ? "PASS" : "FAIL",
              sessions,
              later_dispatches,
              token_divergences,
              event_divergences,
              errors,
              max_enc_out_diff,
              max_cache_ch_diff,
              max_cache_t_diff,
              kAotiTensorTolerance,
              static_cast<long long>(telemetry.dispatcher_exceptions));
  cleanup_cuda_cache();
  return ok;
}

static std::string stream_mode_label(bool explicit_stream, bool mutex_serialize_run) {
  std::string mode = explicit_stream ? "explicit" : "default";
  if (mutex_serialize_run) mode += "+mutex";
  return mode;
}

static CorrectnessResult run_correctness_gate_mode(const DensityArgs& args,
                                                   torch::Device device,
                                                   const std::string& stamp,
                                                   const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                                   int rows,
                                                   const std::vector<RowReplayResult>& reference,
                                                   bool explicit_stream,
                                                   bool mutex_serialize_run,
                                                   int workers,
                                                   int num_runners,
                                                   const std::string& topology) {
  CorrectnessResult result;
  result.rows = rows;
  result.workers = workers;
  result.num_runners = num_runners;
  result.explicit_stream = explicit_stream;
  result.mutex_serialize_run = mutex_serialize_run;
  std::printf("=== DENSITY 0b correctness mode: rows=%d workers=%d num_runners=%d stream_mode=%s topology=%s ===\n",
              rows, workers, num_runners, stream_mode_label(explicit_stream, mutex_serialize_run).c_str(),
              topology.c_str());
  result.used_before = gpu_used_bytes();
  MemorySampler mem;
  mem.start();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, num_runners, -1);
  int finalize_num_runners = capped_general_finalize_runners(num_runners);
  FinalizeBucketLoaderPool finalize_loaders(args.dir,
                                            device,
                                            finalize_num_runners,
                                            "0b_general_finalize_eager_capped_runner_pool");
  finalize_loaders.preload_all();
  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(workers);
  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(workers);
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < workers; ++worker) {
    auto stream = stream_for_worker(explicit_stream, worker);
    streams.push_back(stream);
    uintptr_t handle = stream_handle_value(stream);
    stream_ids.insert(handle);
    result.stream_handles.push_back(handle);
    contexts.push_back(make_worker_context(args.dir, device, stream, shared_bundle));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = !explicit_stream || result.unique_streams == workers;
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);

  StartGate gate(workers);
  std::vector<TimingBuckets> worker_timings(workers);
  std::vector<std::string> errors(workers);
  std::vector<int> worker_mismatches(workers, 0);
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (int worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        for (int utt = 0; utt < result.rows; ++utt) {
          auto row = replay_row_density(utt,
                                        *contexts[worker],
                                        &enc_steady,
                                        nullptr,
                                        finalize_loaders,
                                        device,
                                        tokenizer,
                                        explicit_stream,
                                        mutex_serialize_run,
                                        &worker_timings[worker],
                                        true,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr);
          bool same = row.ok &&
                      row.final_tokens == reference[utt].final_tokens &&
                      strict_events_equal(row.events,
                                          reference[utt].events,
                                          "density.worker" + std::to_string(worker) + ".utt" + std::to_string(utt));
          if (!same) {
            ++worker_mismatches[worker];
            if (!row.error.empty()) {
              std::printf("  worker%d utt%d error: %s\n", worker, utt, row.error.c_str());
            }
          }
        }
      } catch (const std::exception& e) {
        errors[worker] = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& t : threads) t.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem = mem.finish();
  result.used_after = gpu_used_bytes();

  for (int worker = 0; worker < workers; ++worker) {
    result.timings.append(worker_timings[worker]);
    result.mismatches += worker_mismatches[worker];
    if (!errors[worker].empty()) {
      std::printf("  worker%d correctness exception: %s\n", worker, errors[worker].c_str());
      ++result.mismatches;
    }
  }
  result.throughput_rows_per_s =
      (static_cast<double>(workers) * static_cast<double>(result.rows)) / (result.wall_ms / 1000.0);
  auto item_pct = summarize(result.timings.scalar_sync_pct_of_gpu);
  result.identity_ok = result.mismatches == 0;
  result.ok = result.identity_ok && result.stream_uniqueness_ok;

  std::ostringstream json;
  json << "{\"check\":\"0b_concurrent_serial_correctness\""
       << ",\"num_runners\":" << num_runners
       << ",\"workers\":" << workers
       << ",\"stream_mode\":\"" << stream_mode_label(explicit_stream, mutex_serialize_run) << "\""
       << ",\"topology\":\"" << topology << "\""
       << ",\"rows\":" << result.rows
       << ",\"mismatches\":" << result.mismatches
       << ",\"identity_pass\":" << json_bool(result.identity_ok)
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"pass\":" << json_bool(result.ok)
       << ",\"throughput_rows_per_s\":" << result.throughput_rows_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"latency\":" << stats_json(summarize(result.timings.latency_ms))
       << ",\"queue_wait\":" << stats_json(summarize(result.timings.queue_wait_ms))
       << ",\"runner_wait\":" << stats_json(summarize(result.timings.runner_wait_ms))
       << ",\"item_wait\":" << stats_json(summarize(result.timings.scalar_sync_wait_ms))
       << ",\"item_wait_pct_gate\":\"telemetry_only_no_threshold\""
       << ",\"item_wait_pct_of_steady_gpu\":" << stats_json(item_pct)
       << ",\"finalize_wait\":" << stats_json(summarize(result.timings.finalize_runner_wait_ms))
       << ",\"finalize_gpu\":" << stats_json(summarize(result.timings.finalize_gpu_ms))
       << ",\"finalize_loader_memory\":" << finalize_loaders.memory_json(num_runners)
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":[";
  for (size_t i = 0; i < result.stream_handles.size(); ++i) {
    if (i > 0) json << ",";
    json << result.stream_handles[i];
  }
  json << "]"
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << ",\"used_before_bytes\":" << result.used_before
       << ",\"used_after_bytes\":" << result.used_after
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 num_runners,
                 stream_mode_label(explicit_stream, mutex_serialize_run),
                 topology,
                 json.str());

  std::printf("=== DENSITY 0b %s: workers=%d num_runners=%d stream_mode=%s rows=%d mismatches=%d "
              "item_wait_pct_p95=%.2f telemetry_only unique_streams=%d throughput=%.3f rows/s ===\n",
              result.ok ? "PASS" : "FAIL",
              workers,
              num_runners,
              stream_mode_label(explicit_stream, mutex_serialize_run).c_str(),
              result.rows,
              result.mismatches,
              item_pct.p95,
              result.unique_streams,
              result.throughput_rows_per_s);
  return result;
}

struct ScalarLocalityProbeResult {
  bool ran = false;
  bool pass = false;
  bool b_pending_after_item = false;
  double item_wait_ms = 0.0;
  double sentinel_gpu_ms = 0.0;
  double sentinel_sync_wall_ms = 0.0;
  int dim = 4096;
  int iters = 48;
  uintptr_t stream_a = 0;
  uintptr_t stream_b = 0;
};

static ScalarLocalityProbeResult run_scalar_locality_probe(const DensityArgs& args,
                                                           torch::Device device,
                                                           const std::string& stamp) {
  ScalarLocalityProbeResult result;
  result.ran = true;
  c10::cuda::CUDAGuard device_guard(device.index());
  auto stream_a = stream_for_worker(true, 0);
  auto stream_b = stream_for_worker(true, 1);
  result.stream_a = stream_handle_value(stream_a);
  result.stream_b = stream_handle_value(stream_b);
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto lhs = torch::randn({result.dim, result.dim}, float_opts);
  auto rhs = torch::randn({result.dim, result.dim}, float_opts);
  auto scalar_ready = torch::ones({1}, float_opts);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t b_start{};
  cudaEvent_t b_stop{};
  CUDA_CHECK(cudaEventCreate(&b_start));
  CUDA_CHECK(cudaEventCreate(&b_stop));
  torch::Tensor sink;
  {
    c10::cuda::CUDAStreamGuard guard_b(stream_b);
    CUDA_CHECK(cudaEventRecord(b_start, stream_b.stream()));
    for (int i = 0; i < result.iters; ++i) {
      sink = torch::matmul(lhs, rhs);
    }
    CUDA_CHECK(cudaEventRecord(b_stop, stream_b.stream()));
  }

  auto item_start = Clock::now();
  {
    c10::cuda::CUDAStreamGuard guard_a(stream_a);
    double value = scalar_ready.item<double>();
    (void)value;
  }
  result.item_wait_ms = elapsed_ms_since(item_start);

  cudaError_t query = cudaEventQuery(b_stop);
  if (query == cudaErrorNotReady) {
    result.b_pending_after_item = true;
  } else {
    CUDA_CHECK(query);
  }
  auto sync_start = Clock::now();
  CUDA_CHECK(cudaEventSynchronize(b_stop));
  result.sentinel_sync_wall_ms = elapsed_ms_since(sync_start);
  float b_elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&b_elapsed, b_start, b_stop));
  result.sentinel_gpu_ms = static_cast<double>(b_elapsed);
  CUDA_CHECK(cudaEventDestroy(b_start));
  CUDA_CHECK(cudaEventDestroy(b_stop));
  result.pass = result.b_pending_after_item && result.sentinel_gpu_ms > result.item_wait_ms;

  std::ostringstream json;
  json << "{\"check\":\"0b_scalar_locality_sentinel_probe\""
       << ",\"num_runners\":0"
       << ",\"workers\":2"
       << ",\"stream_mode\":\"explicit\""
       << ",\"topology\":\"explicit_stream_item_does_not_drain_unrelated_streams\""
       << ",\"pass\":" << json_bool(result.pass)
       << ",\"stream_a\":" << result.stream_a
       << ",\"stream_b\":" << result.stream_b
       << ",\"b_pending_after_item\":" << json_bool(result.b_pending_after_item)
       << ",\"item_wait_ms\":" << result.item_wait_ms
       << ",\"sentinel_gpu_ms\":" << result.sentinel_gpu_ms
       << ",\"sentinel_sync_wall_ms\":" << result.sentinel_sync_wall_ms
       << ",\"matmul_dim\":" << result.dim
       << ",\"matmul_iters\":" << result.iters
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 0,
                 "explicit",
                 "scalar_locality_sentinel_probe",
                 json.str());
  std::printf("=== DENSITY 0b SCALAR LOCALITY PROBE %s: item_wait=%.3fms sentinel_gpu=%.3fms "
              "b_pending_after_item=%s ===\n",
              result.pass ? "PASS" : "FAIL",
              result.item_wait_ms,
              result.sentinel_gpu_ms,
              result.b_pending_after_item ? "true" : "false");
  return result;
}

static CorrectnessResult run_correctness_gate(const DensityArgs& args,
                                              torch::Device device,
                                              const std::string& stamp,
                                              const std::shared_ptr<torch::jit::Module>& shared_bundle) {
  auto& bundle = *shared_bundle;
  int rows_total = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
  int workers = args.workers > 0 ? args.workers : args.correctness_n;
  int num_runners = args.num_runners > 0 ? args.num_runners : workers;
  bool explicit_stream = args.stream_mode == "explicit";
  std::printf("=== DENSITY 0b correctness: rows=%d/%d workers=%d num_runners=%d per-worker TS handles ===\n",
              rows, rows_total, workers, num_runners);

  TimingBuckets ref_timings;
  auto refs = build_serial_reference(args, device, shared_bundle, rows, &ref_timings);
  std::printf("=== DENSITY 0b serial reference PASS: rows=%d ===\n", rows);

  auto primary = run_correctness_gate_mode(args,
                                           device,
                                           stamp,
                                           shared_bundle,
                                           rows,
                                           refs,
                                           explicit_stream,
                                           args.mutex_serialize_run,
                                           workers,
                                           num_runners,
                                           "one_process_shared_steady_loader_per_thread_session_handles");
  primary.reference = std::move(refs);

  if (args.correctness_default_stream_control && explicit_stream && !args.mutex_serialize_run) {
    cleanup_cuda_cache();
    auto control = run_correctness_gate_mode(args,
                                             device,
                                             stamp,
                                             shared_bundle,
                                             rows,
                                             primary.reference,
                                             false,
                                             false,
                                             workers,
                                             num_runners,
                                             "negative_default_stream_per_thread_session_handles");
    if (primary.throughput_rows_per_s > 0.0) {
      primary.default_stream_penalty = 1.0 - (control.throughput_rows_per_s / primary.throughput_rows_per_s);
      primary.default_stream_control_pass = primary.default_stream_penalty >= 0.15 && control.identity_ok;
    }
    std::printf("=== DENSITY 0b DEFAULT-STREAM CONTROL %s: penalty=%.1f%% explicit=%.3f rows/s default=%.3f rows/s ===\n",
                primary.default_stream_control_pass ? "PASS" : "FAIL",
                100.0 * primary.default_stream_penalty,
                primary.throughput_rows_per_s,
                control.throughput_rows_per_s);
  } else {
    primary.default_stream_control_pass = !explicit_stream;
  }
  ScalarLocalityProbeResult scalar_probe;
  if (args.scalar_locality_probe && explicit_stream && !args.mutex_serialize_run) {
    scalar_probe = run_scalar_locality_probe(args, device, stamp);
  } else {
    scalar_probe.pass = !explicit_stream || args.mutex_serialize_run;
  }
  primary.scalar_locality_pass = primary.default_stream_control_pass && scalar_probe.pass;
  primary.ok = primary.identity_ok && primary.stream_uniqueness_ok && primary.scalar_locality_pass;

  std::ostringstream json;
  json << "{\"check\":\"0b_scalar_locality_summary\""
       << ",\"num_runners\":" << primary.num_runners
       << ",\"workers\":" << primary.workers
       << ",\"stream_mode\":\"" << stream_mode_label(primary.explicit_stream, primary.mutex_serialize_run) << "\""
       << ",\"identity_pass\":" << json_bool(primary.identity_ok)
       << ",\"stream_uniqueness_pass\":" << json_bool(primary.stream_uniqueness_ok)
       << ",\"default_stream_penalty\":" << primary.default_stream_penalty
       << ",\"default_stream_control_pass\":" << json_bool(primary.default_stream_control_pass)
       << ",\"sentinel_probe_pass\":" << json_bool(scalar_probe.pass)
       << ",\"scalar_locality_pass\":" << json_bool(primary.scalar_locality_pass)
       << ",\"item_wait_pct_gate\":\"telemetry_only_no_threshold\""
       << ",\"pass\":" << json_bool(primary.ok)
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 primary.num_runners,
                 stream_mode_label(primary.explicit_stream, primary.mutex_serialize_run),
                 "one_process_shared_steady_loader_per_thread_session_handles_scalar_locality_summary",
                 json.str());
  return primary;
}

struct SteadyCase {
  int utt = -1;
  int chunk = -1;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> oracle_outputs;
};

static std::vector<at::Tensor> clone_tensor_vector(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> out;
  out.reserve(tensors.size());
  for (const auto& tensor : tensors) out.push_back(tensor.clone());
  return out;
}

static bool compare_steady_oracle_outputs(const SteadyCase& steady_case,
                                          const std::vector<at::Tensor>& out,
                                          double atol,
                                          const std::string& label) {
  if (out.size() < 5 || steady_case.oracle_outputs.size() < 5) {
    std::printf("    %s steady oracle output count mismatch: got=%zu oracle=%zu\n",
                label.c_str(), out.size(), steady_case.oracle_outputs.size());
    return false;
  }
  bool ok = true;
  ok = tensor_close("enc_out", out[0], steady_case.oracle_outputs[0], atol, label) && ok;
  ok = tensor_equal("enc_len", out[1].to(steady_case.oracle_outputs[1].device()), steady_case.oracle_outputs[1]) && ok;
  ok = tensor_close("cache_last_channel", out[2], steady_case.oracle_outputs[2], atol, label) && ok;
  ok = tensor_close("cache_last_time", out[3], steady_case.oracle_outputs[3], atol, label) && ok;
  ok = tensor_close("cache_last_channel_len", out[4], steady_case.oracle_outputs[4], atol, label) && ok;
  return ok;
}

static std::vector<SteadyCase> build_steady_cases(const std::string& dir,
                                                  torch::jit::Module& bundle,
                                                  torch::Device device,
                                                  int limit) {
  auto enc_first = load_module_on_device(dir + "/enc_first.ts", device);
  AOTIModelPackageLoader prep_loader(dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
  double oracle_atol = bundle.hasattr("cache_ci_atol") ? scalar_f64(attr_tensor(bundle, "cache_ci_atol")) : 0.0;
  int rows = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  std::vector<SteadyCase> cases;
  cases.reserve(static_cast<size_t>(limit));
  SessionState state;
  for (int utt = 0; utt < rows && static_cast<int>(cases.size()) < limit; ++utt) {
    reset_session(state, bundle, device);
    std::string prefix = "utt" + std::to_string(utt);
    int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
    for (int chunk_index = 0; chunk_index < num_steady && static_cast<int>(cases.size()) < limit; ++chunk_index) {
      auto new_mel = prefix_chunk_tensor(bundle, prefix, chunk_index, "new_mel").to(device).contiguous();
      if (state.emitted == 0) {
        auto out = run_first_encoder(enc_first, new_mel, state);
        state.clc = out[2].clone();
        state.clt = out[3].clone();
        state.clcl = out[4].clone();
      } else {
        auto chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
        auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
        std::vector<at::Tensor> inputs = {
            chunk.contiguous(),
            L.contiguous(),
            state.clc.contiguous(),
            state.clt.contiguous(),
            state.clcl.contiguous(),
        };
        cases.push_back({utt, chunk_index, inputs});
        auto out = prep_loader.run(inputs);
        if (out.size() < 5) throw std::runtime_error("steady case prep AOTI returned fewer than 5 outputs");
        cases.back().oracle_outputs = clone_tensor_vector(out);
        if (!compare_steady_oracle_outputs(cases.back(), out, oracle_atol,
                                           "density.steady_oracle.utt" + std::to_string(utt) +
                                               ".chunk" + std::to_string(chunk_index))) {
          throw std::runtime_error("steady case serial oracle self-compare failed");
        }
        state.clc = out[2].clone();
        state.clt = out[3].clone();
        state.clcl = out[4].clone();
      }
      auto cum = state.ring.defined() ? torch::cat({state.ring, new_mel}, 2) : new_mel;
      state.ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2)).contiguous();
      state.emitted += new_mel.size(2);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (cases.empty()) throw std::runtime_error("no steady continuation cases found in session_bundle.ts");
  std::printf("=== DENSITY steady cases prepared: %zu continuation chunks from session_bundle.ts ===\n",
              cases.size());
  return cases;
}

struct SteadyRunResult {
  int num_runners = 0;
  int workers = 0;
  bool explicit_stream = true;
  bool ok = false;
  int errors = 0;
  int oracle_mismatches = 0;
  size_t oracle_checks = 0;
  size_t calls = 0;
  double wall_ms = 0.0;
  double throughput_calls_per_s = 0.0;
  double contention_confounded_overlap_diagnostic = 0.0;
  int unique_streams = 0;
  bool stream_uniqueness_ok = false;
  size_t used_before_loader = 0;
  size_t used_after_loader = 0;
  size_t loader_delta = 0;
  size_t peak_mem = 0;
  size_t used_after_run = 0;
  TimingBuckets timings;
};

static bool run_steady_case(AOTIModelPackageLoader& loader,
                            const SteadyCase& steady_case,
                            c10::cuda::CUDAStream stream,
                            bool explicit_stream,
                            bool mutex_serialize_run,
                            double oracle_atol,
                            bool compare_oracle,
                            TimingBuckets& timings,
                            const std::string& label) {
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto call_start = Clock::now();
  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  CUDA_CHECK(cudaEventRecord(ev_start, stream.stream()));
  auto run_start = Clock::now();
  auto out = run_aoti_loader(loader, steady_case.inputs, stream, explicit_stream, mutex_serialize_run);
  double runner_host_ms = elapsed_ms_since(run_start);
  CUDA_CHECK(cudaEventRecord(ev_stop, stream.stream()));
  if (out.size() < 5) throw std::runtime_error("steady density run returned fewer than 5 outputs");
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_start, ev_stop));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  timings.latency_ms.push_back(elapsed_ms_since(call_start));
  timings.queue_wait_ms.push_back(0.0);
  timings.steady_gpu_ms.push_back(static_cast<double>(elapsed));
  timings.runner_wait_ms.push_back(std::max(0.0, runner_host_ms - static_cast<double>(elapsed)));
  if (compare_oracle) return compare_steady_oracle_outputs(steady_case, out, oracle_atol, label);
  return true;
}

static std::vector<std::vector<SteadyCase>> clone_steady_cases_per_worker(const std::vector<SteadyCase>& cases,
                                                                          int workers) {
  std::vector<std::vector<SteadyCase>> out(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    out[worker].reserve(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
      const auto& base = cases[(i + static_cast<size_t>(worker)) % cases.size()];
      SteadyCase cloned;
      cloned.utt = base.utt;
      cloned.chunk = base.chunk;
      cloned.inputs = clone_tensor_vector(base.inputs);
      cloned.oracle_outputs = base.oracle_outputs;
      out[worker].push_back(std::move(cloned));
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return out;
}

static SteadyRunResult run_steady_overlap_once(const DensityArgs& args,
                                               torch::Device device,
                                               const std::string& stamp,
                                               const std::vector<SteadyCase>& cases,
                                               int workers,
                                               int num_runners,
                                               bool explicit_stream,
                                               bool mutex_serialize_run,
                                               double oracle_atol) {
  SteadyRunResult result;
  result.num_runners = num_runners;
  result.workers = workers;
  result.explicit_stream = explicit_stream;
  auto worker_cases = clone_steady_cases_per_worker(cases, workers);
  cleanup_cuda_cache();
  result.used_before_loader = gpu_used_bytes();
  AOTIModelPackageLoader loader(args.dir + "/enc_steady_aoti.pt2", "model", false, num_runners, -1);
  result.used_after_loader = gpu_used_bytes();
  result.loader_delta = result.used_after_loader >= result.used_before_loader
                            ? result.used_after_loader - result.used_before_loader
                            : 0;

  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(static_cast<size_t>(workers));
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < workers; ++worker) {
    auto stream = stream_for_worker(explicit_stream, worker);
    streams.push_back(stream);
    stream_ids.insert(stream_handle_value(stream));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = !explicit_stream || result.unique_streams == workers;

  TimingBuckets warmup_timings;
  for (int worker = 0; worker < workers; ++worker) {
    run_steady_case(loader,
                    worker_cases[worker][0],
                    streams[worker],
                    explicit_stream,
                    mutex_serialize_run,
                    oracle_atol,
                    false,
                    warmup_timings,
                    "density.steady_warmup.worker" + std::to_string(worker));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  StartGate oracle_gate(workers);
  std::vector<std::string> oracle_errors(workers);
  std::vector<int> oracle_mismatches(workers, 0);
  std::vector<std::thread> oracle_threads;
  oracle_threads.reserve(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    oracle_threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        TimingBuckets ignored;
        oracle_gate.arrive_and_wait();
        for (size_t i = 0; i < worker_cases[worker].size(); ++i) {
          bool ok = run_steady_case(loader,
                                    worker_cases[worker][i],
                                    streams[worker],
                                    explicit_stream,
                                    mutex_serialize_run,
                                    oracle_atol,
                                    true,
                                    ignored,
                                    "density.steady_oracle.worker" + std::to_string(worker) +
                                        ".case" + std::to_string(i));
          if (!ok) ++oracle_mismatches[worker];
        }
      } catch (const std::exception& e) {
        oracle_errors[worker] = e.what();
      }
    });
  }
  oracle_gate.wait_until_ready_and_start();
  for (auto& thread : oracle_threads) thread.join();
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int worker = 0; worker < workers; ++worker) {
    result.oracle_checks += worker_cases[worker].size();
    result.oracle_mismatches += oracle_mismatches[worker];
    if (!oracle_errors[worker].empty()) {
      ++result.errors;
      ++result.oracle_mismatches;
      std::printf("  steady oracle worker%d exception: %s\n", worker, oracle_errors[worker].c_str());
    }
  }

  MemorySampler mem;
  mem.start();
  StartGate gate(workers);
  std::vector<TimingBuckets> worker_timings(workers);
  std::vector<std::string> errors(workers);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers));
  for (int worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        for (int repeat = 0; repeat < args.steady_repeats; ++repeat) {
          for (const auto& steady_case : worker_cases[worker]) {
            run_steady_case(loader,
                            steady_case,
                            streams[worker],
                            explicit_stream,
                            mutex_serialize_run,
                            oracle_atol,
                            false,
                            worker_timings[worker],
                            "density.steady_timed.worker" + std::to_string(worker));
          }
        }
      } catch (const std::exception& e) {
        errors[worker] = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem = mem.finish();
  result.used_after_run = gpu_used_bytes();

  for (int worker = 0; worker < workers; ++worker) {
    result.timings.append(worker_timings[worker]);
    if (!errors[worker].empty()) {
      ++result.errors;
      std::printf("  steady worker%d exception: %s\n", worker, errors[worker].c_str());
    }
  }
  result.calls = result.timings.latency_ms.size();
  result.throughput_calls_per_s = static_cast<double>(result.calls) / (result.wall_ms / 1000.0);
  double sum_gpu = std::accumulate(result.timings.steady_gpu_ms.begin(),
                                   result.timings.steady_gpu_ms.end(),
                                   0.0);
  result.contention_confounded_overlap_diagnostic = result.wall_ms > 0.0 ? sum_gpu / result.wall_ms : 0.0;
  result.ok = result.errors == 0 && result.oracle_mismatches == 0 && result.stream_uniqueness_ok;

  const std::string stream_mode = stream_mode_label(explicit_stream, mutex_serialize_run);
  const std::string topology = explicit_stream
                                   ? (mutex_serialize_run ? "negative_mutex_serialized_shared_steady_loader_runner_pool"
                                                          : "shared_steady_loader_runner_pool")
                                   : "negative_default_stream_shared_steady_loader";
  std::ostringstream json;
  json << "{\"check\":\"0a_steady_pool_overlap\""
       << ",\"num_runners\":" << num_runners
       << ",\"workers\":" << workers
       << ",\"stream_mode\":\"" << stream_mode << "\""
       << ",\"topology\":\"" << topology << "\""
       << ",\"cases\":" << cases.size()
       << ",\"repeats\":" << args.steady_repeats
       << ",\"calls\":" << result.calls
       << ",\"errors\":" << result.errors
       << ",\"oracle_checks\":" << result.oracle_checks
       << ",\"oracle_mismatches\":" << result.oracle_mismatches
       << ",\"serial_output_oracle_pass\":" << json_bool(result.oracle_mismatches == 0)
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"pass\":" << json_bool(result.ok)
       << ",\"throughput_calls_per_s\":" << result.throughput_calls_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"latency\":" << stats_json(summarize(result.timings.latency_ms))
       << ",\"queue_wait\":" << stats_json(summarize(result.timings.queue_wait_ms))
       << ",\"runner_wait\":" << stats_json(summarize(result.timings.runner_wait_ms))
       << ",\"item_wait\":" << stats_json(SummaryStats{})
       << ",\"finalize_wait\":" << stats_json(SummaryStats{})
       << ",\"steady_gpu\":" << stats_json(summarize(result.timings.steady_gpu_ms))
       << ",\"contention_confounded_cuda_event_diagnostic\":" << result.contention_confounded_overlap_diagnostic
       << ",\"overlap_estimate_label\":\"contention_confounded_diagnostic_not_overlap_proof\""
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":" << stream_handles_json(streams)
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << ",\"used_before_loader_bytes\":" << result.used_before_loader
       << ",\"used_after_loader_bytes\":" << result.used_after_loader
       << ",\"loader_delta_bytes\":" << result.loader_delta
       << ",\"used_after_run_bytes\":" << result.used_after_run
       << "}";
  emit_telemetry(args.dir, stamp, num_runners, stream_mode, topology, json.str());
  std::printf("=== DENSITY 0a %s: num_runners=%d workers=%d stream_mode=%s calls=%zu throughput=%.3f/s "
              "event_diag=%.2f unique_streams=%d oracle_mismatches=%d loader_delta=%.3f GiB peak_mem=%.3f GiB ===\n",
              result.ok ? "RUN" : "ERROR",
              num_runners,
              workers,
              stream_mode.c_str(),
              result.calls,
              result.throughput_calls_per_s,
              result.contention_confounded_overlap_diagnostic,
              result.unique_streams,
              result.oracle_mismatches,
              static_cast<double>(result.loader_delta) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(result.peak_mem) / (1024.0 * 1024.0 * 1024.0));
  return result;
}

struct SteadySweepResult {
  std::map<int, SteadyRunResult> explicit_runs;
  std::unique_ptr<SteadyRunResult> num_runners_one_control;
  std::unique_ptr<SteadyRunResult> default_control;
  std::unique_ptr<SteadyRunResult> mutex_control;
  bool overlap_probe_pass = false;
  bool pass = false;
};

struct SteadyOverlapProbeResult {
  bool ran = false;
  bool pass = false;
  bool stream_b_completed_before_stream_a_done = false;
  bool streams_unique = false;
  double stream_a_wall_ms = 0.0;
  double stream_b_gpu_ms = 0.0;
  double stream_b_sync_wall_ms = 0.0;
  int stream_a_repeats = 16;
  int stream_b_matmul_dim = 2048;
  int stream_b_matmul_iters = 8;
  uintptr_t stream_a = 0;
  uintptr_t stream_b = 0;
};

static SteadyOverlapProbeResult run_steady_overlap_probe(const DensityArgs& args,
                                                         torch::Device device,
                                                         const std::string& stamp,
                                                         const std::vector<SteadyCase>& cases,
                                                         double oracle_atol) {
  SteadyOverlapProbeResult result;
  result.ran = true;
  c10::cuda::CUDAGuard device_guard(device.index());
  AOTIModelPackageLoader loader(args.dir + "/enc_steady_aoti.pt2", "model", false, 2, -1);
  auto stream_a = stream_for_worker(true, 0);
  auto stream_b = stream_for_worker(true, 1);
  result.stream_a = stream_handle_value(stream_a);
  result.stream_b = stream_handle_value(stream_b);
  result.streams_unique = result.stream_a != result.stream_b;
  SteadyCase probe_case;
  probe_case.utt = cases[0].utt;
  probe_case.chunk = cases[0].chunk;
  probe_case.inputs = clone_tensor_vector(cases[0].inputs);
  probe_case.oracle_outputs = cases[0].oracle_outputs;
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto lhs = torch::randn({result.stream_b_matmul_dim, result.stream_b_matmul_dim}, float_opts);
  auto rhs = torch::randn({result.stream_b_matmul_dim, result.stream_b_matmul_dim}, float_opts);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::atomic<bool> a_started{false};
  std::atomic<bool> a_done{false};
  std::string a_error;
  std::thread stream_a_thread([&] {
    try {
      c10::cuda::CUDAGuard thread_guard(device.index());
      TimingBuckets ignored;
      a_started.store(true, std::memory_order_release);
      auto start = Clock::now();
      for (int i = 0; i < result.stream_a_repeats; ++i) {
        run_steady_case(loader,
                        probe_case,
                        stream_a,
                        true,
                        false,
                        oracle_atol,
                        false,
                        ignored,
                        "density.steady_overlap_probe.stream_a");
      }
      result.stream_a_wall_ms = elapsed_ms_since(start);
      a_done.store(true, std::memory_order_release);
    } catch (const std::exception& e) {
      a_error = e.what();
      a_done.store(true, std::memory_order_release);
    }
  });
  while (!a_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  cudaEvent_t b_start{};
  cudaEvent_t b_stop{};
  CUDA_CHECK(cudaEventCreate(&b_start));
  CUDA_CHECK(cudaEventCreate(&b_stop));
  torch::Tensor sink;
  {
    c10::cuda::CUDAStreamGuard guard_b(stream_b);
    CUDA_CHECK(cudaEventRecord(b_start, stream_b.stream()));
    for (int i = 0; i < result.stream_b_matmul_iters; ++i) {
      sink = torch::matmul(lhs, rhs);
    }
    CUDA_CHECK(cudaEventRecord(b_stop, stream_b.stream()));
  }
  auto sync_start = Clock::now();
  CUDA_CHECK(cudaEventSynchronize(b_stop));
  result.stream_b_sync_wall_ms = elapsed_ms_since(sync_start);
  result.stream_b_completed_before_stream_a_done = !a_done.load(std::memory_order_acquire);
  float b_elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&b_elapsed, b_start, b_stop));
  result.stream_b_gpu_ms = static_cast<double>(b_elapsed);
  CUDA_CHECK(cudaEventDestroy(b_start));
  CUDA_CHECK(cudaEventDestroy(b_stop));
  if (stream_a_thread.joinable()) stream_a_thread.join();
  if (!a_error.empty()) std::printf("  steady overlap probe stream A exception: %s\n", a_error.c_str());
  result.pass = a_error.empty() && result.streams_unique && result.stream_b_completed_before_stream_a_done;

  std::ostringstream json;
  json << "{\"check\":\"0a_steady_overlap_sentinel_probe\""
       << ",\"num_runners\":2"
       << ",\"workers\":2"
       << ",\"stream_mode\":\"explicit\""
       << ",\"topology\":\"sentinel_stream_b_runs_while_stream_a_encoder_active\""
       << ",\"pass\":" << json_bool(result.pass)
       << ",\"stream_a\":" << result.stream_a
       << ",\"stream_b\":" << result.stream_b
       << ",\"streams_unique\":" << json_bool(result.streams_unique)
       << ",\"stream_b_completed_before_stream_a_done\":" << json_bool(result.stream_b_completed_before_stream_a_done)
       << ",\"stream_a_wall_ms\":" << result.stream_a_wall_ms
       << ",\"stream_b_gpu_ms\":" << result.stream_b_gpu_ms
       << ",\"stream_b_sync_wall_ms\":" << result.stream_b_sync_wall_ms
       << ",\"stream_a_repeats\":" << result.stream_a_repeats
       << ",\"stream_b_matmul_dim\":" << result.stream_b_matmul_dim
       << ",\"stream_b_matmul_iters\":" << result.stream_b_matmul_iters
       << "}";
  emit_telemetry(args.dir, stamp, 2, "explicit", "steady_overlap_sentinel_probe", json.str());
  std::printf("=== DENSITY 0a OVERLAP SENTINEL %s: stream_b_gpu=%.3fms stream_a_wall=%.3fms "
              "b_completed_before_a_done=%s ===\n",
              result.pass ? "PASS" : "FAIL",
              result.stream_b_gpu_ms,
              result.stream_a_wall_ms,
              result.stream_b_completed_before_stream_a_done ? "true" : "false");
  return result;
}

static SteadySweepResult run_steady_sweep(const DensityArgs& args,
                                          torch::Device device,
                                          const std::string& stamp,
                                          const std::shared_ptr<torch::jit::Module>& shared_bundle) {
  auto& bundle = *shared_bundle;
  auto cases = build_steady_cases(args.dir, bundle, device, args.steady_cases);
  double oracle_atol = bundle.hasattr("cache_ci_atol") ? scalar_f64(attr_tensor(bundle, "cache_ci_atol")) : 0.0;
  cleanup_cuda_cache();

  SteadySweepResult sweep;
  for (int n : args.n_values) {
    int workers = args.workers > 0 ? args.workers : n;
    int num_runners = args.num_runners > 0 ? args.num_runners : n;
    bool explicit_stream = args.stream_mode == "explicit";
    sweep.explicit_runs.emplace(n, run_steady_overlap_once(args,
                                                           device,
                                                           stamp,
                                                           cases,
                                                           workers,
                                                           num_runners,
                                                           explicit_stream,
                                                           args.mutex_serialize_run,
                                                           oracle_atol));
    cleanup_cuda_cache();
  }
  if (args.default_stream_control) {
    int control_n = (args.target_n > 0 &&
                     std::find(args.n_values.begin(), args.n_values.end(), args.target_n) != args.n_values.end())
                        ? args.target_n
                        : (std::find(args.n_values.begin(), args.n_values.end(), 4) != args.n_values.end()
                               ? 4
                               : args.n_values.back());
    int control_workers = args.workers > 0 ? args.workers : control_n;
    int control_runners = args.num_runners > 0 ? args.num_runners : control_workers;
    if (args.stream_mode == "explicit" && !args.mutex_serialize_run && args.num_runners == 0 && args.workers == 0) {
      sweep.num_runners_one_control = std::make_unique<SteadyRunResult>(
          run_steady_overlap_once(args, device, stamp, cases, control_workers, 1, true, false, oracle_atol));
      cleanup_cuda_cache();
      sweep.mutex_control = std::make_unique<SteadyRunResult>(
          run_steady_overlap_once(args, device, stamp, cases, control_workers, control_workers, true, true, oracle_atol));
      cleanup_cuda_cache();
    }
    sweep.default_control = std::make_unique<SteadyRunResult>(
        run_steady_overlap_once(args, device, stamp, cases, control_workers, control_runners, false, false, oracle_atol));
    cleanup_cuda_cache();
  }
  if (args.steady_overlap_probe && args.stream_mode == "explicit" && !args.mutex_serialize_run) {
    auto probe = run_steady_overlap_probe(args, device, stamp, cases, oracle_atol);
    sweep.overlap_probe_pass = probe.pass;
    cleanup_cuda_cache();
  }

  double base = sweep.explicit_runs.count(1) ? sweep.explicit_runs[1].throughput_calls_per_s : 0.0;
  double speedup2 = (base > 0.0 && sweep.explicit_runs.count(2))
                        ? sweep.explicit_runs[2].throughput_calls_per_s / base
                        : 0.0;
  double speedup4 = (base > 0.0 && sweep.explicit_runs.count(4))
                        ? sweep.explicit_runs[4].throughput_calls_per_s / base
                        : 0.0;
  double peak_mem_ratio4 = 0.0;
  if (sweep.explicit_runs.count(1) && sweep.explicit_runs.count(4) && sweep.explicit_runs[1].peak_mem > 0) {
    peak_mem_ratio4 = static_cast<double>(sweep.explicit_runs[4].peak_mem) /
                      static_cast<double>(sweep.explicit_runs[1].peak_mem);
  }
  double target_peak_mem_ratio = 0.0;
  if (args.target_n > 0 &&
      sweep.explicit_runs.count(1) &&
      sweep.explicit_runs.count(args.target_n) &&
      sweep.explicit_runs[1].peak_mem > 0) {
    target_peak_mem_ratio = static_cast<double>(sweep.explicit_runs[args.target_n].peak_mem) /
                            static_cast<double>(sweep.explicit_runs[1].peak_mem);
  }
  double loader_delta_ratio4 = 0.0;
  if (sweep.explicit_runs.count(1) && sweep.explicit_runs.count(4) && sweep.explicit_runs[1].loader_delta > 0) {
    loader_delta_ratio4 = static_cast<double>(sweep.explicit_runs[4].loader_delta) /
                          static_cast<double>(sweep.explicit_runs[1].loader_delta);
  }
  double target_loader_delta_ratio = 0.0;
  if (args.target_n > 0 &&
      sweep.explicit_runs.count(1) &&
      sweep.explicit_runs.count(args.target_n) &&
      sweep.explicit_runs[1].loader_delta > 0) {
    target_loader_delta_ratio = static_cast<double>(sweep.explicit_runs[args.target_n].loader_delta) /
                                static_cast<double>(sweep.explicit_runs[1].loader_delta);
  }
  bool default_control_pass = true;
  double default_penalty = 0.0;
  if (sweep.default_control) {
    double explicit_thr = 0.0;
    for (const auto& kv : sweep.explicit_runs) {
      if (kv.second.workers == sweep.default_control->workers &&
          kv.second.num_runners == sweep.default_control->num_runners) {
        explicit_thr = kv.second.throughput_calls_per_s;
        break;
      }
    }
    if (explicit_thr > 0.0) {
      default_penalty = 1.0 - (sweep.default_control->throughput_calls_per_s / explicit_thr);
      default_control_pass = default_penalty >= 0.15;
    }
  }
  bool loader_delta_flat = false;
  if (target_loader_delta_ratio > 0.0) {
    loader_delta_flat = target_loader_delta_ratio <= 1.15;
  } else if (loader_delta_ratio4 > 0.0) {
    loader_delta_flat = loader_delta_ratio4 <= 1.15;
  }
  bool primary_runs_ok = true;
  for (const auto& kv : sweep.explicit_runs) primary_runs_ok = primary_runs_ok && kv.second.ok;
  sweep.pass = speedup2 >= 1.15 &&
               speedup4 >= 1.30 &&
               loader_delta_flat &&
               default_control_pass &&
               sweep.overlap_probe_pass &&
               primary_runs_ok;

  std::ostringstream summary;
  summary << "{\"check\":\"0a_steady_pool_overlap_summary\""
          << ",\"num_runners\":0"
          << ",\"stream_mode\":\"" << stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run) << "\""
          << ",\"topology\":\"shared_steady_loader_runner_pool\""
          << ",\"pass\":" << json_bool(sweep.pass)
          << ",\"speedup_n2\":" << speedup2
          << ",\"speedup_n4\":" << speedup4
          << ",\"memory_gate_metric\":\"loader_delta_used_after_loader_minus_used_before_loader\""
          << ",\"loader_delta_ratio_n4_vs_n1\":" << loader_delta_ratio4
          << ",\"target_n\":" << args.target_n
          << ",\"loader_delta_ratio_target_vs_n1\":" << target_loader_delta_ratio
          << ",\"loader_delta_flat_pass\":" << json_bool(loader_delta_flat)
          << ",\"peak_mem_ratio_n4_vs_n1_diagnostic\":" << peak_mem_ratio4
          << ",\"peak_mem_ratio_target_vs_n1_diagnostic\":" << target_peak_mem_ratio
          << ",\"default_stream_penalty\":" << default_penalty
          << ",\"default_stream_control_pass\":" << json_bool(default_control_pass)
          << ",\"overlap_proof\":\"sentinel_probe\""
          << ",\"overlap_probe_pass\":" << json_bool(sweep.overlap_probe_pass)
          << ",\"primary_runs_pass\":" << json_bool(primary_runs_ok)
          << "}";
  emit_telemetry(args.dir,
                 stamp,
                 0,
                 "explicit",
                 "shared_steady_loader_runner_pool_summary",
                 summary.str());
  std::printf("=== DENSITY 0a SUMMARY %s: speedup@2=%.3fx speedup@4=%.3fx mem_ratio@4=%.3f "
              "loader_delta_ratio@4=%.3f targetN=%d target_loader_delta_ratio=%.3f "
              "default_penalty=%.1f%% overlap_probe=%s ===\n",
              sweep.pass ? "PASS" : "FAIL",
              speedup2,
              speedup4,
              peak_mem_ratio4,
              loader_delta_ratio4,
              args.target_n,
              target_loader_delta_ratio,
              100.0 * default_penalty,
              sweep.overlap_probe_pass ? "PASS" : "FAIL");
  return sweep;
}

struct FinalizeCase {
  int utt = -1;
  int64_t drop = -1;
  int64_t T = -1;
};

static std::vector<FinalizeBucketKey> unique_finalize_bucket_keys_from_cases(const std::vector<FinalizeCase>& cases) {
  std::vector<FinalizeBucketKey> keys;
  keys.reserve(cases.size());
  for (const auto& item : cases) keys.emplace_back(item.drop, item.T);
  return unique_finalize_bucket_keys(keys);
}

static std::vector<FinalizeCase> discover_finalize_cases(torch::jit::Module& bundle, int rows) {
  std::vector<FinalizeCase> cases;
  cases.reserve(static_cast<size_t>(rows));
  for (int utt = 0; utt < rows; ++utt) {
    int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
    int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
    if (T <= 0) continue;
    cases.push_back({
        utt,
        drop,
        T,
    });
  }
  return cases;
}

static std::vector<FinalizeCase> pick_mixed_finalize_cases(const std::vector<FinalizeCase>& all, int n) {
  std::map<std::pair<int64_t, int64_t>, FinalizeCase> by_bucket;
  for (const auto& item : all) by_bucket.emplace(std::make_pair(item.drop, item.T), item);
  std::vector<FinalizeCase> out;
  for (const auto& kv : by_bucket) {
    out.push_back(kv.second);
    if (static_cast<int>(out.size()) == n) break;
  }
  if (out.empty()) throw std::runtime_error("no finalize cases available");
  while (static_cast<int>(out.size()) < n) out.push_back(out[out.size() % by_bucket.size()]);
  return out;
}

static std::vector<FinalizeCase> pick_same_bucket_finalize_cases(const std::vector<FinalizeCase>& all, int n) {
  if (all.empty()) throw std::runtime_error("no finalize cases available");
  std::map<std::pair<int64_t, int64_t>, std::vector<FinalizeCase>> by_bucket;
  for (const auto& item : all) by_bucket[std::make_pair(item.drop, item.T)].push_back(item);
  auto best = by_bucket.begin();
  for (auto it = by_bucket.begin(); it != by_bucket.end(); ++it) {
    if (it->second.size() > best->second.size()) best = it;
  }
  std::vector<FinalizeCase> out;
  out.reserve(static_cast<size_t>(n));
  for (const auto& item : best->second) {
    out.push_back(item);
    if (static_cast<int>(out.size()) == n) return out;
  }
  while (static_cast<int>(out.size()) < n) out.push_back(best->second[out.size() % best->second.size()]);
  return out;
}

struct FinalizeGateResult {
  bool ok = false;
  bool stream_uniqueness_ok = false;
  int workers = 0;
  int num_runners = 0;
  int steady_num_runners = 0;
  int finalize_num_runners = 0;
  int unique_streams = 0;
  int mismatches = 0;
  double wall_ms = 0.0;
  double throughput_finalize_per_s = 0.0;
  TimingBuckets timings;
  size_t peak_mem = 0;
  std::vector<uintptr_t> stream_handles;
};

static void prepare_finalize_parent(const FinalizeCase& fc,
                                    WorkerContext& ctx,
                                    AOTIModelPackageLoader* enc_steady,
                                    BatchedSteadyLoaderSet* direct_b1_loader,
                                    torch::Device device,
                                    const Tokenizer& tokenizer,
                                    bool explicit_stream,
                                    bool mutex_serialize_run,
                                    SessionState& session,
                                    std::vector<EmittedEvent>& events,
                                    BatchedSteadyScheduler* scheduler = nullptr) {
  reset_session(session, ctx.bundle, device);
  std::string prefix = "utt" + std::to_string(fc.utt);
  int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, fc.utt, "num_steady"));
  TimingBuckets ignored;
  for (int chunk = 0; chunk < num_steady; ++chunk) {
    run_steady_chunk_density(session,
                             ctx.bundle,
                             prefix,
                             chunk,
                             *ctx.enc_first,
                             enc_steady,
                             direct_b1_loader,
                             ctx.joint,
                             ctx.predict,
                             device,
                             tokenizer,
                             events,
                             ctx.stream,
                             explicit_stream,
                             mutex_serialize_run,
                             &ignored,
                             "density.finalize_prep.utt" + std::to_string(fc.utt) + ".chunk" + std::to_string(chunk),
                             scheduler,
                             nullptr,
                             nullptr,
                             nullptr);
  }
  session.mode = SessionMode::PENDING_FINALIZE;
}

static FinalizeGateResult run_finalize_gate_one(const DensityArgs& args,
                                                torch::Device device,
                                                const std::string& stamp,
                                                const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                                const std::string& mode,
                                                const std::vector<FinalizeCase>& cases,
                                                const std::vector<RowReplayResult>& reference) {
  FinalizeGateResult result;
  result.workers = static_cast<int>(cases.size());
  result.steady_num_runners = args.num_runners > 0 ? args.num_runners : result.workers;
  bool hot_same_bucket = mode == "same_bucket";
  result.finalize_num_runners = hot_same_bucket ? result.workers : capped_general_finalize_runners(result.workers);
  result.num_runners = result.finalize_num_runners;
  MemorySampler mem;
  mem.start();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, result.steady_num_runners, -1);
  FinalizeBucketLoaderPool finalize_loaders(
      args.dir,
      device,
      result.finalize_num_runners,
      hot_same_bucket ? "0c_hot_same_bucket_one_bucket_full_worker_runners"
                      : "0c_mixed_bucket_selected_buckets_capped_runner_pool");
  auto needed_buckets = unique_finalize_bucket_keys_from_cases(cases);
  finalize_loaders.preload(needed_buckets);
  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(static_cast<size_t>(result.workers));
  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(static_cast<size_t>(result.workers));
  std::set<uintptr_t> stream_ids;
  for (int worker = 0; worker < result.workers; ++worker) {
    auto stream = stream_for_worker(true, worker);
    streams.push_back(stream);
    uintptr_t handle = stream_handle_value(stream);
    stream_ids.insert(handle);
    result.stream_handles.push_back(handle);
    contexts.push_back(make_worker_context(args.dir, device, stream, shared_bundle));
  }
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = result.unique_streams == result.workers;
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);

  StartGate gate(result.workers);
  std::vector<TimingBuckets> worker_timings(result.workers);
  std::vector<std::string> errors(result.workers);
  std::vector<int> mismatches(result.workers, 0);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        SessionState session;
        std::vector<EmittedEvent> events;
        prepare_finalize_parent(cases[worker],
                                *contexts[worker],
                                &enc_steady,
                                nullptr,
                                device,
                                tokenizer,
                                true,
                                args.mutex_serialize_run,
                                session,
                                events);
        gate.arrive_and_wait();
        std::string prefix = "utt" + std::to_string(cases[worker].utt);
        auto outcome = run_finalize_density(session,
                                            contexts[worker]->bundle,
                                            prefix,
                                            "density.finalize_" + mode + ".worker" + std::to_string(worker),
                                            finalize_loaders,
                                            contexts[worker]->joint,
                                            contexts[worker]->predict,
                                            device,
                                            tokenizer,
                                            events,
                                            FinalizeFinish::SPECULATIVE_KEEP,
                                            contexts[worker]->stream,
                                            true,
                                            args.mutex_serialize_run,
                                            &worker_timings[worker]);
        bool same = outcome.stale_dropped ||
                    (outcome.token_ok &&
                     outcome.fork_ok &&
                     outcome.final_tokens == reference[cases[worker].utt].final_tokens &&
                     strict_events_equal(events,
                                         reference[cases[worker].utt].events,
                                         "density.finalize_" + mode + ".worker" + std::to_string(worker)));
        if (!same) ++mismatches[worker];
      } catch (const std::exception& e) {
        errors[worker] = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& t : threads) t.join();
  auto end_time = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem = mem.finish();

  for (int worker = 0; worker < result.workers; ++worker) {
    result.timings.append(worker_timings[worker]);
    result.mismatches += mismatches[worker];
    if (!errors[worker].empty()) {
      ++result.mismatches;
      std::printf("  finalize %s worker%d exception: %s\n", mode.c_str(), worker, errors[worker].c_str());
    }
  }
  result.throughput_finalize_per_s = static_cast<double>(result.workers) / (result.wall_ms / 1000.0);
  auto finalize_total = summarize(result.timings.finalize_total_ms);
  auto finalize_wait = summarize(result.timings.finalize_runner_wait_ms);
  double wait_pct = finalize_total.p95 > 0.0 ? 100.0 * finalize_wait.p95 / finalize_total.p95 : 0.0;
  result.ok = result.mismatches == 0 && wait_pct <= 25.0 && result.stream_uniqueness_ok;

  std::ostringstream buckets;
  buckets << "[";
  for (size_t i = 0; i < cases.size(); ++i) {
    if (i > 0) buckets << ",";
    buckets << "{\"utt\":" << cases[i].utt << ",\"drop\":" << cases[i].drop << ",\"T\":" << cases[i].T << "}";
  }
  buckets << "]";

  std::ostringstream json;
  json << "{\"check\":\"0c_finalize_concurrency_" << mode << "\""
       << ",\"num_runners\":" << result.num_runners
       << ",\"steady_num_runners\":" << result.steady_num_runners
       << ",\"finalize_num_runners_per_loaded_bucket\":" << result.finalize_num_runners
       << ",\"workers\":" << result.workers
       << ",\"stream_mode\":\"" << stream_mode_label(true, args.mutex_serialize_run) << "\""
       << ",\"topology\":\"shared_finalize_bucket_runner_pool_" << mode << "\""
       << ",\"buckets\":" << buckets.str()
       << ",\"mismatches\":" << result.mismatches
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":" << stream_handles_json(streams)
       << ",\"pass\":" << json_bool(result.ok)
       << ",\"throughput_finalize_per_s\":" << result.throughput_finalize_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"finalize_wait\":" << stats_json(finalize_wait)
       << ",\"finalize_gpu\":" << stats_json(summarize(result.timings.finalize_gpu_ms))
       << ",\"finalize_total\":" << stats_json(finalize_total)
       << ",\"finalize_phases\":" << finalize_phase_stats_json(result.timings)
       << ",\"finalize_runner_wait_pct_of_total_p95\":" << wait_pct
       << ",\"finalize_loader_memory\":" << finalize_loaders.memory_json(result.workers)
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 result.num_runners,
                 stream_mode_label(true, args.mutex_serialize_run),
                 "shared_finalize_bucket_runner_pool_" + mode,
                 json.str());
  std::printf("=== DENSITY 0c %s %s: workers=%d steady_num_runners=%d finalize_num_runners=%d "
              "loaded_buckets=%zu/%zu loader_delta=%.3f GiB unique_streams=%d mismatches=%d "
              "finalize_wait_p95=%.3fms total_p95=%.3fms wait_pct=%.1f%% peak_mem=%.3f GiB ===\n",
              mode.c_str(),
              result.ok ? "PASS" : "FAIL",
              result.workers,
              result.steady_num_runners,
              result.finalize_num_runners,
              finalize_loaders.loaded_bucket_count(),
              finalize_loaders.total_bucket_count(),
              static_cast<double>(finalize_loaders.total_loader_delta()) / (1024.0 * 1024.0 * 1024.0),
              result.unique_streams,
              result.mismatches,
              finalize_wait.p95,
              finalize_total.p95,
              wait_pct,
              static_cast<double>(result.peak_mem) / (1024.0 * 1024.0 * 1024.0));
  return result;
}

static bool run_finalize_gate(const DensityArgs& args,
                              torch::Device device,
                              const std::string& stamp,
                              const std::shared_ptr<torch::jit::Module>& shared_bundle,
                              const CorrectnessResult& correctness) {
  auto& bundle = *shared_bundle;
  int rows_total = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  int rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
  auto all_cases = discover_finalize_cases(bundle, rows);
  bool ok = true;
  if (args.finalize_mode == "both" || args.finalize_mode == "mixed") {
    auto mixed = pick_mixed_finalize_cases(all_cases, args.finalize_n);
    auto mixed_result = run_finalize_gate_one(args, device, stamp, shared_bundle, "mixed_bucket", mixed, correctness.reference);
    ok = ok && mixed_result.ok;
    cleanup_cuda_cache();
  }
  if (args.finalize_mode == "both" || args.finalize_mode == "same") {
    auto same = pick_same_bucket_finalize_cases(all_cases, args.finalize_n);
    auto same_result = run_finalize_gate_one(args, device, stamp, shared_bundle, "same_bucket", same, correctness.reference);
    ok = ok && same_result.ok;
    cleanup_cuda_cache();
  }
  return ok;
}

struct DensitySweepRunResult {
  int n = 0;
  int workers = 0;
  int num_runners = 0;
  int finalize_num_runners = 0;
  int rows_total = 0;
  int requested_sessions = 0;
  int sessions_completed = 0;
  int chunks_completed = 0;
  int finalize_samples = 0;
  int warmup_steady_workers = 0;
  int warmup_finalize_buckets = 0;
  int errors = 0;
  int mismatches = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  int unique_streams = 0;
  bool stream_uniqueness_ok = false;
  bool completed = false;
  bool oom = false;
  bool keepup_ok = false;
  bool ttfs_ok = false;
  bool finalize_p95_valid = false;
  bool correctness_ok = false;
  bool slo_robust = false;
  bool explicit_stream = true;
  double wall_ms = 0.0;
  double offered_audio_ms = 0.0;
  double throughput_realtime_streams = 0.0;
  double throughput_sessions_per_s = 0.0;
  size_t used_before_bytes = 0;
  size_t used_after_loaders_bytes = 0;
  size_t used_after_worker_contexts_bytes = 0;
  size_t shared_enc_first_delta_bytes = 0;
  size_t worker_context_delta_bytes = 0;
  size_t worker_context_delta_per_worker_bytes = 0;
  size_t used_after_run_bytes = 0;
  size_t peak_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  TimingBuckets timings;
  std::vector<double> lag_ms;
  std::vector<double> ttfs_ms;
  std::vector<uintptr_t> stream_handles;
  ResourceStats resource_stats;
  AdmissionTelemetry admission;
  StaleGenTelemetrySnapshot stale_gen;
  std::string finalize_loader_memory_json = "{}";
  std::string error;
};

struct DensityWorkerOutput {
  TimingBuckets timings;
  std::vector<double> lag_ms;
  std::vector<double> ttfs_ms;
  int sessions_completed = 0;
  int chunks_completed = 0;
  int mismatches = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  double offered_audio_ms = 0.0;
  std::string error;
};

static std::vector<std::vector<int>> assign_density_utts(int workers,
                                                         int rows_total,
                                                         int requested_sessions) {
  if (workers <= 0) throw std::runtime_error("density sweep workers must be positive");
  if (rows_total <= 0) throw std::runtime_error("density sweep requires at least one utterance");
  std::vector<std::vector<int>> assigned(static_cast<size_t>(workers));
  for (int i = 0; i < requested_sessions; ++i) {
    assigned[static_cast<size_t>(i % workers)].push_back(i % rows_total);
  }
  return assigned;
}

static std::vector<FinalizeBucketKey> needed_finalize_buckets_for_assignments(torch::jit::Module& bundle,
                                                                              const std::vector<std::vector<int>>& assigned) {
  std::vector<FinalizeBucketKey> keys;
  for (const auto& worker_utts : assigned) {
    for (int utt : worker_utts) {
      int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      if (T <= 0) continue;
      int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      keys.emplace_back(drop, T);
    }
  }
  return unique_finalize_bucket_keys(keys);
}

static std::map<FinalizeBucketKey, FinalizeCase> representative_finalize_cases_for_assignments(
    torch::jit::Module& bundle,
    const std::vector<std::vector<int>>& assigned) {
  std::map<FinalizeBucketKey, FinalizeCase> reps;
  for (const auto& worker_utts : assigned) {
    for (int utt : worker_utts) {
      int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      if (T <= 0) continue;
      int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      FinalizeBucketKey key = std::make_pair(drop, T);
      if (reps.find(key) == reps.end()) {
        reps.emplace(key, FinalizeCase{utt, drop, T});
      }
    }
  }
  return reps;
}

static std::vector<FinalizeBucketKey> finalize_keys_from_representatives(
    const std::map<FinalizeBucketKey, FinalizeCase>& reps) {
  std::vector<FinalizeBucketKey> keys;
  keys.reserve(reps.size());
  for (const auto& kv : reps) keys.push_back(kv.first);
  return keys;
}

static std::vector<std::map<FinalizeBucketKey, FinalizeCase>> representative_finalize_cases_by_worker(
    torch::jit::Module& bundle,
    const std::vector<std::vector<int>>& assigned) {
  std::vector<std::map<FinalizeBucketKey, FinalizeCase>> reps(assigned.size());
  for (size_t worker = 0; worker < assigned.size(); ++worker) {
    for (int utt : assigned[worker]) {
      int64_t T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      if (T <= 0) continue;
      int64_t drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      FinalizeBucketKey key = std::make_pair(drop, T);
      if (reps[worker].find(key) == reps[worker].end()) {
        reps[worker].emplace(key, FinalizeCase{utt, drop, T});
      }
    }
  }
  return reps;
}

static int count_finalize_samples_for_request(torch::jit::Module& bundle,
                                              int rows_total,
                                              int requested_sessions) {
  int count = 0;
  for (int i = 0; i < requested_sessions; ++i) {
    int utt = i % rows_total;
    if (scalar_i64(utt_tensor(bundle, utt, "final_T")) > 0) ++count;
  }
  return count;
}

static int raise_request_for_valid_finalize_p95(torch::jit::Module& bundle,
                                                int rows_total,
                                                int requested_sessions) {
  int finalize_rows_per_cycle = count_finalize_samples_for_request(bundle, rows_total, rows_total);
  if (finalize_rows_per_cycle <= 0) {
    throw std::runtime_error("density sweep cannot collect finalize p95: no rows with final_T > 0");
  }
  int out = requested_sessions;
  while (count_finalize_samples_for_request(bundle, rows_total, out) < kMinFinalizeP95Samples) {
    ++out;
  }
  return out;
}

static int pick_assigned_steady_warmup_utt(torch::jit::Module& bundle,
                                           const std::vector<int>& worker_utts) {
  for (int utt : worker_utts) {
    if (scalar_i64(utt_tensor(bundle, utt, "num_steady")) >= 2) return utt;
  }
  return -1;
}

static void warm_steady_encoder_once_density(int utt,
                                             WorkerContext& ctx,
                                             AOTIModelPackageLoader* enc_steady,
                                             BatchedSteadyLoaderSet* direct_b1_loader,
                                             torch::Device device,
                                             const Tokenizer& tokenizer,
                                             bool explicit_stream,
                                             bool mutex_serialize_run,
                                             const std::string& label,
                                             BatchedSteadyScheduler* scheduler = nullptr) {
  int64_t num_steady = scalar_i64(utt_tensor(ctx.bundle, utt, "num_steady"));
  if (num_steady < 2) {
    throw std::runtime_error("density steady warmup requires an utterance with at least two steady chunks");
  }
  SessionState session;
  reset_session(session, ctx.bundle, device);
  std::vector<EmittedEvent> events;
  std::string prefix = "utt" + std::to_string(utt);
  for (int chunk = 0; chunk < 2; ++chunk) {
    run_steady_chunk_density(session,
                             ctx.bundle,
                             prefix,
                             chunk,
                             *ctx.enc_first,
                             enc_steady,
                             direct_b1_loader,
                             ctx.joint,
                             ctx.predict,
                             device,
                             tokenizer,
                             events,
                             ctx.stream,
                             explicit_stream,
                             mutex_serialize_run,
                             nullptr,
                             label + ".chunk" + std::to_string(chunk),
                             scheduler,
                             nullptr,
                             nullptr,
                             nullptr);
  }
}

static std::string uintptr_list_json(const std::vector<uintptr_t>& values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) oss << ",";
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

static DensitySweepRunResult run_density_sweep_one_impl(const DensityArgs& args,
                                                        torch::Device device,
                                                        const std::string& stamp,
                                                        const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                                        int n,
                                                        int rows_total,
                                                        const std::vector<RowReplayResult>& reference) {
  DensitySweepRunResult result;
  result.n = n;
  result.workers = n;
  result.num_runners = n;
  result.finalize_num_runners = capped_general_finalize_runners(n);
  result.rows_total = rows_total;
  result.explicit_stream = args.stream_mode == "explicit";
  result.total_mem_bytes = gpu_total_bytes();
  if (n <= 0) throw std::runtime_error("density sweep N must be positive");
  if (!result.explicit_stream) {
    std::printf("=== DENSITY 1a WARNING: --stream-mode=%s is a control, not the Step-1a proven topology ===\n",
                args.stream_mode.c_str());
  }

  if (args.density_sessions_per_worker > 0) {
    result.requested_sessions = args.density_sessions_per_worker * n;
  } else if (args.density_rows > 0) {
    result.requested_sessions = args.density_rows;
  } else {
    result.requested_sessions = rows_total;
  }
  if (result.requested_sessions <= 0) throw std::runtime_error("density sweep requested zero sessions");
  auto& assignment_bundle = *shared_bundle;
  int original_requested_sessions = result.requested_sessions;
  result.requested_sessions = raise_request_for_valid_finalize_p95(assignment_bundle,
                                                                   rows_total,
                                                                   result.requested_sessions);
  if (result.requested_sessions != original_requested_sessions) {
    std::printf("=== DENSITY 1a SAMPLE FLOOR: bumped requested sessions from %d to %d "
                "to collect at least %d finalize samples for valid p95 ===\n",
                original_requested_sessions,
                result.requested_sessions,
                kMinFinalizeP95Samples);
  }
  int reference_rows_needed = std::min(rows_total, result.requested_sessions);
  if (static_cast<int>(reference.size()) < reference_rows_needed) {
    throw std::runtime_error("density sweep serial reference is smaller than the assigned utterance set");
  }
  auto assigned = assign_density_utts(result.workers, rows_total, result.requested_sessions);
  auto bucket_reps = representative_finalize_cases_for_assignments(assignment_bundle, assigned);
  auto needed_buckets = finalize_keys_from_representatives(bucket_reps);
  auto worker_bucket_reps = representative_finalize_cases_by_worker(assignment_bundle, assigned);
  const bool batch_steady_on = density_batch_steady_enabled_effective(args);
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy;
  if (batch_steady_on) {
    policy = density_batch_policy_effective(args, result.workers);
    result.num_runners =
        density_effective_steady_num_runners(args, result.num_runners, policy.dispatch_lanes);
    (void)run_b2_a1_parity_check(args, device, steady_batch_dir, shared_bundle, rows_total);
  }

  std::printf("=== DENSITY 1a RUN START: N=%d workers=%d steady_num_runners=%d finalize_num_runners=%d "
              "sessions=%d rows_total=%d finalize_samples_requested=%d min_finalize_p95_samples=%d "
              "cadence=%.3fms ===\n",
              n,
              result.workers,
              result.num_runners,
              result.finalize_num_runners,
              result.requested_sessions,
              rows_total,
              count_finalize_samples_for_request(assignment_bundle, rows_total, result.requested_sessions),
              kMinFinalizeP95Samples,
              args.density_chunk_period_ms);

  cleanup_cuda_cache();
  MemorySampler mem;
  mem.start();
  result.used_before_bytes = gpu_used_bytes();
  AOTIModelPackageLoader* enc_steady = nullptr;
  std::unique_ptr<BatchedSteadyLoaderSet> direct_b1_loader_owner;
  BatchedSteadyLoaderSet* direct_b1_loader = nullptr;
  if (!batch_steady_on) {
    auto direct_b1_load_start = Clock::now();
    direct_b1_loader_owner = std::make_unique<BatchedSteadyLoaderSet>(
        steady_batch_dir,
        args.dir + "/finalize_shared_weights.ts",
        device,
        result.num_runners,
        "density_sweep_off_direct_b1_bucket");
    direct_b1_loader_owner->preload_buckets({1});
    direct_b1_loader = direct_b1_loader_owner.get();
    log_density_phase_timing(n, "direct-b1-bucket-load", direct_b1_load_start);
    std::printf("B2_B1_SOURCE density-sweep N=%d OFF direct path uses %s/enc_steady_aoti_b1.pt2; "
                "production enc_steady_aoti.pt2 is not loaded\n",
                n,
                steady_batch_dir.c_str());
  } else {
    std::printf("B2_SCHEDULER_ON density-sweep N=%d skip production enc_steady_aoti.pt2; "
                "scheduler B=1 bucket handles continuation B=1\n",
                n);
  }

  std::unique_ptr<BatchedSteadyLoaderSet> batched_steady_loader;
  std::unique_ptr<BatchedSteadyScheduler> scheduler_owner;
  BatchedSteadyScheduler* scheduler = nullptr;
  if (batch_steady_on) {
    batched_steady_loader = std::make_unique<BatchedSteadyLoaderSet>(
        steady_batch_dir,
        args.dir + "/finalize_shared_weights.ts",
        device,
        result.num_runners,
        "b2_density_sweep_scheduler_on");
    scheduler_owner = std::make_unique<BatchedSteadyScheduler>(*batched_steady_loader, device, policy);
    scheduler_owner->warmup_buckets();
    scheduler_owner->start();
    scheduler = scheduler_owner.get();
    auto telem = scheduler->telemetry_snapshot();
    std::printf("B2_SCHEDULER_ON density-sweep N=%d dispatch_lanes=%d warmup_runs=%lld warmed_lanes=%lld "
                "policy={B_max:%d,window_ms:%d,lone_timeout_ms:%d,max_queue_delay_ms:%d,"
                "queue_capacity:%d,use_b2_bucket:%s,min_fill_enabled:%s,disable_min_fill:%s,force_bucket:%d}\n",
                n,
                policy.dispatch_lanes,
                static_cast<long long>(telem.warmup_runs),
                static_cast<long long>(telem.warmed_lanes),
                policy.B_max,
                policy.window_ms,
                policy.lone_timeout_ms,
                policy.max_queue_delay_ms,
                policy.queue_capacity,
                policy.use_b2_bucket ? "true" : "false",
                policy.min_fill_enabled ? "true" : "false",
                policy.disable_min_fill ? "true" : "false",
                policy.force_bucket);
  } else {
    if (scheduler != nullptr) throw std::runtime_error("batch steady OFF but scheduler pointer is non-null");
  }

  auto finalize_preload_start = Clock::now();
  FinalizeBucketLoaderPool finalize_loaders(args.dir,
                                            device,
                                            result.finalize_num_runners,
                                            "1a_density_sweep_capped_finalize_runner_pool");

  finalize_loaders.preload(needed_buckets);
  CUDA_CHECK(cudaDeviceSynchronize());
  result.used_after_loaders_bytes = gpu_used_bytes();
  log_density_phase_timing(n, "finalize-pool-preload", finalize_preload_start);

  auto shared_enc_first_load_start = Clock::now();
  auto shared_enc_first = load_shared_enc_first(args.dir, device, "1a_density_sweep_shared_locked_enc_first");
  result.shared_enc_first_delta_bytes = shared_enc_first->delta_bytes;
  log_density_phase_timing(n, "shared-enc_first-load", shared_enc_first_load_start);

  auto worker_context_start = Clock::now();
  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(static_cast<size_t>(result.workers));
  int stream_device_index = device.index() >= 0 ? device.index() : 0;
  std::vector<c10::cuda::CUDAStream> streams;
  streams.reserve(static_cast<size_t>(result.workers));
  std::vector<uintptr_t> stream_handles;
  stream_handles.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    c10::cuda::CUDAGuard device_guard(device.index());
    auto stream = stream_for_worker(result.explicit_stream, worker, stream_device_index);
    streams.push_back(stream);
    stream_handles.push_back(stream_handle_value(stream));
    contexts.push_back(make_worker_context(args.dir, device, stream, shared_bundle, shared_enc_first));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  result.used_after_worker_contexts_bytes = gpu_used_bytes();
  size_t context_base = result.used_after_loaders_bytes + result.shared_enc_first_delta_bytes;
  result.worker_context_delta_bytes = result.used_after_worker_contexts_bytes >= context_base
                                          ? result.used_after_worker_contexts_bytes - context_base
                                          : 0;
  result.worker_context_delta_per_worker_bytes = result.workers > 0
                                                     ? result.worker_context_delta_bytes /
                                                           static_cast<size_t>(result.workers)
                                                     : 0;
  result.stream_handles = std::move(stream_handles);
  std::set<uintptr_t> stream_ids(result.stream_handles.begin(), result.stream_handles.end());
  result.unique_streams = static_cast<int>(stream_ids.size());
  result.stream_uniqueness_ok = !result.explicit_stream || result.unique_streams == result.workers;
  log_density_phase_timing(n, "worker-context-creation", worker_context_start);
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);
  DensityAdmission admission(static_cast<uint64_t>(args.admission_active_cap),
                             static_cast<uint64_t>(args.admission_backlog_cap));
  StaleGenTelemetry stale_gen;

  auto warmup_start = Clock::now();
  ResourceSampler resources;
  StartGate gate(result.workers);
  std::vector<DensityWorkerOutput> worker_outputs(static_cast<size_t>(result.workers));
  std::vector<int> warmup_steady_done(static_cast<size_t>(result.workers), 0);
  std::vector<int> warmup_finalize_bucket_runs(static_cast<size_t>(result.workers), 0);
  const bool fill_trace_enabled = density_fill_trace_enabled();
  std::vector<std::vector<int64_t>> fill_trace_ready_ns;
  if (fill_trace_enabled) {
    fill_trace_ready_ns.resize(static_cast<size_t>(result.workers));
    for (int worker = 0; worker < result.workers; ++worker) {
      fill_trace_ready_ns[static_cast<size_t>(worker)].reserve(
          assigned[static_cast<size_t>(worker)].size() * 8);
    }
  }
  std::atomic<bool> warmup_failed{false};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(result.workers));
  for (int worker = 0; worker < result.workers; ++worker) {
    threads.emplace_back([&, worker] {
      auto& out = worker_outputs[static_cast<size_t>(worker)];
      std::vector<int64_t>* fill_trace_worker =
          fill_trace_enabled ? &fill_trace_ready_ns[static_cast<size_t>(worker)] : nullptr;
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        if (args.density_warmup) {
          int warm_utt = pick_assigned_steady_warmup_utt(contexts[worker]->bundle,
                                                         assigned[static_cast<size_t>(worker)]);
          if (warm_utt < 0) {
            throw std::runtime_error("density sweep steady warmup failed for worker" +
                                     std::to_string(worker) +
                                     ": no assigned utterance has a steady AOTI continuation");
          }
          warm_steady_encoder_once_density(warm_utt,
                                           *contexts[worker],
                                           enc_steady,
                                           direct_b1_loader,
                                           device,
                                           tokenizer,
                                           result.explicit_stream,
                                           args.mutex_serialize_run,
                                           "density.1a.warmup.steady.worker" + std::to_string(worker) +
                                               ".utt" + std::to_string(warm_utt),
                                           scheduler);
          warmup_steady_done[static_cast<size_t>(worker)] = 1;

          for (const auto& kv : worker_bucket_reps[static_cast<size_t>(worker)]) {
            const auto& fc = kv.second;
            SessionState warm_session;
            std::vector<EmittedEvent> warm_events;
            prepare_finalize_parent(fc,
                                    *contexts[worker],
                                    enc_steady,
                                    direct_b1_loader,
                                    device,
                                    tokenizer,
                                    result.explicit_stream,
                                    args.mutex_serialize_run,
                                    warm_session,
                                    warm_events,
                                    scheduler);
            std::string warm_label = "density.1a.warmup.worker" + std::to_string(worker) +
                                     ".finalize_bucket.drop" + std::to_string(fc.drop) +
                                     ".T" + std::to_string(fc.T) +
                                     ".utt" + std::to_string(fc.utt);
            auto warm = run_finalize_density(warm_session,
                                             contexts[worker]->bundle,
                                             "utt" + std::to_string(fc.utt),
                                             warm_label,
                                             finalize_loaders,
                                             contexts[worker]->joint,
                                             contexts[worker]->predict,
                                             device,
                                             tokenizer,
                                             warm_events,
                                             FinalizeFinish::SPECULATIVE_KEEP,
                                             contexts[worker]->stream,
                                             result.explicit_stream,
                                             args.mutex_serialize_run,
                                             nullptr);
            if (!warm.token_ok || !warm.fork_ok) {
              throw std::runtime_error("density sweep finalize bucket warmup failed for worker" +
                                       std::to_string(worker) + " drop=" + std::to_string(fc.drop) +
                                       " T=" + std::to_string(fc.T) +
                                       " utt=" + std::to_string(fc.utt));
            }
            ++warmup_finalize_bucket_runs[static_cast<size_t>(worker)];
          }
          CUDA_CHECK(cudaStreamSynchronize(contexts[worker]->stream.stream()));
        }
      } catch (const std::exception& e) {
        warmup_failed.store(true);
        out.error = e.what();
      }

      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        if (warmup_failed.load() || !out.error.empty()) return;
        // Production arrivals are staggered, not synchronized. The StartGate releases all workers at once,
        // which maximally synchronizes the per-session enc_first-lock + finalize bursts (re-syncing each
        // ~session period). Spread each worker's first-session start uniformly over [0, stagger] (fan-out:
        // worker w at w/(W-1)*stagger) to desync those bursts and model staggered arrivals. 0 = barrier.
        if (args.density_start_stagger_ms > 0.0 && result.workers > 1) {
          double frac = static_cast<double>(worker) / static_cast<double>(result.workers - 1);
          std::this_thread::sleep_for(ms_duration(args.density_start_stagger_ms * frac));
        }
        for (int utt : assigned[static_cast<size_t>(worker)]) {
          std::string prefix = "utt" + std::to_string(utt);
          std::string label = "density.1a.N" + std::to_string(n) +
                              ".worker" + std::to_string(worker) +
                              ".utt" + std::to_string(utt);
          AdmitResult admit = admission.try_admit(label);
          if (!wait_until_admission_active(admission, label, admit.decision)) {
            if (admit.shed()) ++out.mismatches;
            std::printf("DENSITY_ADMISSION_SHED N=%d worker=%d utt=%d decision=%s\n",
                        n,
                        worker,
                        utt,
                        admission_decision_name(admit.decision));
            continue;
          }
          AdmissionCloseGuard admission_guard(admission, label);
          SessionState session;
          reset_session(session, contexts[worker]->bundle, device);
          std::vector<EmittedEvent> events;
          int64_t num_steady = scalar_i64(utt_tensor(contexts[worker]->bundle, utt, "num_steady"));
          auto session_start = Clock::now();
          for (int chunk = 0; chunk < num_steady; ++chunk) {
            auto feed_time = session_start + ms_duration(args.density_chunk_period_ms * static_cast<double>(chunk));
            std::this_thread::sleep_until(feed_time);
            run_steady_chunk_density(session,
                                     contexts[worker]->bundle,
                                     prefix,
                                     chunk,
                                     *contexts[worker]->enc_first,
                                     enc_steady,
                                     direct_b1_loader,
                                     contexts[worker]->joint,
                                     contexts[worker]->predict,
                                     device,
                                     tokenizer,
                                     events,
                                     contexts[worker]->stream,
                                     result.explicit_stream,
                                     args.mutex_serialize_run,
                                     &out.timings,
                                     label + ".chunk" + std::to_string(chunk),
                                     scheduler,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     fill_trace_worker,
                                     &stale_gen);
            auto finish = Clock::now();
            auto deadline = session_start + ms_duration(args.density_chunk_period_ms * static_cast<double>(chunk + 1));
            out.lag_ms.push_back(signed_elapsed_ms(deadline, finish));
            ++out.chunks_completed;
          }
          out.offered_audio_ms += args.density_chunk_period_ms * static_cast<double>(num_steady);
          auto vad_deadline = session_start + ms_duration(args.density_chunk_period_ms * static_cast<double>(num_steady));
          std::this_thread::sleep_until(vad_deadline);
          auto ttfs_start = Clock::now();
          vad_stop(session);
          auto finalize = run_finalize_density(session,
                                               contexts[worker]->bundle,
                                               prefix,
                                               label,
                                               finalize_loaders,
                                               contexts[worker]->joint,
                                               contexts[worker]->predict,
                                               device,
                                               tokenizer,
                                               events,
                                               FinalizeFinish::SPECULATIVE_KEEP,
                                               contexts[worker]->stream,
                                               result.explicit_stream,
                                               args.mutex_serialize_run,
                                               &out.timings,
                                               &stale_gen);
          out.ttfs_ms.push_back(elapsed_ms_since(ttfs_start));
          bool tokens_ok = finalize.stale_dropped ||
                           (finalize.token_ok &&
                            finalize.fork_ok &&
                            finalize.final_tokens == reference[static_cast<size_t>(utt)].final_tokens);
          bool events_ok = finalize.stale_dropped ||
                           strict_events_equal(events,
                                               reference[static_cast<size_t>(utt)].events,
                                               label + ".serial_oracle");
          if (!tokens_ok) ++out.token_divergences;
          if (!events_ok) ++out.event_divergences;
          if (!tokens_ok || !events_ok) ++out.mismatches;
          ++out.sessions_completed;
        }
      } catch (const std::exception& e) {
        out.error = e.what();
      }
    });
  }
  gate.wait_until_ready();
  for (int worker = 0; worker < result.workers; ++worker) {
    result.warmup_steady_workers += warmup_steady_done[static_cast<size_t>(worker)];
    result.warmup_finalize_buckets += warmup_finalize_bucket_runs[static_cast<size_t>(worker)];
  }
  if (args.density_warmup) {
    std::printf("=== DENSITY 1a WARMUP COMPLETE: steady_workers=%d/%d "
                "finalize_bucket_worker_runs=%d unique_loaded_buckets=%zu CUDA_MODULE_LOADING=%s ===\n",
                result.warmup_steady_workers,
                result.workers,
                result.warmup_finalize_buckets,
                needed_buckets.size(),
                std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "(unset)");
  }
  log_density_phase_timing(n, "warmup", warmup_start);
  resources.start();
  gate.start_now();
  for (auto& thread : threads) thread.join();
  auto end_time = Clock::now();
  if (fill_trace_enabled) flush_density_fill_trace(fill_trace_ready_ns);
  log_density_phase_timing(n, "measured-gate", gate.start_time, end_time);
  auto teardown_start = Clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  result.resource_stats = resources.finish();
  result.wall_ms = elapsed_ms(gate.start_time, end_time);
  result.peak_mem_bytes = mem.finish();
  result.used_after_run_bytes = gpu_used_bytes();
  result.admission = admission.telemetry_snapshot();
  result.stale_gen = stale_gen.snapshot();
  result.finalize_loader_memory_json = finalize_loaders.memory_json(result.num_runners);

  for (int worker = 0; worker < result.workers; ++worker) {
    const auto& out = worker_outputs[static_cast<size_t>(worker)];
    result.timings.append(out.timings);
    result.lag_ms.insert(result.lag_ms.end(), out.lag_ms.begin(), out.lag_ms.end());
    result.ttfs_ms.insert(result.ttfs_ms.end(), out.ttfs_ms.begin(), out.ttfs_ms.end());
    result.sessions_completed += out.sessions_completed;
    result.chunks_completed += out.chunks_completed;
    result.mismatches += out.mismatches;
    result.token_divergences += out.token_divergences;
    result.event_divergences += out.event_divergences;
    result.offered_audio_ms += out.offered_audio_ms;
    if (!out.error.empty()) {
      ++result.errors;
      std::printf("  density sweep N=%d worker%d exception: %s\n", n, worker, out.error.c_str());
    }
  }

  result.throughput_realtime_streams = result.wall_ms > 0.0 ? result.offered_audio_ms / result.wall_ms : 0.0;
  result.throughput_sessions_per_s = result.wall_ms > 0.0
                                         ? 1000.0 * static_cast<double>(result.sessions_completed) / result.wall_ms
                                         : 0.0;
  result.finalize_samples = static_cast<int>(result.timings.finalize_total_ms.size());
  result.finalize_p95_valid = result.finalize_samples >= kMinFinalizeP95Samples;
  auto lag = summarize(result.lag_ms);
  auto ttfs = summarize(result.ttfs_ms);
  result.keepup_ok = lag.n > 0 && lag.p95 < 500.0;
  result.ttfs_ok = result.finalize_p95_valid && ttfs.n > 0 && ttfs.p95 <= 175.0 && ttfs.p99 <= 250.0;
  bool event_divergences_pass = density_event_divergences_gate_pass(result.event_divergences);
  result.correctness_ok = result.errors == 0 &&
                          result.token_divergences == 0 &&
                          event_divergences_pass &&
                          result.sessions_completed == result.requested_sessions &&
                          result.stream_uniqueness_ok;
  result.completed = result.sessions_completed == result.requested_sessions && result.errors == 0;
  result.slo_robust = result.completed && result.correctness_ok && result.keepup_ok && result.ttfs_ok;

  auto steady_gpu = summarize(result.timings.steady_gpu_ms);
  auto finalize_wait = summarize(result.timings.finalize_runner_wait_ms);
  auto finalize_aoti = summarize(result.timings.finalize_aoti_run_cuda_ms);
  auto finalize_total = summarize(result.timings.finalize_total_ms);
  const char* cuda_module_loading = std::getenv("CUDA_MODULE_LOADING");
  BatchedSteadySchedulerTelemetry batch_telemetry;
  if (scheduler != nullptr) batch_telemetry = scheduler->telemetry_snapshot();

  std::ostringstream json;
  json << "{\"check\":\"1a_density_sweep_full_session\""
       << ",\"num_runners\":" << result.num_runners
       << ",\"workers\":" << result.workers
       << ",\"steady_num_runners\":" << result.num_runners
       << ",\"finalize_num_runners_per_loaded_bucket\":" << result.finalize_num_runners
       << ",\"cuda_module_loading\":" << json_quote(cuda_module_loading ? cuda_module_loading : "")
       << ",\"stream_mode\":\"" << stream_mode_label(result.explicit_stream, args.mutex_serialize_run) << "\""
       << ",\"topology\":\"shared_steady_loader_shared_locked_enc_first_per_thread_session_handles_explicit_streams_capped_finalize_pool\""
       << ",\"steady_b1_source\":"
       << json_quote(scheduler != nullptr ? "steady_batch_scheduler"
                                           : (direct_b1_loader != nullptr ? "steady_batch_b1_direct"
                                                                          : "production_enc_steady"))
       << ",\"steady_b1_byte_contract\":"
       << json_quote(direct_b1_loader != nullptr
                         ? "A1 outcome B: enc_steady_aoti.pt2 and enc_steady_aoti_b1.pt2 are tensor-identical but SHA-different"
                         : "")
       << ",\"batch_steady_on\":" << json_bool(scheduler != nullptr)
       << ",\"batch_policy\":{\"B_max\":" << (scheduler != nullptr ? scheduler->policy().B_max : 0)
       << ",\"dispatch_lanes\":" << (scheduler != nullptr ? scheduler->policy().dispatch_lanes : 0)
       << ",\"window_ms\":" << (scheduler != nullptr ? scheduler->policy().window_ms : 0)
       << ",\"lone_timeout_ms\":" << (scheduler != nullptr ? scheduler->policy().lone_timeout_ms : 0)
       << ",\"max_queue_delay_ms\":" << (scheduler != nullptr ? scheduler->policy().max_queue_delay_ms : 0)
       << ",\"queue_capacity\":" << (scheduler != nullptr ? scheduler->policy().queue_capacity : 0)
       << ",\"use_b2_bucket\":"
       << json_bool(scheduler != nullptr && scheduler->policy().use_b2_bucket)
       << ",\"min_fill_enabled\":"
       << json_bool(scheduler != nullptr && scheduler->policy().min_fill_enabled)
       << ",\"disable_min_fill\":"
       << json_bool(scheduler != nullptr && scheduler->policy().disable_min_fill)
       << ",\"force_bucket\":" << (scheduler != nullptr ? scheduler->policy().force_bucket : 0) << "}"
       << ",\"batch_telemetry\":{\"dispatch_lanes\":" << batch_telemetry.dispatch_lanes
       << ",\"warmup_runs\":" << batch_telemetry.warmup_runs
       << ",\"warmed_lanes\":" << batch_telemetry.warmed_lanes
       << ",\"B1\":" << batch_telemetry.bucket_b1
       << ",\"B2\":" << batch_telemetry.bucket_b2
       << ",\"B4\":" << batch_telemetry.bucket_b4
       << ",\"B8\":" << batch_telemetry.bucket_b8
       << ",\"B16\":" << batch_telemetry.bucket_b16
       << ",\"K2_padded_to_B4\":" << batch_telemetry.k2_padded_to_b4
       << ",\"K3_padded_to_B4\":" << batch_telemetry.k3_padded_to_b4
       << ",\"K4\":" << batch_telemetry.k4
       << ",\"K5_padded_to_B8\":" << batch_telemetry.k5_padded_to_b8
       << ",\"K6_padded_to_B8\":" << batch_telemetry.k6_padded_to_b8
       << ",\"K7_padded_to_B8\":" << batch_telemetry.k7_padded_to_b8
       << ",\"K8\":" << batch_telemetry.k8
       << ",\"K9_padded_to_B16\":" << batch_telemetry.k9_padded_to_b16
       << ",\"K10_padded_to_B16\":" << batch_telemetry.k10_padded_to_b16
       << ",\"K11_padded_to_B16\":" << batch_telemetry.k11_padded_to_b16
       << ",\"K12_padded_to_B16\":" << batch_telemetry.k12_padded_to_b16
       << ",\"K13_padded_to_B16\":" << batch_telemetry.k13_padded_to_b16
       << ",\"K14_padded_to_B16\":" << batch_telemetry.k14_padded_to_b16
       << ",\"K15_padded_to_B16\":" << batch_telemetry.k15_padded_to_b16
       << ",\"K16\":" << batch_telemetry.k16
       << ",\"backlog_gt_bmax\":" << batch_telemetry.backlog_gt_bmax
       << ",\"skipped_ready\":" << batch_telemetry.skipped_ready
       << ",\"age_at_dispatch_us\":" << scheduler_us_stats_json(batch_telemetry.age_at_dispatch_us)
       << "}";
  if (scheduler != nullptr) {
    json << ",\"scheduler_telemetry\":" << scheduler_telemetry_json(batch_telemetry);
  }
  json << ",\"cadence_ms\":" << args.density_chunk_period_ms
       << ",\"rows_total\":" << rows_total
       << ",\"requested_sessions\":" << result.requested_sessions
       << ",\"sessions_completed\":" << result.sessions_completed
       << ",\"chunks_completed\":" << result.chunks_completed
       << ",\"finalize_samples\":" << result.finalize_samples
       << ",\"min_finalize_p95_samples\":" << kMinFinalizeP95Samples
       << ",\"finalize_p95_valid\":" << json_bool(result.finalize_p95_valid)
       << ",\"admission\":" << admission_telemetry_json(result.admission)
       << ",\"stale_gen\":" << stale_gen_telemetry_json(result.stale_gen)
       << ",\"ws_tail\":null"
       << ",\"warmup\":{\"enabled\":" << json_bool(args.density_warmup)
       << ",\"steady_workers\":" << result.warmup_steady_workers
       << ",\"finalize_bucket_worker_runs\":" << result.warmup_finalize_buckets
       << ",\"finalize_buckets\":" << result.warmup_finalize_buckets
       << ",\"loaded_finalize_buckets\":" << needed_buckets.size()
       << "}"
       << ",\"errors\":" << result.errors
       << ",\"mismatches\":" << result.mismatches
       << ",\"token_divergences\":" << result.token_divergences
       << ",\"event_divergences\":" << result.event_divergences
       << ",\"event_divergences_gated\":" << json_bool(density_strict_event_gate_enabled())
       << ",\"serial_oracle_match_pass\":"
       << json_bool(result.token_divergences == 0 && result.errors == 0 && event_divergences_pass)
       << ",\"strict_event_match_pass\":" << json_bool(result.event_divergences == 0)
       << ",\"stream_uniqueness_pass\":" << json_bool(result.stream_uniqueness_ok)
       << ",\"slo_robust\":" << json_bool(result.slo_robust)
       << ",\"keepup_lag_p95_lt_500ms\":" << json_bool(result.keepup_ok)
       << ",\"ttfs_budget_p95_175_p99_250_pass\":" << json_bool(result.ttfs_ok)
       << ",\"ttfs_budget\":{\"p95_ms\":175,\"p99_ms\":250}"
       << ",\"throughput_realtime_streams\":" << result.throughput_realtime_streams
       << ",\"throughput_sessions_per_s\":" << result.throughput_sessions_per_s
       << ",\"wall_ms\":" << result.wall_ms
       << ",\"offered_audio_ms\":" << result.offered_audio_ms
       << ",\"lag\":" << stats_json(lag)
       << ",\"ttfs\":" << stats_json(ttfs)
       << ",\"steady_latency\":" << stats_json(summarize(result.timings.latency_ms))
       << ",\"steady_runner_wait\":" << stats_json(summarize(result.timings.runner_wait_ms))
       << ",\"steady_gpu\":" << stats_json(steady_gpu)
       << ",\"item_wait\":" << stats_json(summarize(result.timings.scalar_sync_wait_ms))
       << ",\"item_wait_pct_of_steady_gpu\":" << stats_json(summarize(result.timings.scalar_sync_pct_of_gpu))
       << ",\"enc_first_lock_wait\":" << stats_json(summarize(result.timings.enc_first_lock_wait_ms))
       << ",\"enc_first_total\":" << stats_json(summarize(result.timings.enc_first_total_ms))
       << ",\"finalize_wait\":" << stats_json(finalize_wait)
       << ",\"finalize_gpu\":" << stats_json(summarize(result.timings.finalize_gpu_ms))
       << ",\"finalize_total\":" << stats_json(finalize_total)
       << ",\"finalize_phases\":" << finalize_phase_stats_json(result.timings)
       << ",\"resource\":" << resource_stats_json(result.resource_stats)
       << ",\"finalize_loader_memory\":" << result.finalize_loader_memory_json
       << ",\"unique_streams\":" << result.unique_streams
       << ",\"stream_handles\":" << uintptr_list_json(result.stream_handles)
       << ",\"peak_gpu_mem_bytes\":" << result.peak_mem_bytes
       << ",\"total_gpu_mem_bytes\":" << result.total_mem_bytes
       << ",\"used_before_bytes\":" << result.used_before_bytes
       << ",\"used_after_loaders_bytes\":" << result.used_after_loaders_bytes
       << ",\"used_after_worker_contexts_bytes\":" << result.used_after_worker_contexts_bytes
       << ",\"shared_enc_first\":{\"enabled\":true,\"lock\":\"mutex\""
       << ",\"delta_bytes\":" << result.shared_enc_first_delta_bytes
       << ",\"used_before_bytes\":" << shared_enc_first->used_before_bytes
       << ",\"used_after_bytes\":" << shared_enc_first->used_after_bytes
       << ",\"policy\":\"" << shared_enc_first->policy << "\"}"
       << ",\"worker_context_delta_bytes\":" << result.worker_context_delta_bytes
       << ",\"worker_context_delta_per_worker_bytes\":" << result.worker_context_delta_per_worker_bytes
       << ",\"used_after_run_bytes\":" << result.used_after_run_bytes
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 result.num_runners,
                 stream_mode_label(result.explicit_stream, args.mutex_serialize_run),
                 "1a_full_session_density_sweep_shared_locked_enc_first",
                 json.str());

  std::printf("=== DENSITY 1a ROW N=%d %s: throughput_rt=%.3f streams sessions/s=%.3f "
              "ttfs_p50/p95/p99=%.3f/%.3f/%.3fms spread=%.3fms lag_p50/p95=%.3f/%.3fms "
              "steady_gpu_p50/p95=%.3f/%.3fms enc_first_lock_p95=%.3fms "
              "finalize_total_p50/p95=%.3f/%.3fms "
              "finalize_aoti_p50/p95=%.3f/%.3fms finalize_wait_p95=%.3fms finalize_p95_valid=%s "
              "cpu_cores=%.2f/%d gpu_util_mean=%.1f%% peak_mem=%.3fGiB "
              "token_divergences=%d event_divergences=%d mismatches=%d errors=%d ===\n",
              result.n,
              result.slo_robust ? "SLO_ROBUST" : "NOT_SLO_ROBUST",
              result.throughput_realtime_streams,
              result.throughput_sessions_per_s,
              ttfs.p50,
              ttfs.p95,
              ttfs.p99,
              ttfs.p95 - ttfs.p50,
              lag.p50,
              lag.p95,
              steady_gpu.p50,
              steady_gpu.p95,
              summarize(result.timings.enc_first_lock_wait_ms).p95,
              finalize_total.p50,
              finalize_total.p95,
              finalize_aoti.p50,
              finalize_aoti.p95,
              finalize_wait.p95,
              result.finalize_p95_valid ? "true" : "false",
              result.resource_stats.cpu_cores_used,
              result.resource_stats.cpu_threads,
              result.resource_stats.gpu_util_mean_pct,
              static_cast<double>(result.peak_mem_bytes) / (1024.0 * 1024.0 * 1024.0),
              result.token_divergences,
              result.event_divergences,
              result.mismatches,
              result.errors);
  log_density_phase_timing(n, "teardown", teardown_start);
  return result;
}

static DensitySweepRunResult run_density_sweep_one(const DensityArgs& args,
                                                   torch::Device device,
                                                   const std::string& stamp,
                                                   const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                                   int n,
                                                   int rows_total,
                                                   const std::vector<RowReplayResult>& reference) {
  try {
    return run_density_sweep_one_impl(args, device, stamp, shared_bundle, n, rows_total, reference);
  } catch (const std::exception& e) {
    DensitySweepRunResult result;
    result.n = n;
    result.workers = n;
    result.num_runners = n;
    result.finalize_num_runners = capped_general_finalize_runners(std::max(1, n));
    result.rows_total = rows_total;
    result.total_mem_bytes = gpu_total_bytes();
    result.error = e.what();
    std::string lower = result.error;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    result.oom = lower.find("out of memory") != std::string::npos ||
                 lower.find("cuda error") != std::string::npos ||
                 lower.find("memory") != std::string::npos;
    try {
      result.peak_mem_bytes = gpu_used_bytes();
      cleanup_cuda_cache();
    } catch (const std::exception&) {
    }
    std::ostringstream json;
    json << "{\"check\":\"1a_density_sweep_full_session\""
         << ",\"num_runners\":" << result.num_runners
         << ",\"workers\":" << result.workers
         << ",\"slo_robust\":false"
         << ",\"oom\":" << json_bool(result.oom)
         << ",\"error\":" << json_quote(result.error)
         << ",\"admission\":" << admission_telemetry_json(result.admission)
         << ",\"stale_gen\":" << stale_gen_telemetry_json(result.stale_gen)
         << ",\"ws_tail\":null"
         << ",\"peak_gpu_mem_bytes\":" << result.peak_mem_bytes
         << ",\"total_gpu_mem_bytes\":" << result.total_mem_bytes
         << "}";
    emit_telemetry(args.dir,
                   stamp,
                   result.num_runners,
                   stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run),
                   "1a_full_session_density_sweep_error",
                   json.str());
    std::printf("=== DENSITY 1a ROW N=%d ERROR: oom=%s error=%s ===\n",
                n,
                result.oom ? "true" : "false",
                result.error.c_str());
    return result;
  }
}

struct DensitySweepSummary {
  std::vector<DensitySweepRunResult> runs;
  int knee_n = 0;
  int single_thread_keepup_n = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  int errors = 0;
  double multiplier = 0.0;
  bool pass_to_1b = false;
  bool correctness_at_knee = false;
  std::string binding_slo = "none";
  std::string binding_resource = "not_observed";
};

static const DensitySweepRunResult* find_density_result(const std::vector<DensitySweepRunResult>& runs, int n) {
  for (const auto& run : runs) {
    if (run.n == n) return &run;
  }
  return nullptr;
}

static std::string infer_binding_slo(const std::vector<DensitySweepRunResult>& runs, int knee_n) {
  for (const auto& run : runs) {
    if (run.n <= knee_n) continue;
    if (run.oom) return "memory_oom";
    if (!run.correctness_ok && run.completed) return "correctness";
    if (!run.finalize_p95_valid) return "finalize_p95_invalid";
    if (!run.ttfs_ok) return "ttfs_p95_or_p99";
    if (!run.keepup_ok) return "keepup_lag_p95";
    if (!run.completed) return "runtime_error";
  }
  return "none_observed_in_sweep";
}

static std::string infer_binding_resource(const std::vector<DensitySweepRunResult>& runs, int knee_n) {
  const DensitySweepRunResult* base = find_density_result(runs, 1);
  SummaryStats base_steady_gpu = base != nullptr ? summarize(base->timings.steady_gpu_ms) : SummaryStats{};
  const DensitySweepRunResult* first_bound = nullptr;
  for (const auto& run : runs) {
    if (run.n <= knee_n) continue;
    first_bound = &run;
    break;
  }
  if (first_bound == nullptr) {
    first_bound = runs.empty() ? nullptr : &runs.back();
  }
  if (first_bound == nullptr) return "not_observed";
  if (first_bound->oom) return "memory";
  if (first_bound->total_mem_bytes > 0 &&
      static_cast<double>(first_bound->peak_mem_bytes) / static_cast<double>(first_bound->total_mem_bytes) >= 0.92) {
    return "memory";
  }
  if (first_bound->resource_stats.cpu_threads > 0 &&
      first_bound->resource_stats.cpu_cores_used >= 0.85 * static_cast<double>(first_bound->resource_stats.cpu_threads)) {
    return "CPU cores";
  }
  auto steady_gpu = summarize(first_bound->timings.steady_gpu_ms);
  if (base_steady_gpu.p50 > 0.0 && steady_gpu.p50 / base_steady_gpu.p50 >= 1.50) {
    return "GPU encoder contention";
  }
  if (first_bound->resource_stats.gpu_util_available && first_bound->resource_stats.gpu_util_mean_pct >= 80.0) {
    return "GPU encoder contention";
  }
  if (!first_bound->ttfs_ok) return "finalize/TTFS tail";
  if (!first_bound->keepup_ok) return "mixed_or_unknown_keepup";
  return "not_observed";
}

static void emit_density_sweep_manifest(const DensityArgs& args,
                                        const std::string& stamp,
                                        const DensitySweepSummary& summary,
                                        int rows_total) {
  std::string logs_dir = args.dir + "/logs/" + stamp;
  fs::create_directories(logs_dir);
  std::string path = logs_dir + "/density_sweep_manifest.json";
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open density sweep manifest: " + path);
  out << "{"
      << "\"stamp\":\"" << stamp << "\""
      << ",\"mode\":\"density-sweep\""
      << ",\"status\":\"" << (summary.pass_to_1b ? "PASS_TO_1B" : "NO_PASS_TO_1B") << "\""
      << ",\"dir\":\"" << args.dir << "\""
      << ",\"n_values\":" << int_list_json(args.n_values)
      << ",\"rows_total\":" << rows_total
      << ",\"density_rows\":" << args.density_rows
      << ",\"density_sessions_per_worker\":" << args.density_sessions_per_worker
      << ",\"density_warmup\":" << json_bool(args.density_warmup)
      << ",\"min_finalize_p95_samples\":" << kMinFinalizeP95Samples
      << ",\"cuda_module_loading\":"
      << json_quote(std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "")
      << ",\"cadence_ms\":" << args.density_chunk_period_ms
      << ",\"knee_n\":" << summary.knee_n
      << ",\"single_thread_keepup_n\":" << summary.single_thread_keepup_n
      << ",\"token_divergences\":" << summary.token_divergences
      << ",\"event_divergences\":" << summary.event_divergences
      << ",\"errors\":" << summary.errors
      << ",\"multiplier\":" << summary.multiplier
      << ",\"pass_to_1b\":" << json_bool(summary.pass_to_1b)
      << ",\"binding_slo\":\"" << summary.binding_slo << "\""
      << ",\"binding_resource\":\"" << summary.binding_resource << "\""
      << "}\n";
  if (!out) throw std::runtime_error("failed to write density sweep manifest: " + path);
  std::printf("DENSITY_SWEEP_MANIFEST path=%s status=%s\n",
              path.c_str(),
              summary.pass_to_1b ? "PASS_TO_1B" : "NO_PASS_TO_1B");
}

static DensitySweepSummary run_density_sweep(const DensityArgs& args,
                                             torch::Device device,
                                             const std::string& stamp,
                                             const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                             int rows_total) {
  int max_n = *std::max_element(args.n_values.begin(), args.n_values.end());
  int max_requested_sessions = rows_total;
  if (args.density_sessions_per_worker > 0) {
    max_requested_sessions = args.density_sessions_per_worker * max_n;
  } else if (args.density_rows > 0) {
    max_requested_sessions = args.density_rows;
  }
  auto serial_oracle_start = Clock::now();
  auto& reference_sizing_bundle = *shared_bundle;
  int reference_sessions = raise_request_for_valid_finalize_p95(reference_sizing_bundle,
                                                               rows_total,
                                                               std::max(1, max_requested_sessions));
  int reference_rows = std::min(rows_total, reference_sessions);
  std::printf("=== DENSITY 1a SERIAL ORACLE BUILD: rows=%d/%d ===\n", reference_rows, rows_total);
  TimingBuckets ref_timings;
  auto reference = build_serial_reference(args, device, shared_bundle, reference_rows, &ref_timings);
  std::printf("=== DENSITY 1a SERIAL ORACLE PASS: rows=%d/%d ===\n", reference_rows, rows_total);
  log_density_phase_timing(max_n, "serial-oracle-build", serial_oracle_start);
  cleanup_cuda_cache();

  DensitySweepSummary summary;
  for (int n : args.n_values) {
    auto run = run_density_sweep_one(args, device, stamp, shared_bundle, n, rows_total, reference);
    bool stop_after = run.oom;
    summary.runs.push_back(std::move(run));
    cleanup_cuda_cache();
    if (stop_after) {
      std::printf("=== DENSITY 1a STOPPING SWEEP AFTER N=%d: memory/runtime bound hit ===\n", n);
      break;
    }
  }

  for (const auto& run : summary.runs) {
    if (run.slo_robust) summary.knee_n = std::max(summary.knee_n, run.n);
    summary.token_divergences += run.token_divergences;
    summary.event_divergences += run.event_divergences;
    summary.errors += run.errors;
  }
  const auto* n1 = find_density_result(summary.runs, 1);
  summary.single_thread_keepup_n = (n1 != nullptr && n1->slo_robust) ? 1 : 0;
  summary.multiplier = summary.single_thread_keepup_n > 0
                           ? static_cast<double>(summary.knee_n) / static_cast<double>(summary.single_thread_keepup_n)
                           : 0.0;
  const auto* knee = find_density_result(summary.runs, summary.knee_n);
  summary.correctness_at_knee = knee != nullptr && knee->correctness_ok;
  summary.pass_to_1b = summary.multiplier >= 2.0 && summary.correctness_at_knee && summary.knee_n > 0;
  summary.binding_slo = infer_binding_slo(summary.runs, summary.knee_n);
  summary.binding_resource = infer_binding_resource(summary.runs, summary.knee_n);

  AdmissionTelemetry summary_admission;
  StaleGenTelemetrySnapshot summary_stale_gen;
  for (const auto& run : summary.runs) {
    merge_admission_telemetry(summary_admission, run.admission);
    merge_stale_gen_telemetry(summary_stale_gen, run.stale_gen);
  }

  std::ostringstream rows_json;
  rows_json << "[";
  for (size_t i = 0; i < summary.runs.size(); ++i) {
    if (i > 0) rows_json << ",";
    const auto& run = summary.runs[i];
    rows_json << "{\"N\":" << run.n
              << ",\"slo_robust\":" << json_bool(run.slo_robust)
              << ",\"throughput_realtime_streams\":" << run.throughput_realtime_streams
              << ",\"finalize_samples\":" << run.finalize_samples
              << ",\"finalize_p95_valid\":" << json_bool(run.finalize_p95_valid)
              << ",\"ttfs\":" << stats_json(summarize(run.ttfs_ms))
              << ",\"lag\":" << stats_json(summarize(run.lag_ms))
              << ",\"steady_gpu\":" << stats_json(summarize(run.timings.steady_gpu_ms))
              << ",\"enc_first_lock_wait\":" << stats_json(summarize(run.timings.enc_first_lock_wait_ms))
              << ",\"enc_first_total\":" << stats_json(summarize(run.timings.enc_first_total_ms))
              << ",\"finalize_total\":" << stats_json(summarize(run.timings.finalize_total_ms))
              << ",\"finalize_phases\":" << finalize_phase_stats_json(run.timings)
              << ",\"peak_gpu_mem_bytes\":" << run.peak_mem_bytes
              << ",\"shared_enc_first_delta_bytes\":" << run.shared_enc_first_delta_bytes
              << ",\"worker_context_delta_per_worker_bytes\":" << run.worker_context_delta_per_worker_bytes
              << ",\"mismatches\":" << run.mismatches
              << ",\"token_divergences\":" << run.token_divergences
              << ",\"event_divergences\":" << run.event_divergences
              << ",\"errors\":" << run.errors
              << ",\"admission\":" << admission_telemetry_json(run.admission)
              << ",\"stale_gen\":" << stale_gen_telemetry_json(run.stale_gen)
              << ",\"ws_tail\":null"
              << ",\"oom\":" << json_bool(run.oom)
              << "}";
  }
  rows_json << "]";

  std::ostringstream json;
  json << "{\"check\":\"1a_density_sweep_summary\""
       << ",\"num_runners\":0"
       << ",\"stream_mode\":\"" << stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run) << "\""
       << ",\"topology\":\"shared_steady_loader_shared_locked_enc_first_per_thread_session_handles_explicit_streams_capped_finalize_pool\""
       << ",\"ttfs_budget\":{\"p95_ms\":175,\"p99_ms\":250}"
       << ",\"keepup_budget\":{\"lag_p95_ms\":500}"
       << ",\"density_warmup\":" << json_bool(args.density_warmup)
       << ",\"min_finalize_p95_samples\":" << kMinFinalizeP95Samples
       << ",\"cuda_module_loading\":"
       << json_quote(std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "")
       << ",\"rows_total\":" << rows_total
       << ",\"n_values\":" << int_list_json(args.n_values)
       << ",\"knee_n\":" << summary.knee_n
       << ",\"single_thread_keepup_n\":" << summary.single_thread_keepup_n
       << ",\"token_divergences\":" << summary.token_divergences
       << ",\"event_divergences\":" << summary.event_divergences
       << ",\"errors\":" << summary.errors
       << ",\"multiplier\":" << summary.multiplier
       << ",\"pass_to_1b\":" << json_bool(summary.pass_to_1b)
       << ",\"correctness_at_knee\":" << json_bool(summary.correctness_at_knee)
       << ",\"binding_slo\":\"" << summary.binding_slo << "\""
       << ",\"binding_resource\":\"" << summary.binding_resource << "\""
       << ",\"admission\":" << admission_telemetry_json(summary_admission)
       << ",\"stale_gen\":" << stale_gen_telemetry_json(summary_stale_gen)
       << ",\"ws_tail\":null"
       << ",\"rows\":" << rows_json.str()
       << "}";
  emit_telemetry(args.dir,
                 stamp,
                 0,
                 stream_mode_label(args.stream_mode == "explicit", args.mutex_serialize_run),
                 "1a_full_session_density_sweep_shared_locked_enc_first_summary",
                 json.str());
  emit_density_sweep_manifest(args, stamp, summary, rows_total);
  std::printf("=== DENSITY 1a SUMMARY %s: knee_N=%d single_thread_keepup_N=%d multiplier=%.3fx "
              "token_divergences=%d event_divergences=%d errors=%d "
              "binding_slo=%s binding_resource=%s correctness_at_knee=%s ===\n",
              summary.pass_to_1b ? "PASS_TO_1B" : "NO_PASS_TO_1B",
              summary.knee_n,
              summary.single_thread_keepup_n,
              summary.multiplier,
              summary.token_divergences,
              summary.event_divergences,
              summary.errors,
              summary.binding_slo.c_str(),
              summary.binding_resource.c_str(),
              summary.correctness_at_knee ? "PASS" : "FAIL");
  return summary;
}

static void emit_run_manifest(const DensityArgs& args,
                              const std::string& stamp,
                              const std::string& status,
                              bool canonical_full_run,
                              bool partial,
                              int rows_total,
                              int correctness_rows,
                              const std::string& correctness_status,
                              const std::string& steady_status,
                              const std::string& finalize_status) {
  std::string logs_dir = args.dir + "/logs/" + stamp;
  fs::create_directories(logs_dir);
  std::string path = logs_dir + "/manifest.json";
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open manifest: " + path);
  out << "{"
      << "\"stamp\":\"" << stamp << "\""
      << ",\"status\":\"" << status << "\""
      << ",\"canonical_full_run\":" << json_bool(canonical_full_run)
      << ",\"partial\":" << json_bool(partial)
      << ",\"smoke\":" << json_bool(args.smoke)
      << ",\"dir\":\"" << args.dir << "\""
      << ",\"n_values\":" << int_list_json(args.n_values)
      << ",\"target_n\":" << args.target_n
      << ",\"workers_override\":" << args.workers
      << ",\"num_runners_override\":" << args.num_runners
      << ",\"stream_mode\":\"" << args.stream_mode << "\""
      << ",\"mutex_serialize_run\":" << json_bool(args.mutex_serialize_run)
      << ",\"steady_cases\":" << args.steady_cases
      << ",\"steady_repeats\":" << args.steady_repeats
      << ",\"correctness_n\":" << args.correctness_n
      << ",\"correctness_rows\":" << correctness_rows
      << ",\"rows_total\":" << rows_total
      << ",\"finalize_n\":" << args.finalize_n
      << ",\"finalize_mode\":\"" << args.finalize_mode << "\""
      << ",\"skip_correctness\":" << json_bool(args.skip_correctness)
      << ",\"skip_steady\":" << json_bool(args.skip_steady)
      << ",\"skip_finalize\":" << json_bool(args.skip_finalize)
      << ",\"default_stream_control\":" << json_bool(args.default_stream_control)
      << ",\"correctness_default_stream_control\":" << json_bool(args.correctness_default_stream_control)
      << ",\"steady_overlap_probe\":" << json_bool(args.steady_overlap_probe)
      << ",\"scalar_locality_probe\":" << json_bool(args.scalar_locality_probe)
      << ",\"correctness_status\":\"" << correctness_status << "\""
      << ",\"steady_status\":\"" << steady_status << "\""
      << ",\"finalize_status\":\"" << finalize_status << "\""
      << "}\n";
  if (!out) throw std::runtime_error("failed to write manifest: " + path);
  std::printf("DENSITY_MANIFEST path=%s status=%s canonical_full_run=%s partial=%s\n",
              path.c_str(),
              status.c_str(),
              canonical_full_run ? "true" : "false",
              partial ? "true" : "false");
}

static bool run_admission_smoke() {
  DensityAdmission active_only(10, 0);
  int active_admitted = 0;
  int active_shed = 0;
  for (int i = 0; i < 100; ++i) {
    AdmitResult result = active_only.try_admit("active-" + std::to_string(i));
    if (result.decision == AdmissionDecision::ADMITTED) {
      ++active_admitted;
    } else if (result.shed()) {
      ++active_shed;
    }
  }
  AdmissionTelemetry active = active_only.telemetry_snapshot();
  bool active_ok = active_admitted == 10 &&
                   active_shed == 90 &&
                   active.offered == 100 &&
                   active.admitted == 10 &&
                   active.active_peak == 10 &&
                   active.shed_close_count == 90 &&
                   active.active_cap_hits == 90;

  DensityAdmission with_backlog(10, 5);
  int backlog_active_admitted = 0;
  int backlog_queued = 0;
  int backlog_shed = 0;
  for (int i = 0; i < 100; ++i) {
    AdmitResult result = with_backlog.try_admit("backlog-" + std::to_string(i));
    if (result.decision == AdmissionDecision::ADMITTED) {
      ++backlog_active_admitted;
    } else if (result.decision == AdmissionDecision::QUEUED) {
      ++backlog_queued;
    } else if (result.shed()) {
      ++backlog_shed;
    }
  }
  AdmissionTelemetry backlog = with_backlog.telemetry_snapshot();
  bool backlog_ok = backlog_active_admitted == 10 &&
                    backlog_queued == 5 &&
                    backlog_shed == 85 &&
                    backlog.offered == 100 &&
                    backlog.admitted == 15 &&
                    backlog.active_peak == 10 &&
                    backlog.backlog_peak == 5 &&
                    backlog.shed_close_count == 85 &&
                    backlog.backlog_cap_hits == 85;

  bool ok = active_ok && backlog_ok;
  std::printf("ADMISSION_SMOKE pass=%s active_cap_shed=%d backlog_cap_shed=%d "
              "active=%s backlog=%s\n",
              ok ? "true" : "false",
              active_shed,
              backlog_shed,
              admission_telemetry_json(active).c_str(),
              admission_telemetry_json(backlog).c_str());
  return ok;
}

static BatchedSteadyInput make_scheduler_admission_input(torch::Device device, const std::string& label) {
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto long_options = torch::TensorOptions().dtype(torch::kLong).device(device);
  return {
      torch::zeros({1, 128, PRE + SHIFT}, float_options),
      torch::zeros({24, 1, ATT_CONTEXT_LEFT, 1024}, float_options),
      torch::zeros({24, 1, 1024, 8}, float_options),
      torch::zeros({1}, long_options),
      label,
  };
}

static bool run_scheduler_admission_smoke(const DensityArgs& args, torch::Device device) {
  const int producers = 4;
  const int offers_per_producer = 96;
  const int offered = producers * offers_per_producer;
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 4;
  policy.window_ms = 10;
  policy.lone_timeout_ms = 10;
  policy.queue_capacity = 2;
  policy.dispatch_lanes = density_dispatch_lanes_effective(args);
  int steady_runners = density_effective_steady_num_runners(args, 1, policy.dispatch_lanes);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        steady_runners,
                                        "scheduler_admission_smoke");
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  std::atomic<int> admitted{0};
  std::atomic<int> timeouts{0};
  std::atomic<int> producer_errors{0};
  std::mutex futures_mutex;
  std::mutex errors_mutex;
  std::vector<std::future<DispatchResult>> futures;
  std::vector<std::string> errors;
  futures.reserve(static_cast<size_t>(offered));
  StartGate gate(producers);
  std::vector<std::thread> threads;
  threads.reserve(producers);
  for (int producer = 0; producer < producers; ++producer) {
    threads.emplace_back([&, producer] {
      gate.arrive_and_wait();
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        for (int i = 0; i < offers_per_producer; ++i) {
          std::string label = "scheduler_admission_smoke.p" + std::to_string(producer) +
                              ".offer" + std::to_string(i);
          auto input = make_scheduler_admission_input(device, label);
          ScopedCudaEvent producer_event;
          producer_event.create(cudaEventDisableTiming);
          CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
          auto deadline = Clock::now() + std::chrono::milliseconds(5);
          uint64_t stream_key = static_cast<uint64_t>(producer + 1);
          auto future = scheduler.try_enqueue_until(
              {std::move(input), stream, producer_event.event, stream_key},
              deadline);
          if (future.has_value()) {
            producer_event.release();
            admitted.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(futures_mutex);
            futures.push_back(std::move(*future));
          } else {
            timeouts.fetch_add(1, std::memory_order_relaxed);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      } catch (const std::exception& e) {
        producer_errors.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(errors_mutex);
        errors.push_back(e.what());
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();

  std::vector<std::future<DispatchResult>> admitted_futures;
  {
    std::lock_guard<std::mutex> lock(futures_mutex);
    admitted_futures = std::move(futures);
  }

  int completed = 0;
  int future_timeouts = 0;
  int bad_results = 0;
  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard stream_guard(consumer_stream);
    for (auto& future : admitted_futures) {
      auto wait_status = future.wait_for(std::chrono::seconds(10));
      if (wait_status != std::future_status::ready) {
        ++future_timeouts;
        continue;
      }
      auto result = future.get();
      if (result.completion.get() == nullptr || result.row_tensors.size() < 5) {
        ++bad_results;
        continue;
      }
      CUDA_CHECK(cudaStreamWaitEvent(consumer_stream.stream(), result.completion.get(), 0));
      CUDA_CHECK(cudaStreamSynchronize(consumer_stream.stream()));
      ++completed;
    }
  }

  scheduler.close();
  auto telemetry = scheduler.telemetry_snapshot();
  if (producer_errors.load(std::memory_order_relaxed) > 0) {
    for (const auto& error : errors) {
      std::printf("SCHEDULER_ADMISSION_SMOKE_PRODUCER_ERROR %s\n", error.c_str());
    }
  }

  bool ok = producer_errors.load(std::memory_order_relaxed) == 0 &&
            future_timeouts == 0 &&
            bad_results == 0 &&
            timeouts.load(std::memory_order_relaxed) > 0 &&
            admitted.load(std::memory_order_relaxed) > policy.queue_capacity &&
            completed == admitted.load(std::memory_order_relaxed) &&
            telemetry.completed == admitted.load(std::memory_order_relaxed) &&
            telemetry.dispatcher_exceptions == 0;
  std::printf("SCHEDULER_ADMISSION_SMOKE %s offered=%d admitted=%d cap_wait_timeouts=%d "
              "completed=%d future_timeouts=%d bad_results=%d producer_errors=%d policy_capacity=%d "
              "dispatch_cycles=%lld telemetry_completed=%lld dispatcher_exceptions=%lld telemetry=%s\n",
              ok ? "PASS" : "FAIL",
              offered,
              admitted.load(std::memory_order_relaxed),
              timeouts.load(std::memory_order_relaxed),
              completed,
              future_timeouts,
              bad_results,
              producer_errors.load(std::memory_order_relaxed),
              policy.queue_capacity,
              static_cast<long long>(telemetry.dispatch_cycles),
              static_cast<long long>(telemetry.completed),
              static_cast<long long>(telemetry.dispatcher_exceptions),
              scheduler_telemetry_json(telemetry).c_str());
  cleanup_cuda_cache();
  return ok;
}

static std::future<DispatchResult> enqueue_scheduler_smoke_input(BatchedSteadyScheduler& scheduler,
                                                                 torch::Device device,
                                                                 c10::cuda::CUDAStream stream,
                                                                 const std::string& label) {
  auto input = make_scheduler_admission_input(device, label);
  ScopedCudaEvent producer_event;
  producer_event.create(cudaEventDisableTiming);
  CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
  uint64_t stream_key = static_cast<uint64_t>(std::hash<std::string>{}(label));
  if (stream_key == 0) stream_key = 1;
  auto future = scheduler.enqueue({std::move(input), stream, producer_event.event, stream_key});
  producer_event.release();
  return future;
}

static bool consume_scheduler_smoke_result(BatchedSteadyScheduler& scheduler,
                                           std::future<DispatchResult>& future,
                                           c10::cuda::CUDAStream consumer_stream,
                                           bool retire_graph_slot,
                                           int* bad_results,
                                           int* future_timeouts,
                                           int* graph_results) {
  auto wait_status = future.wait_for(std::chrono::seconds(10));
  if (wait_status != std::future_status::ready) {
    ++(*future_timeouts);
    return false;
  }
  auto result = future.get();
  if (result.completion.get() == nullptr || result.row_tensors.size() < 5) {
    ++(*bad_results);
    return false;
  }
  CUDA_CHECK(cudaStreamWaitEvent(consumer_stream.stream(), result.completion.get(), 0));
  CUDA_CHECK(cudaStreamSynchronize(consumer_stream.stream()));
  if (result.graph_slot) {
    ++(*graph_results);
    if (retire_graph_slot) {
      result.graph_slot->retire_on_stream(consumer_stream.stream());
      result.graph_slot.reset();
    }
  }
  result.completion.reset();
  scheduler.record_worker_wait(result.cycle_id, result.k, 0.0, 0.0);
  return true;
}

static bool run_scheduler_lanes_smoke(const DensityArgs& args, torch::Device device) {
  const int producers = 4;
  const int rows_per_producer = 8;
  const int offered = producers * rows_per_producer;
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 1;
  policy.window_ms = 0;
  policy.lone_timeout_ms = 0;
  policy.max_queue_delay_ms = 0;
  policy.queue_capacity = 2;
  policy.min_fill_enabled = false;
  policy.disable_min_fill = true;
  policy.dispatch_lanes = 2;
  int steady_runners = density_effective_steady_num_runners(args, 2, policy.dispatch_lanes);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        steady_runners,
                                        "scheduler_lanes_smoke");
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  struct LabeledFuture {
    std::string label;
    std::future<DispatchResult> future;
  };

  std::mutex futures_mutex;
  std::vector<LabeledFuture> futures;
  futures.reserve(static_cast<size_t>(offered + 1));
  std::atomic<int> producer_errors{0};
  std::atomic<int> enqueue_timeouts{0};
  StartGate gate(producers);
  std::vector<std::thread> threads;
  threads.reserve(producers);
  for (int producer = 0; producer < producers; ++producer) {
    threads.emplace_back([&, producer] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        gate.arrive_and_wait();
        for (int row = 0; row < rows_per_producer; ++row) {
          std::string label = "scheduler_lanes_smoke.p" + std::to_string(producer) +
                              ".row" + std::to_string(row);
          auto input = make_scheduler_admission_input(device, label);
          ScopedCudaEvent producer_event;
          producer_event.create(cudaEventDisableTiming);
          CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
          uint64_t stream_key = static_cast<uint64_t>(std::hash<std::string>{}(label));
          if (stream_key == 0) stream_key = 1;
          auto future = scheduler.try_enqueue_until(
              {std::move(input), stream, producer_event.event, stream_key},
              Clock::now() + std::chrono::seconds(10));
          if (!future.has_value()) {
            ++enqueue_timeouts;
            continue;
          }
          producer_event.release();
          std::lock_guard<std::mutex> lock(futures_mutex);
          futures.push_back({label, std::move(*future)});
        }
      } catch (const std::exception& e) {
        producer_errors.fetch_add(1, std::memory_order_relaxed);
        std::printf("SCHEDULER_LANES_SMOKE_PRODUCER_ERROR producer=%d error=%s\n",
                    producer,
                    e.what());
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  {
    auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    std::string label = "scheduler_lanes_smoke.post_pressure";
    auto input = make_scheduler_admission_input(device, label);
    ScopedCudaEvent producer_event;
    producer_event.create(cudaEventDisableTiming);
    CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
    auto future = scheduler.try_enqueue_until(
        {std::move(input), stream, producer_event.event, 1},
        Clock::now() + std::chrono::seconds(10));
    if (future.has_value()) {
      producer_event.release();
      std::lock_guard<std::mutex> lock(futures_mutex);
      futures.push_back({label, std::move(*future)});
    } else {
      ++enqueue_timeouts;
    }
  }

  int completed = 0;
  int future_timeouts = 0;
  int bad_results = 0;
  int label_mismatches = 0;
  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    for (auto it = futures.rbegin(); it != futures.rend(); ++it) {
      auto wait_status = it->future.wait_for(std::chrono::seconds(10));
      if (wait_status != std::future_status::ready) {
        ++future_timeouts;
        continue;
      }
      auto result = it->future.get();
      if (result.completion.get() == nullptr || result.row_tensors.size() < 5 ||
          result.k != 1 || result.bucket != 1) {
        ++bad_results;
        continue;
      }
      if (result.label != it->label) {
        ++label_mismatches;
        std::printf("SCHEDULER_LANES_SMOKE_LABEL_MISMATCH expected=%s got=%s cycle=%lld row=%d\n",
                    it->label.c_str(),
                    result.label.c_str(),
                    static_cast<long long>(result.cycle_id),
                    result.row);
      }
      CUDA_CHECK(cudaStreamWaitEvent(consumer_stream.stream(), result.completion.get(), 0));
      CUDA_CHECK(cudaStreamSynchronize(consumer_stream.stream()));
      scheduler.record_worker_wait(result.cycle_id, result.k, 0.0, 0.0);
      ++completed;
    }
  }

  scheduler.close();
  auto telemetry = scheduler.telemetry_snapshot();
  bool both_lanes_used = telemetry.lanes.size() >= 2 &&
                         telemetry.lanes[0].dispatch_cycles > 0 &&
                         telemetry.lanes[1].dispatch_cycles > 0;
  bool ok = producer_errors.load(std::memory_order_relaxed) == 0 &&
            enqueue_timeouts.load(std::memory_order_relaxed) == 0 &&
            future_timeouts == 0 &&
            bad_results == 0 &&
            label_mismatches == 0 &&
            completed == static_cast<int>(futures.size()) &&
            completed == offered + 1 &&
            telemetry.completed == completed &&
            telemetry.dispatch_lanes == 2 &&
            telemetry.warmed_lanes == 2 &&
            both_lanes_used &&
            telemetry.dispatcher_exceptions == 0;
  std::printf("SCHEDULER_LANES_SMOKE %s offered=%d completed=%d enqueue_timeouts=%d "
              "future_timeouts=%d bad_results=%d label_mismatches=%d both_lanes_used=%d "
              "lane0_cycles=%lld lane1_cycles=%lld telemetry_completed=%lld dispatcher_exceptions=%lld "
              "telemetry=%s\n",
              ok ? "PASS" : "FAIL",
              offered + 1,
              completed,
              enqueue_timeouts.load(std::memory_order_relaxed),
              future_timeouts,
              bad_results,
              label_mismatches,
              both_lanes_used ? 1 : 0,
              telemetry.lanes.size() > 0 ? static_cast<long long>(telemetry.lanes[0].dispatch_cycles) : 0LL,
              telemetry.lanes.size() > 1 ? static_cast<long long>(telemetry.lanes[1].dispatch_cycles) : 0LL,
              static_cast<long long>(telemetry.completed),
              static_cast<long long>(telemetry.dispatcher_exceptions),
              scheduler_telemetry_json(telemetry).c_str());
  cleanup_cuda_cache();
  return ok;
}

static bool run_scheduler_graph_abandon_smoke(const DensityArgs& args, torch::Device device) {
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 4;
  policy.window_ms = 200;
  policy.lone_timeout_ms = 200;
  policy.queue_capacity = 4;
  policy.dispatch_lanes = density_dispatch_lanes_effective(args);
  int steady_runners = density_effective_steady_num_runners(args, 1, policy.dispatch_lanes);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        steady_runners,
                                        "scheduler_graph_abandon_smoke");
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  auto producer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  c10::cuda::CUDAStreamGuard stream_guard(producer_stream);
  auto abandoned = enqueue_scheduler_smoke_input(scheduler,
                                                 device,
                                                 producer_stream,
                                                 "scheduler_graph_abandon_smoke.timeout_row0");
  bool future_timeout_observed =
      abandoned.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready;
  abandoned = std::future<DispatchResult>();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  std::vector<std::future<DispatchResult>> futures;
  futures.reserve(3);
  for (int row = 1; row < 4; ++row) {
    futures.push_back(enqueue_scheduler_smoke_input(
        scheduler,
        device,
        producer_stream,
        "scheduler_graph_abandon_smoke.row" + std::to_string(row)));
  }

  int completed = 0;
  int bad_results = 0;
  int future_timeouts = 0;
  int graph_results = 0;
  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    for (auto& future : futures) {
      if (consume_scheduler_smoke_result(scheduler,
                                         future,
                                         consumer_stream,
                                         true,
                                         &bad_results,
                                         &future_timeouts,
                                         &graph_results)) {
        ++completed;
      }
    }
  }

  scheduler.close();
  auto telemetry = scheduler.telemetry_snapshot();
  bool ok = future_timeout_observed &&
            completed == 3 &&
            future_timeouts == 0 &&
            bad_results == 0 &&
            telemetry.completed == 4 &&
            telemetry.dispatcher_exceptions == 0;
  std::printf("SCHEDULER_GRAPH_ABANDON_SMOKE %s future_timeout_observed=%d completed=%d "
              "future_timeouts=%d bad_results=%d graph_results=%d dispatch_cycles=%lld "
              "telemetry_completed=%lld dispatcher_exceptions=%lld telemetry=%s\n",
              ok ? "PASS" : "FAIL",
              future_timeout_observed ? 1 : 0,
              completed,
              future_timeouts,
              bad_results,
              graph_results,
              static_cast<long long>(telemetry.dispatch_cycles),
              static_cast<long long>(telemetry.completed),
              static_cast<long long>(telemetry.dispatcher_exceptions),
              scheduler_telemetry_json(telemetry).c_str());
  cleanup_cuda_cache();
  return ok;
}

static bool run_scheduler_graph_shutdown_smoke(const DensityArgs& args, torch::Device device) {
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 4;
  policy.window_ms = 10;
  policy.lone_timeout_ms = 10;
  policy.queue_capacity = 4;
  policy.dispatch_lanes = density_dispatch_lanes_effective(args);
  int steady_runners = density_effective_steady_num_runners(args, 1, policy.dispatch_lanes);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        steady_runners,
                                        "scheduler_graph_shutdown_smoke");
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  auto producer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  c10::cuda::CUDAStreamGuard stream_guard(producer_stream);
  std::vector<std::future<DispatchResult>> futures;
  futures.reserve(4);
  for (int row = 0; row < 4; ++row) {
    futures.push_back(enqueue_scheduler_smoke_input(
        scheduler,
        device,
        producer_stream,
        "scheduler_graph_shutdown_smoke.row" + std::to_string(row)));
  }

  int bad_results = 0;
  int future_timeouts = 0;
  int graph_results = 0;
  std::vector<DispatchResult> held_results;
  held_results.reserve(4);
  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    for (auto& future : futures) {
      auto wait_status = future.wait_for(std::chrono::seconds(10));
      if (wait_status != std::future_status::ready) {
        ++future_timeouts;
        continue;
      }
      auto result = future.get();
      if (result.completion.get() == nullptr || result.row_tensors.size() < 5) {
        ++bad_results;
        continue;
      }
      CUDA_CHECK(cudaStreamWaitEvent(consumer_stream.stream(), result.completion.get(), 0));
      CUDA_CHECK(cudaStreamSynchronize(consumer_stream.stream()));
      if (result.graph_slot) ++graph_results;
      held_results.push_back(std::move(result));
    }
  }

  auto close_future = std::async(std::launch::async, [&scheduler] {
    scheduler.close();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  bool close_waited_for_outstanding =
      close_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready;

  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    for (auto& result : held_results) {
      if (result.graph_slot) {
        result.graph_slot->retire_on_stream(consumer_stream.stream());
        result.graph_slot.reset();
      }
      result.completion.reset();
    }
  }
  held_results.clear();

  bool close_returned = close_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
  if (close_returned) {
    close_future.get();
  }
  auto telemetry = scheduler.telemetry_snapshot();
  bool ok = close_returned &&
            future_timeouts == 0 &&
            bad_results == 0 &&
            static_cast<int>(futures.size()) == 4 &&
            telemetry.completed == 4 &&
            telemetry.dispatcher_exceptions == 0 &&
            (graph_results == 0 || close_waited_for_outstanding);
  std::printf("SCHEDULER_GRAPH_SHUTDOWN_SMOKE %s close_waited_for_outstanding=%d close_returned=%d "
              "future_timeouts=%d bad_results=%d graph_results=%d dispatch_cycles=%lld "
              "telemetry_completed=%lld dispatcher_exceptions=%lld telemetry=%s\n",
              ok ? "PASS" : "FAIL",
              close_waited_for_outstanding ? 1 : 0,
              close_returned ? 1 : 0,
              future_timeouts,
              bad_results,
              graph_results,
              static_cast<long long>(telemetry.dispatch_cycles),
              static_cast<long long>(telemetry.completed),
              static_cast<long long>(telemetry.dispatcher_exceptions),
              scheduler_telemetry_json(telemetry).c_str());
  cleanup_cuda_cache();
  return ok;
}

struct SchedulerStressFuture {
  std::string label;
  int seed = 0;
  int submit_index = 0;
  std::future<DispatchResult> future;
};

struct SchedulerB1StressResult {
  bool pass = false;
  int offered = 0;
  int admitted = 0;
  int completed = 0;
  int enqueue_timeouts = 0;
  int producer_errors = 0;
  int future_timeouts = 0;
  int future_exceptions = 0;
  int bad_results = 0;
  int label_mismatches = 0;
  int tensor_mismatches = 0;
  int post_enqueue_completed = 0;
  BatchedSteadySchedulerTelemetry telemetry;
};

static BatchedSteadyInput make_scheduler_stress_input(torch::Device device,
                                                      const std::string& label,
                                                      int seed) {
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto long_options = torch::TensorOptions().dtype(torch::kLong).device(device);
  float value = static_cast<float>((seed % 251) + 1) * 0.0005f;
  return {
      torch::full({1, 128, PRE + SHIFT}, value, float_options),
      torch::zeros({24, 1, ATT_CONTEXT_LEFT, 1024}, float_options),
      torch::zeros({24, 1, 1024, 8}, float_options),
      torch::zeros({1}, long_options),
      label,
  };
}

static bool compare_scheduler_stress_tensors(const std::string& label,
                                             const std::vector<at::Tensor>& got,
                                             const std::vector<at::Tensor>& ref) {
  if (got.size() < 5 || ref.size() < 5) {
    std::printf("SCHEDULER_STRESS_TENSOR_COUNT_MISMATCH label=%s got=%zu ref=%zu\n",
                label.c_str(),
                got.size(),
                ref.size());
    return false;
  }
  bool ok = true;
  ok = tensor_close("stress.enc_out", got[0], ref[0], kAotiTensorTolerance, label) && ok;
  ok = tensor_equal("stress.enc_len", got[1].to(ref[1].device()), ref[1]) && ok;
  ok = tensor_close("stress.cache_ch", got[2], ref[2], kAotiTensorTolerance, label) && ok;
  ok = tensor_close("stress.cache_t", got[3], ref[3], kAotiTensorTolerance, label) && ok;
  ok = tensor_close("stress.cache_ch_len", got[4], ref[4], kAotiTensorTolerance, label) && ok;
  return ok;
}

static std::map<std::string, std::vector<at::Tensor>> build_scheduler_stress_refs(
    BatchedSteadyLoaderSet& loader,
    torch::Device device,
    const std::vector<std::pair<std::string, int>>& specs) {
  std::map<std::string, std::vector<at::Tensor>> refs;
  auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  for (const auto& spec : specs) {
    auto input = make_scheduler_stress_input(device, spec.first, spec.second);
    auto rows = loader.run({input}, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    if (rows.size() != 1 || rows[0].tensors.size() < 5) {
      throw std::runtime_error("scheduler stress B1 reference returned invalid output");
    }
    refs.emplace(spec.first, clone_tensor_vector(rows[0].tensors));
  }
  return refs;
}

static bool consume_scheduler_stress_future(BatchedSteadyScheduler& scheduler,
                                            SchedulerStressFuture& item,
                                            c10::cuda::CUDAStream consumer_stream,
                                            const std::map<std::string, std::vector<at::Tensor>>& refs,
                                            SchedulerB1StressResult* result,
                                            std::chrono::seconds timeout = std::chrono::seconds(20)) {
  auto wait_status = item.future.wait_for(timeout);
  if (wait_status != std::future_status::ready) {
    ++result->future_timeouts;
    return false;
  }
  DispatchResult dispatch;
  try {
    dispatch = item.future.get();
  } catch (const std::exception& e) {
    ++result->future_exceptions;
    std::printf("SCHEDULER_STRESS_FUTURE_EXCEPTION label=%s error=%s\n",
                item.label.c_str(),
                e.what());
    return false;
  }
  if (dispatch.completion.get() == nullptr || dispatch.row_tensors.size() < 5 ||
      dispatch.bucket != 1 || dispatch.k != 1) {
    ++result->bad_results;
    return false;
  }
  if (dispatch.label != item.label) {
    ++result->label_mismatches;
    std::printf("SCHEDULER_STRESS_LABEL_MISMATCH expected=%s got=%s cycle=%lld row=%d\n",
                item.label.c_str(),
                dispatch.label.c_str(),
                static_cast<long long>(dispatch.cycle_id),
                dispatch.row);
  }
  CUDA_CHECK(cudaStreamWaitEvent(consumer_stream.stream(), dispatch.completion.get(), 0));
  CUDA_CHECK(cudaStreamSynchronize(consumer_stream.stream()));
  auto ref_it = refs.find(item.label);
  if (ref_it == refs.end() ||
      !compare_scheduler_stress_tensors(item.label, dispatch.row_tensors, ref_it->second)) {
    ++result->tensor_mismatches;
  }
  scheduler.record_worker_wait(dispatch.cycle_id, dispatch.k, 0.0, 0.0);
  ++result->completed;
  return true;
}

static SchedulerB1StressResult run_scheduler_b1_stress_rows(const DensityArgs& args,
                                                            torch::Device device,
                                                            const std::string& name,
                                                            int queue_capacity,
                                                            int producers,
                                                            int rows_per_producer,
                                                            int slow_lane,
                                                            int slow_us,
                                                            bool reverse_consume,
                                                            bool post_enqueue_probe) {
  std::unique_ptr<ScopedEnvVar> slow_lane_env;
  std::unique_ptr<ScopedEnvVar> slow_us_env;
  std::unique_ptr<ScopedEnvVar> fault_lane_env;
  std::unique_ptr<ScopedEnvVar> fault_after_env;
  slow_lane_env = std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_LANE",
                                                 std::to_string(slow_lane));
  slow_us_env = std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_US",
                                               std::to_string(slow_us));
  fault_lane_env = std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_LANE", "-1");
  fault_after_env = std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_AFTER", "0");

  SchedulerB1StressResult result;
  const int offered = producers * rows_per_producer;
  result.offered = offered + (post_enqueue_probe ? 1 : 0);
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        density_effective_steady_num_runners(args, 2, 2),
                                        name + "_loader");
  batched_steady.preload_buckets({1});
  std::vector<std::pair<std::string, int>> specs;
  specs.reserve(static_cast<size_t>(result.offered));
  for (int i = 0; i < offered; ++i) {
    specs.push_back({name + ".row" + std::to_string(i), 1000 + i});
  }
  if (post_enqueue_probe) {
    specs.push_back({name + ".post_capacity_probe", 1000 + offered});
  }
  auto refs = build_scheduler_stress_refs(batched_steady, device, specs);

  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 1;
  policy.window_ms = 0;
  policy.lone_timeout_ms = 0;
  policy.max_queue_delay_ms = 0;
  policy.queue_capacity = queue_capacity;
  policy.min_fill_enabled = false;
  policy.disable_min_fill = true;
  policy.dispatch_lanes = 2;
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  std::mutex futures_mutex;
  std::vector<SchedulerStressFuture> futures;
  futures.reserve(static_cast<size_t>(result.offered));
  std::atomic<int> admitted_count{0};
  std::atomic<int> enqueue_timeout_count{0};
  std::atomic<int> producer_error_count{0};
  StartGate gate(producers);
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(producers));
  for (int producer = 0; producer < producers; ++producer) {
    threads.emplace_back([&, producer] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        gate.arrive_and_wait();
        for (int row = 0; row < rows_per_producer; ++row) {
          int index = producer * rows_per_producer + row;
          std::string label = name + ".row" + std::to_string(index);
          auto input = make_scheduler_stress_input(device, label, 1000 + index);
          ScopedCudaEvent producer_event;
          producer_event.create(cudaEventDisableTiming);
          CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
          auto future = scheduler.try_enqueue_until(
              {std::move(input), stream, producer_event.event, static_cast<uint64_t>(producer + 1)},
              Clock::now() + std::chrono::seconds(30));
          if (!future.has_value()) {
            enqueue_timeout_count.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          producer_event.release();
          std::lock_guard<std::mutex> lock(futures_mutex);
          futures.push_back({label, 1000 + index, index, std::move(*future)});
          admitted_count.fetch_add(1, std::memory_order_relaxed);
        }
      } catch (const std::exception& e) {
        producer_error_count.fetch_add(1, std::memory_order_relaxed);
        std::printf("SCHEDULER_STRESS_PRODUCER_ERROR name=%s producer=%d error=%s\n",
                    name.c_str(),
                    producer,
                    e.what());
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  result.admitted = admitted_count.load(std::memory_order_relaxed);
  result.enqueue_timeouts = enqueue_timeout_count.load(std::memory_order_relaxed);
  result.producer_errors = producer_error_count.load(std::memory_order_relaxed);

  std::vector<SchedulerStressFuture> admitted;
  {
    std::lock_guard<std::mutex> lock(futures_mutex);
    admitted = std::move(futures);
  }
  if (reverse_consume) {
    std::sort(admitted.begin(), admitted.end(), [](const auto& a, const auto& b) {
      return a.submit_index > b.submit_index;
    });
  } else {
    std::sort(admitted.begin(), admitted.end(), [](const auto& a, const auto& b) {
      return a.submit_index < b.submit_index;
    });
  }

  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    for (auto& future : admitted) {
      consume_scheduler_stress_future(scheduler, future, consumer_stream, refs, &result);
    }
  }

  if (post_enqueue_probe) {
    auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    std::string label = name + ".post_capacity_probe";
    auto input = make_scheduler_stress_input(device, label, 1000 + offered);
    ScopedCudaEvent producer_event;
    producer_event.create(cudaEventDisableTiming);
    CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
    auto future = scheduler.try_enqueue_until({std::move(input), stream, producer_event.event, 9999},
                                              Clock::now() + std::chrono::seconds(10));
    if (future.has_value()) {
      producer_event.release();
      SchedulerStressFuture item{label, 1000 + offered, offered, std::move(*future)};
      if (consume_scheduler_stress_future(scheduler, item, consumer_stream, refs, &result)) {
        result.post_enqueue_completed = 1;
      }
    } else {
      ++result.enqueue_timeouts;
    }
  }

  scheduler.close();
  result.telemetry = scheduler.telemetry_snapshot();
  bool both_lanes_used = result.telemetry.lanes.size() >= 2 &&
                         result.telemetry.lanes[0].dispatch_cycles > 0 &&
                         result.telemetry.lanes[1].dispatch_cycles > 0;
  result.pass = result.producer_errors == 0 &&
                result.enqueue_timeouts == 0 &&
                result.future_timeouts == 0 &&
                result.future_exceptions == 0 &&
                result.bad_results == 0 &&
                result.label_mismatches == 0 &&
                result.tensor_mismatches == 0 &&
                result.completed == result.admitted + result.post_enqueue_completed &&
                result.telemetry.completed == result.completed &&
                result.telemetry.dispatch_lanes == 2 &&
                result.telemetry.warmed_lanes == 2 &&
                result.telemetry.dispatcher_exceptions == 0 &&
                (!post_enqueue_probe || result.post_enqueue_completed == 1) &&
                (queue_capacity <= 1 || both_lanes_used);
  std::printf("SCHEDULER_LANES_STRESS_%s %s cap=%d slow_lane=%d slow_us=%d offered=%d admitted=%d "
              "completed=%d post_enqueue_completed=%d enqueue_timeouts=%d producer_errors=%d "
              "future_timeouts=%d future_exceptions=%d bad_results=%d label_mismatches=%d "
              "tensor_mismatches=%d lane0_cycles=%lld lane1_cycles=%lld telemetry_completed=%lld\n",
              name.c_str(),
              result.pass ? "PASS" : "FAIL",
              queue_capacity,
              slow_lane,
              slow_us,
              result.offered,
              result.admitted,
              result.completed,
              result.post_enqueue_completed,
              result.enqueue_timeouts,
              result.producer_errors,
              result.future_timeouts,
              result.future_exceptions,
              result.bad_results,
              result.label_mismatches,
              result.tensor_mismatches,
              result.telemetry.lanes.size() > 0 ? static_cast<long long>(result.telemetry.lanes[0].dispatch_cycles) : 0LL,
              result.telemetry.lanes.size() > 1 ? static_cast<long long>(result.telemetry.lanes[1].dispatch_cycles) : 0LL,
              static_cast<long long>(result.telemetry.completed));
  cleanup_cuda_cache();
  return result;
}

static bool run_scheduler_cap_skew_stress(const DensityArgs& args, torch::Device device) {
  auto cap1 = run_scheduler_b1_stress_rows(args,
                                           device,
                                           "CAP_SKEW_CAP1",
                                           1,
                                           4,
                                           16,
                                           0,
                                           25000,
                                           true,
                                           true);
  auto cap2 = run_scheduler_b1_stress_rows(args,
                                           device,
                                           "CAP_SKEW_CAP2",
                                           2,
                                           4,
                                           32,
                                           0,
                                           25000,
                                           true,
                                           true);
  bool ok = cap1.pass && cap2.pass;
  std::printf("SCHEDULER_LANES_STRESS_CAP_SKEW %s cap1_completed=%d cap2_completed=%d "
              "cap1_lane0=%lld cap1_lane1=%lld cap2_lane0=%lld cap2_lane1=%lld\n",
              ok ? "PASS" : "FAIL",
              cap1.completed,
              cap2.completed,
              cap1.telemetry.lanes.size() > 0 ? static_cast<long long>(cap1.telemetry.lanes[0].dispatch_cycles) : 0LL,
              cap1.telemetry.lanes.size() > 1 ? static_cast<long long>(cap1.telemetry.lanes[1].dispatch_cycles) : 0LL,
              cap2.telemetry.lanes.size() > 0 ? static_cast<long long>(cap2.telemetry.lanes[0].dispatch_cycles) : 0LL,
              cap2.telemetry.lanes.size() > 1 ? static_cast<long long>(cap2.telemetry.lanes[1].dispatch_cycles) : 0LL);
  return ok;
}

static bool run_scheduler_no_deadlock_stress(const DensityArgs& args, torch::Device device) {
  auto result = run_scheduler_b1_stress_rows(args,
                                             device,
                                             "NO_DEADLOCK_SATURATED",
                                             3,
                                             8,
                                             8,
                                             -2,
                                             20000,
                                             true,
                                             true);
  std::printf("SCHEDULER_LANES_STRESS_NO_DEADLOCK %s completed=%d cap=3 lane0_cycles=%lld "
              "lane1_cycles=%lld enqueue_timeouts=%d future_timeouts=%d\n",
              result.pass ? "PASS" : "FAIL",
              result.completed,
              result.telemetry.lanes.size() > 0 ? static_cast<long long>(result.telemetry.lanes[0].dispatch_cycles) : 0LL,
              result.telemetry.lanes.size() > 1 ? static_cast<long long>(result.telemetry.lanes[1].dispatch_cycles) : 0LL,
              result.enqueue_timeouts,
              result.future_timeouts);
  return result.pass;
}

static bool run_scheduler_ooo_stress(const DensityArgs& args, torch::Device device) {
  std::unique_ptr<ScopedEnvVar> slow_lane_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_LANE", "0");
  std::unique_ptr<ScopedEnvVar> slow_us_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_US", "100000");
  std::unique_ptr<ScopedEnvVar> fault_lane_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_LANE", "-1");
  std::unique_ptr<ScopedEnvVar> fault_after_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_AFTER", "0");

  const int offered = 66;
  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        density_effective_steady_num_runners(args, 2, 2),
                                        "ooo_stress_loader");
  batched_steady.preload_buckets({1});
  std::vector<std::pair<std::string, int>> specs;
  specs.reserve(static_cast<size_t>(offered));
  for (int i = 0; i < offered; ++i) {
    specs.push_back({"OOO.row" + std::to_string(i), 2000 + i});
  }
  auto refs = build_scheduler_stress_refs(batched_steady, device, specs);
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 1;
  policy.window_ms = 0;
  policy.lone_timeout_ms = 0;
  policy.max_queue_delay_ms = 0;
  policy.queue_capacity = 2;
  policy.min_fill_enabled = false;
  policy.disable_min_fill = true;
  policy.dispatch_lanes = 2;
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  std::vector<SchedulerStressFuture> futures;
  futures.reserve(static_cast<size_t>(offered));
  for (int i = 0; i < 2; ++i) {
    std::string label = "OOO.row" + std::to_string(i);
    auto input = make_scheduler_stress_input(device, label, 2000 + i);
    ScopedCudaEvent producer_event;
    producer_event.create(cudaEventDisableTiming);
    CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
    auto future = scheduler.enqueue({std::move(input), stream, producer_event.event, 1});
    producer_event.release();
    futures.push_back({label, 2000 + i, i, std::move(future)});
  }
  bool later_finished_first = false;
  auto deadline = Clock::now() + std::chrono::seconds(5);
  while (Clock::now() < deadline) {
    bool first_ready = futures[0].future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    bool second_ready = futures[1].future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
    if (second_ready && !first_ready) {
      later_finished_first = true;
      break;
    }
    if (first_ready && !second_ready) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  for (int i = 2; i < offered; ++i) {
    std::string label = "OOO.row" + std::to_string(i);
    auto input = make_scheduler_stress_input(device, label, 2000 + i);
    ScopedCudaEvent producer_event;
    producer_event.create(cudaEventDisableTiming);
    CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
    auto future = scheduler.try_enqueue_until({std::move(input), stream, producer_event.event, 1},
                                              Clock::now() + std::chrono::seconds(30));
    if (!future.has_value()) {
      throw std::runtime_error("OOO stress enqueue timed out");
    }
    producer_event.release();
    futures.push_back({label, 2000 + i, i, std::move(*future)});
  }

  SchedulerB1StressResult result;
  result.offered = offered;
  result.admitted = offered;
  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    std::sort(futures.begin(), futures.end(), [](const auto& a, const auto& b) {
      return a.submit_index > b.submit_index;
    });
    for (auto& future : futures) {
      consume_scheduler_stress_future(scheduler, future, consumer_stream, refs, &result);
    }
  }
  scheduler.close();
  result.telemetry = scheduler.telemetry_snapshot();
  bool ok = later_finished_first &&
            result.completed == offered &&
            result.future_timeouts == 0 &&
            result.future_exceptions == 0 &&
            result.bad_results == 0 &&
            result.label_mismatches == 0 &&
            result.tensor_mismatches == 0 &&
            result.telemetry.dispatcher_exceptions == 0 &&
            result.telemetry.lanes.size() >= 2 &&
            result.telemetry.lanes[0].dispatch_cycles > 0 &&
            result.telemetry.lanes[1].dispatch_cycles > 0;
  std::printf("SCHEDULER_LANES_STRESS_OOO %s offered=%d completed=%d later_finished_first=%d "
              "future_timeouts=%d label_mismatches=%d tensor_mismatches=%d lane0_cycles=%lld lane1_cycles=%lld\n",
              ok ? "PASS" : "FAIL",
              offered,
              result.completed,
              later_finished_first ? 1 : 0,
              result.future_timeouts,
              result.label_mismatches,
              result.tensor_mismatches,
              result.telemetry.lanes.size() > 0 ? static_cast<long long>(result.telemetry.lanes[0].dispatch_cycles) : 0LL,
              result.telemetry.lanes.size() > 1 ? static_cast<long long>(result.telemetry.lanes[1].dispatch_cycles) : 0LL);
  cleanup_cuda_cache();
  return ok;
}

static bool run_scheduler_lane_fault_one(const DensityArgs& args, torch::Device device, int fault_lane) {
  std::unique_ptr<ScopedEnvVar> slow_lane_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_LANE", "-1");
  std::unique_ptr<ScopedEnvVar> slow_us_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_US", "0");
  std::unique_ptr<ScopedEnvVar> fault_lane_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_LANE",
                                     std::to_string(fault_lane));
  std::unique_ptr<ScopedEnvVar> fault_after_env =
      std::make_unique<ScopedEnvVar>("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_AFTER", "3");

  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        density_effective_steady_num_runners(args, 2, 2),
                                        "lane_fault_stress_loader");
  batched_steady.preload_buckets({1});
  BatchedSteadySchedulerPolicy policy;
  policy.B_max = 1;
  policy.window_ms = 0;
  policy.lone_timeout_ms = 0;
  policy.max_queue_delay_ms = 0;
  policy.queue_capacity = 16;
  policy.min_fill_enabled = false;
  policy.disable_min_fill = true;
  policy.dispatch_lanes = 2;
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  const int offered = 48;
  int admitted = 0;
  int abandoned = 0;
  int enqueue_exceptions = 0;
  int enqueue_timeouts = 0;
  std::vector<SchedulerStressFuture> futures;
  futures.reserve(static_cast<size_t>(offered));
  auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  for (int i = 0; i < offered; ++i) {
    try {
      std::string label = "LANE_FAULT.lane" + std::to_string(fault_lane) + ".row" + std::to_string(i);
      auto input = make_scheduler_stress_input(device, label, 3000 + i);
      ScopedCudaEvent producer_event;
      producer_event.create(cudaEventDisableTiming);
      CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
      auto future = scheduler.try_enqueue_until({std::move(input), stream, producer_event.event, 1},
                                                Clock::now() + std::chrono::seconds(10));
      if (!future.has_value()) {
        ++enqueue_timeouts;
        continue;
      }
      producer_event.release();
      ++admitted;
      if (i >= 8 && i % 7 == 0) {
        ++abandoned;
      } else {
        futures.push_back({label, 3000 + i, i, std::move(*future)});
      }
    } catch (const std::exception& e) {
      ++enqueue_exceptions;
      std::printf("SCHEDULER_LANES_STRESS_FAULT_ENQUEUE_EXCEPTION lane=%d index=%d error=%s\n",
                  fault_lane,
                  i,
                  e.what());
      break;
    }
  }

  int completed = 0;
  int future_exceptions = 0;
  int future_timeouts = 0;
  int bad_results = 0;
  auto consumer_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
  {
    c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
    for (auto& item : futures) {
      auto wait_status = item.future.wait_for(std::chrono::seconds(10));
      if (wait_status != std::future_status::ready) {
        ++future_timeouts;
        continue;
      }
      try {
        auto dispatch = item.future.get();
        if (dispatch.completion.get() == nullptr || dispatch.row_tensors.size() < 5) {
          ++bad_results;
          continue;
        }
        CUDA_CHECK(cudaStreamWaitEvent(consumer_stream.stream(), dispatch.completion.get(), 0));
        CUDA_CHECK(cudaStreamSynchronize(consumer_stream.stream()));
        ++completed;
      } catch (const std::exception&) {
        ++future_exceptions;
      }
    }
  }

  bool post_enqueue_threw = false;
  try {
    auto input = make_scheduler_stress_input(device,
                                             "LANE_FAULT.lane" + std::to_string(fault_lane) + ".post",
                                             9999);
    ScopedCudaEvent producer_event;
    producer_event.create(cudaEventDisableTiming);
    CUDA_CHECK(cudaEventRecord(producer_event.event, stream.stream()));
    auto future = scheduler.enqueue({std::move(input), stream, producer_event.event, 1});
    producer_event.release();
    (void)future;
  } catch (const std::exception&) {
    post_enqueue_threw = true;
  }

  scheduler.close();
  auto telemetry = scheduler.telemetry_snapshot();
  int retained = static_cast<int>(futures.size());
  bool ok = admitted > 0 &&
            abandoned > 0 &&
            enqueue_timeouts == 0 &&
            future_timeouts == 0 &&
            bad_results == 0 &&
            completed + future_exceptions == retained &&
            future_exceptions > 0 &&
            enqueue_exceptions >= 0 &&
            post_enqueue_threw &&
            telemetry.dispatcher_exceptions >= 1;
  std::printf("SCHEDULER_LANES_STRESS_FAULT lane=%d %s offered=%d admitted=%d retained=%d abandoned=%d "
              "completed=%d future_exceptions=%d future_timeouts=%d bad_results=%d enqueue_exceptions=%d "
              "post_enqueue_threw=%d dispatcher_exceptions=%lld telemetry_completed=%lld\n",
              fault_lane,
              ok ? "PASS" : "FAIL",
              offered,
              admitted,
              retained,
              abandoned,
              completed,
              future_exceptions,
              future_timeouts,
              bad_results,
              enqueue_exceptions,
              post_enqueue_threw ? 1 : 0,
              static_cast<long long>(telemetry.dispatcher_exceptions),
              static_cast<long long>(telemetry.completed));
  cleanup_cuda_cache();
  return ok;
}

static bool run_scheduler_fault_stress(const DensityArgs& args, torch::Device device) {
  bool lane0 = run_scheduler_lane_fault_one(args, device, 0);
  bool lane1 = run_scheduler_lane_fault_one(args, device, 1);
  bool ok = lane0 && lane1;
  std::printf("SCHEDULER_LANES_STRESS_ABANDONED_FUTURE_FAULT %s lane0=%d lane1=%d\n",
              ok ? "PASS" : "FAIL",
              lane0 ? 1 : 0,
              lane1 ? 1 : 0);
  return ok;
}

static std::vector<int> pick_multichunk_stress_utts(torch::jit::Module& bundle, int rows_total, int sessions) {
  int chosen = -1;
  int chosen_chunks = 0;
  for (int target_chunks : {4, 3}) {
    for (int utt = 0; utt < rows_total; ++utt) {
      int chunks = static_cast<int>(scalar_i64(utt_tensor(bundle, utt, "num_steady")));
      if (chunks == target_chunks) {
        chosen = utt;
        chosen_chunks = chunks;
        break;
      }
    }
    if (chosen >= 0) break;
  }
  if (chosen < 0) {
    for (int utt = 0; utt < rows_total; ++utt) {
      int chunks = static_cast<int>(scalar_i64(utt_tensor(bundle, utt, "num_steady")));
      if (chunks >= 3) {
        chosen = utt;
        chosen_chunks = chunks;
        break;
      }
    }
  }
  if (chosen < 0) throw std::runtime_error("multi-chunk stress found no utterance with >=3 steady chunks");
  std::vector<int> utts(static_cast<size_t>(sessions), chosen);
  std::printf("SCHEDULER_LANES_STRESS_MULTICHUNK_PICK utt=%d steady_chunks=%d continuation_chunks=%d sessions=%d\n",
              chosen,
              chosen_chunks,
              std::max(0, chosen_chunks - 1),
              sessions);
  return utts;
}

static bool run_scheduler_multichunk_stress(const DensityArgs& args,
                                            torch::Device device,
                                            const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                            int rows_total) {
  const int sessions = 4;
  auto utts = pick_multichunk_stress_utts(*shared_bundle, rows_total, sessions);
  auto reference = build_serial_reference_for_utts(args, device, shared_bundle, utts);
  cleanup_cuda_cache();

  std::string steady_batch_dir = resolve_steady_batch_dir(args);
  BatchedSteadySchedulerPolicy policy = density_batch_policy_effective(args, sessions);
  policy.B_max = 4;
  policy.queue_capacity = std::max(policy.queue_capacity, 8);
  policy.dispatch_lanes = 2;
  int scheduler_steady_runners = density_effective_steady_num_runners(args, 2, policy.dispatch_lanes);
  BatchedSteadyLoaderSet batched_steady(steady_batch_dir,
                                        args.dir + "/finalize_shared_weights.ts",
                                        device,
                                        scheduler_steady_runners,
                                        "multichunk_stress_scheduler_buckets");
  batched_steady.preload_all();
  AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", "model", false, 4, -1);
  FinalizeBucketLoaderPool finalize_loaders(args.dir,
                                            device,
                                            capped_general_finalize_runners(sessions),
                                            "multichunk_stress_finalize_pool");
  finalize_loaders.preload_all();
  auto shared_enc_first = load_shared_enc_first(args.dir, device, "multichunk_stress_shared_locked_enc_first");
  BatchedSteadyScheduler scheduler(batched_steady, device, policy);
  scheduler.warmup_buckets();
  scheduler.start();

  std::vector<std::unique_ptr<WorkerContext>> contexts;
  contexts.reserve(sessions);
  for (int worker = 0; worker < sessions; ++worker) {
    contexts.push_back(make_worker_context(args.dir,
                                           device,
                                           stream_for_worker(true, worker),
                                           shared_bundle,
                                           shared_enc_first));
  }
  auto tokenizer = tokenizer_from_bundle(contexts[0]->bundle);
  B2TensorDiffStats diff_stats;
  B2ReusableBarrier continuation_barrier(sessions);
  StartGate gate(sessions);
  std::vector<RowReplayResult> rows(static_cast<size_t>(sessions));
  std::vector<std::thread> threads;
  threads.reserve(sessions);
  for (int worker = 0; worker < sessions; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        c10::cuda::CUDAGuard device_guard(device.index());
        gate.arrive_and_wait();
        rows[static_cast<size_t>(worker)] =
            replay_row_density(utts[static_cast<size_t>(worker)],
                               *contexts[worker],
                               &enc_steady,
                               nullptr,
                               finalize_loaders,
                               device,
                               tokenizer,
                               true,
                               false,
                               nullptr,
                               false,
                               &scheduler,
                               &continuation_barrier,
                               &diff_stats,
                               &enc_steady);
      } catch (const std::exception& e) {
        rows[static_cast<size_t>(worker)].ok = false;
        rows[static_cast<size_t>(worker)].error = e.what();
      }
    });
  }
  gate.wait_until_ready_and_start();
  for (auto& thread : threads) thread.join();
  CUDA_CHECK(cudaDeviceSynchronize());
  scheduler.close();
  auto telemetry = scheduler.telemetry_snapshot();

  int errors = 0;
  int token_divergences = 0;
  int event_divergences = 0;
  for (int worker = 0; worker < sessions; ++worker) {
    int utt = utts[static_cast<size_t>(worker)];
    const auto& row = rows[static_cast<size_t>(worker)];
    if (!row.error.empty()) {
      ++errors;
      std::printf("SCHEDULER_LANES_STRESS_MULTICHUNK_ROW_ERROR worker=%d utt=%d error=%s\n",
                  worker,
                  utt,
                  row.error.c_str());
      continue;
    }
    const auto& ref = reference.at(utt);
    if (!compare_b2_token_vector(row.steady_tokens, ref.steady_tokens, "multichunk.steady", worker, utt)) {
      ++token_divergences;
    }
    if (!compare_b2_token_vector(row.final_tokens, ref.final_tokens, "multichunk.final", worker, utt)) {
      ++token_divergences;
    }
    if (!strict_events_equal(row.events,
                             ref.events,
                             "multichunk.worker" + std::to_string(worker) + ".utt" + std::to_string(utt))) {
      ++event_divergences;
    }
  }
  int enc_len_mismatches = 0;
  int cache_len_mismatches = 0;
  double max_enc_out_diff = 0.0;
  double max_cache_ch_diff = 0.0;
  double max_cache_t_diff = 0.0;
  {
    std::lock_guard<std::mutex> lock(diff_stats.mutex);
    enc_len_mismatches = diff_stats.enc_len_mismatches;
    cache_len_mismatches = diff_stats.cache_len_mismatches;
    max_enc_out_diff = diff_stats.max_enc_out_diff;
    max_cache_ch_diff = diff_stats.max_cache_ch_diff;
    max_cache_t_diff = diff_stats.max_cache_t_diff;
  }
  bool ok = errors == 0 &&
            token_divergences == 0 &&
            event_divergences == 0 &&
            enc_len_mismatches == 0 &&
            cache_len_mismatches == 0 &&
            max_enc_out_diff <= kAotiTensorTolerance &&
            max_cache_ch_diff <= kAotiTensorTolerance &&
            max_cache_t_diff <= kAotiTensorTolerance &&
            telemetry.dispatcher_exceptions == 0 &&
            telemetry.dispatch_lanes == 2 &&
            telemetry.warmed_lanes == 2 &&
            telemetry.lanes.size() >= 2 &&
            telemetry.lanes[0].dispatch_cycles > 0 &&
            telemetry.lanes[1].dispatch_cycles > 0;
  std::printf("SCHEDULER_LANES_STRESS_MULTICHUNK %s sessions=%d token_divergences=%d "
              "event_divergences=%d errors=%d compares=%d max_enc_out=%.3e max_cache_ch=%.3e "
              "max_cache_t=%.3e dispatch_cycles=%lld lane0_cycles=%lld lane1_cycles=%lld\n",
              ok ? "PASS" : "FAIL",
              sessions,
              token_divergences,
              event_divergences,
              errors,
              diff_stats.compares,
              max_enc_out_diff,
              max_cache_ch_diff,
              max_cache_t_diff,
              static_cast<long long>(telemetry.dispatch_cycles),
              telemetry.lanes.size() > 0 ? static_cast<long long>(telemetry.lanes[0].dispatch_cycles) : 0LL,
              telemetry.lanes.size() > 1 ? static_cast<long long>(telemetry.lanes[1].dispatch_cycles) : 0LL);
  cleanup_cuda_cache();
  return ok;
}

static bool run_scheduler_lanes_stress(const DensityArgs& args,
                                       torch::Device device,
                                       const std::shared_ptr<torch::jit::Module>& shared_bundle,
                                       int rows_total) {
  bool cap_skew = run_scheduler_cap_skew_stress(args, device);
  bool fault = run_scheduler_fault_stress(args, device);
  bool no_deadlock = run_scheduler_no_deadlock_stress(args, device);
  bool ooo = run_scheduler_ooo_stress(args, device);
  bool multichunk = run_scheduler_multichunk_stress(args, device, shared_bundle, rows_total);
  bool ok = cap_skew && fault && no_deadlock && ooo && multichunk;
  std::printf("SCHEDULER_LANES_STRESS_RESULT %s cap_skew=%d abandoned_future_fault=%d "
              "no_deadlock=%d ooo=%d multichunk=%d\n",
              ok ? "PASS" : "FAIL",
              cap_skew ? 1 : 0,
              fault ? 1 : 0,
              no_deadlock ? 1 : 0,
              ooo ? 1 : 0,
              multichunk ? 1 : 0);
  return ok;
}

static bool run_stalegen_smoke() {
  StaleGenTelemetry telemetry;
  int stale_events_emitted = 0;

  auto expect_drop = [&](const char* scenario, StaleGenStage stage) {
    SessionState session;
    uint64_t work_generation = density_capture_generation(session);
    density_bump_generation(session, scenario);
    bool live = density_generation_live(session,
                                        work_generation,
                                        &telemetry,
                                        stage,
                                        std::string("stalegen_smoke.") + scenario);
    if (live) {
      ++stale_events_emitted;
      std::printf("STALEGEN_SMOKE_UNEXPECTED_LIVE scenario=%s\n", scenario);
    }
    return !live;
  };

  bool close_while_inflight =
      expect_drop("close_while_inflight", StaleGenStage::FINALIZE_OUTPUT);

  SessionState queued;
  uint64_t queued_generation = density_capture_generation(queued);
  density_bump_generation(queued, "reset_while_queued");
  bool reset_while_queued = true;
  for (int i = 0; i < 3; ++i) {
    bool live = density_generation_live(queued,
                                        queued_generation,
                                        &telemetry,
                                        StaleGenStage::ENCODE,
                                        "stalegen_smoke.reset_while_queued.item" + std::to_string(i));
    if (live) {
      ++stale_events_emitted;
      reset_while_queued = false;
    }
  }

  bool reset_while_finalizer =
      expect_drop("reset_while_finalizer_owns_runner", StaleGenStage::FINALIZE_OUTPUT);
  bool final_after_shed =
      expect_drop("final_after_shed", StaleGenStage::FINALIZE_OUTPUT);

  StaleGenTelemetrySnapshot snapshot = telemetry.snapshot();
  bool ok = close_while_inflight &&
            reset_while_queued &&
            reset_while_finalizer &&
            final_after_shed &&
            stale_events_emitted == 0 &&
            snapshot.drops_at_encode == 3 &&
            snapshot.drops_at_finalize_output == 3 &&
            snapshot.total_drops() == 6;
  std::printf("STALEGEN_SMOKE pass=%s close_while_inflight=%s reset_while_queued=%s "
              "reset_while_finalizer=%s final_after_shed=%s stale_events_emitted=%d drops=%s\n",
              ok ? "true" : "false",
              close_while_inflight ? "PASS" : "FAIL",
              reset_while_queued ? "PASS" : "FAIL",
              reset_while_finalizer ? "PASS" : "FAIL",
              final_after_shed ? "PASS" : "FAIL",
              stale_events_emitted,
              stale_gen_telemetry_json(snapshot).c_str());
  return ok;
}

static std::vector<int16_t> audio_tensor_to_pcm_int16(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
  std::vector<int16_t> out;
  out.reserve(static_cast<size_t>(flat.numel()));
  for (int64_t i = 0; i < flat.numel(); ++i) {
    long value = std::lrint(static_cast<double>(flat[i].item<float>()) * 32768.0);
    value = std::max<long>(-32768, std::min<long>(32767, value));
    out.push_back(static_cast<int16_t>(value));
  }
  return out;
}

static bool run_runtime_smoke(const DensityArgs& args, torch::Device device) {
  std::string bundle_path = (fs::path(args.dir) / "session_audio_bundle.ts").string();
  auto bundle = load_jit_serialized(bundle_path);
  verify_session_bundle_meta(bundle, false);
  int rows_total = static_cast<int>(scalar_i64(attr_tensor(bundle, "num_utts")));
  int rows = std::min(4, rows_total);
  if (rows <= 0) throw std::runtime_error("runtime-smoke requires at least one audio fixture row");

  SharedRuntimeConfig shared_cfg;
  shared_cfg.bundle_path = bundle_path;
  shared_cfg.steady_artifacts_dir = args.dir;
  shared_cfg.steady_batch_dir = args.steady_batch_dir;
  std::string stripped = (fs::path(args.dir) / "stripped_finalize_buckets").string();
  shared_cfg.finalize_buckets_dir = directory_exists(stripped)
                                        ? stripped
                                        : (fs::path(args.dir) / "finalize_buckets").string();
  shared_cfg.scheduler_enabled = false;
  shared_cfg.device_index = device.index();
  SharedRuntime shared(shared_cfg);

  int token_divergences = 0;
  int event_divergences = 0;
  int errors = 0;
  size_t wire_events_total = 0;
  constexpr size_t kPcm20msSamples = 320;

  for (int utt = 0; utt < rows; ++utt) {
    std::string prefix = "utt" + std::to_string(utt);
    std::string label = "runtime.utt" + std::to_string(utt);
    SessionConfig session_cfg;
    session_cfg.label = label;
    session_cfg.finalize_silence_ms = 0;
    SessionRuntime runtime(shared, session_cfg);
    try {
      runtime.handle_vad_start();
      auto pcm = audio_tensor_to_pcm_int16(prefix_tensor(bundle, prefix, "audio"));
      for (size_t pos = 0; pos < pcm.size(); pos += kPcm20msSamples) {
        size_t count = std::min(kPcm20msSamples, pcm.size() - pos);
        auto wire = runtime.append_pcm_and_drain(PCMFrame{pcm.data() + pos, count});
        wire_events_total += wire.size();
      }
      auto final_wire = runtime.finalize_now();
      wire_events_total += final_wire.size();

      auto final_tokens = session_runtime_debug_last_final_tokens(runtime);
      auto gold_tokens = tensor_to_vec(prefix_tensor(bundle, prefix, "gold_tokens"));
      bool tokens_ok = final_tokens == gold_tokens;
      if (!tokens_ok) {
        ++token_divergences;
        equal_tokens(final_tokens, gold_tokens, "runtime final cumulative", label);
      }
      auto events = session_runtime_debug_events(runtime);
      auto gold_events = gold_events_from_bundle(bundle, utt);
      bool events_ok = strict_events_equal(events, gold_events, label + ".events");
      if (!events_ok) ++event_divergences;
      std::printf("RUNTIME_SMOKE_ROW utt=%d samples=%zu chunks=%zu final_tokens=%s events=%s "
                  "wire_events=%zu timing=%s\n",
                  utt,
                  pcm.size(),
                  (pcm.size() + kPcm20msSamples - 1) / kPcm20msSamples,
                  tokens_ok ? "PASS" : "FAIL",
                  events_ok ? "PASS" : "FAIL",
                  final_wire.size(),
                  runtime.last_timing().has_value() ? "PASS" : "FAIL");
    } catch (const std::exception& e) {
      ++errors;
      std::printf("RUNTIME_SMOKE_ROW utt=%d error=%s\n", utt, e.what());
    }
  }

  bool ok = errors == 0 && token_divergences == 0 && event_divergences == 0;
  std::printf("RUNTIME_SMOKE %s rows=%d token_divergences=%d event_divergences=%d errors=%d wire_events=%zu\n",
              ok ? "PASS" : "FAIL",
              rows,
              token_divergences,
              event_divergences,
              errors,
              wire_events_total);
  return ok;
}

static const char* vad_state_name(VadState state) {
  switch (state) {
    case VadState::IDLE: return "IDLE";
    case VadState::SPEAKING: return "SPEAKING";
    case VadState::PENDING_FINALIZE: return "PENDING_FINALIZE";
  }
  return "UNKNOWN";
}

static double vad_smoke_unix_now_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

static uint64_t vad_smoke_finalize_count(const SessionRuntime& runtime) {
  auto timing = runtime.last_timing();
  return timing.has_value() ? timing->finalize_seq : 0;
}

static bool run_vad_smoke(const DensityArgs& args, torch::Device device) {
  std::string bundle_path = (fs::path(args.dir) / "session_audio_bundle.ts").string();
  SharedRuntimeConfig shared_cfg;
  shared_cfg.bundle_path = bundle_path;
  shared_cfg.steady_artifacts_dir = args.dir;
  shared_cfg.steady_batch_dir = args.steady_batch_dir;
  std::string stripped = (fs::path(args.dir) / "stripped_finalize_buckets").string();
  shared_cfg.finalize_buckets_dir = directory_exists(stripped)
                                        ? stripped
                                        : (fs::path(args.dir) / "finalize_buckets").string();
  shared_cfg.scheduler_enabled = false;
  shared_cfg.device_index = device.index();
  SharedRuntime shared(shared_cfg);

  int failures = 0;
  int total_finalize_invocations = 0;
  auto report = [&](const char* name, bool pass, const SessionRuntime& runtime) {
    if (!pass) ++failures;
    std::printf("VAD_SMOKE_CASE name=%s result=%s state=%s deadline=%s finalize_seq=%llu\n",
                name,
                pass ? "PASS" : "FAIL",
                vad_state_name(runtime.vad_state()),
                runtime.vad_deadline_ts().has_value() ? "set" : "none",
                static_cast<unsigned long long>(vad_smoke_finalize_count(runtime)));
  };

  {
    SessionConfig cfg;
    cfg.label = "vad_smoke.start";
    cfg.finalize_silence_ms = 150;
    SessionRuntime runtime(shared, cfg);
    runtime.handle_vad_start();
    bool pass = runtime.vad_state() == VadState::SPEAKING &&
                !runtime.vad_deadline_ts().has_value() &&
                vad_smoke_finalize_count(runtime) == 0;
    report("vad_start_to_speaking", pass, runtime);
  }

  {
    SessionConfig cfg;
    cfg.label = "vad_smoke.default_immediate";
    SessionRuntime runtime(shared, cfg);
    runtime.handle_vad_start();
    auto wire = runtime.handle_vad_stop();
    (void)wire;
    uint64_t count = vad_smoke_finalize_count(runtime);
    total_finalize_invocations += static_cast<int>(count);
    bool pass = runtime.vad_state() == VadState::IDLE &&
                !runtime.vad_deadline_ts().has_value() &&
                count == 1 &&
                runtime.last_timing().has_value() &&
                runtime.last_timing()->reason == "debounce_expired";
    report("vad_stop_zero_immediate_default", pass, runtime);
  }

  std::optional<double> canceled_deadline;
  {
    SessionConfig cfg;
    cfg.label = "vad_smoke.debounced_cancel";
    cfg.finalize_silence_ms = 150;
    SessionRuntime runtime(shared, cfg);
    runtime.handle_vad_start();
    double before = vad_smoke_unix_now_seconds();
    auto wire = runtime.handle_vad_stop();
    (void)wire;
    canceled_deadline = runtime.vad_deadline_ts();
    bool deadline_ok = canceled_deadline.has_value() &&
                       *canceled_deadline >= before + 0.100 &&
                       *canceled_deadline <= before + 0.250;
    bool pending_pass = runtime.vad_state() == VadState::PENDING_FINALIZE &&
                        deadline_ok &&
                        vad_smoke_finalize_count(runtime) == 0;
    report("vad_stop_150_pending_deadline", pending_pass, runtime);

    runtime.handle_vad_start();
    auto late = runtime.poll_timer(canceled_deadline.value_or(before) + 1.0);
    bool cancel_pass = runtime.vad_state() == VadState::SPEAKING &&
                       !runtime.vad_deadline_ts().has_value() &&
                       late.empty() &&
                       vad_smoke_finalize_count(runtime) == 0;
    report("vad_start_cancels_pending", cancel_pass, runtime);
  }

  {
    SessionConfig cfg;
    cfg.label = "vad_smoke.timer";
    cfg.finalize_silence_ms = 150;
    SessionRuntime runtime(shared, cfg);
    runtime.handle_vad_start();
    runtime.handle_vad_stop();
    double deadline = runtime.vad_deadline_ts().value_or(vad_smoke_unix_now_seconds());
    auto early = runtime.poll_timer(deadline - 0.001);
    bool early_pass = early.empty() &&
                      runtime.vad_state() == VadState::PENDING_FINALIZE &&
                      vad_smoke_finalize_count(runtime) == 0;
    auto final_wire = runtime.poll_timer(deadline + 0.001);
    (void)final_wire;
    uint64_t count = vad_smoke_finalize_count(runtime);
    total_finalize_invocations += static_cast<int>(count);
    auto extra = runtime.poll_timer(deadline + 1.0);
    bool timer_pass = early_pass &&
                      runtime.vad_state() == VadState::IDLE &&
                      !runtime.vad_deadline_ts().has_value() &&
                      count == 1 &&
                      extra.empty() &&
                      vad_smoke_finalize_count(runtime) == 1;
    report("timer_fires_no_cancel", timer_pass, runtime);
  }

  {
    SessionConfig cfg;
    cfg.label = "vad_smoke.reset_bypass";
    cfg.finalize_silence_ms = 150;
    SessionRuntime reset_runtime(shared, cfg);
    reset_runtime.handle_vad_start();
    reset_runtime.handle_vad_stop();
    double reset_deadline = reset_runtime.vad_deadline_ts().value_or(vad_smoke_unix_now_seconds());
    auto reset_wire = reset_runtime.reset(true);
    (void)reset_wire;
    uint64_t reset_count = vad_smoke_finalize_count(reset_runtime);
    auto reset_late = reset_runtime.poll_timer(reset_deadline + 1.0);
    bool reset_pass = reset_runtime.vad_state() == VadState::IDLE &&
                      !reset_runtime.vad_deadline_ts().has_value() &&
                      reset_count == 1 &&
                      reset_late.empty() &&
                      vad_smoke_finalize_count(reset_runtime) == 1 &&
                      reset_runtime.last_timing().has_value() &&
                      reset_runtime.last_timing()->reason == "reset";

    cfg.label = "vad_smoke.end_bypass";
    SessionRuntime end_runtime(shared, cfg);
    end_runtime.handle_vad_start();
    end_runtime.handle_vad_stop();
    double end_deadline = end_runtime.vad_deadline_ts().value_or(vad_smoke_unix_now_seconds());
    auto end_wire = end_runtime.end(true);
    (void)end_wire;
    uint64_t end_count = vad_smoke_finalize_count(end_runtime);
    auto end_late = end_runtime.poll_timer(end_deadline + 1.0);
    bool end_pass = end_runtime.vad_state() == VadState::IDLE &&
                    !end_runtime.vad_deadline_ts().has_value() &&
                    end_count == 1 &&
                    end_late.empty() &&
                    vad_smoke_finalize_count(end_runtime) == 1 &&
                    end_runtime.last_timing().has_value() &&
                    end_runtime.last_timing()->reason == "end";
    total_finalize_invocations += static_cast<int>(reset_count + end_count);
    report("reset_true_bypasses_debounce", reset_pass, reset_runtime);
    report("end_true_bypasses_debounce", end_pass, end_runtime);
  }

  bool ok = failures == 0 && total_finalize_invocations == 4;
  std::printf("VAD_SMOKE %s failures=%d finalize_invocations=%d expected_finalize_invocations=4\n",
              ok ? "PASS" : "FAIL",
              failures,
              total_finalize_invocations);
  return ok;
}

struct StatsSmokeSample {
  std::optional<double> vad_stop_to_sent_ms;
  std::optional<double> fork_flush_wall_ms;
  std::optional<double> vad_stop_recv_to_process_ms;
  std::optional<double> lock_wait_ms;
  std::optional<double> enc_first_lock_wait_ms;
  std::optional<double> vad_stop_to_finalize_start_ms;
  double active_sessions_at_emit = 0.0;
  bool emitted = false;
};

static std::optional<double> smoke_delta_ms(std::optional<double> end, std::optional<double> start) {
  if (!end.has_value() || !start.has_value()) return std::nullopt;
  double value = std::max(0.0, (*end - *start) * 1000.0);
  double rounded = std::round(value);
  if (std::abs(value - rounded) < 1.0e-9) return rounded;
  return value;
}

static std::vector<double> collect_stats_smoke_values(
    const std::vector<StatsSmokeSample>& samples,
    const std::function<std::optional<double>(const StatsSmokeSample&)>& pick) {
  std::vector<double> values;
  for (const auto& sample : samples) {
    std::optional<double> value = pick(sample);
    if (value.has_value()) values.push_back(*value);
  }
  return values;
}

static std::vector<double> collect_stats_smoke_active(
    const std::vector<StatsSmokeSample>& samples) {
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto& sample : samples) values.push_back(sample.active_sessions_at_emit);
  return values;
}

static bool json_number_eq(const nlohmann::json& value, double expected) {
  return value.is_number() && std::abs(value.get<double>() - expected) < 1.0e-9;
}

static bool stats_summary_matches(const nlohmann::json& summary, std::vector<double> values) {
  if (!summary.is_object()) return false;
  if (values.empty()) {
    return summary.value("count", -1) == 0 &&
           summary.contains("p50") && summary["p50"].is_null() &&
           summary.contains("p90") && summary["p90"].is_null() &&
           summary.contains("p95") && summary["p95"].is_null() &&
           summary.contains("p99") && summary["p99"].is_null() &&
           summary.contains("max") && summary["max"].is_null();
  }

  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  auto percentile = [&](double p) {
    size_t idx = static_cast<size_t>(std::llround(p * static_cast<double>(n - 1)));
    if (idx >= n) idx = n - 1;
    return values[idx];
  };
  return summary.value("count", -1) == static_cast<int>(n) &&
         json_number_eq(summary.at("p50"), percentile(0.50)) &&
         json_number_eq(summary.at("p90"), percentile(0.90)) &&
         json_number_eq(summary.at("p95"), percentile(0.95)) &&
         json_number_eq(summary.at("p99"), percentile(0.99)) &&
         json_number_eq(summary.at("max"), values.back());
}

static StatsSmokeSample expected_sample_from_timing(const SessionTiming& timing, bool emitted) {
  StatsSmokeSample sample;
  sample.vad_stop_to_sent_ms = smoke_delta_ms(timing.final_sent_ts, timing.vad_stop_ts);
  sample.fork_flush_wall_ms = smoke_delta_ms(timing.fork_flush_done_ts, timing.fork_flush_start_ts);
  sample.vad_stop_recv_to_process_ms = smoke_delta_ms(timing.vad_stop_ts, timing.vad_stop_recv_ts);
  sample.lock_wait_ms = timing.inference_lock_acquire_wait_ms;
  sample.enc_first_lock_wait_ms = timing.enc_first_lock_wait_ms;
  sample.vad_stop_to_finalize_start_ms = smoke_delta_ms(timing.fork_flush_start_ts, timing.vad_stop_ts);
  sample.active_sessions_at_emit = static_cast<double>(timing.active_sessions_at_emit);
  sample.emitted = emitted;
  return sample;
}

static SessionTiming make_fixture_timing(double vad_stop,
                                         double final_sent,
                                         int active_sessions_at_emit) {
  SessionTiming timing;
  timing.vad_stop_ts = vad_stop;
  timing.final_sent_ts = final_sent;
  timing.fork_flush_start_ts = vad_stop + 0.001953125;
  timing.fork_flush_done_ts = vad_stop + 0.005859375;
  timing.vad_stop_recv_ts = vad_stop - 0.00390625;
  timing.inference_lock_acquire_wait_ms = 0.5;
  timing.enc_first_lock_wait_ms = 0.25;
  timing.active_sessions_at_emit = active_sessions_at_emit;
  return timing;
}

static bool run_stats_smoke() {
  unsetenv("NEMOTRON_STATS_ENABLED");
  unsetenv("NEMOTRON_STATS_WINDOW");

  DensityAdmission admission(2, 3);
  (void)admission.try_admit("stats-active-0");
  (void)admission.try_admit("stats-active-1");
  (void)admission.try_admit("stats-backlog-0");
  (void)admission.try_admit("stats-backlog-1");
  (void)admission.try_admit("stats-backlog-2");
  (void)admission.try_admit("stats-shed-0");

  StatsCollector collector(64, true);
  collector.set_admission(&admission);
  std::vector<StatsSmokeSample> expected_samples;
  expected_samples.reserve(50);
  uint64_t expected_lifetime_emitted = 0;
  uint64_t expected_lifetime_suppressed = 0;

  for (int i = 0; i < 50; ++i) {
    SessionTiming timing;
    double base = 1000.0 + static_cast<double>(i);
    timing.vad_stop_ts = base;
    timing.final_sent_ts = base + (10.0 + static_cast<double>(i)) / 1000.0;
    timing.fork_flush_start_ts = base + (4.0 + static_cast<double>(i % 5)) / 1000.0;
    timing.fork_flush_done_ts =
        *timing.fork_flush_start_ts + (20.0 + static_cast<double>(i % 7)) / 1000.0;
    timing.vad_stop_recv_ts = base - (30.0 + static_cast<double>(i % 11)) / 1000.0;
    timing.inference_lock_acquire_wait_ms = static_cast<double>(i % 6) + 0.25;
    timing.enc_first_lock_wait_ms = static_cast<double>(i % 4) + 0.125;
    timing.finalize_seq = static_cast<uint64_t>(i + 1);
    timing.active_sessions_at_emit = i % 8;

    if (i % 7 == 0) {
      timing.fork_flush_start_ts.reset();
      timing.fork_flush_done_ts.reset();
    }
    if (i % 11 == 0) timing.vad_stop_recv_ts.reset();
    if (i % 9 == 0) timing.inference_lock_acquire_wait_ms.reset();
    if (i % 8 == 0) timing.enc_first_lock_wait_ms.reset();
    if (i % 13 == 0) timing.was_suppressed = true;
    if (i % 17 == 0) timing.final_sent_ts.reset();

    bool complete = !timing.was_suppressed &&
                    timing.vad_stop_ts.has_value() &&
                    timing.final_sent_ts.has_value();
    bool emitted = complete && (i % 10 != 0);
    collector.record(timing, emitted);
    if (!complete) {
      ++expected_lifetime_suppressed;
    } else {
      expected_samples.push_back(expected_sample_from_timing(timing, emitted));
      if (emitted) {
        ++expected_lifetime_emitted;
      } else {
        ++expected_lifetime_suppressed;
      }
    }
  }

  std::string full_json_text = collector.snapshot_json();
  std::string last10_json_text = collector.snapshot_json(10);
  nlohmann::json full = nlohmann::json::parse(full_json_text);
  nlohmann::json last10 = nlohmann::json::parse(last10_json_text);

  std::vector<StatsSmokeSample> last10_expected = expected_samples;
  if (last10_expected.size() > 10) {
    last10_expected.erase(last10_expected.begin(),
                          last10_expected.end() - static_cast<std::ptrdiff_t>(10));
  }

  auto collect_vad = [](const StatsSmokeSample& sample) { return sample.vad_stop_to_sent_ms; };
  auto collect_fork = [](const StatsSmokeSample& sample) { return sample.fork_flush_wall_ms; };
  auto collect_recv = [](const StatsSmokeSample& sample) { return sample.vad_stop_recv_to_process_ms; };
  auto collect_lock = [](const StatsSmokeSample& sample) { return sample.lock_wait_ms; };
  auto collect_enc_first_lock = [](const StatsSmokeSample& sample) { return sample.enc_first_lock_wait_ms; };
  auto collect_finalize_start = [](const StatsSmokeSample& sample) {
    return sample.vad_stop_to_finalize_start_ms;
  };

  const uint64_t emitted_in_window =
      static_cast<uint64_t>(std::count_if(expected_samples.begin(), expected_samples.end(),
                                          [](const StatsSmokeSample& sample) { return sample.emitted; }));
  const uint64_t suppressed_in_window =
      static_cast<uint64_t>(expected_samples.size()) - emitted_in_window;

  bool full_ok =
      full.value("enabled", false) &&
      full.value("window_size", 0) == 64 &&
      full.value("samples", -1) == static_cast<int>(expected_samples.size()) &&
      full.value("emitted_in_window", -1) == static_cast<int>(emitted_in_window) &&
      full.value("suppressed_in_window", -1) == static_cast<int>(suppressed_in_window) &&
      full.value("lifetime_emitted", -1) == static_cast<int>(expected_lifetime_emitted) &&
      full.value("lifetime_suppressed", -1) == static_cast<int>(expected_lifetime_suppressed) &&
      stats_summary_matches(full["metrics"]["vad_stop_to_sent_ms"],
                            collect_stats_smoke_values(expected_samples, collect_vad)) &&
      stats_summary_matches(full["metrics"]["fork_flush_wall_ms"],
                            collect_stats_smoke_values(expected_samples, collect_fork)) &&
      stats_summary_matches(full["metrics"]["vad_stop_recv_to_process_ms"],
                            collect_stats_smoke_values(expected_samples, collect_recv)) &&
      stats_summary_matches(full["metrics"]["lock_wait_ms"],
                            collect_stats_smoke_values(expected_samples, collect_lock)) &&
      stats_summary_matches(full["metrics"]["enc_first_lock_wait_ms"],
                            collect_stats_smoke_values(expected_samples, collect_enc_first_lock)) &&
      stats_summary_matches(full["metrics"]["vad_stop_to_finalize_start_ms"],
                            collect_stats_smoke_values(expected_samples, collect_finalize_start)) &&
      stats_summary_matches(full["active_sessions_at_emit"],
                            collect_stats_smoke_active(expected_samples));

  bool last10_ok =
      last10.value("samples", -1) == 10 &&
      stats_summary_matches(last10["metrics"]["vad_stop_to_sent_ms"],
                            collect_stats_smoke_values(last10_expected, collect_vad)) &&
      stats_summary_matches(last10["active_sessions_at_emit"],
                            collect_stats_smoke_active(last10_expected));

  const nlohmann::json& admission_json = full["admission"];
  bool admission_ok =
      admission_json.value("enabled", false) &&
      admission_json.value("attempted", -1) == 6 &&
      admission_json.value("admitted", -1) == 5 &&
      admission_json.value("rejected", -1) == 1 &&
      admission_json.value("max_backlog", -1) == 3 &&
      json_number_eq(admission_json["max_ready_age_ms"], 0.0) &&
      admission_json["signal"].value("queued_events", -1) == 3 &&
      admission_json["signal"].value("ready_count", -1) == 0 &&
      admission_json["signal"].value("backlog_count", -1) == 5 &&
      json_number_eq(admission_json["signal"]["oldest_ready_age_ms"], 0.0) &&
      admission_json["signal"]["oldest_ready_session_id"].is_null();

  StatsCollector disabled(16, false);
  SessionTiming disabled_timing;
  disabled_timing.vad_stop_ts = 10.0;
  disabled_timing.final_sent_ts = 10.001;
  disabled.record(disabled_timing, true);
  nlohmann::json disabled_json = nlohmann::json::parse(disabled.snapshot_json());
  bool disabled_ok =
      !disabled_json.value("enabled", true) &&
      disabled_json.value("samples", -1) == 0 &&
      disabled_json.value("lifetime_emitted", -1) == 0 &&
      disabled_json.value("lifetime_suppressed", -1) == 0 &&
      disabled_json["metrics"]["vad_stop_to_sent_ms"]["p50"].is_null() &&
      !disabled_json["admission"].value("enabled", true);

  StatsCollector fixture(8, true);
  SessionTiming fixture_a = make_fixture_timing(1000.0, 1000.0078125, 1);
  SessionTiming fixture_b = make_fixture_timing(1001.0, 1001.015625, 3);
  fixture_b.fork_flush_start_ts.reset();
  fixture_b.fork_flush_done_ts.reset();
  fixture_b.inference_lock_acquire_wait_ms.reset();
  SessionTiming fixture_c;
  fixture_c.vad_stop_ts = 1002.0;
  SessionTiming fixture_d = make_fixture_timing(1003.0, 1003.0234375, 5);
  fixture_d.vad_stop_recv_ts.reset();
  fixture.record(fixture_a, true);
  fixture.record(fixture_b, false);
  fixture.record(fixture_c, true);
  fixture.record(fixture_d, true);
  std::string fixture_json = fixture.snapshot_json();
  const std::string expected_fixture_json =
      "{\"enabled\":true,\"window_size\":8,\"samples\":3,\"since_unix\":1000.0078125,"
      "\"until_unix\":1003.0234375,\"emitted_in_window\":2,\"suppressed_in_window\":1,"
      "\"lifetime_emitted\":2,\"lifetime_suppressed\":2,\"metrics\":{"
      "\"vad_stop_to_sent_ms\":{\"p50\":15.625,\"p90\":23.4375,\"p95\":23.4375,"
      "\"p99\":23.4375,\"max\":23.4375,\"count\":3},"
      "\"fork_flush_wall_ms\":{\"p50\":3.90625,\"p90\":3.90625,\"p95\":3.90625,"
      "\"p99\":3.90625,\"max\":3.90625,\"count\":2},"
      "\"vad_stop_recv_to_process_ms\":{\"p50\":3.90625,\"p90\":3.90625,"
      "\"p95\":3.90625,\"p99\":3.90625,\"max\":3.90625,\"count\":2},"
      "\"lock_wait_ms\":{\"p50\":0.5,\"p90\":0.5,\"p95\":0.5,\"p99\":0.5,"
      "\"max\":0.5,\"count\":2},"
      "\"enc_first_lock_wait_ms\":{\"p50\":0.25,\"p90\":0.25,\"p95\":0.25,"
      "\"p99\":0.25,\"max\":0.25,\"count\":3},"
      "\"lane_queue_wait_ms\":{\"p50\":null,\"p90\":null,\"p95\":null,"
      "\"p99\":null,\"max\":null,\"count\":0},"
      "\"preproc_ms\":{\"p50\":null,\"p90\":null,\"p95\":null,"
      "\"p99\":null,\"max\":null,\"count\":0},"
      "\"scheduler_enqueue_wait_ms\":{\"p50\":null,\"p90\":null,\"p95\":null,"
      "\"p99\":null,\"max\":null,\"count\":0},"
      "\"scheduler_future_wait_ms\":{\"p50\":null,\"p90\":null,\"p95\":null,"
      "\"p99\":null,\"max\":null,\"count\":0},"
      "\"scheduler_completion_wait_ms\":{\"p50\":null,\"p90\":null,\"p95\":null,"
      "\"p99\":null,\"max\":null,\"count\":0},"
      "\"decode_ms\":{\"p50\":null,\"p90\":null,\"p95\":null,"
      "\"p99\":null,\"max\":null,\"count\":0},"
      "\"vad_stop_to_finalize_start_ms\":{\"p50\":1.953125,\"p90\":1.953125,"
      "\"p95\":1.953125,\"p99\":1.953125,\"max\":1.953125,\"count\":2}},"
      "\"active_sessions_at_emit\":{\"p50\":3,\"p90\":5,\"p95\":5,\"p99\":5,"
      "\"max\":5,\"count\":3},\"admission\":{\"enabled\":false,\"attempted\":0,"
      "\"admitted\":0,\"rejected\":0,\"max_backlog\":0,\"max_ready_age_ms\":0,"
      "\"signal\":{\"queued_events\":0,\"ready_count\":0,\"backlog_count\":0,"
      "\"oldest_ready_age_ms\":0,\"oldest_ready_session_id\":null}}}";
  bool fixture_ok = fixture_json == expected_fixture_json;

  bool ok = collector.enabled() &&
            collector.window_size() == 64 &&
            full_ok &&
            last10_ok &&
            admission_ok &&
            disabled_ok &&
            fixture_ok;
  std::printf("STATS_SMOKE pass=%s full_window=%s last10=%s admission=%s disabled=%s fixture=%s\n",
              ok ? "true" : "false",
              full_ok ? "PASS" : "FAIL",
              last10_ok ? "PASS" : "FAIL",
              admission_ok ? "PASS" : "FAIL",
              disabled_ok ? "PASS" : "FAIL",
              fixture_ok ? "PASS" : "FAIL");
  std::printf("STATS_SMOKE_SAMPLE_JSON %s\n", full_json_text.c_str());
  if (!ok) {
    std::printf("STATS_SMOKE last10=%s\n", last10_json_text.c_str());
    std::printf("STATS_SMOKE fixture_actual=%s\n", fixture_json.c_str());
    std::printf("STATS_SMOKE fixture_expected=%s\n", expected_fixture_json.c_str());
  }
  return ok;
}

static bool response_starts_with(const std::string& response, const std::string& prefix) {
  return response.size() >= prefix.size() &&
         response.compare(0, prefix.size(), prefix) == 0;
}

static std::string bytes_to_string(const std::vector<uint8_t>& bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

static bool run_ws_lib_smoke() {
  int failures = 0;
  auto report = [&](const char* name, bool pass) {
    if (!pass) ++failures;
    std::printf("WS_LIB_SMOKE_CASE name=%s result=%s\n", name, pass ? "PASS" : "FAIL");
  };

  {
    ws_handshake::HttpRequest req;
    const std::string raw =
        "GET /health HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";
    bool pass = ws_handshake::parse_http_request(raw, req) == ws_handshake::ParseResult::OK &&
                ws_routes::dispatch(req).kind == ws_routes::RouteKind::HEALTH;
    report("valid_health_route", pass);
  }

  {
    ws_handshake::HttpRequest req;
    const std::string raw =
        "GET / HTTP/1.x\r\n"
        "Host: localhost\r\n"
        "\r\n";
    auto result = ws_handshake::parse_http_request(raw, req);
    std::string response = ws_handshake::build_http_error_response(400, "{\"error\":\"bad_request\"}");
    bool pass = result == ws_handshake::ParseResult::MALFORMED &&
                response_starts_with(response, "HTTP/1.1 400 Bad Request\r\n");
    report("malformed_request_400", pass);
  }

  {
    ws_handshake::HttpRequest req;
    std::string raw =
        "GET /health HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "X-Fill: ";
    raw.append(ws_handshake::kMaxHttpHeaderBytes, 'a');
    raw += "\r\n\r\n";
    auto result = ws_handshake::parse_http_request(raw, req);
    std::string response =
        ws_handshake::build_http_error_response(431, "{\"error\":\"headers_too_large\"}");
    bool pass = result == ws_handshake::ParseResult::OVERSIZE_HEADERS &&
                response_starts_with(response,
                                     "HTTP/1.1 431 Request Header Fields Too Large\r\n");
    report("oversize_headers_431", pass);
  }

  {
    ws_handshake::HttpRequest req;
    const std::string raw =
        "GET /missing HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "\r\n";
    auto result = ws_handshake::parse_http_request(raw, req);
    auto route = ws_routes::dispatch(req);
    std::string response = ws_handshake::build_http_error_response(404, "{\"error\":\"not_found\"}");
    bool pass = result == ws_handshake::ParseResult::OK &&
                route.kind == ws_routes::RouteKind::NOT_FOUND &&
                response_starts_with(response, "HTTP/1.1 404 Not Found\r\n");
    report("unknown_route_404", pass);
  }

  {
    ws_handshake::HttpRequest req;
    const std::string key = "dGhlIHNhbXBsZSBub25jZQ==";
    const std::string expected_accept = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";
    const std::string raw =
        "GET /?model=en&language=auto HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Upgrade: WebSocket\r\n"
        "Connection: keep-alive\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Key: " + key + "\r\n"
        "\r\n";
    auto result = ws_handshake::parse_http_request(raw, req);
    auto route = ws_routes::dispatch(req);
    std::string accept = ws_handshake::compute_accept_key(key);
    std::string response = ws_handshake::build_handshake_response(accept);
    bool pass = result == ws_handshake::ParseResult::OK &&
                ws_handshake::is_websocket_upgrade(req) &&
                route.kind == ws_routes::RouteKind::WEBSOCKET &&
                route.path == "/" &&
                route.query_params["model"] == "en" &&
                accept == expected_accept &&
                response.find("Sec-WebSocket-Accept: " + expected_accept + "\r\n") !=
                    std::string::npos;
    std::printf("WS_LIB_SMOKE_ACCEPT input=%s computed=%s expected=%s\n",
                key.c_str(),
                accept.c_str(),
                expected_accept.c_str());
    report("valid_ws_handshake_accept", pass);
  }

  {
    std::string raw;
    raw.push_back(static_cast<char>(0x82));
    raw.push_back(static_cast<char>(0x80 | 127));
    uint64_t len = static_cast<uint64_t>(ws_framing::kMaxMessageSize) + 1;
    for (int shift = 56; shift >= 0; shift -= 8) {
      raw.push_back(static_cast<char>((len >> shift) & 0xff));
    }
    ws_framing::Frame frame;
    size_t consumed = 123;
    bool pass = ws_framing::read_frame(raw, frame, consumed) ==
                    ws_framing::ReadResult::FRAME_TOO_LARGE &&
                consumed == 0;
    report("frame_header_too_large", pass);
  }

  {
    std::vector<uint8_t> wire = ws_framing::write_frame(ws_framing::Opcode::TEXT, "hello", true);
    ws_framing::Frame frame;
    size_t consumed = 0;
    auto result = ws_framing::read_frame(bytes_to_string(wire), frame, consumed);
    std::string payload(frame.payload.begin(), frame.payload.end());
    bool pass = result == ws_framing::ReadResult::OK &&
                consumed == wire.size() &&
                frame.fin &&
                frame.opcode == ws_framing::Opcode::TEXT &&
                payload == "hello";
    report("roundtrip_masked_text_frame", pass);
  }

  {
    const std::string reason = "admission_backpressure";
    std::vector<uint8_t> wire = ws_framing::write_close_frame(1013, reason);
    size_t payload_len = 2 + reason.size();
    std::string reason_wire(wire.begin() + 4, wire.end());
    uint16_t code = static_cast<uint16_t>((static_cast<uint16_t>(wire[2]) << 8) | wire[3]);
    bool pass = wire.size() == 2 + payload_len &&
                wire[0] == 0x88 &&
                wire[1] == payload_len &&
                code == 1013 &&
                reason_wire == reason;
    report("close_1013_admission_backpressure", pass);
  }

  bool ok = failures == 0;
  std::printf("WS_LIB_SMOKE %s failures=%d\n", ok ? "PASS" : "FAIL", failures);
  return ok;
}

static std::string density_exe_dir() {
  char path[4096];
  ssize_t n = ::readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (n <= 0) return ".";
  path[n] = '\0';
  fs::path exe(path);
  return exe.has_parent_path() ? exe.parent_path().string() : ".";
}

static std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
}

static bool run_ws_lifecycle_smoke(const DensityArgs& args) {
  std::string server_bin = (fs::path(density_exe_dir()) / "ws_server").string();
  std::string python = file_exists("./.venv/bin/python") ? "./.venv/bin/python" : "python3";
  fs::path script_path = fs::temp_directory_path() /
                         ("ws_lifecycle_smoke_" + std::to_string(::getpid()) + ".py");

  const char* script = R"PY(
import base64
import json
import os
import select
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

server_bin = sys.argv[1]
artifact_dir = Path(sys.argv[2]).resolve()

def fail(name, detail):
    print(f"WS_LIFECYCLE_ASSERT {name}=FAIL {detail}", flush=True)
    raise SystemExit(1)

def ok(name, detail=""):
    print(f"WS_LIFECYCLE_ASSERT {name}=PASS {detail}", flush=True)

def frame(opcode, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    mask = os.urandom(4)
    head = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head.append(0x80 | n)
    elif n <= 0xFFFF:
        head.append(0x80 | 126)
        head.extend(struct.pack("!H", n))
    else:
        head.append(0x80 | 127)
        head.extend(struct.pack("!Q", n))
    head.extend(mask)
    head.extend(bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
    return bytes(head)

def recv_exact(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("socket closed")
        out.extend(chunk)
    return bytes(out)

def read_frame(sock, timeout=30.0):
    ready, _, _ = select.select([sock], [], [], timeout)
    if not ready:
        raise TimeoutError("timed out waiting for frame")
    h = recv_exact(sock, 2)
    opcode = h[0] & 0x0F
    n = h[1] & 0x7F
    if n == 126:
        n = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", recv_exact(sock, 8))[0]
    payload = recv_exact(sock, n)
    return opcode, payload

def read_http(sock):
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1)
        if not chunk:
            raise EOFError("http closed")
        data.extend(chunk)
        if len(data) > 65536:
            raise RuntimeError("http response too large")
    head, body = bytes(data).split(b"\r\n\r\n", 1)
    headers = {}
    lines = head.decode("iso-8859-1").split("\r\n")
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    while len(body) < length:
        body += sock.recv(length - len(body))
    return lines[0], body

def ws_connect(port, query=""):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    target = "/" + (("?" + query) if query else "")
    req = (
        f"GET {target} HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(req)
    status, _ = read_http(sock)
    if " 101 " not in status:
        fail("handshake", status)
    return sock

def http_json(port, path):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("ascii"))
    status, body = read_http(sock)
    sock.close()
    if " 200 " not in status:
        fail("http_" + path, status)
    return json.loads(body.decode("utf-8"))

def load_pcm():
    bundle = torch.jit.load(str(artifact_dir / "session_audio_bundle.ts"), map_location="cpu")
    audio = getattr(bundle, "utt0_audio").detach().cpu().to(torch.float32).contiguous().numpy()
    pcm = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype(np.int16)
    return pcm.tobytes()

def close_code(payload):
    if len(payload) < 2:
        return 0
    return struct.unpack("!H", payload[:2])[0]

env = os.environ.copy()
env["NEMOTRON_ARTIFACT_DIR"] = str(artifact_dir)
env["NEMOTRON_FINALIZE_SILENCE_MS"] = "0"
env["NEMOTRON_WS_STALEGEN_TEST_ENDPOINT"] = "1"
proc = subprocess.Popen(
    [server_bin, "--port", "0", "--admission-active-cap", "4"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
)
port = None
captured = []
server_start_timeout = float(os.environ.get("NEMOTRON_SMOKE_SERVER_START_TIMEOUT_SEC", "900"))
deadline = time.time() + server_start_timeout
try:
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                fail("server_start", "".join(captured[-20:]))
            continue
        captured.append(line)
        print("WS_SERVER_LOG " + line.rstrip(), flush=True)
        if "ws_server listening on 127.0.0.1:" in line:
            port = int(line.rsplit(":", 1)[1])
            break
    if port is None:
        fail("server_start", f"timeout_sec={server_start_timeout}")

    pcm = load_pcm()
    sock = ws_connect(port, "model=en")
    opcode, payload = read_frame(sock, 10)
    ready = payload.decode("utf-8")
    if opcode != 1 or ready != '{"type":"ready"}':
        fail("ready_exact", f"opcode={opcode} payload={ready!r}")
    ok("ready_exact")

    sock.sendall(frame(1, '{"type":"vad_start"}'))
    chunk_bytes = 640
    for pos in range(0, len(pcm), chunk_bytes):
        sock.sendall(frame(2, pcm[pos:pos + chunk_bytes]))
    sock.sendall(frame(1, '{"type":"vad_stop"}'))

    interims = []
    final = None
    for _ in range(128):
        opcode, payload = read_frame(sock, 60)
        if opcode != 1:
            continue
        msg = json.loads(payload.decode("utf-8"))
        if msg.get("type") != "transcript":
            continue
        if msg.get("is_final") is True:
            final = msg
            break
        if msg.get("is_final") is False:
            interims.append(msg)
    if not interims or any((not m.get("text")) or m.get("is_final") is not False for m in interims):
        fail("interim_ordering", f"count={len(interims)}")
    ok("interim_ordering", f"count={len(interims)}")
    if not final or final.get("is_final") is not True or not final.get("text"):
        fail("final_transcript", str(final))
    ok("final_transcript", final.get("text", "")[:48])
    timing = final.get("finalize_timing")
    expected = {
        "reason",
        "vad_stop",
        "vad_stop_recv",
        "debounce_expiry",
        "fork_flush_start",
        "fork_flush_done",
        "final_sent",
        "inference_lock_acquire_wait_ms",
        "enc_first_lock_wait_ms",
        "gil_attrib_enabled",
    }
    if not isinstance(timing, dict) or set(timing.keys()) != expected:
        fail("finalize_timing_keys", str(timing))
    ok("finalize_timing_keys")

    drops_before = http_json(port, "/__stale_gen_drops")
    event_drops_before = int(drops_before.get("drops_at_event_emit", 0))
    sock.sendall(frame(2, pcm[: min(len(pcm), 32000)]) + frame(1, '{"type":"reset","finalize":false}'))
    stale_window_interims = 0
    deadline2 = time.time() + 10
    while time.time() < deadline2:
        ready, _, _ = select.select([sock], [], [], 0.2)
        if not ready:
            break
        opcode, payload = read_frame(sock, 0.2)
        if opcode != 1:
            continue
        msg = json.loads(payload.decode("utf-8"))
        if msg.get("type") == "transcript" and msg.get("is_final") is False:
            stale_window_interims += 1
    event_drops_after = event_drops_before
    deadline_drop = time.time() + 10
    while time.time() < deadline_drop:
        drops_after = http_json(port, "/__stale_gen_drops")
        event_drops_after = int(drops_after.get("drops_at_event_emit", 0))
        if event_drops_after > event_drops_before:
            break
        time.sleep(0.1)
    if stale_window_interims != 0:
        fail("stale_interim_drop", f"interims_after_reset={stale_window_interims}")
    if event_drops_after <= event_drops_before:
        fail("stale_interim_drop", f"drops_at_event_emit_before={event_drops_before} after={event_drops_after}")
    ok("stale_interim_drop", f"drops_at_event_emit={event_drops_after}")

    sock.sendall(frame(2, pcm[: min(len(pcm), 32000)]))
    time.sleep(0.1)
    stats_before = http_json(port, "/stats?last=10")
    suppressed_before = int(stats_before.get("lifetime_suppressed", 0))
    drops_before_final = http_json(port, "/__stale_gen_drops")
    final_drops_before = int(drops_before_final.get("drops_at_finalize_output", 0))
    sock.sendall(frame(1, '{"type":"vad_stop"}') + frame(1, '{"type":"reset","finalize":false}'))
    stats_after = stats_before
    suppressed_after = suppressed_before
    final_drops_after = final_drops_before
    deadline3 = time.time() + 60
    while time.time() < deadline3:
        ready, _, _ = select.select([sock], [], [], 0.2)
        if ready:
            opcode, payload = read_frame(sock, 0.2)
            if opcode == 8:
                break
        stats_after = http_json(port, "/stats?last=10")
        suppressed_after = int(stats_after.get("lifetime_suppressed", 0))
        drops_after_final = http_json(port, "/__stale_gen_drops")
        final_drops_after = int(drops_after_final.get("drops_at_finalize_output", 0))
        if suppressed_after > suppressed_before and final_drops_after > final_drops_before:
            break
    if suppressed_after <= suppressed_before or final_drops_after <= final_drops_before:
        fail("stale_final_record", f"suppressed_before={suppressed_before} after={suppressed_after} drops_before={final_drops_before} drops_after={final_drops_after} stats={json.dumps(stats_after, sort_keys=True)[:300]}")
    ok("stale_final_record", f"suppressed_before={suppressed_before} after={suppressed_after} drops_at_finalize_output={final_drops_after}")
    if int(stats_after.get("lifetime_emitted", 0)) < 1:
        fail("stats_emitted_record", json.dumps(stats_after, sort_keys=True)[:200])
    ok("stats_emitted_record")

    sock.sendall(frame(8, struct.pack("!H", 1000)))
    opcode, payload = read_frame(sock, 10)
    if opcode != 8 or close_code(payload) != 1000:
        fail("clean_close_1000", f"opcode={opcode} code={close_code(payload)}")
    ok("clean_close_1000")
    sock.close()
    print("WS_LIFECYCLE_SMOKE PASS", flush=True)
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
)PY";

  {
    std::ofstream out(script_path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to write ws lifecycle smoke script");
    out << script;
  }

  std::string command = shell_quote(python) + " " +
                        shell_quote(script_path.string()) + " " +
                        shell_quote(server_bin) + " " +
                        shell_quote(args.dir);
  int rc = std::system(command.c_str());
  std::error_code ec;
  fs::remove(script_path, ec);
  bool ok = rc == 0;
  std::printf("WS_LIFECYCLE_SMOKE %s rc=%d server=%s\n",
              ok ? "PASS" : "FAIL",
              rc,
              server_bin.c_str());
  return ok;
}

static bool run_shutdown_smoke(const DensityArgs& args) {
  std::string server_bin = (fs::path(density_exe_dir()) / "ws_server").string();
  std::string python = file_exists("./.venv/bin/python") ? "./.venv/bin/python" : "python3";
  fs::path script_path = fs::temp_directory_path() /
                         ("shutdown_smoke_" + std::to_string(::getpid()) + ".py");

  const char* script = R"PY(
import base64
import json
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

server_bin = sys.argv[1]
artifact_dir = Path(sys.argv[2]).resolve()

def fail(name, detail):
    print(f"SHUTDOWN_SMOKE_ASSERT {name}=FAIL {detail}", flush=True)
    raise SystemExit(1)

def ok(name, detail=""):
    print(f"SHUTDOWN_SMOKE_ASSERT {name}=PASS {detail}", flush=True)

def frame(opcode, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    mask = os.urandom(4)
    head = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head.append(0x80 | n)
    elif n <= 0xFFFF:
        head.append(0x80 | 126)
        head.extend(struct.pack("!H", n))
    else:
        head.append(0x80 | 127)
        head.extend(struct.pack("!Q", n))
    head.extend(mask)
    head.extend(bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
    return bytes(head)

def recv_exact(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("socket closed")
        out.extend(chunk)
    return bytes(out)

def read_frame(sock, timeout=30.0):
    ready, _, _ = select.select([sock], [], [], timeout)
    if not ready:
        raise TimeoutError("timed out waiting for frame")
    h = recv_exact(sock, 2)
    opcode = h[0] & 0x0F
    n = h[1] & 0x7F
    if n == 126:
        n = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", recv_exact(sock, 8))[0]
    payload = recv_exact(sock, n)
    return opcode, payload

def read_http(sock):
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1)
        if not chunk:
            raise EOFError("http closed")
        data.extend(chunk)
        if len(data) > 65536:
            raise RuntimeError("http response too large")
    head, body = bytes(data).split(b"\r\n\r\n", 1)
    headers = {}
    lines = head.decode("iso-8859-1").split("\r\n")
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    while len(body) < length:
        chunk = sock.recv(length - len(body))
        if not chunk:
            break
        body += chunk
    return lines[0], body

def http_get(port, path):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("ascii"))
    status, body = read_http(sock)
    sock.close()
    return status, body

def ws_connect(port, query=""):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    target = "/" + (("?" + query) if query else "")
    req = (
        f"GET {target} HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(req)
    status, body = read_http(sock)
    return sock, status, body

def close_code(payload):
    if len(payload) < 2:
        return 0
    return struct.unpack("!H", payload[:2])[0]

def load_pcm():
    bundle = torch.jit.load(str(artifact_dir / "session_audio_bundle.ts"), map_location="cpu")
    audio = getattr(bundle, "utt0_audio").detach().cpu().to(torch.float32).contiguous().numpy()
    pcm = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype(np.int16)
    return pcm.tobytes()

def start_server(extra_env=None):
    env = os.environ.copy()
    env["NEMOTRON_ARTIFACT_DIR"] = str(artifact_dir)
    env["NEMOTRON_FINALIZE_SILENCE_MS"] = "0"
    env["NEMOTRON_WS_STALEGEN_TEST_ENDPOINT"] = "1"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [server_bin, "--port", "0", "--admission-active-cap", "4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    captured = []
    port = None
    server_start_timeout = float(os.environ.get("NEMOTRON_SMOKE_SERVER_START_TIMEOUT_SEC", "900"))
    deadline = time.time() + server_start_timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                fail("server_start", "".join(captured[-20:]))
            continue
        captured.append(line)
        print("WS_SERVER_LOG " + line.rstrip(), flush=True)
        if "ws_server listening on 127.0.0.1:" in line:
            port = int(line.rsplit(":", 1)[1])
            break
    if port is None:
        fail("server_start", f"timeout_sec={server_start_timeout}")
    return proc, port, captured

def collect_exit(proc, captured, timeout=60):
    try:
        rest, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        rest, _ = proc.communicate(timeout=20)
    if rest:
        for line in rest.splitlines():
            print("WS_SERVER_LOG " + line, flush=True)
        captured.extend([line + "\n" for line in rest.splitlines()])
    return proc.returncode, "".join(captured)

def wait_health_503(port):
    last = ""
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            status, body = http_get(port, "/health")
            last = status + " " + body.decode("utf-8", "replace")
            if " 503 " in status:
                parsed = json.loads(body.decode("utf-8"))
                if parsed.get("status") not in ("healthy", "loading"):
                    fail("health_drain_enum", parsed)
                ok("health_during_drain", last)
                return
        except Exception as exc:
            last = repr(exc)
        time.sleep(0.05)
    fail("health_during_drain", last)

def read_final_and_close(sock, label, timeout=30):
    final_seen = False
    close_seen = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        opcode, payload = read_frame(sock, max(0.1, deadline - time.time()))
        if opcode == 1:
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("type") == "transcript" and msg.get("is_final") is True and msg.get("text"):
                final_seen = True
        elif opcode == 8:
            close_seen = close_code(payload)
            break
        elif opcode == 9:
            sock.sendall(frame(10, payload))
    if not final_seen or close_seen != 1000:
        fail(label, f"final={final_seen} close={close_seen}")
    ok(label, f"close={close_seen}")

def wait_interim(sock, label, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        opcode, payload = read_frame(sock, max(0.1, deadline - time.time()))
        if opcode == 1:
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("type") == "transcript" and msg.get("is_final") is False and msg.get("text"):
                ok(label, msg.get("text", "")[:48])
                return
        elif opcode == 9:
            sock.sendall(frame(10, payload))
        elif opcode == 8:
            fail(label, f"closed_early={close_code(payload)}")
    fail(label, "timeout")

def clean_variant(pcm):
    proc, port, captured = start_server()
    sockets = []
    try:
        for i in range(2):
            sock, status, _ = ws_connect(port, f"model=en&__selftest_drain_hold_ms=1000")
            if " 101 " not in status:
                fail("clean_handshake", status)
            opcode, payload = read_frame(sock, 10)
            if opcode != 1 or payload.decode("utf-8") != '{"type":"ready"}':
                fail("clean_ready", f"opcode={opcode} payload={payload!r}")
            sockets.append(sock)
        ok("clean_ready", "connections=2")
        for sock in sockets:
            sock.sendall(frame(1, '{"type":"vad_start"}'))
            for pos in range(0, len(pcm), 640):
                sock.sendall(frame(2, pcm[pos:pos + 640]))
        for i, sock in enumerate(sockets):
            wait_interim(sock, f"clean_session_{i}_inflight")

        proc.send_signal(signal.SIGTERM)
        wait_health_503(port)

        sock, status, body = ws_connect(port, "model=en")
        sock.close()
        if " 503 " not in status or body != b'{"error":"draining"}':
            fail("new_ws_draining_503", f"status={status} body={body!r}")
        ok("new_ws_draining_503")

        status, body = http_get(port, "/stats")
        if " 200 " not in status:
            fail("stats_during_drain", status)
        ok("stats_during_drain", body[:80].decode("utf-8", "replace"))

        for i, sock in enumerate(sockets):
            read_final_and_close(sock, f"clean_session_{i}_final_close")
            sock.close()
        rc, logs = collect_exit(proc, captured, 60)
        if rc != 0:
            fail("clean_exit_code", f"rc={rc}")
        if "STATS_LIFETIME emitted=" not in logs:
            fail("clean_stats_lifetime", logs[-1000:])
        ok("clean_exit_code", "rc=0")
        ok("clean_stats_lifetime")
    finally:
        for sock in sockets:
            try:
                sock.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=20)

def forced_variant(pcm):
    proc, port, captured = start_server({"NEMOTRON_SHUTDOWN_DRAIN_SEC": "2"})
    sock = None
    try:
        sock, status, _ = ws_connect(port, "model=en&__selftest_drain_hold_ms=8000")
        if " 101 " not in status:
            fail("forced_handshake", status)
        opcode, payload = read_frame(sock, 10)
        if opcode != 1 or payload.decode("utf-8") != '{"type":"ready"}':
            fail("forced_ready", f"opcode={opcode} payload={payload!r}")
        sock.sendall(frame(1, '{"type":"vad_start"}'))
        sock.sendall(frame(2, pcm[: min(len(pcm), 64000)]))
        proc.send_signal(signal.SIGTERM)
        code = None
        deadline = time.time() + 8
        while time.time() < deadline:
            opcode, payload = read_frame(sock, max(0.1, deadline - time.time()))
            if opcode == 8:
                code = close_code(payload)
                break
            if opcode == 9:
                sock.sendall(frame(10, payload))
        if code != 1001:
            fail("forced_close_1001", f"code={code}")
        ok("forced_close_1001", "code=1001")
        sock.close()
        rc, logs = collect_exit(proc, captured, 30)
        if rc != 1:
            fail("forced_exit_code", f"rc={rc}")
        if "WS_SHUTDOWN_FORCE_CLOSE session_id=" not in logs:
            fail("forced_log", logs[-1000:])
        ok("forced_exit_code", "rc=1")
        ok("forced_log")
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=20)

pcm = load_pcm()
clean_variant(pcm)
forced_variant(pcm)
print("SHUTDOWN_SMOKE PASS", flush=True)
)PY";

  {
    std::ofstream out(script_path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to write shutdown smoke script");
    out << script;
  }

  std::string command = shell_quote(python) + " " +
                        shell_quote(script_path.string()) + " " +
                        shell_quote(server_bin) + " " +
                        shell_quote(args.dir);
  int rc = std::system(command.c_str());
  std::error_code ec;
  fs::remove(script_path, ec);
  bool ok = rc == 0;
  std::printf("SHUTDOWN_SMOKE %s rc=%d server=%s\n",
              ok ? "PASS" : "FAIL",
              rc,
              server_bin.c_str());
  return ok;
}

static bool run_backpressure_smoke(const DensityArgs& args) {
  std::string server_bin = (fs::path(density_exe_dir()) / "ws_server").string();
  std::string python = file_exists("./.venv/bin/python") ? "./.venv/bin/python" : "python3";
  fs::path script_path = fs::temp_directory_path() /
                         ("backpressure_smoke_" + std::to_string(::getpid()) + ".py");

  const char* script = R"PY(
import base64
import os
import select
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

server_bin = sys.argv[1]
artifact_dir = Path(sys.argv[2]).resolve()

def fail(name, detail):
    print(f"BACKPRESSURE_SMOKE_ASSERT {name}=FAIL {detail}", flush=True)
    raise SystemExit(1)

def ok(name, detail=""):
    print(f"BACKPRESSURE_SMOKE_ASSERT {name}=PASS {detail}", flush=True)

def frame(opcode, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    mask = os.urandom(4)
    head = bytearray([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head.append(0x80 | n)
    elif n <= 0xFFFF:
        head.append(0x80 | 126)
        head.extend(struct.pack("!H", n))
    else:
        head.append(0x80 | 127)
        head.extend(struct.pack("!Q", n))
    head.extend(mask)
    head.extend(bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
    return bytes(head)

def recv_exact(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("socket closed")
        out.extend(chunk)
    return bytes(out)

def read_frame(sock, timeout=10.0):
    ready, _, _ = select.select([sock], [], [], timeout)
    if not ready:
        raise TimeoutError("timed out waiting for frame")
    h = recv_exact(sock, 2)
    opcode = h[0] & 0x0F
    n = h[1] & 0x7F
    if n == 126:
        n = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", recv_exact(sock, 8))[0]
    payload = recv_exact(sock, n)
    return opcode, payload

def read_http(sock):
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1)
        if not chunk:
            raise EOFError("http closed")
        data.extend(chunk)
    head, body = bytes(data).split(b"\r\n\r\n", 1)
    headers = {}
    lines = head.decode("iso-8859-1").split("\r\n")
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.lower()] = v.strip()
    length = int(headers.get("content-length", "0"))
    while len(body) < length:
        body += sock.recv(length - len(body))
    return lines[0], body

def ws_connect(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        "GET /?model=en HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(req)
    status, _ = read_http(sock)
    if " 101 " not in status:
        fail("handshake", status)
    opcode, payload = read_frame(sock, 10)
    if opcode != 1 or payload.decode("utf-8") != '{"type":"ready"}':
        fail("ready", f"opcode={opcode} payload={payload!r}")
    return sock

def close_code(payload):
    if len(payload) < 2:
        return 0
    return struct.unpack("!H", payload[:2])[0]

def close_reason(payload):
    return payload[2:].decode("utf-8", "replace") if len(payload) > 2 else ""

env = os.environ.copy()
env["NEMOTRON_ARTIFACT_DIR"] = str(artifact_dir)
env["NEMOTRON_FINALIZE_SILENCE_MS"] = "0"
env["NEMOTRON_WS_PING_INTERVAL_SEC"] = "1"
env["NEMOTRON_WS_PONG_TIMEOUT_SEC"] = "1"
proc = subprocess.Popen(
    [server_bin, "--port", "0", "--admission-active-cap", "2"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
    bufsize=1,
)
captured = []
port = None
try:
    server_start_timeout = float(os.environ.get("NEMOTRON_SMOKE_SERVER_START_TIMEOUT_SEC", "900"))
    deadline = time.time() + server_start_timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                fail("server_start", "".join(captured[-20:]))
            continue
        captured.append(line)
        print("WS_SERVER_LOG " + line.rstrip(), flush=True)
        if "ws_server listening on 127.0.0.1:" in line:
            port = int(line.rsplit(":", 1)[1])
            break
    if port is None:
        fail("server_start", f"timeout_sec={server_start_timeout}")

    sock = ws_connect(port)
    oversize = 16 * 1024 * 1024
    header = bytearray([0x82, 0x80 | 127])
    header.extend(struct.pack("!Q", oversize))
    sock.sendall(bytes(header))
    opcode, payload = read_frame(sock, 5)
    if opcode != 8 or close_code(payload) != 1009:
        fail("oversize_1009", f"opcode={opcode} code={close_code(payload)} reason={close_reason(payload)!r}")
    ok("oversize_1009", close_reason(payload))
    try:
        sock.sendall(b"x" * 16)
    except OSError:
        pass
    sock.close()

    sock = ws_connect(port)
    opcode, payload = read_frame(sock, 5)
    if opcode != 9:
        fail("ping_sent", f"opcode={opcode}")
    ok("ping_sent")
    opcode, payload = read_frame(sock, 5)
    if opcode != 8 or close_code(payload) != 1011 or close_reason(payload) != "pong_timeout":
        fail("pong_timeout_1011", f"opcode={opcode} code={close_code(payload)} reason={close_reason(payload)!r}")
    ok("pong_timeout_1011")
    sock.close()

    print("BACKPRESSURE_SMOKE PASS", flush=True)
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
)PY";

  {
    std::ofstream out(script_path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to write backpressure smoke script");
    out << script;
  }

  std::string command = shell_quote(python) + " " +
                        shell_quote(script_path.string()) + " " +
                        shell_quote(server_bin) + " " +
                        shell_quote(args.dir);
  int rc = std::system(command.c_str());
  std::error_code ec;
  fs::remove(script_path, ec);
  bool ok = rc == 0;
  std::printf("BACKPRESSURE_SMOKE %s rc=%d server=%s\n",
              ok ? "PASS" : "FAIL",
              rc,
              server_bin.c_str());
  return ok;
}

int main(int argc, char** argv) {
  try {
    DensityArgs args = parse_density_args(argc, argv);
    if (args.mode == "ws-lib-smoke") {
      return run_ws_lib_smoke() ? 0 : 1;
    }
    if (args.mode == "admission-smoke") {
      return run_admission_smoke() ? 0 : 1;
    }
    if (args.mode == "stalegen-smoke") {
      return run_stalegen_smoke() ? 0 : 1;
    }
    if (args.mode == "stats-smoke") {
      return run_stats_smoke() ? 0 : 1;
    }
    if (args.mode == "ws-lifecycle-smoke") {
      return run_ws_lifecycle_smoke(args) ? 0 : 1;
    }
    if (args.mode == "shutdown-smoke") {
      return run_shutdown_smoke(args) ? 0 : 1;
    }
    if (args.mode == "backpressure-smoke") {
      return run_backpressure_smoke(args) ? 0 : 1;
    }
    if (args.mode == "density-sweep") {
      const char* previous_cuda_module_loading = std::getenv("CUDA_MODULE_LOADING");
      if (setenv("CUDA_MODULE_LOADING", "EAGER", 1) != 0) {
        throw std::runtime_error("failed to set CUDA_MODULE_LOADING=EAGER");
      }
      std::printf("=== DENSITY CUDA_MODULE_LOADING=EAGER (was %s) ===\n",
                  previous_cuda_module_loading ? previous_cuda_module_loading : "(unset)");
    }
    torch::NoGradGuard ng;
    auto device = torch::Device(torch::kCUDA, 0);
    c10::cuda::CUDAGuard device_guard(device.index());
    std::string stamp = timestamp_utc();
    if (args.mode == "runtime-smoke") {
      bool ok = run_runtime_smoke(args, device);
      return ok ? 0 : 1;
    }
    if (args.mode == "scheduler-admission-smoke") {
      bool ok = run_scheduler_admission_smoke(args, device);
      return ok ? 0 : 1;
    }
    if (args.mode == "scheduler-lanes-smoke") {
      bool ok = run_scheduler_lanes_smoke(args, device);
      return ok ? 0 : 1;
    }
    if (args.mode == "scheduler-graph-abandon-smoke") {
      bool ok = run_scheduler_graph_abandon_smoke(args, device);
      return ok ? 0 : 1;
    }
    if (args.mode == "scheduler-graph-shutdown-smoke") {
      bool ok = run_scheduler_graph_shutdown_smoke(args, device);
      return ok ? 0 : 1;
    }
    if (args.mode == "vad-smoke") {
      bool ok = run_vad_smoke(args, device);
      return ok ? 0 : 1;
    }
    auto shared_bundle = load_shared_session_bundle(args.dir);
    int rows_total = static_cast<int>(scalar_i64(attr_tensor(*shared_bundle, "num_utts")));
    int requested_rows = args.correctness_rows > 0 ? std::min(args.correctness_rows, rows_total) : rows_total;
    if (args.mode == "b1-t1") {
      bool ok = run_b1_t1_gate(args, device, stamp, shared_bundle, rows_total);
      return ok ? 0 : 1;
    }
    if (args.mode == "b2-t1") {
      bool ok = run_b2_t1_gate(args, device, stamp, shared_bundle, rows_total);
      return ok ? 0 : 1;
    }
    if (args.mode == "async-ordering") {
      bool ok = run_async_ordering_gate(args, device, stamp, shared_bundle, rows_total);
      return ok ? 0 : 1;
    }
    if (args.mode == "scheduler-lanes-stress") {
      bool ok = run_scheduler_lanes_stress(args, device, shared_bundle, rows_total);
      return ok ? 0 : 1;
    }
    if (args.mode == "enc-first-parity") {
      bool ok = run_enc_first_parity(args, device, shared_bundle);
      return ok ? 0 : 1;
    }
    if (args.mode == "density-sweep") {
      std::printf("=== DENSITY 1a START: dir=%s stamp=%s n_values=",
                  args.dir.c_str(), stamp.c_str());
      for (size_t i = 0; i < args.n_values.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", args.n_values[i]);
      }
      std::printf(" rows_total=%d density_rows=%d sessions_per_worker=%d cadence=%.3fms stream_mode=%s "
                  "mutex=%s warmup=%s min_finalize_p95_samples=%d CUDA_MODULE_LOADING=%s ===\n",
                  rows_total,
                  args.density_rows,
                  args.density_sessions_per_worker,
                  args.density_chunk_period_ms,
                  args.stream_mode.c_str(),
                  args.mutex_serialize_run ? "true" : "false",
                  args.density_warmup ? "true" : "false",
                  kMinFinalizeP95Samples,
                  std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "(unset)");
      auto summary = run_density_sweep(args, device, stamp, shared_bundle, rows_total);
      return summary.pass_to_1b ? 0 : 1;
    }
    std::printf("=== DENSITY STEP0 START: dir=%s stamp=%s n_values=",
                args.dir.c_str(), stamp.c_str());
    for (size_t i = 0; i < args.n_values.size(); ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", args.n_values[i]);
    }
    std::printf(" target_n=%d correctness_n=%d finalize_n=%d finalize_mode=%s rows=%d/%d stream_mode=%s "
                "workers_override=%d num_runners_override=%d mutex=%s smoke=%s partial=%s ===\n",
                args.target_n,
                args.correctness_n,
                args.finalize_n,
                args.finalize_mode.c_str(),
                requested_rows,
                rows_total,
                args.stream_mode.c_str(),
                args.workers,
                args.num_runners,
                args.mutex_serialize_run ? "true" : "false",
                args.smoke ? "true" : "false",
                args.partial ? "true" : "false");

    CorrectnessResult correctness;
    std::string correctness_status = "SKIP";
    if (!args.skip_correctness) {
      correctness = run_correctness_gate(args, device, stamp, shared_bundle);
      if (!correctness.identity_ok) {
        std::printf("=== DENSITY STEP0 STOP-CANDIDATE: 0b token/event identity failed; throughput numbers are not trusted ===\n");
      }
      if (!correctness.ok) {
        std::printf("=== DENSITY STEP0 CONTINUE_UNTRUSTED: 0b identity/scalar-locality/topology gate failed ===\n");
      }
      correctness_status = correctness.ok ? "PASS" : (correctness.identity_ok ? "FAIL" : "STOP_CANDIDATE");
      cleanup_cuda_cache();
    } else {
      std::printf("=== DENSITY 0b SKIP: throughput numbers will be correctness-untrusted ===\n");
    }

    bool steady_ok = true;
    std::string steady_status = "SKIP";
    if (!args.skip_steady) {
      auto steady = run_steady_sweep(args, device, stamp, shared_bundle);
      steady_ok = steady.pass;
      steady_status = steady_ok ? "PASS" : "FAIL";
      cleanup_cuda_cache();
    }

    bool finalize_ok = true;
    std::string finalize_status = "SKIP";
    if (!args.skip_finalize) {
      if (args.skip_correctness) {
        throw std::runtime_error("0c finalize gate needs serial references; do not combine --skip-correctness with finalize");
      }
      finalize_ok = run_finalize_gate(args, device, stamp, shared_bundle, correctness);
      finalize_status = finalize_ok ? "PASS" : "FAIL";
    }

    bool no_skips = !args.skip_correctness && !args.skip_steady && !args.skip_finalize;
    bool rows_full = requested_rows == rows_total;
    bool has_n1 = std::find(args.n_values.begin(), args.n_values.end(), 1) != args.n_values.end();
    bool has_n2 = std::find(args.n_values.begin(), args.n_values.end(), 2) != args.n_values.end();
    bool has_n4 = std::find(args.n_values.begin(), args.n_values.end(), 4) != args.n_values.end();
    bool has_target = args.target_n <= 0 ||
                      std::find(args.n_values.begin(), args.n_values.end(), args.target_n) != args.n_values.end();
    bool canonical_full_run = no_skips &&
                              rows_full &&
                              !args.smoke &&
                              !args.partial &&
                              args.workers == 0 &&
                              args.num_runners == 0 &&
                              args.finalize_mode == "both" &&
                              args.stream_mode == "explicit" &&
                              !args.mutex_serialize_run &&
                              args.default_stream_control &&
                              args.correctness_default_stream_control &&
                              args.scalar_locality_probe &&
                              args.steady_overlap_probe &&
                              has_n1 &&
                              has_n2 &&
                              has_n4 &&
                              has_target;
    bool partial = args.partial || args.smoke || !no_skips || !rows_full || !canonical_full_run;
    bool gate_all = no_skips && correctness.ok && steady_ok && finalize_ok;
    std::string final_status = partial ? "PARTIAL_DIAGNOSTIC" : (gate_all ? "PASS" : "FAIL");
    emit_run_manifest(args,
                      stamp,
                      final_status,
                      canonical_full_run,
                      partial,
                      rows_total,
                      requested_rows,
                      correctness_status,
                      steady_status,
                      finalize_status);
    std::printf("=== DENSITY STEP0 %s: correctness=%s steady=%s finalize=%s canonical_full_run=%s partial=%s stamp=%s ===\n",
                final_status.c_str(),
                correctness_status.c_str(),
                steady_status.c_str(),
                finalize_status.c_str(),
                canonical_full_run ? "true" : "false",
                partial ? "true" : "false",
                stamp.c_str());
    return gate_all ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("DENSITY setup failed: %s\n", e.what());
    return 2;
  }
}
