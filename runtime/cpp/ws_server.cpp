#include "lib/admission/density_admission.h"
#include "lib/runtime_io/jit_load.h"
#include "lib/runtime_io/json.hpp"
#include "lib/runtime_io/steady_batch_dir.h"
#include "lib/scheduler/batched_steady_scheduler.h"
#include "lib/session/runtime.h"
#include "lib/telemetry/stats_collector.h"
#include "lib/ws/framing.h"
#include "lib/ws/handshake.h"
#include "lib/ws/routes.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace ws_server {

struct StaleGenDrops {
  uint64_t drops_at_event_emit = 0;
  uint64_t drops_at_finalize_output = 0;
};

namespace {
std::atomic<uint64_t> g_drops_at_event_emit{0};
std::atomic<uint64_t> g_drops_at_finalize_output{0};
}  // namespace

StaleGenDrops stale_gen_drops() {
  StaleGenDrops out;
  out.drops_at_event_emit = g_drops_at_event_emit.load(std::memory_order_acquire);
  out.drops_at_finalize_output = g_drops_at_finalize_output.load(std::memory_order_acquire);
  return out;
}

void reset_stale_gen_drops_for_test() {
  g_drops_at_event_emit.store(0, std::memory_order_release);
  g_drops_at_finalize_output.store(0, std::memory_order_release);
}

}  // namespace ws_server

namespace {
std::atomic<bool> g_sigterm_requested{false};

void handle_sigterm(int) {
  g_sigterm_requested.store(true, std::memory_order_release);
}

void install_sigterm_handler() {
  ::signal(SIGTERM, handle_sigterm);
}

constexpr int kAdminWorkers = 2;
constexpr size_t kAdminQueueDepth = 16;
constexpr int kDefaultAdmissionBacklogCap = 12;
constexpr int kDefaultBatchMax = 16;
constexpr int kDefaultBatchWindowMs = 10;  // W=10 burst-coalescing buffer; W=0 regressed the L40S knee -2 (admission timeouts under bursts at the GPU wall) — proj-2026-05-31-1050/phaseB-l40s-retest.md
constexpr int kDefaultBatchMaxQueueDelayMs = 2;
constexpr int kDefaultBatchLoneTimeoutMs = 0;
constexpr int kDefaultBatchQueueCapacity = 32;
constexpr size_t kDefaultStatsWindow = 2048;
constexpr size_t kDefaultWsMaxMessageSize = ws_framing::kMaxMessageSize;
constexpr int kDefaultWsSendTimeoutSec = 5;
constexpr int kDefaultWsPingIntervalSec = 60;
constexpr int kDefaultWsPongTimeoutSec = 30;
constexpr int kDefaultShutdownDrainSec = 30;
constexpr int kDefaultFinalizeSilenceMs = 0;
constexpr const char* kStaleGenTestPath = "/__stale_gen_drops";

std::string json_quote(const std::string& value) {
  return nlohmann::json(value).dump();
}

const char* json_bool(bool value) {
  return value ? "true" : "false";
}

double unix_now_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

std::string ascii_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::string trim_ascii(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
  return value.substr(begin, end - begin);
}

std::string http_reason(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 431:
      return "Request Header Fields Too Large";
    case 503:
      return "Service Unavailable";
    default:
      return "Error";
  }
}

std::string build_json_response(int status, const std::string& body) {
  std::ostringstream oss;
  oss << "HTTP/1.1 " << status << " " << http_reason(status) << "\r\n"
      << "Content-Type: application/json\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n"
      << "\r\n"
      << body;
  return oss.str();
}

enum class SendResult {
  OK,
  TIMEOUT,
  ERROR,
};

bool is_send_timeout_errno(int value) {
  return value == EAGAIN || value == EWOULDBLOCK;
}

SendResult send_all_result(int fd, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  size_t sent = 0;
  while (sent < size) {
    ssize_t n = ::send(fd, p + sent, size - sent, MSG_NOSIGNAL);
    if (n > 0) {
      sent += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && is_send_timeout_errno(errno)) return SendResult::TIMEOUT;
    return SendResult::ERROR;
  }
  return SendResult::OK;
}

bool send_all(int fd, const void* data, size_t size) {
  return send_all_result(fd, data, size) == SendResult::OK;
}

bool send_all(int fd, const std::string& data) {
  return send_all(fd, data.data(), data.size());
}

bool send_all(int fd, const std::vector<uint8_t>& data) {
  return send_all(fd, data.data(), data.size());
}

bool configure_send_timeout(int fd, int timeout_sec) {
  timeval timeout{};
  timeout.tv_sec = timeout_sec;
  timeout.tv_usec = 0;
  return ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0;
}

bool configure_tcp_nodelay(int fd) {
  int yes = 1;
  return ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0;
}

void close_fd(int* fd) {
  if (fd != nullptr && *fd >= 0) {
    ::close(*fd);
    *fd = -1;
  }
}

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
  }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  ~UniqueFd() {
    reset();
  }
  int get() const { return fd_; }
  int release() {
    int fd = fd_;
    fd_ = -1;
    return fd;
  }
  void reset(int next = -1) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = next;
  }

 private:
  int fd_ = -1;
};

int parse_int_strict(const std::string& text, const char* label) {
  if (text.empty()) throw std::runtime_error(std::string(label) + " requires an integer");
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string(label) + " must be an integer: " + text);
  }
  return static_cast<int>(value);
}

uint64_t parse_u64_strict(const std::string& text, const char* label) {
  if (text.empty()) throw std::runtime_error(std::string(label) + " requires an integer");
  if (!std::all_of(text.begin(), text.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
    throw std::runtime_error(std::string(label) + " must be a non-negative integer: " + text);
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    throw std::runtime_error(std::string(label) + " must be a non-negative integer: " + text);
  }
  return static_cast<uint64_t>(value);
}

int read_env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return parse_int_strict(raw, name);
}

std::optional<int> read_env_int_optional(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return std::nullopt;
  return parse_int_strict(raw, name);
}

std::optional<uint64_t> read_env_u64_optional(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return std::nullopt;
  return parse_u64_strict(raw, name);
}

uint64_t read_env_u64(const char* name, uint64_t fallback) {
  auto value = read_env_u64_optional(name);
  return value.value_or(fallback);
}

int capped_general_finalize_runners(uint64_t active_cap) {
  uint64_t clamped = std::max<uint64_t>(1, std::min<uint64_t>(active_cap, 2));
  if (clamped > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(clamped);
}

size_t read_env_size_t(const char* name, size_t fallback) {
  uint64_t value = read_env_u64(name, fallback);
  if (value == 0 || value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    const char* raw = std::getenv(name);
    throw std::runtime_error(std::string("invalid positive integer env var ") + name + "=" +
                             (raw != nullptr ? raw : std::to_string(value)));
  }
  return static_cast<size_t>(value);
}

bool read_env_enabled(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::string(raw) != "0";
}

bool read_env_on_off(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  std::string value = ascii_lower(trim_ascii(raw));
  if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
  if (value == "0" || value == "false" || value == "no" || value == "off") return false;
  throw std::runtime_error(std::string(name) + " must be unset/off/0 or on/1");
}

bool stale_gen_test_endpoint_enabled() {
  return read_env_enabled("NEMOTRON_WS_STALEGEN_TEST_ENDPOINT", false);
}

bool file_exists(const fs::path& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec);
}

bool dir_exists(const fs::path& path) {
  std::error_code ec;
  return fs::is_directory(path, ec);
}

std::string weakly_canonical_string(const fs::path& path) {
  std::error_code ec;
  fs::path out = fs::weakly_canonical(path, ec);
  if (ec) return path.lexically_normal().string();
  return out.string();
}

bool artifact_dir_valid(const fs::path& dir) {
  return file_exists(dir / "session_audio_bundle.ts") &&
         file_exists(dir / "preproc.ts") &&
         file_exists(dir / "enc_first.ts") &&
         file_exists(dir / "enc_steady_aoti.pt2") &&
         file_exists(dir / "finalize_shared_weights.ts");
}

std::vector<fs::path> ancestor_bases(const fs::path& start) {
  std::vector<fs::path> out;
  fs::path cur = start;
  for (int i = 0; i < 8 && !cur.empty(); ++i) {
    out.push_back(cur);
    fs::path parent = cur.parent_path();
    if (parent == cur) break;
    cur = parent;
  }
  return out;
}

std::string resolve_artifact_dir(const std::string& argv0) {
  std::vector<fs::path> candidates;
  const char* env = std::getenv("NEMOTRON_ARTIFACT_DIR");
  if (env != nullptr && env[0] != '\0') candidates.emplace_back(env);

  std::error_code ec;
  fs::path cwd = fs::current_path(ec);
  if (!ec) {
    for (const auto& base : ancestor_bases(cwd)) {
      candidates.push_back(base / "artifacts");
      candidates.push_back(base / "runtime" / "artifacts");
    }
  }
  if (!argv0.empty()) {
    fs::path exe = fs::absolute(argv0, ec);
    if (!ec) {
      fs::path base = exe.has_parent_path() ? exe.parent_path() : exe;
      for (const auto& ancestor : ancestor_bases(base)) {
        candidates.push_back(ancestor / "artifacts");
        candidates.push_back(ancestor / "runtime" / "artifacts");
      }
    }
  }
  candidates.emplace_back("../artifacts");
  candidates.emplace_back("artifacts");
  candidates.emplace_back("runtime/artifacts");

  for (const auto& candidate : candidates) {
    if (artifact_dir_valid(candidate)) return weakly_canonical_string(candidate);
  }
  throw std::runtime_error("could not resolve runtime artifacts directory; set NEMOTRON_ARTIFACT_DIR");
}

bool steady_batch_dir_valid(const fs::path& dir) {
  return runtime_io::steady_batch_dir_has_declared_packages(dir.string());
}

std::string resolve_steady_batch_dir(const std::string& configured,
                                     bool explicit_path,
                                     const std::string& artifact_dir,
                                     const std::string& argv0,
                                     bool required) {
  std::vector<fs::path> candidates;
  if (!configured.empty()) candidates.emplace_back(configured);

  if (!explicit_path) {
    fs::path artifact_path(artifact_dir);
    if (artifact_path.has_parent_path()) {
      candidates.push_back(artifact_path.parent_path() / "steady_b_artifacts");
    }

    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
      for (const auto& base : ancestor_bases(cwd)) {
        candidates.push_back(base / "steady_b_artifacts");
        candidates.push_back(base / "runtime" / "steady_b_artifacts");
      }
    }
    if (!argv0.empty()) {
      fs::path exe = fs::absolute(argv0, ec);
      if (!ec) {
        fs::path base = exe.has_parent_path() ? exe.parent_path() : exe;
        for (const auto& ancestor : ancestor_bases(base)) {
          candidates.push_back(ancestor / "steady_b_artifacts");
          candidates.push_back(ancestor / "runtime" / "steady_b_artifacts");
        }
      }
    }
    candidates.emplace_back("steady_b_artifacts");
    candidates.emplace_back("../steady_b_artifacts");
    candidates.emplace_back("runtime/steady_b_artifacts");
  }

  for (const auto& candidate : candidates) {
    if (steady_batch_dir_valid(candidate)) return weakly_canonical_string(candidate);
  }

  if (required) {
    fs::path first = configured.empty() ? fs::path("./steady_b_artifacts") : fs::path(configured);
    auto validation = runtime_io::validate_steady_batch_dir(first.string());
    if (!validation.ok) throw std::runtime_error(validation.error);
    throw std::runtime_error("steady batch artifacts invalid under: " + first.string());
  }
  return configured.empty() ? "./steady_b_artifacts" : configured;
}

struct Summary {
  size_t n = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double mean = 0.0;
  double max = 0.0;
};

Summary summarize(std::vector<double> values) {
  Summary out;
  out.n = values.size();
  if (values.empty()) return out;
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    size_t idx = static_cast<size_t>(std::llround(p * static_cast<double>(values.size() - 1)));
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
  };
  out.p50 = percentile(0.50);
  out.p95 = percentile(0.95);
  out.p99 = percentile(0.99);
  out.max = values.back();
  out.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  return out;
}

std::string value_stats_json(const Summary& s) {
  std::ostringstream oss;
  oss << std::setprecision(17);
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

std::string scheduler_us_stats_json(const std::vector<double>& values) {
  Summary s = summarize(values);
  std::ostringstream oss;
  oss << std::setprecision(17);
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

double pct_clamped(double numerator, double denominator) {
  if (denominator <= 0.0) return 0.0;
  double pct = 100.0 * numerator / denominator;
  if (pct < 0.0) return 0.0;
  if (pct > 100.0) return 100.0;
  return pct;
}

std::string scheduler_telemetry_json(const BatchedSteadySchedulerTelemetry& telemetry) {
  std::ostringstream oss;
  oss << std::setprecision(17);
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
      << ",\"queue_depth\":" << value_stats_json(summarize(telemetry.queue_depth))
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
        << ",\"queue_depth\":" << value_stats_json(summarize(lane.queue_depth))
        << "}";
  }
  oss << "]}";
  return oss.str();
}

std::string python_admission_json(const DensityAdmission& admission) {
  AdmissionTelemetry telemetry = admission.telemetry_snapshot();
  uint64_t backlog_count = telemetry.active_count + telemetry.backlog_count;
  std::ostringstream oss;
  oss << "{\"enabled\":true"
      << ",\"attempted\":" << telemetry.offered
      << ",\"admitted\":" << telemetry.admitted
      << ",\"rejected\":" << telemetry.shed_close_count
      << ",\"max_backlog\":" << telemetry.backlog_peak
      << ",\"max_ready_age_ms\":0"
      << ",\"signal\":{"
      << "\"queued_events\":" << telemetry.backlog_count
      << ",\"ready_count\":0"
      << ",\"backlog_count\":" << backlog_count
      << ",\"oldest_ready_age_ms\":0"
      << ",\"oldest_ready_session_id\":null"
      << "}}";
  return oss.str();
}

struct ServerConfig {
  int port = -1;
  bool port_set = false;
  uint64_t admission_active_cap = 0;
  bool admission_active_cap_set = false;
  uint64_t admission_backlog_cap = kDefaultAdmissionBacklogCap;
  bool admission_backlog_cap_set = false;
  std::string steady_batch_dir = "./steady_b_artifacts";
  std::string effective_steady_batch_dir = "./steady_b_artifacts";
  bool steady_batch_dir_explicit = false;
  std::string process_label;
  bool selftest_and_exit = false;
  bool print_config = false;
  std::string artifact_dir;
  std::string argv0;

  bool scheduler_enabled = false;
  bool steady_shadow_enabled = false;
  int batch_b_max = kDefaultBatchMax;
  int batch_window_ms = kDefaultBatchWindowMs;
  int batch_lone_timeout_ms = kDefaultBatchLoneTimeoutMs;
  int batch_max_queue_delay_ms = kDefaultBatchMaxQueueDelayMs;
  int batch_queue_capacity = kDefaultBatchQueueCapacity;
  bool batch_min_fill_enabled = true;
  bool batch_disable_min_fill = false;
  int batch_force_bucket = 0;
  int device_index = 0;
  int steady_num_runners = 1;
  int steady_dispatch_lanes = 1;
  int finalize_num_runners = 1;
  int ws_lanes = 1;
  bool background_warmup_enabled = true;
  int warm_sync_lanes = 4;

  bool stats_enabled = true;
  size_t stats_window = kDefaultStatsWindow;
  size_t ws_max_message_size = kDefaultWsMaxMessageSize;
  int ws_send_timeout_sec = kDefaultWsSendTimeoutSec;
  int ws_ping_interval_sec = kDefaultWsPingIntervalSec;
  int ws_pong_timeout_sec = kDefaultWsPongTimeoutSec;
  int shutdown_drain_sec = kDefaultShutdownDrainSec;
  int finalize_silence_ms = kDefaultFinalizeSilenceMs;

  int selftest_ws_close_delay_ms = 0;
  bool selftest_lightweight_runtime = false;
};

void populate_env_config(ServerConfig* cfg) {
  cfg->scheduler_enabled = read_env_enabled("NEMOTRON_WS_SCHEDULER", true);
  cfg->steady_shadow_enabled = read_env_enabled("NEMOTRON_WS_STEADY_SHADOW", false);
  cfg->batch_b_max = read_env_int("NEMOTRON_DENSITY_BATCH_MAX", kDefaultBatchMax);
  cfg->batch_window_ms = read_env_int("NEMOTRON_DENSITY_BATCH_WINDOW_MS", kDefaultBatchWindowMs);
  cfg->batch_lone_timeout_ms =
      read_env_int("NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS", kDefaultBatchLoneTimeoutMs);
  cfg->batch_max_queue_delay_ms =
      read_env_int("NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS", kDefaultBatchMaxQueueDelayMs);
  cfg->batch_queue_capacity =
      read_env_int("NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY", kDefaultBatchQueueCapacity);
  cfg->batch_min_fill_enabled = read_env_enabled("NEMOTRON_DENSITY_BATCH_MIN_FILL", true);
  cfg->batch_disable_min_fill = read_env_enabled("NEMOTRON_DENSITY_BATCH_DISABLE_MIN_FILL", false);
  cfg->batch_force_bucket = read_env_int("NEMOTRON_DENSITY_BATCH_FORCE_BUCKET", 0);
  cfg->device_index = read_env_int("NEMOTRON_DENSITY_DEVICE_INDEX", 0);
  cfg->steady_num_runners = read_env_int("NEMOTRON_DENSITY_STEADY_RUNNERS", 1);
  cfg->steady_dispatch_lanes = read_env_int("NEMOTRON_DENSITY_DISPATCH_LANES", cfg->steady_dispatch_lanes);
  cfg->stats_enabled = read_env_enabled("NEMOTRON_STATS_ENABLED", true);
  cfg->stats_window = read_env_size_t("NEMOTRON_STATS_WINDOW", kDefaultStatsWindow);
  cfg->background_warmup_enabled = read_env_enabled("NEMOTRON_WS_BACKGROUND_WARMUP", true);
  if (cfg->background_warmup_enabled) {
    cfg->warm_sync_lanes = read_env_int("NEMOTRON_WS_WARM_SYNC_LANES", 4);
  }
  cfg->ws_max_message_size =
      read_env_size_t("NEMOTRON_WS_MAX_MESSAGE_SIZE", kDefaultWsMaxMessageSize);
  cfg->ws_send_timeout_sec = read_env_int("NEMOTRON_WS_SEND_TIMEOUT_SEC", kDefaultWsSendTimeoutSec);
  cfg->ws_ping_interval_sec = read_env_int("NEMOTRON_WS_PING_INTERVAL_SEC", kDefaultWsPingIntervalSec);
  cfg->ws_pong_timeout_sec = read_env_int("NEMOTRON_WS_PONG_TIMEOUT_SEC", kDefaultWsPongTimeoutSec);
  cfg->shutdown_drain_sec = read_env_int("NEMOTRON_SHUTDOWN_DRAIN_SEC", kDefaultShutdownDrainSec);
  cfg->finalize_silence_ms = read_env_int("NEMOTRON_FINALIZE_SILENCE_MS", kDefaultFinalizeSilenceMs);

  if (!cfg->admission_active_cap_set) {
    auto env_active = read_env_u64_optional("NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP");
    if (env_active.has_value()) {
      cfg->admission_active_cap = *env_active;
      cfg->admission_active_cap_set = true;
    }
  }
  if (!cfg->admission_backlog_cap_set) {
    cfg->admission_backlog_cap = read_env_u64("NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP",
                                              kDefaultAdmissionBacklogCap);
  }

  auto env_lanes = read_env_int_optional("NEMOTRON_WS_LANES");
  if (env_lanes.has_value()) {
    cfg->ws_lanes = *env_lanes;
    if (cfg->ws_lanes <= 0) throw std::runtime_error("NEMOTRON_WS_LANES must be positive");
    cfg->admission_active_cap = static_cast<uint64_t>(cfg->ws_lanes);
    cfg->admission_active_cap_set = true;
  } else if (cfg->admission_active_cap_set) {
    if (cfg->admission_active_cap > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("--admission-active-cap is too large for NEMOTRON_WS_LANES default");
    }
    cfg->ws_lanes = static_cast<int>(cfg->admission_active_cap);
  }

  auto env_finalize_runners = read_env_int_optional("NEMOTRON_WS_FINALIZE_RUNNERS");
  if (env_finalize_runners.has_value()) {
    cfg->finalize_num_runners = *env_finalize_runners;
  } else {
    uint64_t active_cap = cfg->admission_active_cap_set
                              ? cfg->admission_active_cap
                              : static_cast<uint64_t>(cfg->ws_lanes);
    cfg->finalize_num_runners = capped_general_finalize_runners(active_cap);
  }
}

void validate_config(ServerConfig* cfg, bool require_port_and_admission) {
  populate_env_config(cfg);
  cfg->artifact_dir = resolve_artifact_dir(cfg->argv0);
  bool scheduler_artifacts_required = cfg->scheduler_enabled && !cfg->selftest_lightweight_runtime;
  cfg->effective_steady_batch_dir = resolve_steady_batch_dir(cfg->steady_batch_dir,
                                                             cfg->steady_batch_dir_explicit,
                                                             cfg->artifact_dir,
                                                             cfg->argv0,
                                                             scheduler_artifacts_required);
  if (require_port_and_admission && !cfg->port_set) {
    throw std::runtime_error("warning: --port is required; no compiled default is provided");
  }
  if (cfg->port_set && (cfg->port < 0 || cfg->port > 65535)) {
    throw std::runtime_error("--port must be in [0, 65535]");
  }
  if (require_port_and_admission && !cfg->admission_active_cap_set) {
    throw std::runtime_error("--admission-active-cap or NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP is required");
  }
  if (cfg->steady_shadow_enabled && !cfg->scheduler_enabled) {
    throw std::runtime_error("NEMOTRON_WS_STEADY_SHADOW requires NEMOTRON_WS_SCHEDULER=1");
  }
  if (cfg->admission_active_cap_set && cfg->admission_active_cap == 0) {
    throw std::runtime_error("--admission-active-cap must be positive");
  }
  if (cfg->ws_lanes <= 0) {
    throw std::runtime_error("NEMOTRON_WS_LANES must be positive");
  }
  if (cfg->background_warmup_enabled && cfg->warm_sync_lanes <= 0) {
    throw std::runtime_error("NEMOTRON_WS_WARM_SYNC_LANES must be positive");
  }
  if (cfg->batch_b_max != 1 && cfg->batch_b_max != 2 && cfg->batch_b_max != 4 &&
      cfg->batch_b_max != 8 && cfg->batch_b_max != 16) {
    throw std::runtime_error("NEMOTRON_DENSITY_BATCH_MAX must be one of 1, 2, 4, 8, 16");
  }
  if (cfg->batch_window_ms < 0 || cfg->batch_lone_timeout_ms < 0 ||
      cfg->batch_max_queue_delay_ms < 0 ||
      cfg->batch_queue_capacity <= 0) {
    throw std::runtime_error("batch scheduler timing/capacity env vars must be non-negative, with capacity > 0");
  }
  if (cfg->batch_force_bucket != 0 && cfg->batch_force_bucket != 1 && cfg->batch_force_bucket != 2 &&
      cfg->batch_force_bucket != 4 && cfg->batch_force_bucket != 8 &&
      cfg->batch_force_bucket != 16) {
    throw std::runtime_error("NEMOTRON_DENSITY_BATCH_FORCE_BUCKET must be 0 or one of 1, 2, 4, 8, 16");
  }
  if (scheduler_artifacts_required) {
    BatchedSteadySchedulerPolicy policy;
    policy.B_max = cfg->batch_b_max;
    policy.force_bucket = cfg->batch_force_bucket;
    const auto required_buckets = BatchedSteadyScheduler::required_buckets_for_policy(policy);
    auto validation = runtime_io::validate_steady_batch_dir(cfg->effective_steady_batch_dir);
    std::vector<int> missing;
    for (int bucket : required_buckets) {
      if (std::find(validation.buckets.begin(), validation.buckets.end(), bucket) ==
          validation.buckets.end()) {
        missing.push_back(bucket);
      }
    }
    if (!missing.empty()) {
      std::ostringstream oss;
      oss << "steady batch dir " << cfg->effective_steady_batch_dir
          << " is missing scheduler-required bucket(s)";
      for (int bucket : missing) oss << " B=" << bucket;
      if (cfg->batch_b_max >= 16 || cfg->batch_force_bucket == 16) {
        oss << "; production default B_max=16 requires bucket 16 "
            << "(enc_steady_aoti_b16.pt2). Deploy steady_b_artifacts_b16.";
      }
      throw std::runtime_error(oss.str());
    }
  }
  if (cfg->steady_num_runners <= 0 || cfg->finalize_num_runners <= 0) {
    throw std::runtime_error("runner counts must be positive");
  }
  if (cfg->steady_dispatch_lanes <= 0 || cfg->steady_dispatch_lanes > 2) {
    throw std::runtime_error("NEMOTRON_DENSITY_DISPATCH_LANES must be 1 or 2");
  }
  if (cfg->scheduler_enabled && cfg->steady_dispatch_lanes > 1 &&
      read_env_on_off("NEMOTRON_WS_STEADY_CUDAGRAPH", false)) {
    throw std::runtime_error(
        "NEMOTRON_DENSITY_DISPATCH_LANES > 1 is incompatible with NEMOTRON_WS_STEADY_CUDAGRAPH=1");
  }
  if (cfg->scheduler_enabled && cfg->steady_num_runners < cfg->steady_dispatch_lanes) {
    std::cout << "N1 auto-raise: steady_runners " << cfg->steady_num_runners
              << " -> " << cfg->steady_dispatch_lanes
              << " for steady_dispatch_lanes=" << cfg->steady_dispatch_lanes
              << " model=shared_loader_num_runners\n";
    cfg->steady_num_runners = cfg->steady_dispatch_lanes;
  }
  if (cfg->ws_send_timeout_sec <= 0 || cfg->ws_ping_interval_sec <= 0 ||
      cfg->ws_pong_timeout_sec <= 0 || cfg->shutdown_drain_sec < 0) {
    throw std::runtime_error("WS timeout env vars must be positive; shutdown drain must be non-negative");
  }
}

std::string config_table(const ServerConfig& cfg) {
  std::ostringstream oss;
  oss << "[runtime]\n"
      << "  scheduler_env = NEMOTRON_WS_SCHEDULER\n"
      << "  scheduler_owner = SharedRuntime\n"
      << "  scheduler_enabled = " << json_bool(cfg.scheduler_enabled) << "\n"
      << "  prewarm_env = NEMOTRON_WS_PREWARM\n"
      << "  prewarm_enabled = " << json_bool(read_env_enabled("NEMOTRON_WS_PREWARM", true)) << "\n"
      << "  steady_shadow_env = NEMOTRON_WS_STEADY_SHADOW\n"
      << "  steady_shadow_enabled = " << json_bool(cfg.steady_shadow_enabled) << "\n"
      << "  steady_shadow_timing = INVALID_WHEN_ENABLED\n"
      << "  lanes = " << cfg.ws_lanes << "\n"
      << "  background_warmup_enabled = " << json_bool(cfg.background_warmup_enabled) << "\n"
      << "  warm_sync_lanes = " << cfg.warm_sync_lanes << "\n"
      << "  steady_runners = " << cfg.steady_num_runners << "\n"
      << "  steady_dispatch_lanes = " << cfg.steady_dispatch_lanes << "\n"
      << "  steady_runner_model = shared_loader_num_runners>=dispatch_lanes\n"
      << "  finalize_runners = " << cfg.finalize_num_runners << "\n"
      << "  steady_batch_dir = " << cfg.effective_steady_batch_dir << "\n"
      << "  batch_max = " << cfg.batch_b_max << "\n"
      << "  batch_window_ms = " << cfg.batch_window_ms << "\n"
      << "  batch_lone_timeout_ms = " << cfg.batch_lone_timeout_ms << "\n"
      << "  batch_max_queue_delay_ms = " << cfg.batch_max_queue_delay_ms << "\n"
      << "  batch_min_fill_enabled = " << json_bool(cfg.batch_min_fill_enabled) << "\n"
      << "  batch_disable_min_fill = " << json_bool(cfg.batch_disable_min_fill) << "\n"
      << "  batch_force_bucket = " << cfg.batch_force_bucket << "\n"
      << "  steady_cuda_graph_default = off\n"
      << "\n[admission]\n"
      << "  active_cap = " << cfg.admission_active_cap << "\n"
      << "  backlog_cap = " << cfg.admission_backlog_cap << "\n"
      << "\n[stats]\n"
      << "  enabled = " << json_bool(cfg.stats_enabled) << "\n"
      << "  window_size = " << cfg.stats_window << "\n"
      << "\n[ws]\n"
      << "  max_message_size = " << cfg.ws_max_message_size << "\n"
      << "  send_timeout_sec = " << cfg.ws_send_timeout_sec << "\n"
      << "  ping_interval_sec = " << cfg.ws_ping_interval_sec << "\n"
      << "  pong_timeout_sec = " << cfg.ws_pong_timeout_sec << "\n"
      << "\n[shutdown]\n"
      << "  drain_sec = " << cfg.shutdown_drain_sec << "\n";
  return oss.str();
}

ServerConfig parse_args(int argc, char** argv) {
  ServerConfig cfg;
  if (argc > 0 && argv[0] != nullptr) cfg.argv0 = argv[0];
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " requires a value");
      return argv[++i];
    };
    if (arg == "--port") {
      cfg.port = parse_int_strict(need_value("--port"), "--port");
      cfg.port_set = true;
    } else if (arg == "--admission-active-cap") {
      cfg.admission_active_cap = parse_u64_strict(need_value("--admission-active-cap"),
                                                  "--admission-active-cap");
      cfg.admission_active_cap_set = true;
    } else if (arg == "--admission-backlog-cap") {
      cfg.admission_backlog_cap = parse_u64_strict(need_value("--admission-backlog-cap"),
                                                   "--admission-backlog-cap");
      cfg.admission_backlog_cap_set = true;
    } else if (arg == "--steady-batch-dir") {
      cfg.steady_batch_dir = need_value("--steady-batch-dir");
      cfg.steady_batch_dir_explicit = true;
    } else if (arg == "--steady-dispatch-lanes") {
      cfg.steady_dispatch_lanes = parse_int_strict(need_value("--steady-dispatch-lanes"),
                                                   "--steady-dispatch-lanes");
    } else if (arg == "--process-label") {
      cfg.process_label = need_value("--process-label");
    } else if (arg == "--selftest-and-exit") {
      cfg.selftest_and_exit = true;
    } else if (arg == "--print-config") {
      cfg.print_config = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "usage: ws_server --port <int> --admission-active-cap <int> [options]\n"
          << "options:\n"
          << "  --admission-backlog-cap <int>   default 12 or env override\n"
          << "  --steady-batch-dir <path>       default ./steady_b_artifacts\n"
          << "  --steady-dispatch-lanes <int>   default 1 or NEMOTRON_DENSITY_DISPATCH_LANES\n"
          << "  --process-label <str>\n"
          << "  --print-config\n"
          << "  --selftest-and-exit\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return cfg;
}

struct ReadHttpOutcome {
  ws_handshake::ParseResult result = ws_handshake::ParseResult::NEED_MORE;
  ws_handshake::HttpRequest request;
};

ReadHttpOutcome read_http_request_from_socket(int fd) {
  ReadHttpOutcome out;
  std::string buffer;
  buffer.reserve(ws_handshake::kMaxHttpHeaderBytes);
  auto deadline = Clock::now() + std::chrono::milliseconds(ws_handshake::kHttpReadTimeoutMs);

  for (;;) {
    out.result = ws_handshake::parse_http_request(buffer, out.request);
    if (out.result != ws_handshake::ParseResult::NEED_MORE) return out;

    auto now = Clock::now();
    if (now >= deadline) {
      out.result = ws_handshake::ParseResult::MALFORMED;
      return out;
    }
    int timeout_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
    if (timeout_ms < 0) timeout_ms = 0;
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    int pr = ::poll(&pfd, 1, timeout_ms);
    if (pr == 0) {
      out.result = ws_handshake::ParseResult::MALFORMED;
      return out;
    }
    if (pr < 0) {
      if (errno == EINTR) continue;
      out.result = ws_handshake::ParseResult::MALFORMED;
      return out;
    }
    char chunk[1024];
    ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
    if (n > 0) {
      buffer.append(chunk, static_cast<size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    out.result = ws_handshake::ParseResult::MALFORMED;
    return out;
  }
}

std::optional<size_t> parse_last_query(const ws_routes::Route& route, std::string* error_body) {
  auto it = route.query_params.find("last");
  if (it == route.query_params.end() || it->second.empty()) return std::nullopt;
  const std::string& raw = it->second;
  bool digits = std::all_of(raw.begin(), raw.end(), [](unsigned char ch) { return std::isdigit(ch); });
  if (!digits) {
    *error_body = "{\"error\":" + json_quote("invalid 'last': '" + raw + "'") + "}";
    return std::nullopt;
  }
  uint64_t parsed = 0;
  try {
    parsed = parse_u64_strict(raw, "last");
  } catch (const std::exception&) {
    *error_body = "{\"error\":" + json_quote("invalid 'last': '" + raw + "'") + "}";
    return std::nullopt;
  }
  if (parsed == 0 || parsed > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    *error_body = "{\"error\":" + json_quote("invalid 'last': '" + raw + "'") + "}";
    return std::nullopt;
  }
  return static_cast<size_t>(parsed);
}

struct ActiveWsSession {
  ActiveWsSession(std::string id, int socket_fd) : stream_id(std::move(id)), fd(socket_fd) {}

  std::string stream_id;
  int fd = -1;
  std::atomic<bool> drain_requested{false};
  std::atomic<bool> forced{false};
  std::atomic<bool> closed{false};
  std::mutex send_mutex;
};

struct ServerState {
  explicit ServerState(ServerConfig c) : cfg(std::move(c)) {}

  ServerConfig cfg;
  std::unique_ptr<SharedRuntime> shared_runtime;
  std::unique_ptr<DensityAdmission> admission;
  std::unique_ptr<StatsCollector> stats;
  std::atomic<bool> model_loaded{false};
  std::atomic<uint64_t> next_stream_id{1};
  std::atomic<bool> shutting_down{false};
  std::mutex sessions_mutex;
  std::unordered_map<std::string, std::shared_ptr<ActiveWsSession>> sessions;
};

bool is_shutting_down(const ServerState& state) {
  return state.shutting_down.load(std::memory_order_acquire) ||
         (state.admission && state.admission->is_shutting_down());
}

std::shared_ptr<ActiveWsSession> register_session(const std::shared_ptr<ServerState>& state,
                                                  const std::string& stream_id,
                                                  int fd) {
  auto session = std::make_shared<ActiveWsSession>(stream_id, fd);
  if (is_shutting_down(*state)) {
    session->drain_requested.store(true, std::memory_order_release);
  }
  std::lock_guard<std::mutex> lock(state->sessions_mutex);
  state->sessions[stream_id] = session;
  return session;
}

void unregister_session(const std::shared_ptr<ServerState>& state,
                        const std::shared_ptr<ActiveWsSession>& session) {
  if (!session) return;
  session->closed.store(true, std::memory_order_release);
  std::lock_guard<std::mutex> lock(state->sessions_mutex);
  auto it = state->sessions.find(session->stream_id);
  if (it != state->sessions.end() && it->second == session) {
    state->sessions.erase(it);
  }
}

std::vector<std::shared_ptr<ActiveWsSession>> session_snapshot(const std::shared_ptr<ServerState>& state) {
  std::vector<std::shared_ptr<ActiveWsSession>> out;
  std::lock_guard<std::mutex> lock(state->sessions_mutex);
  out.reserve(state->sessions.size());
  for (const auto& kv : state->sessions) out.push_back(kv.second);
  return out;
}

size_t active_session_count(const std::shared_ptr<ServerState>& state) {
  std::lock_guard<std::mutex> lock(state->sessions_mutex);
  return state->sessions.size();
}

void request_session_drains(const std::shared_ptr<ServerState>& state) {
  for (const auto& session : session_snapshot(state)) {
    session->drain_requested.store(true, std::memory_order_release);
  }
}

std::string health_json(const ServerState& state) {
  bool loaded = state.model_loaded.load(std::memory_order_acquire);
  std::ostringstream oss;
  oss << "{\"status\":\"" << (loaded ? "healthy" : "loading") << "\""
      << ",\"model_loaded\":" << json_bool(loaded);
  if (state.admission) {
    oss << ",\"admission\":" << python_admission_json(*state.admission);
  }
  if (!state.cfg.process_label.empty()) {
    oss << ",\"pid\":" << static_cast<long long>(::getpid())
        << ",\"process_label\":" << json_quote(state.cfg.process_label);
  }
  oss << "}";
  return oss.str();
}

std::string stale_gen_drops_json() {
  ws_server::StaleGenDrops drops = ws_server::stale_gen_drops();
  std::ostringstream oss;
  oss << "{\"drops_at_event_emit\":" << drops.drops_at_event_emit
      << ",\"drops_at_finalize_output\":" << drops.drops_at_finalize_output
      << "}";
  return oss.str();
}

struct AdminJob {
  int fd = -1;
  ws_routes::Route route;
};

class AdminHandlerPool {
 public:
  explicit AdminHandlerPool(std::shared_ptr<ServerState> state) : state_(std::move(state)) {}
  AdminHandlerPool(const AdminHandlerPool&) = delete;
  AdminHandlerPool& operator=(const AdminHandlerPool&) = delete;

  void start() {
    for (int i = 0; i < kAdminWorkers; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  bool try_enqueue(AdminJob job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || queue_.size() >= kAdminQueueDepth) return false;
    queue_.push_back(std::move(job));
    cv_.notify_one();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
    workers_.clear();
  }

 private:
  void worker_loop() {
    for (;;) {
      AdminJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stopping_) break;
          continue;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      handle_job(std::move(job));
    }
  }

  void handle_job(AdminJob job) {
    UniqueFd fd(job.fd);
    std::string body;
    int status = 200;
    try {
      if (job.route.path == kStaleGenTestPath && stale_gen_test_endpoint_enabled()) {
        body = stale_gen_drops_json();
      } else switch (job.route.kind) {
        case ws_routes::RouteKind::HEALTH:
          body = health_json(*state_);
          if (is_shutting_down(*state_)) status = 503;
          break;
        case ws_routes::RouteKind::STATS: {
          std::string error_body;
          std::optional<size_t> last = parse_last_query(job.route, &error_body);
          if (!error_body.empty()) {
            status = 400;
            body = std::move(error_body);
          } else {
            body = state_->stats->snapshot_json(last);
          }
          break;
        }
        case ws_routes::RouteKind::SCHEDULER_TELEMETRY:
          if (!state_->shared_runtime || !state_->shared_runtime->has_scheduler()) {
            status = 404;
            body = "{\"error\":\"no scheduler\"}";
          } else {
            body = scheduler_telemetry_json(state_->shared_runtime->scheduler_telemetry_snapshot());
          }
          break;
        default:
          status = 404;
          body = "{\"error\":\"not_found\"}";
          break;
      }
    } catch (const std::exception& e) {
      status = 503;
      body = "{\"error\":" + json_quote(e.what()) + "}";
    }
    (void)send_all(fd.get(), build_json_response(status, body));
  }

  std::shared_ptr<ServerState> state_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<AdminJob> queue_;
  bool stopping_ = false;
};

class AdmissionCloseGuard {
 public:
  AdmissionCloseGuard(DensityAdmission* admission, std::string stream_id)
      : admission_(admission), stream_id_(std::move(stream_id)) {}
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

std::string wire_event_json(const WireEvent& event) {
  std::ostringstream oss;
  oss << "{\"type\":" << json_quote(event.type);
  if (event.text.has_value()) oss << ",\"text\":" << json_quote(*event.text);
  if (event.is_final.has_value()) oss << ",\"is_final\":" << json_bool(*event.is_final);
  if (event.finalize.has_value()) oss << ",\"finalize\":" << json_bool(*event.finalize);
  if (event.finalize_timing.has_value()) {
    oss << ",\"finalize_timing\":" << event.finalize_timing->dump();
  }
  if (event.message.has_value()) oss << ",\"message\":" << json_quote(*event.message);
  if (event.language.has_value()) {
    // Mirror the Python server payload, which carries both keys.
    oss << ",\"language\":" << json_quote(*event.language)
        << ",\"language_tag\":" << json_quote(*event.language);
  }
  oss << "}";
  return oss.str();
}

std::string error_event_json(const std::string& message) {
  WireEvent event;
  event.type = "error";
  event.message = message;
  return wire_event_json(event);
}

bool send_ws_frame_checked(int fd,
                           const std::vector<uint8_t>& frame,
                           const std::shared_ptr<ActiveWsSession>& active,
                           bool* protocol_close_sent) {
  std::unique_lock<std::mutex> lock;
  if (active) lock = std::unique_lock<std::mutex>(active->send_mutex);
  SendResult result = send_all_result(fd, frame.data(), frame.size());
  if (result == SendResult::OK) return true;
  if (result == SendResult::TIMEOUT) {
    if (protocol_close_sent != nullptr) *protocol_close_sent = true;
    std::vector<uint8_t> close = ws_framing::write_close_frame(1011, "send_timeout");
    (void)send_all_result(fd, close.data(), close.size());
  }
  return false;
}

bool send_ws_text(int fd,
                  const std::string& json,
                  const std::shared_ptr<ActiveWsSession>& active,
                  bool* protocol_close_sent) {
  return send_ws_frame_checked(fd,
                               ws_framing::write_frame(ws_framing::Opcode::TEXT, json),
                               active,
                               protocol_close_sent);
}

bool send_ws_close(int fd,
                   uint16_t code,
                   std::string_view reason,
                   const std::shared_ptr<ActiveWsSession>& active,
                   bool* protocol_close_sent) {
  if (protocol_close_sent != nullptr) *protocol_close_sent = true;
  return send_ws_frame_checked(fd, ws_framing::write_close_frame(code, reason), active, protocol_close_sent);
}

bool send_ws_control(int fd,
                     ws_framing::Opcode opcode,
                     std::string_view payload,
                     const std::shared_ptr<ActiveWsSession>& active,
                     bool* protocol_close_sent) {
  return send_ws_frame_checked(fd, ws_framing::write_frame(opcode, payload), active, protocol_close_sent);
}

void best_effort_force_close(const std::shared_ptr<ActiveWsSession>& session) {
  if (!session || session->closed.load(std::memory_order_acquire)) return;
  session->forced.store(true, std::memory_order_release);
  std::vector<uint8_t> close = ws_framing::write_close_frame(1001, "going_away");
  std::unique_lock<std::mutex> lock(session->send_mutex, std::try_to_lock);
  if (lock.owns_lock()) {
    (void)::send(session->fd, close.data(), close.size(), MSG_NOSIGNAL | MSG_DONTWAIT);
  }
  (void)::shutdown(session->fd, SHUT_RDWR);
}

void drain_peer_after_clean_close(int fd, int timeout_ms) {
  (void)::shutdown(fd, SHUT_WR);
  auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
  char discard[4096];
  for (;;) {
    auto now = Clock::now();
    if (now >= deadline) return;
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    int wait_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
    if (wait_ms < 0) wait_ms = 0;
    int pr = ::poll(&pfd, 1, wait_ms);
    if (pr <= 0) return;
    if (pfd.revents & (POLLERR | POLLNVAL)) return;
    if (pfd.revents & (POLLIN | POLLHUP)) {
      ssize_t n = ::recv(fd, discard, sizeof(discard), 0);
      if (n > 0) continue;
      return;
    }
  }
}

std::string accepted_model_aliases_text(const std::vector<std::string>& aliases) {
  std::ostringstream oss;
  for (size_t i = 0; i < aliases.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << aliases[i];
  }
  return oss.str();
}

std::vector<std::string> accepted_model_aliases(const ServerState& state) {
  std::vector<std::string> aliases;
  auto add = [&](std::string value) {
    value = trim_ascii(std::move(value));
    if (value.empty()) return;
    value = ascii_lower(std::move(value));
    if (std::find(aliases.begin(), aliases.end(), value) == aliases.end()) {
      aliases.push_back(std::move(value));
    }
  };

  const char* env_model = std::getenv("NEMOTRON_MODEL_NAME");
  if (env_model != nullptr) add(env_model);
  add(MODEL_ID);
  fs::path model_path(MODEL_ID);
  add(model_path.filename().string());
  add(model_path.stem().string());
  add("english");
  add("en");
  add("nvidia/nemotron-speech-streaming-en-0.6b");
  add("nemotron-speech-streaming-en-0.6b");
  (void)state;
  return aliases;
}

std::optional<std::string> validate_ws_query(const ws_routes::Route& route,
                                             const ServerState& state) {
  auto model_it = route.query_params.find("model");
  std::string model = model_it == route.query_params.end() ? std::string() : trim_ascii(model_it->second);
  if (!model.empty()) {
    std::vector<std::string> aliases = accepted_model_aliases(state);
    std::string requested = ascii_lower(model);
    if (std::find(aliases.begin(), aliases.end(), requested) == aliases.end()) {
      return "model mismatch: requested " + model + "; server accepts: " +
             accepted_model_aliases_text(aliases);
    }
  }

  auto language_it = route.query_params.find("language");
  std::string language =
      language_it == route.query_params.end() ? std::string() : trim_ascii(language_it->second);
  if (!language.empty()) {
    if (!MODEL_PROMPTED) {
      return "this model does not accept a language argument";
    }
    const PromptTable& table = prompt_runtime_table();
    if (table.lang_to_index.find(language) == table.lang_to_index.end()) {
      std::vector<std::string> supported;
      supported.reserve(table.lang_to_index.size());
      for (const auto& kv : table.lang_to_index) supported.push_back(kv.first);
      std::sort(supported.begin(), supported.end());
      std::string joined;
      for (const auto& lang : supported) {
        if (!joined.empty()) joined += ", ";
        joined += lang;
      }
      return "unsupported language " + language + "; supported: " + joined;
    }
  }
  return std::nullopt;
}

int selftest_drain_hold_ms(const ws_routes::Route& route) {
  if (!stale_gen_test_endpoint_enabled()) return 0;
  auto it = route.query_params.find("__selftest_drain_hold_ms");
  if (it == route.query_params.end() || it->second.empty()) return 0;
  int value = parse_int_strict(it->second, "__selftest_drain_hold_ms");
  return std::max(0, value);
}

struct PendingOutput {
  enum class Kind {
    EVENT,
    FINALIZE,
  };

  Kind kind = Kind::EVENT;
  uint64_t generation = 0;
  std::vector<WireEvent> events;
  std::optional<SessionTiming> timing;
};

bool event_is_finalize_output(const WireEvent& event) {
  return event.is_final.value_or(false) && event.finalize.value_or(false);
}

void enqueue_event_outputs(std::deque<PendingOutput>* pending,
                           std::vector<WireEvent> events,
                           uint64_t generation) {
  if (events.empty()) return;
  PendingOutput output;
  output.kind = PendingOutput::Kind::EVENT;
  output.generation = generation;
  output.events = std::move(events);
  pending->push_back(std::move(output));
}

void enqueue_finalize_output(std::deque<PendingOutput>* pending,
                             std::vector<WireEvent> events,
                             uint64_t generation,
                             const SessionRuntime& session,
                             uint64_t* last_enqueued_finalize_seq) {
  std::optional<SessionTiming> timing = session.last_timing();
  if (!timing.has_value() || timing->finalize_seq == *last_enqueued_finalize_seq) {
    enqueue_event_outputs(pending, std::move(events), generation);
    return;
  }

  PendingOutput output;
  output.kind = PendingOutput::Kind::FINALIZE;
  output.generation = generation;
  output.timing = *timing;
  for (auto& event : events) {
    if (event_is_finalize_output(event)) {
      output.events.push_back(std::move(event));
    }
  }
  *last_enqueued_finalize_seq = timing->finalize_seq;
  pending->push_back(std::move(output));
}

bool flush_pending_outputs(int fd,
                           const SessionRuntime& session,
                           StatsCollector& stats_collector,
                           std::deque<PendingOutput>* pending,
                           const std::shared_ptr<ActiveWsSession>& active,
                           bool* protocol_close_sent) {
  while (!pending->empty()) {
    PendingOutput output = std::move(pending->front());
    pending->pop_front();

    const bool stale = session.generation() != output.generation;
    if (output.kind == PendingOutput::Kind::EVENT) {
      if (stale) {
        ws_server::g_drops_at_event_emit.fetch_add(output.events.size(), std::memory_order_relaxed);
        continue;
      }
      for (const auto& event : output.events) {
        if (!send_ws_text(fd, wire_event_json(event), active, protocol_close_sent)) return false;
      }
      continue;
    }

    bool emitted = false;
    if (stale) {
      if (!output.events.empty()) {
        ws_server::g_drops_at_finalize_output.fetch_add(output.events.size(), std::memory_order_relaxed);
      }
    } else {
      emitted = !output.events.empty();
      for (const auto& event : output.events) {
        if (!send_ws_text(fd, wire_event_json(event), active, protocol_close_sent)) {
          emitted = false;
          break;
        }
      }
    }

    if (output.timing.has_value()) {
      SessionTiming timing = *output.timing;
      if (stale) timing.was_suppressed = true;
      stats_collector.record(timing, emitted && !stale);
    }
    if (!stale && !emitted && !output.events.empty()) return false;
  }
  return true;
}

int poll_timeout_ms(const SessionRuntime& session) {
  if (session.vad_state() != VadState::PENDING_FINALIZE) return -1;
  std::optional<double> deadline = session.vad_deadline_ts();
  if (!deadline.has_value()) return -1;
  double remaining_ms = (*deadline - unix_now_seconds()) * 1000.0;
  if (remaining_ms <= 0.0) return 0;
  if (remaining_ms > static_cast<double>(std::numeric_limits<int>::max())) return -1;
  return static_cast<int>(std::ceil(remaining_ms));
}

std::optional<bool> control_finalize_flag(const nlohmann::json& parsed) {
  auto it = parsed.find("finalize");
  if (it == parsed.end()) return std::nullopt;
  if (!it->is_boolean()) return std::nullopt;
  return it->get<bool>();
}

bool process_text_control(const ws_framing::Message& message,
                          SessionRuntime& session,
                          std::deque<PendingOutput>* pending,
                          uint64_t* last_enqueued_finalize_seq,
                          bool* should_close) {
  std::string text(message.payload.begin(), message.payload.end());
  nlohmann::json parsed = nlohmann::json::parse(text, nullptr, false);
  if (parsed.is_discarded() || !parsed.is_object()) return true;
  auto type_it = parsed.find("type");
  if (type_it == parsed.end() || !type_it->is_string()) return true;
  std::string type = type_it->get<std::string>();

  if (type == "vad_start") {
    session.handle_vad_start();
    return true;
  }

  if (type == "vad_stop") {
    uint64_t finalize_gen = session.generation();
    auto events = session.handle_vad_stop();
    enqueue_finalize_output(pending,
                            std::move(events),
                            finalize_gen,
                            session,
                            last_enqueued_finalize_seq);
    return true;
  }

  if (type == "reset") {
    bool finalize = control_finalize_flag(parsed).value_or(true);
    auto events = session.reset(finalize);
    uint64_t generation = session.generation();
    if (finalize) {
      enqueue_finalize_output(pending,
                              std::move(events),
                              generation,
                              session,
                              last_enqueued_finalize_seq);
    } else {
      enqueue_event_outputs(pending, std::move(events), generation);
    }
    return true;
  }

  if (type == "end") {
    bool finalize = control_finalize_flag(parsed).value_or(true);
    auto events = session.end(finalize);
    uint64_t generation = session.generation();
    if (finalize) {
      enqueue_finalize_output(pending,
                              std::move(events),
                              generation,
                              session,
                              last_enqueued_finalize_seq);
    } else {
      enqueue_event_outputs(pending, std::move(events), generation);
    }
    *should_close = true;
    return true;
  }

  return true;
}

bool process_binary_pcm(const ws_framing::Message& message,
                        SessionRuntime& session,
                        std::deque<PendingOutput>* pending,
                        bool* protocol_close_sent,
                        int fd,
                        const std::shared_ptr<ActiveWsSession>& active) {
  if ((message.payload.size() % 2) != 0) {
    (void)send_ws_close(fd, 1003, "unsupported_data", active, protocol_close_sent);
    return false;
  }

  uint64_t interim_gen = session.generation();
  PCMFrame pcm;
  pcm.samples = reinterpret_cast<const int16_t*>(message.payload.data());
  pcm.count = message.payload.size() / sizeof(int16_t);
  auto events = session.append_pcm_and_drain(pcm);
  enqueue_event_outputs(pending, std::move(events), interim_gen);
  return true;
}

void cap_poll_timeout(int* timeout_ms, int cap_ms) {
  if (cap_ms < 0) return;
  if (*timeout_ms < 0 || cap_ms < *timeout_ms) *timeout_ms = cap_ms;
}

int ms_until(Clock::time_point deadline, Clock::time_point now) {
  if (deadline <= now) return 0;
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
  if (ms > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
  return static_cast<int>(ms);
}

bool enqueue_shutdown_finalize(SessionRuntime& session,
                               std::deque<PendingOutput>* pending,
                               uint64_t* last_enqueued_finalize_seq) {
  auto events = session.reset(true);
  uint64_t generation = session.generation();
  enqueue_finalize_output(pending,
                          std::move(events),
                          generation,
                          session,
                          last_enqueued_finalize_seq);
  return true;
}

void ws_worker(int fd, std::shared_ptr<ServerState> state, ws_routes::Route route) {
  UniqueFd conn(fd);
  std::shared_ptr<ActiveWsSession> active;
  bool protocol_close_sent = false;
  try {
  if (!configure_send_timeout(conn.get(), state->cfg.ws_send_timeout_sec)) {
    return;
  }
  std::string stream_id = "ws-" + std::to_string(::getpid()) + "-" +
                          std::to_string(state->next_stream_id.fetch_add(1));

  if (auto query_error = validate_ws_query(route, *state); query_error.has_value()) {
    (void)send_ws_text(conn.get(), error_event_json(*query_error), nullptr, &protocol_close_sent);
    (void)send_ws_control(conn.get(), ws_framing::Opcode::CLOSE, "", nullptr, &protocol_close_sent);
    return;
  }

  AdmitResult admit = state->admission->try_admit(stream_id);
  if (admit.shed()) {
    (void)send_ws_close(conn.get(), 1013, "admission_backpressure", nullptr, &protocol_close_sent);
    return;
  }
  if (admit.decision == AdmissionDecision::QUEUED) {
    while (!state->admission->try_admit_complete(stream_id)) {
      if (is_shutting_down(*state)) {
        (void)send_ws_close(conn.get(), 1000, "", nullptr, &protocol_close_sent);
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  AdmissionCloseGuard guard(state->admission.get(), stream_id);
  if (!state->shared_runtime) {
    (void)send_ws_close(conn.get(), 1011, "runtime_unavailable", nullptr, &protocol_close_sent);
    return;
  }

  SessionConfig session_cfg;
  session_cfg.finalize_silence_ms = state->cfg.finalize_silence_ms;
  session_cfg.active_sessions_at_emit =
      static_cast<int>(state->admission->telemetry_snapshot().active_count);
  session_cfg.label = stream_id;
  if (MODEL_PROMPTED) {
    auto language_it = route.query_params.find("language");
    if (language_it != route.query_params.end()) {
      session_cfg.language = trim_ascii(language_it->second);
    }
  }
  SessionRuntime session(*state->shared_runtime, session_cfg);
  active = register_session(state, stream_id, conn.get());

  if (!send_ws_text(conn.get(), "{\"type\":\"ready\"}", active, &protocol_close_sent)) {
    unregister_session(state, active);
    return;
  }

  std::string recv_buffer;
  recv_buffer.reserve(4096);
  ws_framing::MessageAssembler message_assembler(state->cfg.ws_max_message_size);
  std::deque<PendingOutput> pending_outputs;
  uint64_t last_enqueued_finalize_seq = 0;
  bool should_close = false;
  bool shutdown_finalize_requested = false;
  int drain_hold_ms = selftest_drain_hold_ms(route);
  bool ping_waiting_for_pong = false;
  uint64_t ping_sequence = 0;
  auto last_recv_activity = Clock::now();
  auto ping_sent_at = Clock::time_point{};

  auto drain_recv_buffer = [&]() -> bool {
    for (;;) {
      ws_framing::Frame frame;
      size_t consumed = 0;
      ws_framing::ReadResult read =
          ws_framing::read_frame(recv_buffer, frame, consumed, state->cfg.ws_max_message_size);
      if (read == ws_framing::ReadResult::NEED_MORE) return true;
      if (read == ws_framing::ReadResult::FRAME_TOO_LARGE) {
        (void)send_ws_close(conn.get(), 1009, "message_too_big", active, &protocol_close_sent);
        should_close = true;
        return false;
      }
      if (read == ws_framing::ReadResult::MALFORMED) {
        (void)send_ws_close(conn.get(), 1003, "unsupported_data", active, &protocol_close_sent);
        should_close = true;
        return false;
      }
      recv_buffer.erase(0, consumed);

      ws_framing::Message message;
      ws_framing::ReadResult assembled = message_assembler.push_frame(frame, message);
      if (assembled == ws_framing::ReadResult::NEED_MORE) continue;
      if (assembled == ws_framing::ReadResult::FRAME_TOO_LARGE) {
        (void)send_ws_close(conn.get(), 1009, "message_too_big", active, &protocol_close_sent);
        should_close = true;
        return false;
      }
      if (assembled == ws_framing::ReadResult::MALFORMED) {
        (void)send_ws_close(conn.get(), 1003, "unsupported_data", active, &protocol_close_sent);
        should_close = true;
        return false;
      }

      if (message.opcode == ws_framing::Opcode::BINARY) {
        if (!process_binary_pcm(message,
                                session,
                                &pending_outputs,
                                &protocol_close_sent,
                                conn.get(),
                                active)) {
          should_close = true;
          return false;
        }
      } else if (message.opcode == ws_framing::Opcode::TEXT) {
        if (!process_text_control(message,
                                  session,
                                  &pending_outputs,
                                  &last_enqueued_finalize_seq,
                                  &should_close)) {
          should_close = true;
          return false;
        }
      } else if (message.opcode == ws_framing::Opcode::PING) {
        std::string payload(message.payload.begin(), message.payload.end());
        if (!send_ws_control(conn.get(), ws_framing::Opcode::PONG, payload, active, &protocol_close_sent)) {
          should_close = true;
          return false;
        }
      } else if (message.opcode == ws_framing::Opcode::PONG) {
        ping_waiting_for_pong = false;
      } else if (message.opcode == ws_framing::Opcode::CLOSE) {
        should_close = true;
      }

      if (!flush_pending_outputs(conn.get(),
                                 session,
                                 *state->stats,
                                 &pending_outputs,
                                 active,
                                 &protocol_close_sent)) {
        should_close = true;
        return false;
      }
      if (should_close) return false;
    }
  };

  while (!should_close) {
    if (active->forced.load(std::memory_order_acquire)) {
      protocol_close_sent = true;
      break;
    }
    if ((active->drain_requested.load(std::memory_order_acquire) || is_shutting_down(*state)) &&
        !shutdown_finalize_requested) {
      active->drain_requested.store(true, std::memory_order_release);
      if (drain_hold_ms > 0) {
        auto hold_deadline = Clock::now() + std::chrono::milliseconds(drain_hold_ms);
        while (!active->forced.load(std::memory_order_acquire) && Clock::now() < hold_deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      }
      if (active->forced.load(std::memory_order_acquire)) {
        protocol_close_sent = true;
        break;
      }
      shutdown_finalize_requested = enqueue_shutdown_finalize(session,
                                                              &pending_outputs,
                                                              &last_enqueued_finalize_seq);
      should_close = true;
    }
    if (!flush_pending_outputs(conn.get(),
                               session,
                               *state->stats,
                               &pending_outputs,
                               active,
                               &protocol_close_sent)) {
      break;
    }
    if (should_close) break;

    if (!recv_buffer.empty()) {
      if (!drain_recv_buffer()) break;
      if (should_close) break;
    }

    auto now = Clock::now();
    if (ping_waiting_for_pong &&
        now - ping_sent_at >= std::chrono::seconds(state->cfg.ws_pong_timeout_sec)) {
      (void)send_ws_close(conn.get(), 1011, "pong_timeout", active, &protocol_close_sent);
      break;
    }
    if (!ping_waiting_for_pong &&
        now - last_recv_activity >= std::chrono::seconds(state->cfg.ws_ping_interval_sec)) {
      std::string payload = "ping:" + std::to_string(++ping_sequence);
      if (!send_ws_control(conn.get(), ws_framing::Opcode::PING, payload, active, &protocol_close_sent)) {
        break;
      }
      ping_waiting_for_pong = true;
      ping_sent_at = Clock::now();
    }

    pollfd pfd{};
    pfd.fd = conn.get();
    pfd.events = POLLIN;
    int timeout_ms = poll_timeout_ms(session);
    now = Clock::now();
    if (ping_waiting_for_pong) {
      cap_poll_timeout(&timeout_ms,
                       ms_until(ping_sent_at + std::chrono::seconds(state->cfg.ws_pong_timeout_sec), now));
    } else {
      cap_poll_timeout(&timeout_ms,
                       ms_until(last_recv_activity + std::chrono::seconds(state->cfg.ws_ping_interval_sec), now));
    }
    cap_poll_timeout(&timeout_ms, 250);
    if (active->drain_requested.load(std::memory_order_acquire) || is_shutting_down(*state)) {
      cap_poll_timeout(&timeout_ms, 100);
    }
    int pr = ::poll(&pfd, 1, timeout_ms);
    if (pr < 0) {
      if (errno == EINTR) continue;
      (void)send_ws_close(conn.get(), 1011, "poll_failed", active, &protocol_close_sent);
      break;
    }

    if (pr > 0 && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      should_close = true;
    }

    if (pr > 0 && (pfd.revents & POLLIN)) {
      char buf[8192];
      ssize_t n = ::recv(conn.get(), buf, sizeof(buf), 0);
      if (n > 0) {
        recv_buffer.append(buf, static_cast<size_t>(n));
        last_recv_activity = Clock::now();
      } else if (n == 0) {
        should_close = true;
      } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        should_close = true;
      }

      if (!drain_recv_buffer()) break;
    }

    if (!flush_pending_outputs(conn.get(),
                               session,
                               *state->stats,
                               &pending_outputs,
                               active,
                               &protocol_close_sent)) break;

    if (!should_close) {
      uint64_t finalize_gen = session.generation();
      auto timer_events = session.poll_timer(unix_now_seconds());
      enqueue_finalize_output(&pending_outputs,
                              std::move(timer_events),
                              finalize_gen,
                              session,
                              &last_enqueued_finalize_seq);
      if (!flush_pending_outputs(conn.get(),
                                 session,
                                 *state->stats,
                                 &pending_outputs,
                                 active,
                                 &protocol_close_sent)) break;
    }
  }

  bool sent_clean_close = false;
  if (!protocol_close_sent && !active->forced.load(std::memory_order_acquire)) {
    sent_clean_close = send_ws_close(conn.get(), 1000, "", active, &protocol_close_sent);
  }
  if (sent_clean_close) {
    drain_peer_after_clean_close(conn.get(), 1000);
  }
  unregister_session(state, active);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "WS_SESSION_ERROR error=%s\n", e.what());
    if (conn.get() >= 0 && !protocol_close_sent) {
      (void)send_ws_text(conn.get(), error_event_json(e.what()), active, &protocol_close_sent);
    }
    if (conn.get() >= 0 && !protocol_close_sent) {
      (void)send_ws_close(conn.get(), 1011, "session_error", active, &protocol_close_sent);
    }
    unregister_session(state, active);
  } catch (...) {
    std::fprintf(stderr, "WS_SESSION_ERROR error=unknown\n");
    if (conn.get() >= 0 && !protocol_close_sent) {
      (void)send_ws_text(conn.get(), error_event_json("unknown session error"), active, &protocol_close_sent);
    }
    if (conn.get() >= 0 && !protocol_close_sent) {
      (void)send_ws_close(conn.get(), 1011, "session_error", active, &protocol_close_sent);
    }
    unregister_session(state, active);
  }
}

class WsServer {
 public:
  explicit WsServer(ServerConfig cfg) : state_(std::make_shared<ServerState>(std::move(cfg))) {}
  WsServer(const WsServer&) = delete;
  WsServer& operator=(const WsServer&) = delete;
  ~WsServer() {
    stop();
  }

  void start() {
    if (running_.load()) return;
    construct_runtime();
    listen_fd_.reset(create_listener(state_->cfg.port));
    state_->cfg.port = bound_port(listen_fd_.get());
    admin_pool_ = std::make_unique<AdminHandlerPool>(state_);
    admin_pool_->start();
    running_.store(true, std::memory_order_release);
    accept_thread_ = std::thread([this] { accept_loop(); });
  }

  void stop() {
    if (!state_) return;
    running_.store(false, std::memory_order_release);
    stop_accepting();
    stop_admin_pool();
    join_ws_threads();
    state_.reset();
  }

  void begin_drain() {
    if (!state_) return;
    bool expected = false;
    if (state_->shutting_down.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      if (state_->admission) state_->admission->shutting_down(true);
      request_session_drains(state_);
      std::cout << "WS_SHUTDOWN_DRAIN_BEGIN drain_sec=" << state_->cfg.shutdown_drain_sec << "\n";
      std::cout.flush();
    } else {
      request_session_drains(state_);
    }
  }

  bool drain_and_stop() {
    if (!state_) return false;
    begin_drain();
    const auto deadline = Clock::now() + std::chrono::seconds(state_->cfg.shutdown_drain_sec);
    while (active_session_count(state_) > 0 && Clock::now() < deadline) {
      request_session_drains(state_);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    bool forced = active_session_count(state_) > 0;
    if (forced) {
      force_close_active_sessions();
    }

    stop_accepting();
    stop_admin_pool();
    join_ws_threads();
    emit_stats_lifetime();
    state_.reset();
    return forced;
  }

  int port() const {
    return state_->cfg.port;
  }

  const std::string& bind_host() const {
    return bind_host_;
  }

 private:
  void construct_runtime() {
    auto stats = std::make_unique<StatsCollector>(state_->cfg.stats_window, state_->cfg.stats_enabled);

    std::string ws_lanes_env = std::to_string(state_->cfg.ws_lanes);
    ::setenv("NEMOTRON_WS_LANES", ws_lanes_env.c_str(), 1);

    SharedRuntimeConfig shared_cfg;
    shared_cfg.bundle_path = (fs::path(state_->cfg.artifact_dir) / "session_audio_bundle.ts").string();
    shared_cfg.steady_artifacts_dir = state_->cfg.artifact_dir;
    // Honor --steady-batch-dir for the scheduler steady loader: pass the
    // already-resolved effective dir so runtime.cpp does not re-search from the
    // artifact-dir parent (which silently ignored an explicit --steady-batch-dir).
    shared_cfg.steady_batch_dir = state_->cfg.effective_steady_batch_dir;
    std::string stripped = (fs::path(state_->cfg.artifact_dir) / "stripped_finalize_buckets").string();
    shared_cfg.finalize_buckets_dir = dir_exists(stripped)
                                          ? stripped
                                          : (fs::path(state_->cfg.artifact_dir) / "finalize_buckets").string();
    shared_cfg.b_max = state_->cfg.batch_b_max;
    shared_cfg.batch_window_ms = state_->cfg.batch_window_ms;
    shared_cfg.batch_lone_timeout_ms = state_->cfg.batch_lone_timeout_ms;
    shared_cfg.batch_max_queue_delay_ms = state_->cfg.batch_max_queue_delay_ms;
    shared_cfg.batch_queue_capacity = state_->cfg.batch_queue_capacity;
    shared_cfg.batch_min_fill_enabled = state_->cfg.batch_min_fill_enabled;
    shared_cfg.batch_disable_min_fill = state_->cfg.batch_disable_min_fill;
    shared_cfg.batch_force_bucket = state_->cfg.batch_force_bucket;
    shared_cfg.device_index = state_->cfg.device_index;
    shared_cfg.steady_num_runners = state_->cfg.steady_num_runners;
    shared_cfg.steady_dispatch_lanes = state_->cfg.steady_dispatch_lanes;
    shared_cfg.finalize_num_runners = state_->cfg.finalize_num_runners;
    shared_cfg.scheduler_enabled = state_->cfg.scheduler_enabled;
    shared_cfg.steady_shadow_enabled = state_->cfg.steady_shadow_enabled;
    shared_cfg.background_warmup_enabled = state_->cfg.background_warmup_enabled;
    shared_cfg.warm_sync_lanes = state_->cfg.warm_sync_lanes;

    state_->admission = std::make_unique<DensityAdmission>(state_->cfg.admission_active_cap,
                                                           state_->cfg.admission_backlog_cap);
    stats->set_admission(state_->admission.get());
    state_->stats = std::move(stats);
    if (!state_->cfg.selftest_lightweight_runtime) {
      state_->shared_runtime = std::make_unique<SharedRuntime>(shared_cfg);
      if (state_->cfg.background_warmup_enabled) {
        SharedRuntime* runtime = state_->shared_runtime.get();
        state_->admission->set_active_cap_provider([runtime]() {
          return runtime != nullptr ? runtime->warmed_lane_count() : 0;
        });
      }
    } else if (state_->cfg.scheduler_enabled) {
      std::printf("shared runtime scheduler skipped: selftest_lightweight_runtime=true\n");
      std::fflush(stdout);
    }

    state_->model_loaded.store(true, std::memory_order_release);
  }

  int create_listener(int port) {
    UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
    if (fd.get() < 0) throw std::runtime_error(std::string("socket failed: ") + std::strerror(errno));
    int yes = 1;
    if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
      throw std::runtime_error(std::string("setsockopt SO_REUSEADDR failed: ") + std::strerror(errno));
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    // Bind host is configurable via NEMOTRON_WS_HOST (default 127.0.0.1 to preserve the
    // same-box compat-oracle behavior). Set "0.0.0.0" to serve remote clients (production
    // deploy, or an off-box load driver). Any other value is parsed as a dotted-quad.
    const char* host_env = std::getenv("NEMOTRON_WS_HOST");
    std::string bind_host = (host_env != nullptr && host_env[0] != '\0') ? host_env : "127.0.0.1";
    if (bind_host == "127.0.0.1" || bind_host == "localhost") {
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (bind_host == "0.0.0.0" || bind_host == "*") {
      addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) != 1) {
      throw std::runtime_error("NEMOTRON_WS_HOST must be 0.0.0.0, 127.0.0.1, or a dotted-quad IPv4: " + bind_host);
    }
    bind_host_ = bind_host;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      throw std::runtime_error(std::string("bind failed: ") + std::strerror(errno));
    }
    if (::listen(fd.get(), 128) != 0) {
      throw std::runtime_error(std::string("listen failed: ") + std::strerror(errno));
    }
    return fd.release();
  }

  int bound_port(int fd) const {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
      throw std::runtime_error(std::string("getsockname failed: ") + std::strerror(errno));
    }
    return ntohs(addr.sin_port);
  }

  void accept_loop() {
    while (running_.load(std::memory_order_acquire)) {
      if (g_sigterm_requested.load(std::memory_order_acquire)) begin_drain();
      sockaddr_in peer{};
      socklen_t peer_len = sizeof(peer);
      int fd = ::accept(listen_fd_.get(), reinterpret_cast<sockaddr*>(&peer), &peer_len);
      if (fd < 0) {
        if (errno == EINTR) continue;
        if (!running_.load(std::memory_order_acquire)) break;
        continue;
      }
      handle_accepted(fd);
    }
  }

  void stop_accepting() {
    running_.store(false, std::memory_order_release);
    if (listen_fd_.get() >= 0) {
      (void)::shutdown(listen_fd_.get(), SHUT_RDWR);
    }
    listen_fd_.reset();
    if (accept_thread_.joinable()) accept_thread_.join();
  }

  void stop_admin_pool() {
    if (admin_pool_) {
      admin_pool_->stop();
      admin_pool_.reset();
    }
  }

  void join_ws_threads() {
    std::lock_guard<std::mutex> lock(ws_threads_mutex_);
    for (auto& worker : ws_threads_) {
      if (worker.joinable()) worker.join();
    }
    ws_threads_.clear();
  }

  void emit_stats_lifetime() const {
    uint64_t emitted = 0;
    uint64_t suppressed = 0;
    if (state_ && state_->stats) {
      nlohmann::json stats = nlohmann::json::parse(state_->stats->snapshot_json(), nullptr, false);
      if (!stats.is_discarded()) {
        emitted = stats.value("lifetime_emitted", 0ULL);
        suppressed = stats.value("lifetime_suppressed", 0ULL);
      }
    }
    std::cout << "STATS_LIFETIME emitted=" << emitted << " suppressed=" << suppressed << "\n";
    std::cout.flush();
  }

  void force_close_active_sessions() {
    std::vector<std::shared_ptr<ActiveWsSession>> sessions = session_snapshot(state_);
    for (const auto& session : sessions) {
      if (!session || session->closed.load(std::memory_order_acquire)) continue;
      std::cout << "WS_SHUTDOWN_FORCE_CLOSE session_id=" << session->stream_id << "\n";
      best_effort_force_close(session);
    }
    std::cout.flush();
  }

  void handle_accepted(int fd) {
    UniqueFd conn(fd);
    (void)configure_tcp_nodelay(conn.get());
    ReadHttpOutcome read = read_http_request_from_socket(conn.get());
    if (read.result == ws_handshake::ParseResult::MALFORMED) {
      (void)send_all(conn.get(), build_json_response(400, "{\"error\":\"bad_request\"}"));
      return;
    }
    if (read.result == ws_handshake::ParseResult::OVERSIZE_HEADERS) {
      (void)send_all(conn.get(), build_json_response(431, "{\"error\":\"headers_too_large\"}"));
      return;
    }
    if (read.result != ws_handshake::ParseResult::OK) {
      (void)send_all(conn.get(), build_json_response(400, "{\"error\":\"bad_request\"}"));
      return;
    }

    ws_routes::Route route = ws_routes::dispatch(read.request);
    bool stale_gen_test_endpoint =
        route.path == kStaleGenTestPath && stale_gen_test_endpoint_enabled();
    if (route.kind == ws_routes::RouteKind::HEALTH ||
        route.kind == ws_routes::RouteKind::STATS ||
        route.kind == ws_routes::RouteKind::SCHEDULER_TELEMETRY ||
        stale_gen_test_endpoint) {
      AdminJob job;
      job.fd = conn.release();
      job.route = std::move(route);
      if (!admin_pool_->try_enqueue(std::move(job))) {
        UniqueFd rejected(job.fd);
        (void)send_all(rejected.get(), build_json_response(503, "{\"error\":\"admin_queue_full\"}"));
      }
      return;
    }

    if (route.kind == ws_routes::RouteKind::WEBSOCKET) {
      if (is_shutting_down(*state_)) {
        (void)send_all(conn.get(), build_json_response(503, "{\"error\":\"draining\"}"));
        return;
      }
      auto key_it = read.request.headers.find("sec-websocket-key");
      std::string accept_key = ws_handshake::compute_accept_key(key_it->second);
      if (!send_all(conn.get(), ws_handshake::build_handshake_response(accept_key))) return;
      std::lock_guard<std::mutex> lock(ws_threads_mutex_);
      ws_threads_.emplace_back([state = state_, fd = conn.release(), route = std::move(route)]() mutable {
        ws_worker(fd, state, std::move(route));
      });
      return;
    }

    if (route.kind == ws_routes::RouteKind::BAD_REQUEST) {
      (void)send_all(conn.get(), build_json_response(400, "{\"error\":\"bad_request\"}"));
    } else {
      (void)send_all(conn.get(), build_json_response(404, "{\"error\":\"not_found\"}"));
    }
  }

  std::shared_ptr<ServerState> state_;
  std::atomic<bool> running_{false};
  std::string bind_host_ = "127.0.0.1";
  UniqueFd listen_fd_;
  std::unique_ptr<AdminHandlerPool> admin_pool_;
  std::thread accept_thread_;
  std::mutex ws_threads_mutex_;
  std::vector<std::thread> ws_threads_;
};

class ScopedEnv {
 public:
  void set(const std::string& name, const std::string& value) {
    remember(name);
    ::setenv(name.c_str(), value.c_str(), 1);
  }
  void unset(const std::string& name) {
    remember(name);
    ::unsetenv(name.c_str());
  }
  ~ScopedEnv() {
    for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
      if (it->second.has_value()) {
        ::setenv(it->first.c_str(), it->second->c_str(), 1);
      } else {
        ::unsetenv(it->first.c_str());
      }
    }
  }

 private:
  void remember(const std::string& name) {
    if (std::any_of(saved_.begin(), saved_.end(), [&](const auto& item) {
          return item.first == name;
        })) {
      return;
    }
    const char* raw = std::getenv(name.c_str());
    if (raw == nullptr) {
      saved_.push_back({name, std::nullopt});
    } else {
      saved_.push_back({name, std::string(raw)});
    }
  }

  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

void clear_selftest_env(ScopedEnv* env) {
  for (const char* name : {
           "NEMOTRON_STATS_ENABLED",
           "NEMOTRON_STATS_WINDOW",
           "NEMOTRON_WS_SCHEDULER",
           "NEMOTRON_WS_STEADY_SHADOW",
           "NEMOTRON_DENSITY_BATCH_STEADY",
           "NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP",
           "NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP",
           "NEMOTRON_DENSITY_BATCH_MAX",
           "NEMOTRON_DENSITY_BATCH_WINDOW_MS",
           "NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS",
           "NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS",
           "NEMOTRON_DENSITY_BATCH_QUEUE_CAPACITY",
           "NEMOTRON_DENSITY_BATCH_MIN_FILL",
           "NEMOTRON_DENSITY_BATCH_DISABLE_MIN_FILL",
           "NEMOTRON_DENSITY_BATCH_FORCE_BUCKET",
           "NEMOTRON_DENSITY_DISPATCH_LANES",
           "NEMOTRON_WS_LANES",
           "NEMOTRON_WS_FINALIZE_RUNNERS",
           "NEMOTRON_WS_SEND_TIMEOUT_SEC",
           "NEMOTRON_WS_PING_INTERVAL_SEC",
           "NEMOTRON_WS_PONG_TIMEOUT_SEC",
           "NEMOTRON_SHUTDOWN_DRAIN_SEC",
           "NEMOTRON_WS_STALEGEN_TEST_ENDPOINT",
       }) {
    env->unset(name);
  }
}

struct HttpClientResponse {
  int status = 0;
  std::string body;
  std::string raw;
};

UniqueFd connect_localhost(int port) {
  UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (fd.get() < 0) throw std::runtime_error(std::string("client socket failed: ") + std::strerror(errno));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::connect(fd.get(), reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    throw std::runtime_error(std::string("client connect failed: ") + std::strerror(errno));
  }
  return fd;
}

std::string recv_until_close(int fd) {
  std::string out;
  char buf[4096];
  for (;;) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    break;
  }
  return out;
}

HttpClientResponse http_request(int port, const std::string& request) {
  UniqueFd fd = connect_localhost(port);
  if (!send_all(fd.get(), request)) throw std::runtime_error("client send failed");
  std::string raw = recv_until_close(fd.get());
  HttpClientResponse out;
  out.raw = raw;
  size_t line_end = raw.find("\r\n");
  if (line_end == std::string::npos) throw std::runtime_error("HTTP response missing status line");
  std::istringstream status_line(raw.substr(0, line_end));
  std::string http_version;
  status_line >> http_version >> out.status;
  size_t header_end = raw.find("\r\n\r\n");
  if (header_end != std::string::npos) out.body = raw.substr(header_end + 4);
  return out;
}

struct ClientFrame {
  uint8_t opcode = 0;
  std::vector<uint8_t> payload;
};

bool recv_exact(int fd, void* data, size_t size) {
  char* p = static_cast<char*>(data);
  size_t got = 0;
  while (got < size) {
    ssize_t n = ::recv(fd, p + got, size - got, 0);
    if (n > 0) {
      got += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

ClientFrame read_server_frame(int fd) {
  uint8_t hdr[2]{};
  if (!recv_exact(fd, hdr, sizeof(hdr))) throw std::runtime_error("missing websocket frame header");
  bool masked = (hdr[1] & 0x80) != 0;
  uint64_t len = hdr[1] & 0x7f;
  if (len == 126) {
    uint8_t ext[2]{};
    if (!recv_exact(fd, ext, sizeof(ext))) throw std::runtime_error("missing websocket frame ext16");
    len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
  } else if (len == 127) {
    uint8_t ext[8]{};
    if (!recv_exact(fd, ext, sizeof(ext))) throw std::runtime_error("missing websocket frame ext64");
    len = 0;
    for (uint8_t b : ext) len = (len << 8) | b;
  }
  uint8_t mask[4]{};
  if (masked && !recv_exact(fd, mask, sizeof(mask))) throw std::runtime_error("missing websocket mask");
  if (len > 1024 * 1024) throw std::runtime_error("selftest websocket frame too large");
  ClientFrame frame;
  frame.opcode = hdr[0] & 0x0f;
  frame.payload.resize(static_cast<size_t>(len));
  if (len > 0 && !recv_exact(fd, frame.payload.data(), frame.payload.size())) {
    throw std::runtime_error("missing websocket payload");
  }
  if (masked) {
    for (size_t i = 0; i < frame.payload.size(); ++i) frame.payload[i] ^= mask[i % 4];
  }
  return frame;
}

ClientFrame read_server_frame_timeout(int fd, int timeout_ms) {
  pollfd pfd{};
  pfd.fd = fd;
  pfd.events = POLLIN;
  int pr = ::poll(&pfd, 1, timeout_ms);
  if (pr <= 0) throw std::runtime_error("timed out waiting for websocket frame");
  return read_server_frame(fd);
}

bool send_client_text(int fd, const std::string& payload) {
  return send_all(fd, ws_framing::write_frame(ws_framing::Opcode::TEXT, payload, true));
}

bool send_client_binary(int fd, const std::vector<int16_t>& pcm) {
  std::string payload(reinterpret_cast<const char*>(pcm.data()), pcm.size() * sizeof(int16_t));
  return send_all(fd, ws_framing::write_frame(ws_framing::Opcode::BINARY, payload, true));
}

bool send_client_close(int fd) {
  std::string payload;
  payload.push_back(static_cast<char>((1000 >> 8) & 0xff));
  payload.push_back(static_cast<char>(1000 & 0xff));
  return send_all(fd, ws_framing::write_frame(ws_framing::Opcode::CLOSE, payload, true));
}

UniqueFd websocket_connect(int port, int* http_status) {
  UniqueFd fd = connect_localhost(port);
  std::string request =
      "GET / HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Version: 13\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
      "\r\n";
  if (!send_all(fd.get(), request)) throw std::runtime_error("websocket handshake send failed");

  std::string header;
  char ch = '\0';
  while (header.find("\r\n\r\n") == std::string::npos) {
    ssize_t n = ::recv(fd.get(), &ch, 1, 0);
    if (n == 1) {
      header.push_back(ch);
      if (header.size() > 8192) throw std::runtime_error("websocket handshake response too large");
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    throw std::runtime_error("websocket handshake response closed early");
  }
  size_t line_end = header.find("\r\n");
  std::istringstream status_line(header.substr(0, line_end));
  std::string http_version;
  status_line >> http_version >> *http_status;
  return fd;
}

uint16_t close_code(const ClientFrame& frame) {
  if (frame.payload.size() < 2) return 0;
  return static_cast<uint16_t>((static_cast<uint16_t>(frame.payload[0]) << 8) |
                               static_cast<uint16_t>(frame.payload[1]));
}

std::string close_reason(const ClientFrame& frame) {
  if (frame.payload.size() <= 2) return {};
  return std::string(frame.payload.begin() + 2, frame.payload.end());
}

std::vector<int16_t> selftest_audio_tensor_to_pcm_int16(torch::Tensor tensor) {
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

bool has_raw_finalize_timing_keys(const nlohmann::json& event) {
  if (!event.contains("finalize_timing") || !event["finalize_timing"].is_object()) return false;
  const nlohmann::json& timing = event["finalize_timing"];
  static const std::vector<std::string> keys = {
      "reason",
      "vad_stop",
      "vad_stop_recv",
      "debounce_expiry",
      "fork_flush_start",
      "fork_flush_done",
      "final_sent",
      "inference_lock_acquire_wait_ms",
      "enc_first_lock_wait_ms",
      "lane_queue_wait_ms",
      "preproc_ms",
      "scheduler_enqueue_wait_ms",
      "scheduler_future_wait_ms",
      "scheduler_completion_wait_ms",
      "decode_ms",
      "gil_attrib_enabled",
  };
  if (timing.size() != keys.size()) return false;
  for (const auto& key : keys) {
    if (!timing.contains(key)) return false;
  }
  return timing["gil_attrib_enabled"].is_boolean();
}

struct SelftestResult {
  int id = 0;
  std::string name;
  bool pass = false;
  std::string diagnostic;
};

ServerConfig selftest_config(const ServerConfig& base, bool lightweight_runtime = true) {
  ServerConfig cfg = base;
  cfg.port = 0;
  cfg.port_set = true;
  cfg.admission_active_cap = 4;
  cfg.admission_active_cap_set = true;
  cfg.admission_backlog_cap = kDefaultAdmissionBacklogCap;
  cfg.admission_backlog_cap_set = true;
  cfg.process_label.clear();
  cfg.selftest_ws_close_delay_ms = 0;
  cfg.selftest_lightweight_runtime = lightweight_runtime;
  validate_config(&cfg, true);
  return cfg;
}

SelftestResult run_case(int id, const std::string& name, const std::function<void(SelftestResult*)>& fn) {
  SelftestResult result;
  result.id = id;
  result.name = name;
  try {
    fn(&result);
    if (result.diagnostic.empty()) result.diagnostic = "ok";
  } catch (const std::exception& e) {
    result.pass = false;
    result.diagnostic = e.what();
  }
  std::cout << "SELFTEST " << id << " " << (result.pass ? "PASS" : "FAIL")
            << " - " << name << " - " << result.diagnostic << "\n";
  std::cout.flush();
  return result;
}

bool json_has_bool(const std::string& text, const std::string& key, bool expected) {
  nlohmann::json parsed = nlohmann::json::parse(text);
  return parsed.contains(key) && parsed[key].is_boolean() && parsed[key].get<bool>() == expected;
}

int run_selftest(const ServerConfig& parsed) {
  ServerConfig base = parsed;
  std::vector<SelftestResult> results;

  results.push_back(run_case(1, "Default env, valid artifacts", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base, false);
    WsServer server(cfg);
    server.start();
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = health.status == 200 && json_has_bool(health.body, "model_loaded", true);
    r->diagnostic = "health_status=" + std::to_string(health.status);
  }));

  results.push_back(run_case(2, "NEMOTRON_STATS_ENABLED=0", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_STATS_ENABLED", "0");
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    HttpClientResponse stats = http_request(server.port(), "GET /stats HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = stats.status == 200 && json_has_bool(stats.body, "enabled", false);
    r->diagnostic = "stats_status=" + std::to_string(stats.status) + " body=" + stats.body.substr(0, 64);
  }));

  results.push_back(run_case(3, "NEMOTRON_STATS_WINDOW=abc startup failure", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_STATS_WINDOW", "abc");
    ServerConfig cfg = base;
    cfg.port = 0;
    cfg.port_set = true;
    cfg.admission_active_cap = 4;
    cfg.admission_active_cap_set = true;
    try {
      validate_config(&cfg, true);
      WsServer server(cfg);
      server.start();
      server.stop();
      r->pass = false;
      r->diagnostic = "startup unexpectedly succeeded";
    } catch (const std::exception& e) {
      r->pass = std::string(e.what()).find("NEMOTRON_STATS_WINDOW") != std::string::npos;
      r->diagnostic = e.what();
    }
  }));

  results.push_back(run_case(4, "Scheduler ON missing MANIFEST startup failure", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_WS_SCHEDULER", "1");
    fs::path tmp = fs::temp_directory_path() /
                   ("ws-server-missing-manifest-" + std::to_string(::getpid()));
    fs::create_directories(tmp);
    ServerConfig cfg = base;
    cfg.port = 0;
    cfg.port_set = true;
    cfg.admission_active_cap = 4;
    cfg.admission_active_cap_set = true;
    cfg.steady_batch_dir = tmp.string();
    cfg.steady_batch_dir_explicit = true;
    try {
      validate_config(&cfg, true);
      WsServer server(cfg);
      server.start();
      server.stop();
      r->pass = false;
      r->diagnostic = "startup unexpectedly succeeded";
    } catch (const std::exception& e) {
      r->pass = std::string(e.what()).find("MANIFEST") != std::string::npos;
      r->diagnostic = e.what();
    }
    fs::remove_all(tmp);
  }));

  results.push_back(run_case(5, "Scheduler ON valid artifacts", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    env.set("NEMOTRON_WS_SCHEDULER", "1");
    ServerConfig cfg = selftest_config(base, false);
    WsServer server(cfg);
    server.start();
    HttpClientResponse telemetry =
        http_request(server.port(), "GET /scheduler_telemetry HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = telemetry.status == 200 && telemetry.body.find("\"counts\"") != std::string::npos;
    r->diagnostic = "scheduler_status=" + std::to_string(telemetry.status);
  }));

  results.push_back(run_case(6, "--port 0 auto-bind", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    int bound = server.port();
    HttpClientResponse health = http_request(bound, "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = bound > 0 && health.status == 200;
    r->diagnostic = "bound_port=" + std::to_string(bound) + " health_status=" + std::to_string(health.status);
  }));

  results.push_back(run_case(7, "--admission-active-cap 0 startup failure", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = base;
    cfg.port = 0;
    cfg.port_set = true;
    cfg.admission_active_cap = 0;
    cfg.admission_active_cap_set = true;
    try {
      validate_config(&cfg, true);
      r->pass = false;
      r->diagnostic = "validation unexpectedly succeeded";
    } catch (const std::exception& e) {
      r->pass = std::string(e.what()).find("positive") != std::string::npos;
      r->diagnostic = e.what();
    }
  }));

  results.push_back(run_case(8, "Bound port health + stats + WS lifecycle PCM+vad_stop", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base, false);
    auto bundle = load_jit_serialized((fs::path(cfg.artifact_dir) / "session_audio_bundle.ts").string());
    std::vector<int16_t> pcm =
        selftest_audio_tensor_to_pcm_int16(prefix_tensor(bundle, "utt0", "audio"));
    WsServer server(cfg);
    server.start();
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    HttpClientResponse stats = http_request(server.port(), "GET /stats?last=1 HTTP/1.1\r\nHost: localhost\r\n\r\n");
    int status = 0;
    UniqueFd ws = websocket_connect(server.port(), &status);
    ClientFrame ready = read_server_frame_timeout(ws.get(), 5000);
    if (!send_client_text(ws.get(), "{\"type\":\"vad_start\"}") ||
        !send_client_binary(ws.get(), pcm) ||
        !send_client_text(ws.get(), "{\"type\":\"vad_stop\"}")) {
      throw std::runtime_error("failed to send lifecycle websocket frames");
    }

    bool interim_ok = false;
    bool final_ok = false;
    bool timing_ok = false;
    for (int i = 0; i < 64 && !final_ok; ++i) {
      ClientFrame frame = read_server_frame_timeout(ws.get(), 30000);
      if (frame.opcode != static_cast<uint8_t>(ws_framing::Opcode::TEXT)) continue;
      std::string payload(frame.payload.begin(), frame.payload.end());
      nlohmann::json event = nlohmann::json::parse(payload);
      if (event.value("type", "") != "transcript") continue;
      if (event.value("is_final", false)) {
        final_ok = event.contains("text") && event["text"].is_string() &&
                   !event["text"].get<std::string>().empty() &&
                   event.value("finalize", false);
        timing_ok = has_raw_finalize_timing_keys(event);
      } else {
        interim_ok = event.contains("text") && event["text"].is_string() &&
                     !event["text"].get<std::string>().empty();
      }
    }
    if (!send_client_close(ws.get())) throw std::runtime_error("failed to send close");
    ClientFrame close = read_server_frame_timeout(ws.get(), 5000);
    server.stop();
    std::string ready_payload(ready.payload.begin(), ready.payload.end());
    r->pass = health.status == 200 &&
              stats.status == 200 &&
              json_has_bool(stats.body, "enabled", true) &&
              status == 101 &&
              ready.opcode == static_cast<uint8_t>(ws_framing::Opcode::TEXT) &&
              ready_payload == "{\"type\":\"ready\"}" &&
              final_ok &&
              timing_ok &&
              close.opcode == static_cast<uint8_t>(ws_framing::Opcode::CLOSE) &&
              close_code(close) == 1000;
    r->diagnostic = "health=" + std::to_string(health.status) +
                    " stats=" + std::to_string(stats.status) +
                    " ws_status=" + std::to_string(status) +
                    " interim=" + std::string(interim_ok ? "true" : "false") +
                    " final=" + std::string(final_ok ? "true" : "false") +
                    " timing=" + std::string(timing_ok ? "true" : "false") +
                    " close=" + std::to_string(close_code(close));
  }));

  results.push_back(run_case(9, "Cap=1, two WS connections, second post-handshake WS-1013", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base, false);
    cfg.admission_active_cap = 1;
    cfg.admission_backlog_cap = 0;
    WsServer server(cfg);
    server.start();
    int first_status = 0;
    UniqueFd first = websocket_connect(server.port(), &first_status);
    ClientFrame first_ready = read_server_frame_timeout(first.get(), 5000);
    int second_status = 0;
    UniqueFd second = websocket_connect(server.port(), &second_status);
    ClientFrame second_close = read_server_frame_timeout(second.get(), 5000);
    if (!send_client_close(first.get())) throw std::runtime_error("failed to close first ws");
    ClientFrame first_close = read_server_frame_timeout(first.get(), 5000);
    server.stop();
    r->pass = first_status == 101 &&
              first_ready.opcode == static_cast<uint8_t>(ws_framing::Opcode::TEXT) &&
              second_status == 101 &&
              second_close.opcode == static_cast<uint8_t>(ws_framing::Opcode::CLOSE) &&
              close_code(second_close) == 1013 &&
              close_reason(second_close) == "admission_backpressure" &&
              close_code(first_close) == 1000;
    r->diagnostic = "first_status=" + std::to_string(first_status) +
                    " second_status=" + std::to_string(second_status) +
                    " second_close=" + std::to_string(close_code(second_close)) +
                    " reason=" + close_reason(second_close);
  }));

  results.push_back(run_case(10, "Malformed first HTTP request line", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    HttpClientResponse bad = http_request(server.port(), "not http\r\n\r\n");
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = bad.status == 400 && health.status == 200;
    r->diagnostic = "bad_status=" + std::to_string(bad.status) +
                    " followup_health=" + std::to_string(health.status);
  }));

  results.push_back(run_case(11, "Oversize headers", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg = selftest_config(base);
    WsServer server(cfg);
    server.start();
    std::string request = "GET /health HTTP/1.1\r\nHost: localhost\r\nX-Fill: " +
                          std::string(ws_handshake::kMaxHttpHeaderBytes + 100, 'a') + "\r\n\r\n";
    HttpClientResponse large = http_request(server.port(), request);
    HttpClientResponse health = http_request(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    server.stop();
    r->pass = large.status == 431 && health.status == 200;
    r->diagnostic = "large_status=" + std::to_string(large.status) +
                    " followup_health=" + std::to_string(health.status);
  }));

  results.push_back(run_case(12, "Two ws_server instances on different ports both healthy", [&](SelftestResult* r) {
    ScopedEnv env;
    clear_selftest_env(&env);
    ServerConfig cfg_a = selftest_config(base);
    ServerConfig cfg_b = selftest_config(base);
    cfg_a.process_label = "selftest-a";
    cfg_b.process_label = "selftest-b";
    WsServer server_a(cfg_a);
    WsServer server_b(cfg_b);
    server_a.start();
    server_b.start();
    HttpClientResponse health_a =
        http_request(server_a.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    HttpClientResponse health_b =
        http_request(server_b.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
    int port_a = server_a.port();
    int port_b = server_b.port();
    server_b.stop();
    server_a.stop();
    r->pass = port_a > 0 && port_b > 0 && port_a != port_b &&
              health_a.status == 200 && health_b.status == 200 &&
              health_a.body.find("\"process_label\":\"selftest-a\"") != std::string::npos &&
              health_b.body.find("\"process_label\":\"selftest-b\"") != std::string::npos;
    r->diagnostic = "port_a=" + std::to_string(port_a) +
                    " port_b=" + std::to_string(port_b) +
                    " health_a=" + std::to_string(health_a.status) +
                    " health_b=" + std::to_string(health_b.status);
  }));

  bool all_pass = std::all_of(results.begin(), results.end(), [](const SelftestResult& result) {
    return result.pass;
  });
  std::cout << "SELFTEST_SUMMARY pass=" << json_bool(all_pass)
            << " passed=" << std::count_if(results.begin(), results.end(), [](const SelftestResult& r) {
                 return r.pass;
               })
            << " total=" << results.size() << "\n";
  return all_pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ServerConfig cfg = parse_args(argc, argv);
    if (cfg.selftest_and_exit) {
      return run_selftest(cfg);
    }

    validate_config(&cfg, true);
    if (cfg.print_config) {
      std::cout << config_table(cfg);
    }
    install_sigterm_handler();

    WsServer server(cfg);
    server.start();
    std::cout << "ws_server listening on " << server.bind_host() << ":" << server.port() << "\n";
    std::cout.flush();

    while (!g_sigterm_requested.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    bool forced = server.drain_and_stop();
    return forced ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << "ws_server startup error: " << e.what() << "\n";
    return 1;
  }
}
