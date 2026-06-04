#include "stats_collector.h"

#include "lib/admission/density_admission.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

size_t read_env_size_t(const char* name, size_t fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  errno = 0;
  char* end = nullptr;
  unsigned long long value = std::strtoull(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value == 0 ||
      value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(std::string("invalid positive integer env var ") + name + "=" + raw);
  }
  return static_cast<size_t>(value);
}

bool read_env_enabled(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::string(raw) != "0";
}

void append_optional_number(std::ostringstream& oss, std::optional<double> value) {
  if (value.has_value()) {
    oss << *value;
  } else {
    oss << "null";
  }
}

std::optional<double> delta_ms(std::optional<double> end, std::optional<double> start) {
  if (!end.has_value() || !start.has_value()) return std::nullopt;
  double value = std::max(0.0, (*end - *start) * 1000.0);
  double rounded = std::round(value);
  if (std::abs(value - rounded) < 1.0e-9) return rounded;
  return value;
}

std::string quantile_summary_json(std::vector<double> values) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  if (values.empty()) {
    return "{\"p50\":null,\"p90\":null,\"p95\":null,\"p99\":null,\"max\":null,\"count\":0}";
  }
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  auto percentile = [&](double p) {
    size_t idx = static_cast<size_t>(std::llround(p * static_cast<double>(n - 1)));
    if (idx >= n) idx = n - 1;
    return values[idx];
  };
  oss << "{\"p50\":" << percentile(0.50)
      << ",\"p90\":" << percentile(0.90)
      << ",\"p95\":" << percentile(0.95)
      << ",\"p99\":" << percentile(0.99)
      << ",\"max\":" << values.back()
      << ",\"count\":" << n
      << "}";
  return oss.str();
}

void append_admission_json(std::ostringstream& oss, const DensityAdmission* admission) {
  uint64_t attempted = 0;
  uint64_t admitted = 0;
  uint64_t rejected = 0;
  uint64_t max_backlog = 0;
  uint64_t queued_events = 0;
  uint64_t ready_count = 0;
  uint64_t backlog_count = 0;

  if (admission != nullptr) {
    AdmissionTelemetry telemetry = admission->telemetry_snapshot();
    attempted = telemetry.offered;
    admitted = telemetry.admitted;
    rejected = telemetry.shed_close_count;
    max_backlog = telemetry.backlog_peak;
    queued_events = telemetry.backlog_count;
    backlog_count = telemetry.active_count + telemetry.backlog_count;
  }

  oss << "{\"enabled\":" << (admission != nullptr ? "true" : "false")
      << ",\"attempted\":" << attempted
      << ",\"admitted\":" << admitted
      << ",\"rejected\":" << rejected
      << ",\"max_backlog\":" << max_backlog
      << ",\"max_ready_age_ms\":0"
      << ",\"signal\":{"
      << "\"queued_events\":" << queued_events
      << ",\"ready_count\":" << ready_count
      << ",\"backlog_count\":" << backlog_count
      << ",\"oldest_ready_age_ms\":0"
      << ",\"oldest_ready_session_id\":null"
      << "}}";
}

}  // namespace

StatsCollector::StatsCollector(size_t window_size, bool enabled)
    : enabled_(read_env_enabled("NEMOTRON_STATS_ENABLED", enabled)),
      window_size_(read_env_size_t("NEMOTRON_STATS_WINDOW", window_size)) {
  if (window_size_ == 0) {
    throw std::runtime_error("StatsCollector window_size must be > 0");
  }
  if (!enabled) enabled_ = false;
}

void StatsCollector::record(SessionTiming timing, bool emitted) {
  if (!enabled_) return;

  const bool complete = !timing.was_suppressed &&
                        timing.vad_stop_ts.has_value() &&
                        timing.final_sent_ts.has_value();

  std::lock_guard<std::mutex> lock(mutex_);
  if (!complete) {
    ++lifetime_suppressed_;
    return;
  }

  Sample sample;
  sample.ts_unix = *timing.final_sent_ts;
  sample.vad_stop_to_sent_ms = delta_ms(timing.final_sent_ts, timing.vad_stop_ts);
  sample.fork_flush_wall_ms = delta_ms(timing.fork_flush_done_ts, timing.fork_flush_start_ts);
  sample.vad_stop_recv_to_process_ms = delta_ms(timing.vad_stop_ts, timing.vad_stop_recv_ts);
  sample.lock_wait_ms = timing.inference_lock_acquire_wait_ms;
  sample.enc_first_lock_wait_ms = timing.enc_first_lock_wait_ms;
  sample.lane_queue_wait_ms = timing.lane_queue_wait_ms;
  sample.preproc_ms = timing.preproc_ms;
  sample.scheduler_enqueue_wait_ms = timing.scheduler_enqueue_wait_ms;
  sample.scheduler_future_wait_ms = timing.scheduler_future_wait_ms;
  sample.scheduler_completion_wait_ms = timing.scheduler_completion_wait_ms;
  sample.decode_ms = timing.decode_ms;
  sample.vad_stop_to_finalize_start_ms = delta_ms(timing.fork_flush_start_ts, timing.vad_stop_ts);
  sample.active_sessions_at_emit = timing.active_sessions_at_emit;
  sample.emitted = emitted;

  samples_.push_back(sample);
  while (samples_.size() > window_size_) samples_.pop_front();
  if (emitted) {
    ++lifetime_emitted_;
  } else {
    ++lifetime_suppressed_;
  }
}

void StatsCollector::set_admission(const DensityAdmission* admission) {
  std::lock_guard<std::mutex> lock(mutex_);
  admission_ = admission;
}

bool StatsCollector::enabled() const {
  return enabled_;
}

size_t StatsCollector::window_size() const {
  return window_size_;
}

std::string StatsCollector::snapshot_json(std::optional<size_t> last_n) const {
  std::vector<Sample> samples;
  uint64_t lifetime_emitted = 0;
  uint64_t lifetime_suppressed = 0;
  const DensityAdmission* admission = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    lifetime_emitted = lifetime_emitted_;
    lifetime_suppressed = lifetime_suppressed_;
    admission = admission_;
    size_t start = 0;
    if (last_n.has_value() && *last_n > 0 && *last_n < samples_.size()) {
      start = samples_.size() - *last_n;
    }
    samples.assign(samples_.begin() + static_cast<std::ptrdiff_t>(start), samples_.end());
  }

  std::vector<double> vad_stop_to_sent;
  std::vector<double> fork_flush_wall;
  std::vector<double> vad_stop_recv_to_process;
  std::vector<double> lock_wait;
  std::vector<double> enc_first_lock_wait;
  std::vector<double> lane_queue_wait;
  std::vector<double> preproc;
  std::vector<double> scheduler_enqueue_wait;
  std::vector<double> scheduler_future_wait;
  std::vector<double> scheduler_completion_wait;
  std::vector<double> decode;
  std::vector<double> vad_stop_to_finalize_start;
  std::vector<double> active_sessions;
  vad_stop_to_sent.reserve(samples.size());
  fork_flush_wall.reserve(samples.size());
  vad_stop_recv_to_process.reserve(samples.size());
  lock_wait.reserve(samples.size());
  enc_first_lock_wait.reserve(samples.size());
  lane_queue_wait.reserve(samples.size());
  preproc.reserve(samples.size());
  scheduler_enqueue_wait.reserve(samples.size());
  scheduler_future_wait.reserve(samples.size());
  scheduler_completion_wait.reserve(samples.size());
  decode.reserve(samples.size());
  vad_stop_to_finalize_start.reserve(samples.size());
  active_sessions.reserve(samples.size());

  for (const auto& sample : samples) {
    if (sample.vad_stop_to_sent_ms) vad_stop_to_sent.push_back(*sample.vad_stop_to_sent_ms);
    if (sample.fork_flush_wall_ms) fork_flush_wall.push_back(*sample.fork_flush_wall_ms);
    if (sample.vad_stop_recv_to_process_ms) {
      vad_stop_recv_to_process.push_back(*sample.vad_stop_recv_to_process_ms);
    }
    if (sample.lock_wait_ms) lock_wait.push_back(*sample.lock_wait_ms);
    if (sample.enc_first_lock_wait_ms) enc_first_lock_wait.push_back(*sample.enc_first_lock_wait_ms);
    if (sample.lane_queue_wait_ms) lane_queue_wait.push_back(*sample.lane_queue_wait_ms);
    if (sample.preproc_ms) preproc.push_back(*sample.preproc_ms);
    if (sample.scheduler_enqueue_wait_ms) scheduler_enqueue_wait.push_back(*sample.scheduler_enqueue_wait_ms);
    if (sample.scheduler_future_wait_ms) scheduler_future_wait.push_back(*sample.scheduler_future_wait_ms);
    if (sample.scheduler_completion_wait_ms) {
      scheduler_completion_wait.push_back(*sample.scheduler_completion_wait_ms);
    }
    if (sample.decode_ms) decode.push_back(*sample.decode_ms);
    if (sample.vad_stop_to_finalize_start_ms) {
      vad_stop_to_finalize_start.push_back(*sample.vad_stop_to_finalize_start_ms);
    }
    active_sessions.push_back(static_cast<double>(sample.active_sessions_at_emit));
  }

  std::optional<double> since;
  std::optional<double> until;
  if (!samples.empty()) {
    since = samples.front().ts_unix;
    until = samples.back().ts_unix;
  }
  const uint64_t emitted_in_window =
      static_cast<uint64_t>(std::count_if(samples.begin(), samples.end(), [](const Sample& sample) {
        return sample.emitted;
      }));
  const uint64_t suppressed_in_window = static_cast<uint64_t>(samples.size()) - emitted_in_window;

  std::ostringstream oss;
  oss << std::setprecision(17);
  oss << "{\"enabled\":" << (enabled_ ? "true" : "false")
      << ",\"window_size\":" << window_size_
      << ",\"samples\":" << samples.size()
      << ",\"since_unix\":";
  append_optional_number(oss, since);
  oss << ",\"until_unix\":";
  append_optional_number(oss, until);
  oss << ",\"emitted_in_window\":" << emitted_in_window
      << ",\"suppressed_in_window\":" << suppressed_in_window
      << ",\"lifetime_emitted\":" << lifetime_emitted
      << ",\"lifetime_suppressed\":" << lifetime_suppressed
      << ",\"metrics\":{"
      << "\"vad_stop_to_sent_ms\":" << quantile_summary_json(std::move(vad_stop_to_sent))
      << ",\"fork_flush_wall_ms\":" << quantile_summary_json(std::move(fork_flush_wall))
      << ",\"vad_stop_recv_to_process_ms\":" << quantile_summary_json(std::move(vad_stop_recv_to_process))
      << ",\"lock_wait_ms\":" << quantile_summary_json(std::move(lock_wait))
      << ",\"enc_first_lock_wait_ms\":" << quantile_summary_json(std::move(enc_first_lock_wait))
      << ",\"lane_queue_wait_ms\":" << quantile_summary_json(std::move(lane_queue_wait))
      << ",\"preproc_ms\":" << quantile_summary_json(std::move(preproc))
      << ",\"scheduler_enqueue_wait_ms\":" << quantile_summary_json(std::move(scheduler_enqueue_wait))
      << ",\"scheduler_future_wait_ms\":" << quantile_summary_json(std::move(scheduler_future_wait))
      << ",\"scheduler_completion_wait_ms\":" << quantile_summary_json(std::move(scheduler_completion_wait))
      << ",\"decode_ms\":" << quantile_summary_json(std::move(decode))
      << ",\"vad_stop_to_finalize_start_ms\":" << quantile_summary_json(std::move(vad_stop_to_finalize_start))
      << "},\"active_sessions_at_emit\":" << quantile_summary_json(std::move(active_sessions))
      << ",\"admission\":";
  append_admission_json(oss, admission);
  oss << "}";
  return oss.str();
}
