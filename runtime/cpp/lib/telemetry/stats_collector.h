#pragma once

#include "session_timing.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

class DensityAdmission;

class StatsCollector {
 public:
  explicit StatsCollector(size_t window_size = 2048, bool enabled = true);

  void record(SessionTiming timing, bool emitted);
  void set_admission(const DensityAdmission* admission);
  std::string snapshot_json(std::optional<size_t> last_n = std::nullopt) const;

  bool enabled() const;
  size_t window_size() const;

 private:
  struct Sample {
    double ts_unix = 0.0;
    std::optional<double> vad_stop_to_sent_ms;
    std::optional<double> fork_flush_wall_ms;
    std::optional<double> vad_stop_recv_to_process_ms;
    std::optional<double> lock_wait_ms;
    std::optional<double> enc_first_lock_wait_ms;
    std::optional<double> lane_queue_wait_ms;
    std::optional<double> preproc_ms;
    std::optional<double> scheduler_enqueue_wait_ms;
    std::optional<double> scheduler_future_wait_ms;
    std::optional<double> scheduler_completion_wait_ms;
    std::optional<double> decode_ms;
    std::optional<double> vad_stop_to_finalize_start_ms;
    int active_sessions_at_emit = 0;
    bool emitted = false;
  };

  bool enabled_ = true;
  size_t window_size_ = 2048;
  mutable std::mutex mutex_;
  std::deque<Sample> samples_;
  uint64_t lifetime_emitted_ = 0;
  uint64_t lifetime_suppressed_ = 0;
  const DensityAdmission* admission_ = nullptr;
};
