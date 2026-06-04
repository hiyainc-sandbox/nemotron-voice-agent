#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

enum class AdmissionDecision {
  ADMITTED,
  QUEUED,
  SHED_ACTIVE_CAP,
  SHED_BACKLOG_CAP,
};

struct AdmitResult {
  AdmissionDecision decision = AdmissionDecision::SHED_ACTIVE_CAP;

  bool accepted() const {
    return decision == AdmissionDecision::ADMITTED || decision == AdmissionDecision::QUEUED;
  }

  bool shed() const {
    return decision == AdmissionDecision::SHED_ACTIVE_CAP ||
           decision == AdmissionDecision::SHED_BACKLOG_CAP;
  }
};

struct AdmissionTelemetry {
  uint64_t active_cap = 0;
  uint64_t backlog_cap = 0;
  uint64_t offered = 0;
  uint64_t admitted = 0;
  uint64_t active_count = 0;
  uint64_t backlog_count = 0;
  uint64_t active_peak = 0;
  uint64_t backlog_peak = 0;
  uint64_t active_cap_hits = 0;
  uint64_t backlog_cap_hits = 0;
  uint64_t shed_close_count = 0;
  double shed_close_rate = 0.0;
};

class DensityAdmission {
 public:
  DensityAdmission(uint64_t active_cap, uint64_t backlog_cap);

  AdmitResult try_admit(const std::string& stream_id = {});
  bool try_admit_complete(const std::string& stream_id = {});
  void on_admit_complete(const std::string& stream_id = {});
  void on_close(const std::string& stream_id = {});
  void set_active_cap_provider(std::function<uint64_t()> provider);
  AdmissionTelemetry telemetry_snapshot() const;
  // Step 10 extension: process-level drain gate for SIGTERM shutdown.
  void shutting_down(bool value);
  bool is_shutting_down() const;

 private:
  enum class StreamSlot { ACTIVE, BACKLOG };

  bool try_increment_below(std::atomic<uint64_t>& counter, uint64_t cap, uint64_t* value_after);
  void decrement_if_positive(std::atomic<uint64_t>& counter);
  void update_peak(std::atomic<uint64_t>& peak, uint64_t value);
  void remember_stream(const std::string& stream_id, StreamSlot slot);
  uint64_t effective_active_cap() const;

  const uint64_t active_cap_;
  const uint64_t backlog_cap_;

  std::atomic<uint64_t> active_count_{0};
  std::atomic<uint64_t> backlog_count_{0};
  std::atomic<uint64_t> offered_{0};
  std::atomic<uint64_t> admitted_{0};
  std::atomic<uint64_t> active_peak_{0};
  std::atomic<uint64_t> backlog_peak_{0};
  std::atomic<uint64_t> active_cap_hits_{0};
  std::atomic<uint64_t> backlog_cap_hits_{0};
  std::atomic<uint64_t> shed_close_count_{0};
  std::atomic<bool> shutting_down_{false};

  mutable std::mutex active_cap_provider_mutex_;
  std::function<uint64_t()> active_cap_provider_;

  mutable std::mutex streams_mutex_;
  std::unordered_map<std::string, StreamSlot> streams_;
};
