#pragma once

#include "lib/runtime_io/json.hpp"
#include "lib/session/session.h"
#include "lib/telemetry/session_timing.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct BatchedSteadySchedulerTelemetry;

namespace nemotron_runtime_detail {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
constexpr bool kNativeLittleEndian = true;
#else
constexpr bool kNativeLittleEndian = false;
#endif
}  // namespace nemotron_runtime_detail

static_assert(nemotron_runtime_detail::kNativeLittleEndian,
              "PCMFrame requires native-endian int16 PCM on a little-endian target");

struct PCMFrame {
  const int16_t* samples = nullptr;
  size_t count = 0;
};

struct WireEvent {
  std::string type;
  std::optional<std::string> text;
  std::optional<bool> is_final;
  std::optional<bool> finalize;
  std::optional<nlohmann::json> finalize_timing;
  std::optional<std::string> message;
  // Prompted (multilingual) profile: resolved language for this transcript —
  // the last complete <xx-XX> tag observed, else the session's target language.
  std::optional<std::string> language;
};

struct SharedRuntimeConfig {
  std::string bundle_path;
  std::string finalize_buckets_dir;
  std::string steady_artifacts_dir;
  // Explicit steady-batch package dir (honors --steady-batch-dir). Empty =>
  // resolve_steady_batch_dir falls back to <artifact_dir>/../steady_b_artifacts
  // search (the historical behavior). Distinct from steady_artifacts_dir, which
  // is the general artifact dir consumed by artifact_dir_from_config().
  std::string steady_batch_dir;
  bool scheduler_enabled = false;
  bool steady_shadow_enabled = false;
  int b_max = 16;
  int batch_window_ms = 10;
  int batch_lone_timeout_ms = 0;
  int batch_max_queue_delay_ms = 2;
  int batch_queue_capacity = 32;
  bool batch_min_fill_enabled = true;
  bool batch_disable_min_fill = false;
  int batch_force_bucket = 0;
  int device_index = 0;
  int steady_num_runners = 1;
  int steady_dispatch_lanes = 1;
  int finalize_num_runners = 1;
  bool verify_tokenizer = true;
  bool gil_attrib_enabled = false;
  bool background_warmup_enabled = true;
  int warm_sync_lanes = 4;
};

struct SessionConfig {
  int finalize_silence_ms = 0;
  int active_sessions_at_emit = 0;
  std::string label = "session";
  bool gil_attrib_enabled = false;
  // Prompted (multilingual) profile: resolved per-connection language. Must be
  // a key of the bundle's prompt table (e.g. "es-ES" or "auto"); empty selects
  // the bundle default.
  std::string language;
};

enum class VadState { IDLE, SPEAKING, PENDING_FINALIZE };

class SessionRuntime;

class SharedRuntime {
 public:
  explicit SharedRuntime(SharedRuntimeConfig cfg);
  ~SharedRuntime();

  SharedRuntime(const SharedRuntime&) = delete;
  SharedRuntime& operator=(const SharedRuntime&) = delete;

  const Tokenizer& tokenizer() const;
  const SharedRuntimeConfig& config() const;
  bool has_scheduler() const noexcept;
  uint64_t warmed_lane_count() const noexcept;
  BatchedSteadySchedulerTelemetry scheduler_telemetry_snapshot() const;

 private:
  friend class SessionRuntime;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class SessionRuntime {
 public:
  SessionRuntime(const SharedRuntime& shared, SessionConfig cfg);
  ~SessionRuntime();

  SessionRuntime(const SessionRuntime&) = delete;
  SessionRuntime& operator=(const SessionRuntime&) = delete;

  std::vector<WireEvent> append_pcm_and_drain(const PCMFrame& frame);

  // Client-side VAD control hooks. vad_stop either finalizes immediately when
  // finalize_silence_ms is 0, or arms a per-session debounce deadline.
  void handle_vad_start();
  std::vector<WireEvent> handle_vad_stop();
  std::vector<WireEvent> poll_timer(double now_unix_ts);

  VadState vad_state() const noexcept;
  std::optional<double> vad_deadline_ts() const noexcept;

  std::vector<WireEvent> reset(bool finalize);
  std::vector<WireEvent> end(bool finalize);
  std::vector<WireEvent> finalize_now();

  uint64_t generation() const noexcept;
  void bump_generation() noexcept;

  std::optional<SessionTiming> last_timing() const;

 private:
  friend std::vector<EmittedEvent> session_runtime_debug_events(const SessionRuntime& runtime);
  friend std::vector<int64_t> session_runtime_debug_last_final_tokens(const SessionRuntime& runtime);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};
