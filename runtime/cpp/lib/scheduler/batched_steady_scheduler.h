#pragma once

#include "lib/scheduler/steady_batch_primitive.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct BatchedSteadySchedulerPolicy {
  int window_ms = 10;
  int lone_timeout_ms = 0;
  int max_queue_delay_ms = 10;
  int B_max = 4;
  int queue_capacity = 16;
  bool use_b2_bucket = false;
  bool min_fill_enabled = false;
  bool disable_min_fill = false;
  int force_bucket = 0;
  int dispatch_lanes = 1;
};

struct EnqueueRequest {
  BatchedSteadyInput input;
  c10::cuda::CUDAStream producer;
  cudaEvent_t producer_event = nullptr;
  uint64_t stream_key = 0;
};

struct DispatchResult {
  struct CompletionEvent {
    cudaEvent_t event = nullptr;
    bool owns = true;

    CompletionEvent() = default;
    explicit CompletionEvent(cudaEvent_t e, bool take_ownership = true) : event(e), owns(take_ownership) {}
    CompletionEvent(const CompletionEvent&) = delete;
    CompletionEvent& operator=(const CompletionEvent&) = delete;

    CompletionEvent(CompletionEvent&& other) noexcept : event(other.event), owns(other.owns) {
      other.event = nullptr;
      other.owns = true;
    }

    CompletionEvent& operator=(CompletionEvent&& other) noexcept {
      if (this != &other) {
        reset();
        event = other.event;
        owns = other.owns;
        other.event = nullptr;
        other.owns = true;
      }
      return *this;
    }

    ~CompletionEvent() {
      reset();
    }

    cudaEvent_t get() const {
      return event;
    }

    void reset(cudaEvent_t next = nullptr, bool take_ownership = true) noexcept {
      if (event != nullptr && owns) cudaEventDestroy(event);
      event = next;
      owns = take_ownership;
    }
  };

  struct GraphSlotLease {
    GraphSlotLease() = default;
    GraphSlotLease(std::function<void(cudaStream_t)> retire_fn,
                   std::function<void()> abandon_fn)
        : retire_fn_(std::move(retire_fn)), abandon_fn_(std::move(abandon_fn)) {}
    GraphSlotLease(const GraphSlotLease&) = delete;
    GraphSlotLease& operator=(const GraphSlotLease&) = delete;
    ~GraphSlotLease() {
      abandon();
    }

    void retire_on_stream(cudaStream_t consumer_stream) {
      std::function<void(cudaStream_t)> fn;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) return;
        done_ = true;
        fn = std::move(retire_fn_);
        abandon_fn_ = nullptr;
      }
      if (fn) fn(consumer_stream);
    }

    void abandon() noexcept {
      std::function<void()> fn;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) return;
        done_ = true;
        fn = std::move(abandon_fn_);
        retire_fn_ = nullptr;
      }
      if (fn) {
        try {
          fn();
        } catch (...) {
        }
      }
    }

   private:
    std::mutex mutex_;
    bool done_ = false;
    std::function<void(cudaStream_t)> retire_fn_;
    std::function<void()> abandon_fn_;
  };

  std::vector<at::Tensor> row_tensors;
  int bucket = 0;
  int row = 0;
  int k = 0;
  int64_t cycle_id = 0;
  CompletionEvent completion;
  double gather_wait_us = 0.0;
  double service_wait_us = 0.0;
  double cuda_run_us = 0.0;
  std::string label;
  std::shared_ptr<GraphSlotLease> graph_slot;
};

struct BatchedSteadySchedulerTelemetry {
  struct Lane {
    int lane_id = 0;
    int64_t completed = 0;
    int64_t dispatch_cycles = 0;
    int64_t warmup_runs = 0;
    double dispatcher_cpu_us = 0.0;
    double dispatcher_wall_us = 0.0;
    double dispatcher_stream_run_us = 0.0;
    std::vector<double> cuda_run_us;
    std::vector<double> queue_depth;
  };

  int dispatch_lanes = 1;
  int64_t enqueued = 0;
  int64_t completed = 0;
  int64_t dispatch_cycles = 0;
  int64_t warmup_runs = 0;
  int64_t warmed_lanes = 0;
  int64_t bucket_b1 = 0;
  int64_t bucket_b2 = 0;
  int64_t bucket_b4 = 0;
  int64_t bucket_b8 = 0;
  int64_t bucket_b16 = 0;
  int64_t k2_padded_to_b4 = 0;
  int64_t k3_padded_to_b4 = 0;
  int64_t k4 = 0;
  int64_t k5_padded_to_b8 = 0;
  int64_t k6_padded_to_b8 = 0;
  int64_t k7_padded_to_b8 = 0;
  int64_t k8 = 0;
  int64_t k9_padded_to_b16 = 0;
  int64_t k10_padded_to_b16 = 0;
  int64_t k11_padded_to_b16 = 0;
  int64_t k12_padded_to_b16 = 0;
  int64_t k13_padded_to_b16 = 0;
  int64_t k14_padded_to_b16 = 0;
  int64_t k15_padded_to_b16 = 0;
  int64_t k16 = 0;
  int64_t backlog_gt_bmax = 0;
  int64_t skipped_ready = 0;
  int64_t dispatcher_exceptions = 0;
  double dispatcher_cpu_us = 0.0;
  double dispatcher_wall_us = 0.0;
  double dispatcher_stream_run_us = 0.0;
  std::vector<double> age_at_dispatch_us;
  std::vector<double> gather_wait_us;
  std::vector<double> service_wait_us;
  std::vector<double> cuda_run_us;
  std::vector<double> output_sync_us;
  std::vector<double> worker_blocked_us;
  std::vector<double> completion_wait_us;
  std::vector<double> window_wakeup_jitter_us;
  std::vector<double> queue_depth;
  std::vector<double> per_stream_fairness_spread_us;
  std::vector<Lane> lanes;
};

class BatchedSteadyScheduler {
 public:
  BatchedSteadyScheduler(BatchedSteadyLoaderSet& loader_set,
                         torch::Device device,
                         BatchedSteadySchedulerPolicy policy);
  BatchedSteadyScheduler(const BatchedSteadyScheduler&) = delete;
  BatchedSteadyScheduler& operator=(const BatchedSteadyScheduler&) = delete;
  ~BatchedSteadyScheduler();

  std::future<DispatchResult> enqueue(EnqueueRequest&& request);
  std::optional<std::future<DispatchResult>> try_enqueue_until(
      EnqueueRequest&& request,
      std::chrono::steady_clock::time_point deadline);
  void start();
  void close();
  void warmup_buckets();
  void record_worker_wait(int64_t cycle_id,
                          int k,
                          double output_sync_us,
                          double worker_blocked_us,
                          double completion_wait_us = -1.0);

  BatchedSteadySchedulerTelemetry telemetry_snapshot() const;
  const BatchedSteadySchedulerPolicy& policy() const { return policy_; }
  int future_timeout_ms() const;
  static std::vector<int> required_buckets_for_policy(const BatchedSteadySchedulerPolicy& policy);

 private:
  using Clock = std::chrono::steady_clock;

  struct QueueItem {
    explicit QueueItem(EnqueueRequest&& r) : request(std::move(r)) {}

    EnqueueRequest request;
    std::promise<DispatchResult> promise;
    Clock::time_point enqueue_time;
    Clock::time_point pop_time;
    int64_t sequence = 0;
  };

  struct FormedBatch {
    std::vector<std::shared_ptr<QueueItem>> items;
    double queue_depth_at_form = -1.0;

    bool empty() const {
      return items.empty();
    }
  };

  struct Scratch {
    bool initialized = false;
    std::vector<int64_t> chunk_shape;
    torch::Tensor chunks;
    torch::Tensor length;
    torch::Tensor cache_ch;
    torch::Tensor cache_t;
    torch::Tensor cache_ch_len;
    std::vector<torch::Tensor> row_indices;
    std::vector<torch::Tensor> length_scalars;
  };

  enum class DispatchTimingMode { Sync, Poll, Off };

  struct PendingDispatchTiming {
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    int k = 0;
  };

  struct PendingDispatch {
    cudaEvent_t dispatch_done = nullptr;
    bool owns_dispatch_done = true;
    std::vector<std::shared_ptr<QueueItem>> input_owners;
    int capacity_tokens = 0;

    PendingDispatch() = default;
    PendingDispatch(const PendingDispatch&) = delete;
    PendingDispatch& operator=(const PendingDispatch&) = delete;

    PendingDispatch(PendingDispatch&& other) noexcept
        : dispatch_done(other.dispatch_done),
          owns_dispatch_done(other.owns_dispatch_done),
          input_owners(std::move(other.input_owners)),
          capacity_tokens(other.capacity_tokens) {
      other.dispatch_done = nullptr;
      other.owns_dispatch_done = true;
      other.capacity_tokens = 0;
    }

    PendingDispatch& operator=(PendingDispatch&& other) noexcept {
      if (this != &other) {
        reset();
        dispatch_done = other.dispatch_done;
        owns_dispatch_done = other.owns_dispatch_done;
        input_owners = std::move(other.input_owners);
        capacity_tokens = other.capacity_tokens;
        other.dispatch_done = nullptr;
        other.owns_dispatch_done = true;
        other.capacity_tokens = 0;
      }
      return *this;
    }

    ~PendingDispatch() {
      reset();
    }

    void reset() noexcept {
      if (dispatch_done != nullptr && owns_dispatch_done) cudaEventDestroy(dispatch_done);
      dispatch_done = nullptr;
      owns_dispatch_done = true;
      input_owners.clear();
      capacity_tokens = 0;
    }
  };

  struct DispatchLane {
    int lane_id = 0;
    c10::cuda::CUDAStream dispatcher_stream;
    std::deque<PendingDispatchTiming> pending_dispatch_timings;
    std::deque<PendingDispatch> pending_dispatches;
    std::map<int, Scratch> scratch;
    std::deque<FormedBatch> inbox;
    bool free = true;
    bool telemetry_measurement_started = false;
    Clock::time_point telemetry_measurement_wall_start;
    double telemetry_measurement_cpu_start_us = 0.0;
    int64_t test_dispatch_attempts = 0;

    DispatchLane(int lane_id, c10::cuda::CUDAStream dispatcher_stream)
        : lane_id(lane_id), dispatcher_stream(dispatcher_stream) {}
    DispatchLane(const DispatchLane&) = delete;
    DispatchLane& operator=(const DispatchLane&) = delete;
    DispatchLane(DispatchLane&&) noexcept = default;
    DispatchLane& operator=(DispatchLane&&) noexcept = default;
  };

  enum class GraphSlotState { Free, StagingReplay, Published, Retiring };

  struct GraphRowRetirement {
    bool active = false;
    bool fence_recorded = false;
    bool zero_consumer = false;
  };

  struct GraphSlot {
    int bucket = 0;
    int slot_id = 0;
    GraphSlotState state = GraphSlotState::Free;
    bool captured = false;
    std::unique_ptr<AOTIModelPackageLoader> graph_loader;
    Scratch scratch;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaEvent_t completion_event = nullptr;
    cudaEvent_t dispatch_done_event = nullptr;
    std::vector<cudaEvent_t> consumer_fence_events;
    std::vector<std::array<at::Tensor, 5>> output_rows;
    std::vector<at::Tensor> raw_outputs;
    std::vector<GraphRowRetirement> row_retirements;
    int published_rows = 0;
    int retired_rows = 0;
    int64_t generation = 0;

    GraphSlot() = default;
    GraphSlot(const GraphSlot&) = delete;
    GraphSlot& operator=(const GraphSlot&) = delete;
    GraphSlot(GraphSlot&& other) noexcept;
    GraphSlot& operator=(GraphSlot&& other) noexcept;
    ~GraphSlot();
    void reset() noexcept;
  };

  struct GraphStaging {
    bool initialized = false;
    std::vector<int64_t> chunk_shape;
    torch::Tensor chunks;
    torch::Tensor length;
    torch::Tensor cache_ch;
    torch::Tensor cache_t;
    torch::Tensor cache_ch_len;
    std::vector<torch::Tensor> row_indices;
    std::vector<torch::Tensor> length_scalars;
  };

  struct GraphBucketState {
    int bucket = 0;
    bool initialized = false;
    bool disabled = false;
    std::string disabled_reason;
    GraphStaging staging;
    std::vector<GraphSlot> slots;
    std::vector<int64_t> chunk_shape;
    std::vector<int64_t> cache_ch_shape;
    std::vector<int64_t> cache_t_shape;
    std::vector<int64_t> cache_ch_len_shape;
  };

  void dispatcher_loop();
  void former_loop();
  void lane_loop(size_t lane_index);
  std::vector<int> required_buckets() const;
  int effective_max_batch_size() const;
  bool min_fill_active() const;
  int dispatch_bucket_for_k(int k) const;
  FormedBatch gather_batch(DispatchLane* lane);
  bool assign_batch_to_free_lane(FormedBatch formed_batch);
  void fail_formed_batch(FormedBatch formed_batch, std::exception_ptr ep) noexcept;
  void fail_formed_batches(std::vector<FormedBatch>* formed_batches, std::exception_ptr ep) noexcept;
  void mark_lane_free_for_assignment(DispatchLane& lane);
  void drain_lane_inboxes_locked(std::vector<FormedBatch>* formed_batches);
  void request_fault_shutdown(std::exception_ptr ep) noexcept;
  void notify_shutdown_waiters() noexcept;
  void join_threads_except_current();
  void force_drain_pending_dispatches_all();
  bool any_pending_dispatches_locked() const;
  void drain_pending_dispatches_all_locked(bool force,
                                           std::vector<std::shared_ptr<QueueItem>>* deferred_owners);
  void drain_dispatch_timing_events_all(bool force);
  void execute_dispatch_lane(DispatchLane& lane,
                             FormedBatch formed_batch,
                             std::thread::id dispatcher_thread_id);
  void dispatch_batch(DispatchLane& lane,
                      std::vector<std::shared_ptr<QueueItem>> batch,
                      std::thread::id dispatcher_thread_id,
                      double queue_depth_at_form);
  bool drain_one_dispatch_timing_event(DispatchLane& lane, bool force);
  void drain_dispatch_timing_events(DispatchLane& lane, bool force);
  void cap_pending_dispatch_timing_events(DispatchLane& lane);
  bool capacity_available_locked() const;
  void release_capacity_tokens_locked(int tokens);
  bool drain_one_pending_dispatch_locked(DispatchLane& lane,
                                         bool force,
                                         std::vector<std::shared_ptr<QueueItem>>* deferred_owners);
  void drain_pending_dispatches_locked(DispatchLane& lane,
                                       bool force,
                                       std::vector<std::shared_ptr<QueueItem>>* deferred_owners);
  void cap_pending_dispatches_locked(DispatchLane& lane,
                                     std::vector<std::shared_ptr<QueueItem>>* deferred_owners);
  bool graph_enabled_from_env() const;
  int graph_slot_count() const;
  void initialize_graph_runtime();
  bool graph_bucket_replayable(int bucket) const;
  bool should_use_graph_for_batch(const std::vector<BatchedSteadyInput>& ready, int bucket);
  bool dispatch_batch_graph(DispatchLane& lane,
                            std::vector<std::shared_ptr<QueueItem>>& batch,
                            const std::vector<BatchedSteadyInput>& ready,
                            int bucket,
                            int64_t cycle_id,
                            const std::vector<double>& gather_waits,
                            const std::vector<double>& service_waits,
                            bool backlog,
                            double queue_depth_at_form,
                            Clock::time_point dispatch_wall_start,
                            double dispatch_cpu_start_us);
  GraphBucketState& ensure_graph_bucket_state(int bucket, const BatchedSteadyInput& first);
  GraphSlot* acquire_graph_slot(DispatchLane& lane, int bucket);
  void initialize_graph_slot(GraphBucketState& state, GraphSlot& slot);
  void capture_graph_slot(GraphBucketState& state, GraphSlot& slot);
  Scratch& ensure_graph_slot_scratch(GraphBucketState& state, GraphSlot& slot);
  std::vector<at::Tensor> pack_staging_into_scratch_for_graph(GraphBucketState& state, GraphSlot& slot);
  void copy_ready_into_graph_staging(GraphBucketState& state,
                                     const std::vector<BatchedSteadyInput>& ready);
  void copy_raw_to_graph_slot(const std::vector<at::Tensor>& raw, GraphSlot& slot);
  std::vector<at::Tensor> graph_slot_row_tensors(GraphSlot& slot, int row);
  bool drain_one_graph_slot_locked(bool force);
  void drain_graph_slots_locked(bool force);
  bool graph_slots_active_locked() const;
  void wait_for_graph_slots_shutdown();
  void retire_graph_slot_consumer(int bucket, int slot_id, int row, cudaStream_t consumer_stream);
  void abandon_graph_slot_consumer(int bucket, int slot_id, int row) noexcept;
  void mark_graph_slot_free_locked(GraphSlot& slot);
  std::vector<at::Tensor> pack_into_scratch(DispatchLane& lane,
                                            const std::vector<BatchedSteadyInput>& ready,
                                            int bucket);
  Scratch& ensure_scratch(DispatchLane& lane, int bucket, const BatchedSteadyInput& first);
  void set_pending_exception_locked(std::vector<std::shared_ptr<QueueItem>>* pending);
  void set_item_exception(const std::shared_ptr<QueueItem>& item, std::exception_ptr ep);
  void apply_test_lane_hooks(DispatchLane& lane, const char* stage);
  void add_dispatch_telemetry(int lane_id,
                              int bucket,
                              int64_t cycle_id,
                              int k,
                              bool backlog,
                              const std::vector<double>& gather_wait_us,
                              const std::vector<double>& service_wait_us,
                              double cuda_run_us,
                              double wakeup_jitter_us,
                              double queue_depth_at_form,
                              Clock::time_point dispatch_wall_start,
                              Clock::time_point dispatch_wall_end,
                              double dispatch_cpu_start_us,
                              double dispatch_cpu_end_us);
  void add_dispatch_timing_telemetry_locked(int lane_id,
                                            int k,
                                            double cuda_run_us,
                                            Clock::time_point timing_wall_end);
  BatchedSteadySchedulerTelemetry::Lane& lane_telemetry_locked(int lane_id);
  DispatchLane& graph_lane();
  const DispatchLane& graph_lane() const;
  static void cuda_check(cudaError_t err, const char* expr, const char* file, int line);
  static DispatchTimingMode dispatch_timing_mode_from_env();

  BatchedSteadyLoaderSet& loader_set_;
  torch::Device device_;
  BatchedSteadySchedulerPolicy policy_;
  DispatchTimingMode dispatch_timing_mode_;
  bool graph_enabled_ = false;
  int graph_slots_per_bucket_ = 0;
  std::vector<DispatchLane> dispatch_lanes_;

  std::mutex close_mutex_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable cv_capacity_;
  std::condition_variable cv_graph_slots_;
  std::mutex lane_mutex_;
  std::condition_variable cv_lane_;
  bool lane_shutdown_ = false;
  std::deque<std::shared_ptr<QueueItem>> queue_;
  bool started_ = false;
  bool closing_ = false;
  bool closed_ = false;
  int64_t next_sequence_ = 0;
  int64_t next_cycle_id_ = 0;
  int capacity_tokens_in_use_ = 0;
  std::exception_ptr fault_;
  int test_slow_lane_ = -1;
  int test_slow_us_ = 0;
  int test_fault_lane_ = -1;
  int test_fault_after_ = 0;
  std::thread dispatcher_thread_;
  std::vector<std::thread> lane_threads_;
  std::map<int, GraphBucketState> graph_buckets_;
  int64_t next_graph_slot_generation_ = 1;

  mutable std::mutex telemetry_mutex_;
  BatchedSteadySchedulerTelemetry telemetry_;
  bool dispatcher_measurement_started_ = false;
  Clock::time_point dispatcher_measurement_wall_start_;
  double dispatcher_measurement_cpu_start_us_ = 0.0;
  std::map<int64_t, int> worker_wait_expected_by_cycle_;
  std::map<int64_t, std::vector<double>> worker_waits_by_cycle_;
};
