#include "lib/scheduler/batched_steady_scheduler.h"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <pthread.h>
#include <sstream>

#define B2_CUDA_CHECK(expr) BatchedSteadyScheduler::cuda_check((expr), #expr, __FILE__, __LINE__)

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kMaxPendingDispatchTimings = 64;
constexpr size_t kMaxPendingDispatches = 64;
constexpr int kDispatchTimingIdlePollMs = 2;
constexpr int kPendingDispatchIdlePollMs = 2;

struct CudaEventOwner {
  cudaEvent_t event = nullptr;

  CudaEventOwner() = default;
  CudaEventOwner(const CudaEventOwner&) = delete;
  CudaEventOwner& operator=(const CudaEventOwner&) = delete;

  ~CudaEventOwner() {
    if (event != nullptr) cudaEventDestroy(event);
  }

  cudaEvent_t release() noexcept {
    cudaEvent_t out = event;
    event = nullptr;
    return out;
  }
};

template <typename Fn>
class ScopeExit {
 public:
  explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() noexcept {
    if (active_) fn_();
  }
  void dismiss() noexcept {
    active_ = false;
  }

 private:
  Fn fn_;
  bool active_ = true;
};

template <typename Fn>
ScopeExit<Fn> make_scope_exit(Fn fn) {
  return ScopeExit<Fn>(std::move(fn));
}

double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

std::chrono::milliseconds ms_duration(int ms) {
  return std::chrono::milliseconds(std::max(0, ms));
}

int min_fill_count_for_bucket(int bucket) {
  if (bucket <= 4) return 1;
  return (bucket * 3 + 3) / 4;
}

torch::TensorOptions long_options_for(const torch::Tensor& tensor) {
  return torch::TensorOptions().dtype(torch::kLong).device(tensor.device());
}

double timespec_to_us(const timespec& ts) {
  return static_cast<double>(ts.tv_sec) * 1000000.0 + static_cast<double>(ts.tv_nsec) / 1000.0;
}

double current_thread_cpu_us() {
  clockid_t clock_id;
  if (pthread_getcpuclockid(pthread_self(), &clock_id) != 0) {
    clock_id = CLOCK_THREAD_CPUTIME_ID;
  }
  timespec ts{};
  if (clock_gettime(clock_id, &ts) != 0) return -1.0;
  return timespec_to_us(ts);
}

bool aliases_any_raw_output(const at::Tensor& tensor, const std::vector<at::Tensor>& raw) {
  if (!tensor.defined() || !tensor.has_storage()) return false;
  for (const auto& candidate : raw) {
    if (candidate.defined() && candidate.has_storage() && tensor.is_alias_of(candidate)) return true;
  }
  return false;
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

int scheduler_env_int(const char* name, int default_value) {
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

std::vector<int64_t> tensor_shape_vec(const at::Tensor& tensor) {
  return std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
}

at::Tensor raw_row_tensor(const std::vector<at::Tensor>& raw, int output_index, int64_t row) {
  switch (output_index) {
    case 0:
      return raw[0].select(0, row).unsqueeze(0);
    case 1:
      return raw[1].select(0, row).reshape({1});
    case 2:
      return raw[2].select(1, row).unsqueeze(1);
    case 3:
      return raw[3].select(1, row).unsqueeze(1);
    case 4:
      return raw[4].select(0, row).reshape({1});
    default:
      throw std::runtime_error("bad steady graph raw output index");
  }
}

int64_t tensor_bytes(const at::Tensor& tensor) {
  return tensor.defined() ? tensor.numel() * tensor.element_size() : 0;
}

double bytes_to_mib(int64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::vector<at::Tensor> own_row_output_tensors(std::vector<at::Tensor>&& row_tensors,
                                               const std::vector<at::Tensor>& raw) {
  // unpack_prepacked_outputs() uses .contiguous(), but that is a no-op for
  // already-contiguous row views such as B=1 slices. Clone any tensor still
  // aliasing the raw AOTI outputs while the lane stream is current, before
  // the per-row completion event is recorded.
  for (auto& tensor : row_tensors) {
    if (aliases_any_raw_output(tensor, raw)) {
      tensor = tensor.clone(at::MemoryFormat::Preserve);
    }
  }
  return std::move(row_tensors);
}
}  // namespace

BatchedSteadyScheduler::GraphSlot::GraphSlot(GraphSlot&& other) noexcept
    : bucket(other.bucket),
      slot_id(other.slot_id),
      state(other.state),
      captured(other.captured),
      graph_loader(std::move(other.graph_loader)),
      scratch(std::move(other.scratch)),
      graph(other.graph),
      exec(other.exec),
      completion_event(other.completion_event),
      dispatch_done_event(other.dispatch_done_event),
      consumer_fence_events(std::move(other.consumer_fence_events)),
      output_rows(std::move(other.output_rows)),
      raw_outputs(std::move(other.raw_outputs)),
      row_retirements(std::move(other.row_retirements)),
      published_rows(other.published_rows),
      retired_rows(other.retired_rows),
      generation(other.generation) {
  other.graph = nullptr;
  other.exec = nullptr;
  other.completion_event = nullptr;
  other.dispatch_done_event = nullptr;
  other.state = GraphSlotState::Free;
  other.captured = false;
  other.published_rows = 0;
  other.retired_rows = 0;
  other.generation = 0;
}

BatchedSteadyScheduler::GraphSlot& BatchedSteadyScheduler::GraphSlot::operator=(GraphSlot&& other) noexcept {
  if (this != &other) {
    reset();
    bucket = other.bucket;
    slot_id = other.slot_id;
    state = other.state;
    captured = other.captured;
    graph_loader = std::move(other.graph_loader);
    scratch = std::move(other.scratch);
    graph = other.graph;
    exec = other.exec;
    completion_event = other.completion_event;
    dispatch_done_event = other.dispatch_done_event;
    consumer_fence_events = std::move(other.consumer_fence_events);
    output_rows = std::move(other.output_rows);
    raw_outputs = std::move(other.raw_outputs);
    row_retirements = std::move(other.row_retirements);
    published_rows = other.published_rows;
    retired_rows = other.retired_rows;
    generation = other.generation;
    other.graph = nullptr;
    other.exec = nullptr;
    other.completion_event = nullptr;
    other.dispatch_done_event = nullptr;
    other.state = GraphSlotState::Free;
    other.captured = false;
    other.published_rows = 0;
    other.retired_rows = 0;
    other.generation = 0;
  }
  return *this;
}

BatchedSteadyScheduler::GraphSlot::~GraphSlot() {
  reset();
}

void BatchedSteadyScheduler::GraphSlot::reset() noexcept {
  if (exec != nullptr) cudaGraphExecDestroy(exec);
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (completion_event != nullptr) cudaEventDestroy(completion_event);
  if (dispatch_done_event != nullptr) cudaEventDestroy(dispatch_done_event);
  for (cudaEvent_t event : consumer_fence_events) {
    if (event != nullptr) cudaEventDestroy(event);
  }
  graph = nullptr;
  exec = nullptr;
  completion_event = nullptr;
  dispatch_done_event = nullptr;
  consumer_fence_events.clear();
  output_rows.clear();
  raw_outputs.clear();
  row_retirements.clear();
  graph_loader.reset();
  scratch.initialized = false;
  scratch.chunk_shape.clear();
  scratch.chunks = torch::Tensor();
  scratch.length = torch::Tensor();
  scratch.cache_ch = torch::Tensor();
  scratch.cache_t = torch::Tensor();
  scratch.cache_ch_len = torch::Tensor();
  scratch.row_indices.clear();
  scratch.length_scalars.clear();
  state = GraphSlotState::Free;
  captured = false;
  published_rows = 0;
  retired_rows = 0;
  generation = 0;
}

BatchedSteadyScheduler::BatchedSteadyScheduler(BatchedSteadyLoaderSet& loader_set,
                                               torch::Device device,
                                               BatchedSteadySchedulerPolicy policy)
    : loader_set_(loader_set),
      device_(device),
      policy_(policy),
      dispatch_timing_mode_(dispatch_timing_mode_from_env()),
      graph_enabled_(graph_enabled_from_env()),
      graph_slots_per_bucket_(graph_enabled_ ? graph_slot_count() : 0) {
  if (policy_.window_ms < 0) throw std::runtime_error("batch steady window_ms must be non-negative");
  if (policy_.lone_timeout_ms < 0) throw std::runtime_error("batch steady lone_timeout_ms must be non-negative");
  if (policy_.max_queue_delay_ms < 0) {
    throw std::runtime_error("batch steady max_queue_delay_ms must be non-negative");
  }
  if (policy_.dispatch_lanes <= 0 || policy_.dispatch_lanes > 2) {
    throw std::runtime_error("batch steady dispatch_lanes must be in [1,2]");
  }
  if (policy_.dispatch_lanes > 1 && graph_enabled_) {
    throw std::runtime_error(
        "batch steady dispatch_lanes > 1 is incompatible with NEMOTRON_WS_STEADY_CUDAGRAPH=1; "
        "steady CUDA graph is single-lane only");
  }
  if (loader_set_.num_runners() < policy_.dispatch_lanes) {
    std::ostringstream oss;
    oss << "batch steady shared-loader model requires steady_num_runners >= dispatch_lanes"
        << " (steady_num_runners=" << loader_set_.num_runners()
        << ", dispatch_lanes=" << policy_.dispatch_lanes << ")";
    throw std::runtime_error(oss.str());
  }
  if (policy_.B_max != 1 && policy_.B_max != 2 && policy_.B_max != 4 &&
      policy_.B_max != 8 && policy_.B_max != 16) {
    throw std::runtime_error("batch steady B_max must be one of {1,2,4,8,16}");
  }
  if (policy_.force_bucket != 0 && policy_.force_bucket != 1 && policy_.force_bucket != 2 &&
      policy_.force_bucket != 4 && policy_.force_bucket != 8 && policy_.force_bucket != 16) {
    throw std::runtime_error("batch steady force_bucket must be 0 or one of {1,2,4,8,16}");
  }
  if (policy_.queue_capacity <= 0) policy_.queue_capacity = 16;
  test_slow_lane_ = scheduler_env_int("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_LANE", -1);
  test_slow_us_ = scheduler_env_int("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_US", 0);
  test_fault_lane_ = scheduler_env_int("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_LANE", -1);
  test_fault_after_ = scheduler_env_int("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_AFTER", 0);
  if (test_slow_lane_ < -2 || test_slow_lane_ >= policy_.dispatch_lanes) {
    throw std::runtime_error("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_LANE out of range");
  }
  if (test_slow_us_ < 0) {
    throw std::runtime_error("NEMOTRON_DENSITY_SCHEDULER_TEST_SLOW_US must be non-negative");
  }
  if (test_fault_lane_ < -1 || test_fault_lane_ >= policy_.dispatch_lanes) {
    throw std::runtime_error("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_LANE out of range");
  }
  if (test_fault_after_ < 0) {
    throw std::runtime_error("NEMOTRON_DENSITY_SCHEDULER_TEST_FAULT_AFTER must be non-negative");
  }
  dispatch_lanes_.reserve(static_cast<size_t>(policy_.dispatch_lanes));
  for (int lane_id = 0; lane_id < policy_.dispatch_lanes; ++lane_id) {
    dispatch_lanes_.emplace_back(
        lane_id,
        c10::cuda::getStreamFromPool(/*isHighPriority=*/false,
                                     device.index() >= 0 ? device.index() : 0));
  }
  if (dispatch_lanes_.empty()) throw std::runtime_error("batch steady constructed with no dispatch lanes");
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    telemetry_.dispatch_lanes = policy_.dispatch_lanes;
    telemetry_.lanes.resize(static_cast<size_t>(policy_.dispatch_lanes));
    for (int lane_id = 0; lane_id < policy_.dispatch_lanes; ++lane_id) {
      telemetry_.lanes[static_cast<size_t>(lane_id)].lane_id = lane_id;
    }
  }
  auto buckets = required_buckets();
  loader_set_.preload_buckets(buckets);
  if (!loader_set_.sealed()) throw std::runtime_error("batch steady loader set failed to seal after preload_buckets()");
  std::ostringstream stream_list;
  for (size_t i = 0; i < dispatch_lanes_.size(); ++i) {
    if (i > 0) stream_list << ",";
    stream_list << reinterpret_cast<void*>(dispatch_lanes_[i].dispatcher_stream.stream());
  }
  std::printf("B2_SCHEDULER_CONSTRUCT policy={B_max:%d,window_ms:%d,lone_timeout_ms:%d,max_queue_delay_ms:%d,"
              "queue_capacity:%d,use_b2_bucket:%s,min_fill_enabled:%s,disable_min_fill:%s,force_bucket:%d} "
              "loaded_buckets=%d dispatch_lanes=%d dispatcher_streams=[%s] "
              "steady_cuda_graph=%s graph_slots_per_bucket=%d\n",
              policy_.B_max,
              policy_.window_ms,
              policy_.lone_timeout_ms,
              policy_.max_queue_delay_ms,
              policy_.queue_capacity,
              policy_.use_b2_bucket ? "true" : "false",
              policy_.min_fill_enabled ? "true" : "false",
              policy_.disable_min_fill ? "true" : "false",
              policy_.force_bucket,
              loader_set_.loaded_bucket_count(),
              policy_.dispatch_lanes,
              stream_list.str().c_str(),
              graph_enabled_ ? "enabled" : "off",
              graph_slots_per_bucket_);
}

BatchedSteadyScheduler::~BatchedSteadyScheduler() {
  close();
}

void BatchedSteadyScheduler::cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr
        << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

BatchedSteadyScheduler::DispatchTimingMode BatchedSteadyScheduler::dispatch_timing_mode_from_env() {
  const char* raw = std::getenv("NEMOTRON_WS_DISPATCH_TIMING");
  if (raw == nullptr || raw[0] == '\0') return DispatchTimingMode::Poll;
  if (std::string(raw) == "sync") return DispatchTimingMode::Sync;
  if (std::string(raw) == "poll") return DispatchTimingMode::Poll;
  if (std::string(raw) == "off") return DispatchTimingMode::Off;
  throw std::runtime_error("NEMOTRON_WS_DISPATCH_TIMING must be one of {sync,poll,off}");
}

bool BatchedSteadyScheduler::graph_enabled_from_env() const {
  const char* raw = std::getenv("NEMOTRON_WS_STEADY_CUDAGRAPH");
  if (raw == nullptr || raw[0] == '\0') return false;
  std::string value = lowercase_ascii(raw);
  if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
  if (value == "0" || value == "false" || value == "no" || value == "off") return false;
  throw std::runtime_error("NEMOTRON_WS_STEADY_CUDAGRAPH must be unset/off/0 or on/1");
}

int BatchedSteadyScheduler::graph_slot_count() const {
  const char* raw = std::getenv("NEMOTRON_WS_STEADY_CUDAGRAPH_SLOTS");
  if (raw == nullptr || raw[0] == '\0') return 4;
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value <= 0 || value > 64) {
    throw std::runtime_error("NEMOTRON_WS_STEADY_CUDAGRAPH_SLOTS must be an integer in [1,64]");
  }
  return static_cast<int>(value);
}

std::future<DispatchResult> BatchedSteadyScheduler::enqueue(EnqueueRequest&& request) {
  if (request.producer_event == nullptr) {
    throw std::runtime_error("batch steady enqueue requires a producer-ready CUDA event");
  }
  auto item = std::make_shared<QueueItem>(std::move(request));
  item->enqueue_time = Clock::now();
  auto future = item->promise.get_future();

  std::vector<std::shared_ptr<QueueItem>> deferred_owners;
  std::unique_lock<std::mutex> lock(mutex_);
  auto capacity_ready = [&] {
    drain_pending_dispatches_all_locked(false, &deferred_owners);
    drain_graph_slots_locked(false);
    return closing_ || fault_ || capacity_available_locked();
  };
  while (!capacity_ready()) {
    cv_capacity_.wait_for(lock, ms_duration(kPendingDispatchIdlePollMs), capacity_ready);
  }
  if (fault_) std::rethrow_exception(fault_);
  if (closing_) throw std::runtime_error("batch steady enqueue after scheduler close");
  item->sequence = next_sequence_++;
  ++capacity_tokens_in_use_;
  queue_.push_back(item);
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    ++telemetry_.enqueued;
  }
  cv_.notify_one();
  return future;
}

std::optional<std::future<DispatchResult>> BatchedSteadyScheduler::try_enqueue_until(
    EnqueueRequest&& request,
    std::chrono::steady_clock::time_point deadline) {
  if (request.producer_event == nullptr) {
    throw std::runtime_error("batch steady enqueue requires a producer-ready CUDA event");
  }
  auto item = std::make_shared<QueueItem>(std::move(request));
  item->enqueue_time = Clock::now();
  auto future = item->promise.get_future();

  std::vector<std::shared_ptr<QueueItem>> deferred_owners;
  std::unique_lock<std::mutex> lock(mutex_);
  auto capacity_ready = [&] {
    drain_pending_dispatches_all_locked(false, &deferred_owners);
    drain_graph_slots_locked(false);
    return closing_ || fault_ || capacity_available_locked();
  };
  while (!capacity_ready()) {
    auto now = Clock::now();
    if (now >= deadline) return std::nullopt;
    auto wait_deadline = std::min(deadline, now + ms_duration(kPendingDispatchIdlePollMs));
    cv_capacity_.wait_until(lock, wait_deadline, capacity_ready);
  }
  if (fault_) std::rethrow_exception(fault_);
  if (closing_) throw std::runtime_error("batch steady enqueue after scheduler close");
  item->sequence = next_sequence_++;
  ++capacity_tokens_in_use_;
  queue_.push_back(item);
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    ++telemetry_.enqueued;
  }
  cv_.notify_one();
  return std::optional<std::future<DispatchResult>>(std::move(future));
}

void BatchedSteadyScheduler::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) throw std::runtime_error("batch steady start after close");
  if (started_) return;
  started_ = true;
  {
    std::lock_guard<std::mutex> lane_lock(lane_mutex_);
    lane_shutdown_ = false;
  }
  if (dispatch_lanes_.size() == 1) {
    dispatcher_thread_ = std::thread([this] { dispatcher_loop(); });
  } else {
    lane_threads_.reserve(dispatch_lanes_.size());
    for (size_t lane_index = 0; lane_index < dispatch_lanes_.size(); ++lane_index) {
      lane_threads_.emplace_back([this, lane_index] { lane_loop(lane_index); });
    }
    dispatcher_thread_ = std::thread([this] { former_loop(); });
  }
}

void BatchedSteadyScheduler::notify_shutdown_waiters() noexcept {
  cv_.notify_all();
  cv_capacity_.notify_all();
  cv_graph_slots_.notify_all();
  cv_lane_.notify_all();
}

void BatchedSteadyScheduler::drain_lane_inboxes_locked(std::vector<FormedBatch>* formed_batches) {
  for (auto& lane : dispatch_lanes_) {
    while (!lane.inbox.empty()) {
      if (formed_batches != nullptr) {
        formed_batches->push_back(std::move(lane.inbox.front()));
      }
      lane.inbox.pop_front();
    }
    lane.free = true;
  }
}

void BatchedSteadyScheduler::fail_formed_batches(std::vector<FormedBatch>* formed_batches,
                                                 std::exception_ptr ep) noexcept {
  if (formed_batches == nullptr) return;
  for (auto& formed_batch : *formed_batches) {
    fail_formed_batch(std::move(formed_batch), ep);
  }
  formed_batches->clear();
}

void BatchedSteadyScheduler::request_fault_shutdown(std::exception_ptr ep) noexcept {
  if (!ep) {
    try {
      ep = std::make_exception_ptr(std::runtime_error("batch steady scheduler fault"));
    } catch (...) {
      ep = std::current_exception();
    }
  }
  std::vector<std::shared_ptr<QueueItem>> pending;
  std::vector<FormedBatch> inboxed;
  try {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!fault_) fault_ = ep;
      closing_ = true;
      set_pending_exception_locked(&pending);
    }
    {
      std::lock_guard<std::mutex> lane_lock(lane_mutex_);
      lane_shutdown_ = true;
      drain_lane_inboxes_locked(&inboxed);
    }
  } catch (...) {
  }
  for (const auto& item : pending) set_item_exception(item, ep);
  fail_formed_batches(&inboxed, ep);
  notify_shutdown_waiters();
}

void BatchedSteadyScheduler::join_threads_except_current() {
  const std::thread::id current = std::this_thread::get_id();
  if (dispatcher_thread_.joinable() && dispatcher_thread_.get_id() != current) {
    dispatcher_thread_.join();
  }
  for (auto& thread : lane_threads_) {
    if (thread.joinable() && thread.get_id() != current) {
      thread.join();
    }
  }
}

void BatchedSteadyScheduler::force_drain_pending_dispatches_all() {
  while (true) {
    std::vector<std::shared_ptr<QueueItem>> deferred_owners;
    bool done = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      drain_pending_dispatches_all_locked(true, &deferred_owners);
      drain_graph_slots_locked(true);
      done = !any_pending_dispatches_locked() && !graph_slots_active_locked();
    }
    deferred_owners.clear();
    if (done) break;
  }
}

void BatchedSteadyScheduler::close() {
  std::unique_lock<std::mutex> close_lock(close_mutex_);
  std::vector<std::shared_ptr<QueueItem>> pending;
  std::vector<FormedBatch> inboxed;
  auto ep = std::make_exception_ptr(std::runtime_error("batch steady scheduler closed"));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    if (!fault_) fault_ = ep;
    closing_ = true;
    set_pending_exception_locked(&pending);
  }
  {
    std::lock_guard<std::mutex> lane_lock(lane_mutex_);
    lane_shutdown_ = true;
    drain_lane_inboxes_locked(&inboxed);
  }
  for (const auto& item : pending) {
    set_item_exception(item, ep);
  }
  fail_formed_batches(&inboxed, ep);
  notify_shutdown_waiters();
  join_threads_except_current();
  {
    std::lock_guard<std::mutex> lane_lock(lane_mutex_);
    drain_lane_inboxes_locked(&inboxed);
  }
  fail_formed_batches(&inboxed, ep);
  wait_for_graph_slots_shutdown();
  force_drain_pending_dispatches_all();
  drain_dispatch_timing_events_all(true);
  pending.clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    set_pending_exception_locked(&pending);
    closed_ = true;
  }
  for (const auto& item : pending) set_item_exception(item, ep);
}

int BatchedSteadyScheduler::future_timeout_ms() const {
  int backlog_budget_ms = std::max(policy_.queue_capacity, 1) * 50;
  int gather_budget_ms = min_fill_active()
                             ? std::max(policy_.window_ms, policy_.max_queue_delay_ms)
                             : policy_.window_ms;
  return std::max(1000, gather_budget_ms + backlog_budget_ms + 200);
}

void BatchedSteadyScheduler::warmup_buckets() {
  c10::cuda::CUDAGuard device_guard(device_.index());
  torch::NoGradGuard no_grad;
  int warmed_lanes = 0;
  for (auto& lane : dispatch_lanes_) {
    c10::cuda::CUDAStreamGuard stream_guard(lane.dispatcher_stream);
    for (int bucket : required_buckets()) {
      std::vector<BatchedSteadyInput> ready;
      ready.reserve(static_cast<size_t>(bucket));
      auto chunk = torch::zeros({1, 128, 25}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      auto cache_ch = torch::zeros({24, 1, 70, 1024}, chunk.options());
      auto cache_t = torch::zeros({24, 1, 1024, 8}, chunk.options());
      auto cache_ch_len = torch::zeros({1}, torch::TensorOptions().dtype(torch::kLong).device(device_));
      for (int row = 0; row < bucket; ++row) {
        ready.push_back({chunk.clone(), cache_ch.clone(), cache_t.clone(), cache_ch_len.clone(),
                         "warmup.lane" + std::to_string(lane.lane_id) + ".B" + std::to_string(bucket) +
                             ".row" + std::to_string(row)});
      }
      auto inputs = pack_into_scratch(lane, ready, bucket);
      auto raw = loader_set_.run_raw_prepacked(inputs, bucket, lane.dispatcher_stream);
      auto rows = loader_set_.unpack_prepacked_outputs(raw, ready, bucket);
      (void)rows;
      B2_CUDA_CHECK(cudaStreamSynchronize(lane.dispatcher_stream.stream()));
      {
        std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
        ++telemetry_.warmup_runs;
        auto& lane_telem = lane_telemetry_locked(lane.lane_id);
        ++lane_telem.warmup_runs;
      }
      std::printf("B2_SCHEDULER_WARMUP lane=%d bucket=%d rows=%zu\n",
                  lane.lane_id,
                  bucket,
                  ready.size());
    }
    ++warmed_lanes;
  }
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    telemetry_.warmed_lanes = warmed_lanes;
  }
  std::printf("B2_SCHEDULER_WARMUP_DONE warmed_lanes=%d dispatch_lanes=%d\n",
              warmed_lanes,
              policy_.dispatch_lanes);
  if (warmed_lanes != policy_.dispatch_lanes) {
    throw std::runtime_error("batch steady warmup did not warm every dispatch lane");
  }
  if (graph_enabled_) {
    initialize_graph_runtime();
  }
}

std::vector<int> BatchedSteadyScheduler::required_buckets() const {
  return required_buckets_for_policy(policy_);
}

std::vector<int> BatchedSteadyScheduler::required_buckets_for_policy(
    const BatchedSteadySchedulerPolicy& policy) {
  std::vector<int> buckets{1};
  auto add = [&](int bucket) {
    if (std::find(buckets.begin(), buckets.end(), bucket) == buckets.end()) {
      buckets.push_back(bucket);
    }
  };
  if (policy.B_max != 1 && policy.B_max != 2 && policy.B_max != 4 &&
      policy.B_max != 8 && policy.B_max != 16) {
    throw std::runtime_error("batch steady B_max must be one of {1,2,4,8,16}");
  }
  if (policy.force_bucket != 0 && policy.force_bucket != 1 && policy.force_bucket != 2 &&
      policy.force_bucket != 4 && policy.force_bucket != 8 && policy.force_bucket != 16) {
    throw std::runtime_error("batch steady force_bucket must be 0 or one of {1,2,4,8,16}");
  }
  int max_bucket = std::max(policy.B_max, policy.force_bucket);
  if (max_bucket >= 2) add(policy.use_b2_bucket ? 2 : 4);
  if (max_bucket >= 4) add(4);
  if (max_bucket >= 8) add(8);
  if (max_bucket >= 16) add(16);
  if (policy.force_bucket > 0) add(policy.force_bucket);
  return buckets;
}

int BatchedSteadyScheduler::effective_max_batch_size() const {
  return std::max(policy_.B_max, policy_.force_bucket);
}

bool BatchedSteadyScheduler::min_fill_active() const {
  return policy_.min_fill_enabled && !policy_.disable_min_fill && policy_.force_bucket == 0;
}

int BatchedSteadyScheduler::dispatch_bucket_for_k(int k) const {
  if (policy_.force_bucket > 0) {
    if (k > policy_.force_bucket) {
      throw std::runtime_error("batch steady force_bucket is smaller than dispatch k");
    }
    return policy_.force_bucket;
  }
  int bucket = BatchedSteadyLoaderSet::bucket_for_k_public(k);
  if (bucket > policy_.B_max) bucket = policy_.B_max;
  bucket = BatchedSteadyLoaderSet::bucket_for_k_public(bucket);
  if (!policy_.use_b2_bucket && bucket == 2) return 4;
  return bucket;
}

void BatchedSteadyScheduler::record_worker_wait(int64_t cycle_id,
                                                int k,
                                                double output_sync_us,
                                                double worker_blocked_us,
                                                double completion_wait_us) {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  telemetry_.output_sync_us.push_back(output_sync_us);
  telemetry_.worker_blocked_us.push_back(worker_blocked_us);
  if (completion_wait_us >= 0.0) telemetry_.completion_wait_us.push_back(completion_wait_us);
  if (cycle_id < 0 || k <= 0) return;

  auto expected_it = worker_wait_expected_by_cycle_.find(cycle_id);
  if (expected_it == worker_wait_expected_by_cycle_.end()) {
    expected_it = worker_wait_expected_by_cycle_.emplace(cycle_id, k).first;
  } else {
    expected_it->second = std::max(expected_it->second, k);
  }

  auto& waits = worker_waits_by_cycle_[cycle_id];
  waits.push_back(worker_blocked_us);
  if (static_cast<int>(waits.size()) >= expected_it->second) {
    auto minmax = std::minmax_element(waits.begin(), waits.end());
    telemetry_.per_stream_fairness_spread_us.push_back(*minmax.second - *minmax.first);
    worker_waits_by_cycle_.erase(cycle_id);
    worker_wait_expected_by_cycle_.erase(cycle_id);
  }
}

BatchedSteadySchedulerTelemetry BatchedSteadyScheduler::telemetry_snapshot() const {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  return telemetry_;
}

void BatchedSteadyScheduler::dispatcher_loop() {
  const std::thread::id dispatcher_thread_id = std::this_thread::get_id();
  auto& lane = dispatch_lanes_.front();
  try {
    c10::cuda::CUDAGuard device_guard(device_.index());
    c10::cuda::CUDAStreamGuard stream_guard(lane.dispatcher_stream);
    torch::NoGradGuard no_grad;
    while (true) {
      {
        std::vector<std::shared_ptr<QueueItem>> deferred_owners;
        std::lock_guard<std::mutex> lock(mutex_);
        drain_pending_dispatches_locked(lane, false, &deferred_owners);
        drain_graph_slots_locked(false);
      }
      drain_dispatch_timing_events(lane, false);
      auto formed_batch = gather_batch(&lane);
      if (formed_batch.empty()) break;
      execute_dispatch_lane(lane, std::move(formed_batch), dispatcher_thread_id);
    }
    {
      std::vector<std::shared_ptr<QueueItem>> deferred_owners;
      std::lock_guard<std::mutex> lock(mutex_);
      drain_pending_dispatches_locked(lane, true, &deferred_owners);
      drain_graph_slots_locked(true);
    }
    drain_dispatch_timing_events(lane, true);
  } catch (...) {
    std::exception_ptr ep = std::current_exception();
    request_fault_shutdown(ep);
    try {
      drain_dispatch_timing_events(lane, true);
    } catch (...) {
    }
    try {
      wait_for_graph_slots_shutdown();
    } catch (...) {
    }
    {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      ++telemetry_.dispatcher_exceptions;
    }
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      std::printf("B2_SCHEDULER_DISPATCHER_FATAL error=%s\n", e.what());
    } catch (...) {
      std::printf("B2_SCHEDULER_DISPATCHER_FATAL error=unknown\n");
    }
    std::fflush(stdout);
    return;
  }
}

void BatchedSteadyScheduler::former_loop() {
  try {
    c10::cuda::CUDAGuard device_guard(device_.index());
    torch::NoGradGuard no_grad;
    while (true) {
      {
        std::vector<std::shared_ptr<QueueItem>> deferred_owners;
        std::lock_guard<std::mutex> lock(mutex_);
        drain_pending_dispatches_all_locked(false, &deferred_owners);
        drain_graph_slots_locked(false);
      }
      auto formed_batch = gather_batch(nullptr);
      if (formed_batch.empty()) break;
      if (!assign_batch_to_free_lane(std::move(formed_batch))) break;
    }
  } catch (...) {
    std::exception_ptr ep = std::current_exception();
    request_fault_shutdown(ep);
    {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      ++telemetry_.dispatcher_exceptions;
    }
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      std::printf("B2_SCHEDULER_FORMER_FATAL error=%s\n", e.what());
    } catch (...) {
      std::printf("B2_SCHEDULER_FORMER_FATAL error=unknown\n");
    }
    std::fflush(stdout);
  }
}

void BatchedSteadyScheduler::lane_loop(size_t lane_index) {
  auto& lane = dispatch_lanes_.at(lane_index);
  const std::thread::id dispatcher_thread_id = std::this_thread::get_id();
  try {
    c10::cuda::CUDAGuard device_guard(device_.index());
    c10::cuda::CUDAStreamGuard stream_guard(lane.dispatcher_stream);
    torch::NoGradGuard no_grad;
    while (true) {
      FormedBatch formed_batch;
      bool fail_without_dispatch = false;
      {
        std::unique_lock<std::mutex> lane_lock(lane_mutex_);
        cv_lane_.wait(lane_lock, [&] {
          return lane_shutdown_ || !lane.inbox.empty();
        });
        if (lane.inbox.empty()) break;
        formed_batch = std::move(lane.inbox.front());
        lane.inbox.pop_front();
        fail_without_dispatch = lane_shutdown_;
      }
      if (fail_without_dispatch) {
        fail_formed_batch(
            std::move(formed_batch),
            std::make_exception_ptr(std::runtime_error("batch steady scheduler closed before lane dispatch")));
        continue;
      }
      execute_dispatch_lane(lane, std::move(formed_batch), dispatcher_thread_id);
      drain_dispatch_timing_events(lane, false);
    }
    drain_dispatch_timing_events(lane, true);
  } catch (...) {
    try {
      drain_dispatch_timing_events(lane, true);
    } catch (...) {
    }
    std::exception_ptr ep = std::current_exception();
    request_fault_shutdown(ep);
    {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      ++telemetry_.dispatcher_exceptions;
    }
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      std::printf("B2_SCHEDULER_LANE_FATAL lane=%d error=%s\n", lane.lane_id, e.what());
    } catch (...) {
      std::printf("B2_SCHEDULER_LANE_FATAL lane=%d error=unknown\n", lane.lane_id);
    }
    std::fflush(stdout);
  }
}

BatchedSteadyScheduler::FormedBatch BatchedSteadyScheduler::gather_batch(DispatchLane* lane) {
  FormedBatch formed_batch;
  auto& batch = formed_batch.items;
  std::vector<std::shared_ptr<QueueItem>> deferred_owners;
  std::unique_lock<std::mutex> lock(mutex_);
  auto drain_pending = [&] {
    if (lane != nullptr) {
      drain_pending_dispatches_locked(*lane, false, &deferred_owners);
    } else {
      drain_pending_dispatches_all_locked(false, &deferred_owners);
    }
  };
  auto ready_pred = [&] {
    drain_pending();
    drain_graph_slots_locked(false);
    return closing_ || fault_ || !queue_.empty();
  };
  while (!ready_pred()) {
    bool needs_timed_poll = any_pending_dispatches_locked() ||
                            graph_enabled_ ||
                            (lane != nullptr &&
                             dispatch_timing_mode_ == DispatchTimingMode::Poll &&
                             !lane->pending_dispatch_timings.empty());
    if (needs_timed_poll) {
      lock.unlock();
      if (lane != nullptr && dispatch_timing_mode_ == DispatchTimingMode::Poll) {
        drain_dispatch_timing_events(*lane, false);
      }
      lock.lock();
      if (ready_pred()) break;
      cv_.wait_for(lock,
                   ms_duration(std::min(kDispatchTimingIdlePollMs, kPendingDispatchIdlePollMs)),
                   ready_pred);
    } else {
      cv_.wait(lock, ready_pred);
    }
  }
  if (fault_ && !closing_) std::rethrow_exception(fault_);
  if (closing_ && queue_.empty()) return formed_batch;
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    formed_batch.queue_depth_at_form = static_cast<double>(queue_.size());
    telemetry_.queue_depth.push_back(formed_batch.queue_depth_at_form);
  }

  auto pop_one = [&] {
    auto item = queue_.front();
    queue_.pop_front();
    item->pop_time = Clock::now();
    batch.push_back(item);
  };

  int max_batch_size = effective_max_batch_size();
  if (min_fill_active()) {
    auto select_ready = [&](int limit, int64_t* skipped_ready) {
      std::vector<std::shared_ptr<QueueItem>> selected;
      std::vector<uint64_t> selected_keys;
      selected.reserve(static_cast<size_t>(limit));
      selected_keys.reserve(static_cast<size_t>(limit));
      for (const auto& item : queue_) {
        uint64_t key = item->request.stream_key;
        if (key != 0 &&
            std::find(selected_keys.begin(), selected_keys.end(), key) != selected_keys.end()) {
          if (skipped_ready != nullptr) ++(*skipped_ready);
          continue;
        }
        selected.push_back(item);
        if (key != 0) selected_keys.push_back(key);
        if (static_cast<int>(selected.size()) >= limit) break;
      }
      return selected;
    };

    auto choose_dispatch_count = [&](int ready_count) {
      int dispatch_count = std::max(1, ready_count);
      while (true) {
        int bucket = dispatch_bucket_for_k(dispatch_count);
        if (bucket <= 4 || dispatch_count >= min_fill_count_for_bucket(bucket)) break;
        dispatch_count = std::max(1, bucket / 2);
      }
      return std::min(dispatch_count, ready_count);
    };

    while (true) {
      if (fault_ && !closing_) std::rethrow_exception(fault_);
      if (closing_ && queue_.empty()) return formed_batch;

      auto selected = select_ready(max_batch_size, nullptr);
      if (selected.empty()) break;
      int ready_count = static_cast<int>(selected.size());
      bool lone_fast_path = ready_count == 1 && policy_.lone_timeout_ms <= 0;
      bool full = ready_count >= max_batch_size;
      auto age_deadline = queue_.front()->enqueue_time + ms_duration(policy_.max_queue_delay_ms);
      bool expired = Clock::now() >= age_deadline;
      if (lone_fast_path || full || expired || closing_) {
        int64_t skipped_ready = 0;
        selected = select_ready(max_batch_size, &skipped_ready);
        ready_count = static_cast<int>(selected.size());
        int dispatch_count = choose_dispatch_count(ready_count);
        auto pop_time = Clock::now();
        int removed = 0;
        for (auto it = queue_.begin(); it != queue_.end() && removed < dispatch_count;) {
          auto selected_it = std::find(selected.begin(), selected.begin() + dispatch_count, *it);
          if (selected_it == selected.begin() + dispatch_count) {
            ++it;
            continue;
          }
          auto item = *it;
          it = queue_.erase(it);
          item->pop_time = pop_time;
          batch.push_back(std::move(item));
          ++removed;
        }
        if (removed != dispatch_count) {
          throw std::runtime_error("batch steady adaptive gather failed to pop selected rows");
        }
        if (skipped_ready > 0) {
          std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
          telemetry_.skipped_ready += skipped_ready;
        }
        return formed_batch;
      }

      auto before_wait = Clock::now();
      bool woke = false;
      while (Clock::now() < age_deadline) {
        drain_pending();
        drain_graph_slots_locked(false);
        if (closing_ || fault_ || static_cast<int>(select_ready(max_batch_size, nullptr).size()) >= max_batch_size) {
          woke = true;
          break;
        }
        auto poll_deadline = std::min(age_deadline, Clock::now() + ms_duration(kPendingDispatchIdlePollMs));
        woke = cv_.wait_until(lock, poll_deadline, [&] {
          drain_pending();
          drain_graph_slots_locked(false);
          return closing_ || fault_ ||
                 static_cast<int>(select_ready(max_batch_size, nullptr).size()) >= max_batch_size;
        });
        if (woke) break;
      }
      auto after_wait = Clock::now();
      if (fault_ && !closing_) std::rethrow_exception(fault_);
      if (woke) continue;
      double jitter = elapsed_us(age_deadline, after_wait);
      if (jitter > 0.0) {
        std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
        telemetry_.window_wakeup_jitter_us.push_back(jitter);
      }
      (void)before_wait;
    }
    return formed_batch;
  }

  auto first_pop_time = Clock::now();
  (void)first_pop_time;
  pop_one();

  auto gather_start = Clock::now();
  auto window_deadline = gather_start + ms_duration(policy_.window_ms);
  while (static_cast<int>(batch.size()) < max_batch_size) {
    if (!queue_.empty()) {
      pop_one();
      continue;
    }
    if (closing_) break;

    int wait_ms = static_cast<int>(batch.size()) == 1 ? policy_.lone_timeout_ms : policy_.window_ms;
    if (wait_ms <= 0) break;
    auto deadline = static_cast<int>(batch.size()) == 1
                        ? Clock::now() + ms_duration(wait_ms)
                        : window_deadline;
    auto before_wait = Clock::now();
    bool woke = false;
    while (Clock::now() < deadline) {
      drain_pending();
      drain_graph_slots_locked(false);
      if (closing_ || fault_ || !queue_.empty()) {
        woke = true;
        break;
      }
      auto poll_deadline = std::min(deadline, Clock::now() + ms_duration(kPendingDispatchIdlePollMs));
      woke = cv_.wait_until(lock, poll_deadline, [&] {
        drain_pending();
        drain_graph_slots_locked(false);
        return closing_ || fault_ || !queue_.empty();
      });
      if (woke) break;
    }
    auto after_wait = Clock::now();
    if (fault_ && !closing_) std::rethrow_exception(fault_);
    if (woke) {
      continue;
    }
    double jitter = elapsed_us(deadline, after_wait);
    if (jitter > 0.0) {
      std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
      telemetry_.window_wakeup_jitter_us.push_back(jitter);
    }
    (void)before_wait;
    break;
  }
  return formed_batch;
}

bool BatchedSteadyScheduler::assign_batch_to_free_lane(FormedBatch formed_batch) {
  if (formed_batch.empty()) return true;
  std::unique_lock<std::mutex> lane_lock(lane_mutex_);
  cv_lane_.wait(lane_lock, [&] {
    if (lane_shutdown_) return true;
    for (const auto& lane : dispatch_lanes_) {
      if (lane.free && lane.inbox.empty()) return true;
    }
    return false;
  });
  if (lane_shutdown_) {
    lane_lock.unlock();
    fail_formed_batch(
        std::move(formed_batch),
        std::make_exception_ptr(std::runtime_error("batch steady scheduler closed before lane assignment")));
    return false;
  }
  for (auto& lane : dispatch_lanes_) {
    if (!lane.free || !lane.inbox.empty()) continue;
    lane.free = false;
    lane.inbox.push_back(std::move(formed_batch));
    lane_lock.unlock();
    cv_lane_.notify_all();
    return true;
  }
  throw std::runtime_error("batch steady free-lane wait woke without a free lane");
}

void BatchedSteadyScheduler::fail_formed_batch(FormedBatch formed_batch, std::exception_ptr ep) noexcept {
  if (formed_batch.empty()) return;
  if (!ep) {
    try {
      ep = std::make_exception_ptr(std::runtime_error("batch steady formed batch abandoned"));
    } catch (...) {
      ep = std::current_exception();
    }
  }
  int tokens = static_cast<int>(formed_batch.items.size());
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    release_capacity_tokens_locked(tokens);
  } catch (...) {
  }
  for (const auto& item : formed_batch.items) {
    set_item_exception(item, ep);
  }
}

void BatchedSteadyScheduler::mark_lane_free_for_assignment(DispatchLane& lane) {
  {
    std::lock_guard<std::mutex> lane_lock(lane_mutex_);
    lane.free = true;
  }
  cv_lane_.notify_all();
}

void BatchedSteadyScheduler::execute_dispatch_lane(DispatchLane& lane,
                                                   FormedBatch formed_batch,
                                                   std::thread::id dispatcher_thread_id) {
  dispatch_batch(lane,
                 std::move(formed_batch.items),
                 dispatcher_thread_id,
                 formed_batch.queue_depth_at_form);
}

void BatchedSteadyScheduler::dispatch_batch(DispatchLane& lane,
                                            std::vector<std::shared_ptr<QueueItem>> batch,
                                            std::thread::id dispatcher_thread_id,
                                            double queue_depth_at_form) {
  if (batch.empty()) return;
  int k = static_cast<int>(batch.size());
  bool popped_batch_guard_armed = true;
  auto cleanup_popped_batch = [&](std::exception_ptr ep) noexcept {
    if (!popped_batch_guard_armed) return;
    popped_batch_guard_armed = false;

    CudaEventOwner cleanup_done;
    if (cudaEventCreateWithFlags(&cleanup_done.event, cudaEventDisableTiming) == cudaSuccess &&
        cudaEventRecord(cleanup_done.event, lane.dispatcher_stream.stream()) == cudaSuccess) {
      (void)cudaEventSynchronize(cleanup_done.event);
    }

    for (const auto& item : batch) {
      if (item->request.producer_event != nullptr) {
        (void)cudaEventDestroy(item->request.producer_event);
        item->request.producer_event = nullptr;
      }
    }

    try {
      std::lock_guard<std::mutex> lock(mutex_);
      release_capacity_tokens_locked(k);
    } catch (...) {
    }

    if (!ep) {
      try {
        ep = std::make_exception_ptr(
            std::runtime_error("batch steady dispatch abandoned before pending record install"));
      } catch (...) {
        ep = std::current_exception();
      }
    }
    for (const auto& item : batch) {
      set_item_exception(item, ep);
    }
    batch.clear();
  };
  auto popped_batch_guard = make_scope_exit([&]() noexcept {
    cleanup_popped_batch(nullptr);
  });

  try {
    assert(lane.dispatcher_stream.stream() != nullptr);
    assert(dispatcher_thread_id == std::this_thread::get_id());
    auto dispatch_wall_start = Clock::now();
    double dispatch_cpu_start_us = current_thread_cpu_us();
    c10::cuda::CUDAStreamGuard stream_guard(lane.dispatcher_stream);
    int bucket = dispatch_bucket_for_k(k);

    bool backlog = false;
    {
      std::vector<std::shared_ptr<QueueItem>> deferred_owners;
      std::lock_guard<std::mutex> lock(mutex_);
      backlog = !queue_.empty();
      drain_pending_dispatches_locked(lane, false, &deferred_owners);
      drain_graph_slots_locked(false);
    }

    drain_dispatch_timing_events(lane, false);

    std::vector<BatchedSteadyInput> ready;
    ready.reserve(batch.size());
    for (const auto& item : batch) {
      B2_CUDA_CHECK(cudaStreamWaitEvent(lane.dispatcher_stream.stream(), item->request.producer_event, 0));
      ready.push_back(item->request.input);
    }
    for (const auto& item : batch) {
      B2_CUDA_CHECK(cudaEventDestroy(item->request.producer_event));
      item->request.producer_event = nullptr;
    }
    ++lane.test_dispatch_attempts;
    apply_test_lane_hooks(lane, "after_producer_wait");

    int64_t cycle_id = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cycle_id = next_cycle_id_++;
    }

    auto compute_waits = [&](Clock::time_point service_start,
                             std::vector<double>& gather_waits,
                             std::vector<double>& service_waits) {
      gather_waits.reserve(batch.size());
      service_waits.reserve(batch.size());
      for (const auto& item : batch) {
        gather_waits.push_back(elapsed_us(item->enqueue_time, item->pop_time));
        service_waits.push_back(elapsed_us(item->pop_time, service_start));
      }
    };

    if (graph_enabled_ && should_use_graph_for_batch(ready, bucket)) {
      std::vector<double> graph_gather_waits;
      std::vector<double> graph_service_waits;
      compute_waits(Clock::now(), graph_gather_waits, graph_service_waits);
      if (dispatch_batch_graph(lane,
                               batch,
                               ready,
                               bucket,
                               cycle_id,
                               graph_gather_waits,
                               graph_service_waits,
                               backlog,
                               queue_depth_at_form,
                               dispatch_wall_start,
                               dispatch_cpu_start_us)) {
        popped_batch_guard_armed = false;
        popped_batch_guard.dismiss();
        return;
      }
    }

    auto inputs = pack_into_scratch(lane, ready, bucket);
    auto service_start = Clock::now();
    std::vector<double> gather_waits;
    std::vector<double> service_waits;
    compute_waits(service_start, gather_waits, service_waits);
    cudaEvent_t ev_start{};
    cudaEvent_t ev_stop{};
    const bool timing_enabled = dispatch_timing_mode_ != DispatchTimingMode::Off;
    if (timing_enabled) {
      B2_CUDA_CHECK(cudaEventCreate(&ev_start));
      B2_CUDA_CHECK(cudaEventCreate(&ev_stop));
      B2_CUDA_CHECK(cudaEventRecord(ev_start, lane.dispatcher_stream.stream()));
    }
    // Stream-order invariant: scratch reuse is safe because pack_into_scratch()
    // above, run_raw_prepacked(), unpack_prepacked_outputs(), and the next
    // pack_into_scratch() are all issued by this single dispatcher thread onto
    // the lane stream. The timing sync below is telemetry only; correctness
    // does not rely on it.
    auto raw = loader_set_.run_raw_prepacked(inputs, bucket, lane.dispatcher_stream);
    double cuda_run_us = -1.0;
    if (timing_enabled) {
      B2_CUDA_CHECK(cudaEventRecord(ev_stop, lane.dispatcher_stream.stream()));
      if (dispatch_timing_mode_ == DispatchTimingMode::Sync) {
        // This host-sync is only for CUDA elapsed-time telemetry. It remains
        // available as an opt-in control mode; the default poll path records
        // timing without blocking this dispatcher thread.
        B2_CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float elapsed_ms = 0.0f;
        B2_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        B2_CUDA_CHECK(cudaEventDestroy(ev_start));
        B2_CUDA_CHECK(cudaEventDestroy(ev_stop));
        ev_start = nullptr;
        ev_stop = nullptr;
        cuda_run_us = static_cast<double>(elapsed_ms) * 1000.0;
      } else {
        lane.pending_dispatch_timings.push_back({ev_start, ev_stop, k});
        ev_start = nullptr;
        ev_stop = nullptr;
        cap_pending_dispatch_timing_events(lane);
      }
    }

    auto rows = loader_set_.unpack_prepacked_outputs(raw, ready, bucket);
    if (rows.size() != batch.size()) {
      throw std::runtime_error("batch steady dispatch returned wrong row count");
    }

    auto dispatch_wall_end = Clock::now();
    double dispatch_cpu_end_us = current_thread_cpu_us();
    add_dispatch_telemetry(lane.lane_id,
                           bucket,
                           cycle_id,
                           k,
                           backlog,
                           gather_waits,
                           service_waits,
                           cuda_run_us,
                           0.0,
                           queue_depth_at_form,
                           dispatch_wall_start,
                           dispatch_wall_end,
                           dispatch_cpu_start_us,
                           dispatch_cpu_end_us);

    std::vector<DispatchResult> results;
    results.reserve(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      CudaEventOwner completion;
      B2_CUDA_CHECK(cudaEventCreateWithFlags(&completion.event, cudaEventDisableTiming));
      DispatchResult result;
      result.completion.reset(completion.release());
      result.row_tensors = own_row_output_tensors(std::move(rows[i].tensors), raw);
      B2_CUDA_CHECK(cudaEventRecord(result.completion.get(), lane.dispatcher_stream.stream()));
      result.bucket = bucket;
      result.row = static_cast<int>(i);
      result.k = k;
      result.cycle_id = cycle_id;
      result.gather_wait_us = gather_waits[i];
      result.service_wait_us = service_waits[i];
      result.cuda_run_us = cuda_run_us >= 0.0 ? cuda_run_us : 0.0;
      result.label = rows[i].label;
      results.push_back(std::move(result));
    }

    CudaEventOwner dispatch_done;
    B2_CUDA_CHECK(cudaEventCreateWithFlags(&dispatch_done.event, cudaEventDisableTiming));
    B2_CUDA_CHECK(cudaEventRecord(dispatch_done.event, lane.dispatcher_stream.stream()));
    {
      PendingDispatch pending;
      pending.dispatch_done = dispatch_done.release();
      pending.input_owners = batch;
      pending.capacity_tokens = k;
      std::vector<std::shared_ptr<QueueItem>> deferred_owners;
      std::lock_guard<std::mutex> lock(mutex_);
      lane.pending_dispatches.push_back(std::move(pending));
      popped_batch_guard_armed = false;
      popped_batch_guard.dismiss();
      cap_pending_dispatches_locked(lane, &deferred_owners);
      drain_pending_dispatches_locked(lane, false, &deferred_owners);
    }
    if (dispatch_lanes_.size() > 1) {
      mark_lane_free_for_assignment(lane);
    }

    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i]->promise.set_value(std::move(results[i]));
    }
  } catch (...) {
    cleanup_popped_batch(std::current_exception());
    throw;
  }
}

bool BatchedSteadyScheduler::drain_one_dispatch_timing_event(DispatchLane& lane, bool force) {
  if (lane.pending_dispatch_timings.empty()) return false;
  auto& pending = lane.pending_dispatch_timings.front();
  if (!force) {
    cudaError_t ready = cudaEventQuery(pending.ev_stop);
    if (ready == cudaErrorNotReady) return false;
    B2_CUDA_CHECK(ready);
  } else {
    B2_CUDA_CHECK(cudaEventSynchronize(pending.ev_stop));
  }
  float elapsed_ms = 0.0f;
  B2_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, pending.ev_start, pending.ev_stop));
  B2_CUDA_CHECK(cudaEventDestroy(pending.ev_start));
  B2_CUDA_CHECK(cudaEventDestroy(pending.ev_stop));
  {
    std::lock_guard<std::mutex> telemetry_lock(telemetry_mutex_);
    add_dispatch_timing_telemetry_locked(lane.lane_id,
                                         pending.k,
                                         static_cast<double>(elapsed_ms) * 1000.0,
                                         Clock::now());
  }
  lane.pending_dispatch_timings.pop_front();
  return true;
}

void BatchedSteadyScheduler::drain_dispatch_timing_events(DispatchLane& lane, bool force) {
  while (drain_one_dispatch_timing_event(lane, force)) {
  }
}

void BatchedSteadyScheduler::cap_pending_dispatch_timing_events(DispatchLane& lane) {
  while (lane.pending_dispatch_timings.size() > kMaxPendingDispatchTimings) {
    (void)drain_one_dispatch_timing_event(lane, true);
  }
}

void BatchedSteadyScheduler::drain_dispatch_timing_events_all(bool force) {
  for (auto& lane : dispatch_lanes_) {
    drain_dispatch_timing_events(lane, force);
  }
}

bool BatchedSteadyScheduler::capacity_available_locked() const {
  return capacity_tokens_in_use_ < policy_.queue_capacity;
}

void BatchedSteadyScheduler::release_capacity_tokens_locked(int tokens) {
  if (tokens <= 0) return;
  assert(capacity_tokens_in_use_ >= tokens);
  capacity_tokens_in_use_ = std::max(0, capacity_tokens_in_use_ - tokens);
  cv_capacity_.notify_all();
}

bool BatchedSteadyScheduler::drain_one_pending_dispatch_locked(
    DispatchLane& lane,
    bool force,
    std::vector<std::shared_ptr<QueueItem>>* deferred_owners) {
  if (lane.pending_dispatches.empty()) return false;
  auto& pending = lane.pending_dispatches.front();
  if (!force) {
    cudaError_t ready = cudaEventQuery(pending.dispatch_done);
    if (ready == cudaErrorNotReady) return false;
    B2_CUDA_CHECK(ready);
  } else {
    B2_CUDA_CHECK(cudaEventSynchronize(pending.dispatch_done));
  }

  PendingDispatch retired = std::move(pending);
  lane.pending_dispatches.pop_front();
  if (retired.owns_dispatch_done) B2_CUDA_CHECK(cudaEventDestroy(retired.dispatch_done));
  retired.dispatch_done = nullptr;
  retired.owns_dispatch_done = true;
  release_capacity_tokens_locked(retired.capacity_tokens);
  if (deferred_owners != nullptr) {
    for (auto& owner : retired.input_owners) {
      deferred_owners->push_back(std::move(owner));
    }
    retired.input_owners.clear();
  }
  return true;
}

void BatchedSteadyScheduler::drain_pending_dispatches_locked(
    DispatchLane& lane,
    bool force,
    std::vector<std::shared_ptr<QueueItem>>* deferred_owners) {
  while (drain_one_pending_dispatch_locked(lane, force, deferred_owners)) {
  }
}

void BatchedSteadyScheduler::cap_pending_dispatches_locked(
    DispatchLane& lane,
    std::vector<std::shared_ptr<QueueItem>>* deferred_owners) {
  while (lane.pending_dispatches.size() > kMaxPendingDispatches) {
    (void)drain_one_pending_dispatch_locked(lane, true, deferred_owners);
  }
}

bool BatchedSteadyScheduler::any_pending_dispatches_locked() const {
  for (const auto& lane : dispatch_lanes_) {
    if (!lane.pending_dispatches.empty()) return true;
  }
  return false;
}

void BatchedSteadyScheduler::drain_pending_dispatches_all_locked(
    bool force,
    std::vector<std::shared_ptr<QueueItem>>* deferred_owners) {
  for (auto& lane : dispatch_lanes_) {
    drain_pending_dispatches_locked(lane, force, deferred_owners);
  }
}

void BatchedSteadyScheduler::initialize_graph_runtime() {
  if (!graph_enabled_) return;
  auto& lane = graph_lane();
  c10::cuda::CUDAGuard device_guard(device_.index());
  c10::cuda::CUDAStreamGuard stream_guard(lane.dispatcher_stream);
  torch::NoGradGuard no_grad;
  for (int bucket : required_buckets()) {
    if (!graph_bucket_replayable(bucket)) {
      std::printf("B2_STEADY_CUDAGRAPH_SKIP bucket=%d reason=not_replayable\n", bucket);
      continue;
    }
    if (bucket != 4) {
      std::printf("B2_STEADY_CUDAGRAPH_SKIP bucket=%d reason=unsupported_bucket\n", bucket);
      continue;
    }
    try {
      auto chunk = torch::zeros({1, 128, 25}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      auto cache_ch = torch::zeros({24, 1, 70, 1024}, chunk.options());
      auto cache_t = torch::zeros({24, 1, 1024, 8}, chunk.options());
      auto cache_ch_len = torch::full({1}, 3, torch::TensorOptions().dtype(torch::kLong).device(device_));
      std::vector<BatchedSteadyInput> ready;
      ready.reserve(static_cast<size_t>(bucket));
      for (int row = 0; row < bucket; ++row) {
        ready.push_back({chunk.clone(), cache_ch.clone(), cache_t.clone(), cache_ch_len.clone(),
                         "graph_warmup.B" + std::to_string(bucket) + ".row" + std::to_string(row)});
      }
      auto& state = ensure_graph_bucket_state(bucket, ready.front());
      copy_ready_into_graph_staging(state, ready);
      for (auto& slot : state.slots) {
        initialize_graph_slot(state, slot);
        if (!slot.captured) capture_graph_slot(state, slot);
      }
      B2_CUDA_CHECK(cudaStreamSynchronize(lane.dispatcher_stream.stream()));

      int64_t staging_bytes = tensor_bytes(state.staging.chunks) + tensor_bytes(state.staging.length) +
                              tensor_bytes(state.staging.cache_ch) + tensor_bytes(state.staging.cache_t) +
                              tensor_bytes(state.staging.cache_ch_len);
      int64_t scratch_bytes_per_slot = 0;
      int64_t raw_bytes_per_slot = 0;
      int64_t output_bytes_per_slot = 0;
      if (!state.slots.empty()) {
        const auto& scratch = state.slots.front().scratch;
        scratch_bytes_per_slot = tensor_bytes(scratch.chunks) + tensor_bytes(scratch.length) +
                                 tensor_bytes(scratch.cache_ch) + tensor_bytes(scratch.cache_t) +
                                 tensor_bytes(scratch.cache_ch_len);
        for (const auto& tensor : state.slots.front().raw_outputs) raw_bytes_per_slot += tensor_bytes(tensor);
        for (const auto& row : state.slots.front().output_rows) {
          for (const auto& tensor : row) output_bytes_per_slot += tensor_bytes(tensor);
        }
      }
      int64_t total = staging_bytes +
                      static_cast<int64_t>(state.slots.size()) *
                          (scratch_bytes_per_slot + raw_bytes_per_slot + output_bytes_per_slot);
      std::printf("B2_STEADY_CUDAGRAPH_READY bucket=%d slots=%zu graphs=%zu staging_mib=%.2f scratch_mib_per_slot=%.2f "
                  "raw_mib_per_slot=%.2f output_mib_per_slot=%.2f total_mib=%.2f\n",
                  bucket,
                  state.slots.size(),
                  state.slots.size(),
                  bytes_to_mib(staging_bytes),
                  bytes_to_mib(scratch_bytes_per_slot),
                  bytes_to_mib(raw_bytes_per_slot),
                  bytes_to_mib(output_bytes_per_slot),
                  bytes_to_mib(total));
    } catch (const std::exception& e) {
      auto& state = graph_buckets_[bucket];
      state.bucket = bucket;
      state.disabled = true;
      state.disabled_reason = e.what();
      std::printf("B2_STEADY_CUDAGRAPH_DISABLE bucket=%d reason=%s\n", bucket, e.what());
      std::fflush(stdout);
    }
  }
}

bool BatchedSteadyScheduler::graph_bucket_replayable(int bucket) const {
  return bucket == 4;
}

bool BatchedSteadyScheduler::should_use_graph_for_batch(const std::vector<BatchedSteadyInput>& ready, int bucket) {
  if (!graph_enabled_) return false;
  if (!graph_bucket_replayable(bucket) || static_cast<int>(ready.size()) != bucket) return false;
  auto state_it = graph_buckets_.find(bucket);
  if (state_it == graph_buckets_.end() || state_it->second.disabled) return false;
  return !ready.empty() && static_cast<int>(ready.size()) <= bucket;
}

BatchedSteadyScheduler::GraphBucketState& BatchedSteadyScheduler::ensure_graph_bucket_state(
    int bucket,
    const BatchedSteadyInput& first) {
  auto& state = graph_buckets_[bucket];
  if (state.disabled) return state;
  if (state.initialized) {
    if (tensor_shape_vec(first.chunk) != state.chunk_shape ||
        tensor_shape_vec(first.cache_ch) != state.cache_ch_shape ||
        tensor_shape_vec(first.cache_t) != state.cache_t_shape ||
        tensor_shape_vec(first.cache_ch_len) != state.cache_ch_len_shape) {
      state.disabled = true;
      state.disabled_reason = "graph bucket shape changed";
    }
    return state;
  }

  state.bucket = bucket;
  state.chunk_shape = tensor_shape_vec(first.chunk);
  state.cache_ch_shape = tensor_shape_vec(first.cache_ch);
  state.cache_t_shape = tensor_shape_vec(first.cache_t);
  state.cache_ch_len_shape = tensor_shape_vec(first.cache_ch_len);

  auto chunk_options = first.chunk.options();
  auto long_options = long_options_for(first.chunk);
  state.staging.initialized = true;
  state.staging.chunk_shape = state.chunk_shape;
  state.staging.chunks = torch::empty({bucket, first.chunk.size(1), first.chunk.size(2)}, chunk_options);
  state.staging.length = torch::empty({bucket}, long_options);
  state.staging.cache_ch = torch::empty({first.cache_ch.size(0), bucket, first.cache_ch.size(2), first.cache_ch.size(3)},
                                        first.cache_ch.options());
  state.staging.cache_t = torch::empty({first.cache_t.size(0), bucket, first.cache_t.size(2), first.cache_t.size(3)},
                                       first.cache_t.options());
  state.staging.cache_ch_len = torch::empty({bucket}, long_options);
  state.staging.row_indices.reserve(static_cast<size_t>(bucket));
  state.staging.length_scalars.reserve(static_cast<size_t>(bucket));
  for (int row = 0; row < bucket; ++row) {
    state.staging.row_indices.push_back(torch::full({1}, row, long_options));
    state.staging.length_scalars.push_back(torch::full({1}, first.chunk.size(2), long_options));
  }
  state.staging.chunks.zero_();
  state.staging.cache_ch.zero_();
  state.staging.cache_t.zero_();
  state.staging.cache_ch_len.zero_();
  state.staging.length.fill_(first.chunk.size(2));

  state.slots.reserve(static_cast<size_t>(graph_slots_per_bucket_));
  for (int slot_id = 0; slot_id < graph_slots_per_bucket_; ++slot_id) {
    GraphSlot slot;
    slot.bucket = bucket;
    slot.slot_id = slot_id;
    state.slots.push_back(std::move(slot));
  }
  state.initialized = true;
  return state;
}

void BatchedSteadyScheduler::initialize_graph_slot(GraphBucketState& state, GraphSlot& slot) {
  if (slot.completion_event == nullptr) {
    B2_CUDA_CHECK(cudaEventCreateWithFlags(&slot.completion_event, cudaEventDisableTiming));
  }
  if (slot.dispatch_done_event == nullptr) {
    B2_CUDA_CHECK(cudaEventCreateWithFlags(&slot.dispatch_done_event, cudaEventDisableTiming));
  }
  if (slot.consumer_fence_events.empty()) {
    slot.consumer_fence_events.resize(static_cast<size_t>(state.bucket), nullptr);
    for (auto& event : slot.consumer_fence_events) {
      B2_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
  }
  if (slot.graph_loader == nullptr) {
    slot.graph_loader = loader_set_.create_dedicated_loader_for_bucket(
        state.bucket, 1, "shared_runtime_scheduler_cuda_graph_slot");
  }
  if (!slot.output_rows.empty()) return;

  std::vector<at::Tensor> warm_raw;
  for (int warm = 0; warm < 2; ++warm) {
    auto inputs = pack_staging_into_scratch_for_graph(state, slot);
    warm_raw = slot.graph_loader->run(inputs, reinterpret_cast<void*>(graph_lane().dispatcher_stream.stream()));
    if (warm_raw.size() < 5) throw std::runtime_error("steady graph warm raw returned fewer than 5 outputs");
    B2_CUDA_CHECK(cudaStreamSynchronize(graph_lane().dispatcher_stream.stream()));
  }
  slot.output_rows.resize(static_cast<size_t>(state.bucket));
  for (int row = 0; row < state.bucket; ++row) {
    for (int out = 0; out < 5; ++out) {
      slot.output_rows[static_cast<size_t>(row)][static_cast<size_t>(out)] =
          torch::empty_like(raw_row_tensor(warm_raw, out, row), at::MemoryFormat::Contiguous);
    }
  }
  slot.row_retirements.resize(static_cast<size_t>(state.bucket));
}

void BatchedSteadyScheduler::capture_graph_slot(GraphBucketState& state, GraphSlot& slot) {
  if (slot.exec != nullptr) {
    B2_CUDA_CHECK(cudaGraphExecDestroy(slot.exec));
    slot.exec = nullptr;
  }
  if (slot.graph != nullptr) {
    B2_CUDA_CHECK(cudaGraphDestroy(slot.graph));
    slot.graph = nullptr;
  }
  cudaGraph_t captured = nullptr;
  if (slot.graph_loader == nullptr) {
    throw std::runtime_error("steady graph capture requested before slot graph loader initialization");
  }
  B2_CUDA_CHECK(cudaStreamBeginCapture(graph_lane().dispatcher_stream.stream(), cudaStreamCaptureModeRelaxed));
  try {
    auto inputs = pack_staging_into_scratch_for_graph(state, slot);
    slot.raw_outputs = slot.graph_loader->run(inputs,
                                              reinterpret_cast<void*>(graph_lane().dispatcher_stream.stream()));
    if (slot.raw_outputs.size() < 5) throw std::runtime_error("steady graph capture raw returned fewer than 5 outputs");
    copy_raw_to_graph_slot(slot.raw_outputs, slot);
  } catch (...) {
    (void)cudaStreamEndCapture(graph_lane().dispatcher_stream.stream(), &captured);
    if (captured != nullptr) cudaGraphDestroy(captured);
    throw;
  }
  B2_CUDA_CHECK(cudaStreamEndCapture(graph_lane().dispatcher_stream.stream(), &slot.graph));
  B2_CUDA_CHECK(cudaGraphInstantiate(&slot.exec, slot.graph, 0));
  slot.captured = true;
}

BatchedSteadyScheduler::Scratch& BatchedSteadyScheduler::ensure_graph_slot_scratch(GraphBucketState& state,
                                                                                   GraphSlot& slot) {
  if (!state.staging.initialized) throw std::runtime_error("steady graph staging is not initialized");
  BatchedSteadyInput first{
      state.staging.chunks.select(0, 0).unsqueeze(0),
      state.staging.cache_ch.select(1, 0).unsqueeze(1),
      state.staging.cache_t.select(1, 0).unsqueeze(1),
      state.staging.cache_ch_len.select(0, 0).reshape({1}),
      "graph.staging.first",
  };
  auto& scratch = slot.scratch;
  std::vector<int64_t> chunk_shape(first.chunk.sizes().begin(), first.chunk.sizes().end());
  if (scratch.initialized && scratch.chunk_shape == chunk_shape) return scratch;

  auto chunk_options = first.chunk.options();
  auto long_options = long_options_for(first.chunk);
  scratch.chunk_shape = std::move(chunk_shape);
  scratch.chunks = torch::empty({state.bucket, first.chunk.size(1), first.chunk.size(2)}, chunk_options);
  scratch.length = torch::empty({state.bucket}, long_options);
  scratch.cache_ch = torch::empty({first.cache_ch.size(0), state.bucket, first.cache_ch.size(2), first.cache_ch.size(3)},
                                  first.cache_ch.options());
  scratch.cache_t = torch::empty({first.cache_t.size(0), state.bucket, first.cache_t.size(2), first.cache_t.size(3)},
                                 first.cache_t.options());
  scratch.cache_ch_len = torch::empty({state.bucket}, long_options);
  scratch.row_indices.clear();
  scratch.length_scalars.clear();
  scratch.row_indices.reserve(static_cast<size_t>(state.bucket));
  scratch.length_scalars.reserve(static_cast<size_t>(state.bucket));
  for (int row = 0; row < state.bucket; ++row) {
    scratch.row_indices.push_back(torch::full({1}, row, long_options));
    scratch.length_scalars.push_back(torch::full({1}, first.chunk.size(2), long_options));
  }
  scratch.initialized = true;
  return scratch;
}

std::vector<at::Tensor> BatchedSteadyScheduler::pack_staging_into_scratch_for_graph(GraphBucketState& state,
                                                                                   GraphSlot& slot) {
  if (!state.staging.initialized) throw std::runtime_error("steady graph staging is not initialized");
  auto& scratch = ensure_graph_slot_scratch(state, slot);
  scratch.chunks.copy_(state.staging.chunks);
  scratch.length.copy_(state.staging.length);
  scratch.cache_ch.copy_(state.staging.cache_ch);
  scratch.cache_t.copy_(state.staging.cache_t);
  scratch.cache_ch_len.copy_(state.staging.cache_ch_len);
  return {
      scratch.chunks,
      scratch.length,
      scratch.cache_ch,
      scratch.cache_t,
      scratch.cache_ch_len,
  };
}

void BatchedSteadyScheduler::copy_ready_into_graph_staging(GraphBucketState& state,
                                                          const std::vector<BatchedSteadyInput>& ready) {
  if (ready.empty()) throw std::runtime_error("steady graph staging copy called with no rows");
  for (int row = 0; row < state.bucket; ++row) {
    const auto& src = ready[static_cast<size_t>(row < static_cast<int>(ready.size()) ? row : 0)];
    if (tensor_shape_vec(src.chunk) != state.chunk_shape ||
        tensor_shape_vec(src.cache_ch) != state.cache_ch_shape ||
        tensor_shape_vec(src.cache_t) != state.cache_t_shape ||
        tensor_shape_vec(src.cache_ch_len) != state.cache_ch_len_shape) {
      throw std::runtime_error("steady graph staging shape mismatch");
    }
    state.staging.chunks.select(0, row).unsqueeze(0).copy_(src.chunk);
    state.staging.cache_ch.select(1, row).unsqueeze(1).copy_(src.cache_ch);
    state.staging.cache_t.select(1, row).unsqueeze(1).copy_(src.cache_t);
    state.staging.cache_ch_len.narrow(0, row, 1).copy_(src.cache_ch_len);
    state.staging.length.narrow(0, row, 1).copy_(state.staging.length_scalars[static_cast<size_t>(row)]);
  }
}

void BatchedSteadyScheduler::copy_raw_to_graph_slot(const std::vector<at::Tensor>& raw, GraphSlot& slot) {
  for (int row = 0; row < slot.bucket; ++row) {
    for (int out = 0; out < 5; ++out) {
      slot.output_rows[static_cast<size_t>(row)][static_cast<size_t>(out)].copy_(raw_row_tensor(raw, out, row));
    }
  }
}

std::vector<at::Tensor> BatchedSteadyScheduler::graph_slot_row_tensors(GraphSlot& slot, int row) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(5);
  auto& output_row = slot.output_rows[static_cast<size_t>(row)];
  for (int out = 0; out < 5; ++out) {
    tensors.push_back(output_row[static_cast<size_t>(out)]);
  }
  return tensors;
}

BatchedSteadyScheduler::GraphSlot* BatchedSteadyScheduler::acquire_graph_slot(DispatchLane& lane, int bucket) {
  std::vector<std::shared_ptr<QueueItem>> deferred_owners;
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = graph_buckets_.find(bucket);
  if (it == graph_buckets_.end() || it->second.disabled) return nullptr;
  while (true) {
    drain_pending_dispatches_locked(lane, false, &deferred_owners);
    drain_graph_slots_locked(false);
    if (!deferred_owners.empty()) {
      lock.unlock();
      deferred_owners.clear();
      lock.lock();
      continue;
    }
    for (auto& slot : it->second.slots) {
      if (slot.captured && slot.state == GraphSlotState::Free) {
        slot.state = GraphSlotState::StagingReplay;
        slot.published_rows = 0;
        slot.retired_rows = 0;
        slot.generation = next_graph_slot_generation_++;
        for (auto& retirement : slot.row_retirements) retirement = GraphRowRetirement{};
        return &slot;
      }
    }
    if (closing_) return nullptr;
    if (fault_) std::rethrow_exception(fault_);
    cv_graph_slots_.wait_for(lock, ms_duration(kPendingDispatchIdlePollMs));
  }
}

bool BatchedSteadyScheduler::dispatch_batch_graph(DispatchLane& lane,
                                                  std::vector<std::shared_ptr<QueueItem>>& batch,
                                                  const std::vector<BatchedSteadyInput>& ready,
                                                  int bucket,
                                                  int64_t cycle_id,
                                                  const std::vector<double>& gather_waits,
                                                  const std::vector<double>& service_waits,
                                                  bool backlog,
                                                  double queue_depth_at_form,
                                                  Clock::time_point dispatch_wall_start,
                                                  double dispatch_cpu_start_us) {
  auto state_it = graph_buckets_.find(bucket);
  if (state_it == graph_buckets_.end() || state_it->second.disabled) return false;
  GraphBucketState& state = state_it->second;
  GraphSlot* slot = acquire_graph_slot(lane, bucket);
  if (slot == nullptr) return false;
  bool slot_published = false;
  auto slot_guard = make_scope_exit([&]() noexcept {
    if (slot_published || slot == nullptr) return;
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      mark_graph_slot_free_locked(*slot);
    } catch (...) {
    }
  });

  copy_ready_into_graph_staging(state, ready);
  B2_CUDA_CHECK(cudaGraphLaunch(slot->exec, lane.dispatcher_stream.stream()));
  B2_CUDA_CHECK(cudaEventRecord(slot->completion_event, lane.dispatcher_stream.stream()));
  B2_CUDA_CHECK(cudaEventRecord(slot->dispatch_done_event, lane.dispatcher_stream.stream()));

  const int k = static_cast<int>(batch.size());
  auto dispatch_wall_end = Clock::now();
  double dispatch_cpu_end_us = current_thread_cpu_us();
  add_dispatch_telemetry(lane.lane_id,
                         bucket,
                         cycle_id,
                         k,
                         backlog,
                         gather_waits,
                         service_waits,
                         0.0,
                         0.0,
                         queue_depth_at_form,
                         dispatch_wall_start,
                         dispatch_wall_end,
                         dispatch_cpu_start_us,
                         dispatch_cpu_end_us);

  std::vector<DispatchResult> results;
  results.reserve(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    DispatchResult result;
    result.completion.reset(slot->completion_event, false);
    result.row_tensors = graph_slot_row_tensors(*slot, static_cast<int>(i));
    result.bucket = bucket;
    result.row = static_cast<int>(i);
    result.k = k;
    result.cycle_id = cycle_id;
    result.gather_wait_us = gather_waits[i];
    result.service_wait_us = service_waits[i];
    result.cuda_run_us = 0.0;
    result.label = ready[i].label;
    const int slot_id = slot->slot_id;
    const int row = static_cast<int>(i);
    result.graph_slot = std::make_shared<DispatchResult::GraphSlotLease>(
        [this, bucket, slot_id, row](cudaStream_t consumer_stream) {
          retire_graph_slot_consumer(bucket, slot_id, row, consumer_stream);
        },
        [this, bucket, slot_id, row]() noexcept {
          abandon_graph_slot_consumer(bucket, slot_id, row);
        });
    results.push_back(std::move(result));
  }

  {
    std::vector<std::shared_ptr<QueueItem>> deferred_owners;
    std::lock_guard<std::mutex> lock(mutex_);
    slot->state = GraphSlotState::Published;
    slot->published_rows = k;
    slot->retired_rows = 0;
    for (int row = 0; row < slot->bucket; ++row) {
      slot->row_retirements[static_cast<size_t>(row)] = GraphRowRetirement{};
      slot->row_retirements[static_cast<size_t>(row)].active = row < k;
    }

    PendingDispatch pending;
    pending.dispatch_done = slot->dispatch_done_event;
    pending.owns_dispatch_done = false;
    pending.input_owners = batch;
    pending.capacity_tokens = k;
    lane.pending_dispatches.push_back(std::move(pending));
    cap_pending_dispatches_locked(lane, &deferred_owners);
    drain_pending_dispatches_locked(lane, false, &deferred_owners);
  }
  slot_published = true;
  slot_guard.dismiss();

  for (size_t i = 0; i < batch.size(); ++i) {
    batch[i]->promise.set_value(std::move(results[i]));
  }
  return true;
}

bool BatchedSteadyScheduler::drain_one_graph_slot_locked(bool force) {
  for (auto& bucket_entry : graph_buckets_) {
    auto& state = bucket_entry.second;
    for (auto& slot : state.slots) {
      if (slot.state != GraphSlotState::Retiring) continue;
      bool ready = true;
      for (int row = 0; row < slot.published_rows; ++row) {
        const auto& retirement = slot.row_retirements[static_cast<size_t>(row)];
        if (!retirement.active) continue;
        cudaEvent_t event = retirement.fence_recorded
                                ? slot.consumer_fence_events[static_cast<size_t>(row)]
                                : slot.dispatch_done_event;
        if (event == nullptr) continue;
        if (force) {
          B2_CUDA_CHECK(cudaEventSynchronize(event));
        } else {
          cudaError_t status = cudaEventQuery(event);
          if (status == cudaErrorNotReady) {
            ready = false;
            break;
          }
          B2_CUDA_CHECK(status);
        }
      }
      if (!ready) continue;
      mark_graph_slot_free_locked(slot);
      return true;
    }
  }
  return false;
}

void BatchedSteadyScheduler::drain_graph_slots_locked(bool force) {
  while (drain_one_graph_slot_locked(force)) {
  }
}

bool BatchedSteadyScheduler::graph_slots_active_locked() const {
  for (const auto& bucket_entry : graph_buckets_) {
    const auto& state = bucket_entry.second;
    for (const auto& slot : state.slots) {
      if (slot.state == GraphSlotState::StagingReplay ||
          slot.state == GraphSlotState::Published ||
          slot.state == GraphSlotState::Retiring) {
        return true;
      }
    }
  }
  return false;
}

void BatchedSteadyScheduler::wait_for_graph_slots_shutdown() {
  std::vector<std::shared_ptr<QueueItem>> deferred_owners;
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    drain_pending_dispatches_all_locked(true, &deferred_owners);
    drain_graph_slots_locked(true);
    if (!deferred_owners.empty()) {
      lock.unlock();
      deferred_owners.clear();
      lock.lock();
      continue;
    }
    if (!graph_slots_active_locked()) break;
    cv_graph_slots_.wait_for(lock, ms_duration(kPendingDispatchIdlePollMs));
  }
}

void BatchedSteadyScheduler::mark_graph_slot_free_locked(GraphSlot& slot) {
  slot.state = GraphSlotState::Free;
  slot.published_rows = 0;
  slot.retired_rows = 0;
  slot.generation = 0;
  for (auto& retirement : slot.row_retirements) retirement = GraphRowRetirement{};
  cv_graph_slots_.notify_all();
}

void BatchedSteadyScheduler::retire_graph_slot_consumer(int bucket,
                                                        int slot_id,
                                                        int row,
                                                        cudaStream_t consumer_stream) {
  cudaEvent_t fence = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto state_it = graph_buckets_.find(bucket);
    if (state_it == graph_buckets_.end()) throw std::runtime_error("steady graph retire unknown bucket");
    auto& slots = state_it->second.slots;
    if (slot_id < 0 || slot_id >= static_cast<int>(slots.size())) {
      throw std::runtime_error("steady graph retire unknown slot");
    }
    auto& slot = slots[static_cast<size_t>(slot_id)];
    if (row < 0 || row >= slot.published_rows) throw std::runtime_error("steady graph retire unknown row");
    fence = slot.consumer_fence_events[static_cast<size_t>(row)];
  }

  B2_CUDA_CHECK(cudaEventRecord(fence, consumer_stream));

  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = graph_buckets_.at(bucket).slots[static_cast<size_t>(slot_id)];
    auto& retirement = slot.row_retirements[static_cast<size_t>(row)];
    if (!retirement.active || retirement.fence_recorded || retirement.zero_consumer) return;
    retirement.fence_recorded = true;
    ++slot.retired_rows;
    if (slot.retired_rows >= slot.published_rows) {
      slot.state = GraphSlotState::Retiring;
    }
    drain_graph_slots_locked(false);
  }
  cv_graph_slots_.notify_all();
}

void BatchedSteadyScheduler::abandon_graph_slot_consumer(int bucket, int slot_id, int row) noexcept {
  try {
    std::lock_guard<std::mutex> lock(mutex_);
    auto state_it = graph_buckets_.find(bucket);
    if (state_it == graph_buckets_.end()) return;
    if (slot_id < 0 || slot_id >= static_cast<int>(state_it->second.slots.size())) return;
    auto& slot = state_it->second.slots[static_cast<size_t>(slot_id)];
    if (row < 0 || row >= slot.published_rows) return;
    auto& retirement = slot.row_retirements[static_cast<size_t>(row)];
    if (!retirement.active || retirement.fence_recorded || retirement.zero_consumer) return;
    retirement.zero_consumer = true;
    ++slot.retired_rows;
    if (slot.retired_rows >= slot.published_rows) {
      slot.state = GraphSlotState::Retiring;
    }
    drain_graph_slots_locked(false);
  } catch (...) {
  }
  cv_graph_slots_.notify_all();
}

std::vector<at::Tensor> BatchedSteadyScheduler::pack_into_scratch(DispatchLane& lane,
                                                                  const std::vector<BatchedSteadyInput>& ready,
                                                                  int bucket) {
  if (ready.empty()) throw std::runtime_error("batch steady scratch pack called with no rows");
  auto& scratch = ensure_scratch(lane, bucket, ready.front());
  for (int row = 0; row < bucket; ++row) {
    const auto& src = ready[static_cast<size_t>(row < static_cast<int>(ready.size()) ? row : 0)];
    if (src.chunk.sizes() != ready.front().chunk.sizes()) throw std::runtime_error("batch steady scratch chunk shape mismatch");
    if (src.cache_ch.sizes() != ready.front().cache_ch.sizes()) throw std::runtime_error("batch steady scratch cache_ch shape mismatch");
    if (src.cache_t.sizes() != ready.front().cache_t.sizes()) throw std::runtime_error("batch steady scratch cache_t shape mismatch");
    if (src.cache_ch_len.sizes() != ready.front().cache_ch_len.sizes()) {
      throw std::runtime_error("batch steady scratch cache_ch_len shape mismatch");
    }
    const auto& idx = scratch.row_indices[static_cast<size_t>(row)];
    scratch.chunks.index_copy_(0, idx, src.chunk);
    scratch.cache_ch.index_copy_(1, idx, src.cache_ch);
    scratch.cache_t.index_copy_(1, idx, src.cache_t);
    scratch.cache_ch_len.index_copy_(0, idx, src.cache_ch_len);
    auto len = torch::full({1}, src.chunk.size(2), long_options_for(src.chunk));
    scratch.length.index_copy_(0, idx, len);
  }
  return {
      scratch.chunks,
      scratch.length,
      scratch.cache_ch,
      scratch.cache_t,
      scratch.cache_ch_len,
  };
}

BatchedSteadyScheduler::Scratch& BatchedSteadyScheduler::ensure_scratch(DispatchLane& lane,
                                                                        int bucket,
                                                                        const BatchedSteadyInput& first) {
  auto& scratch = lane.scratch[bucket];
  std::vector<int64_t> chunk_shape(first.chunk.sizes().begin(), first.chunk.sizes().end());
  if (scratch.initialized && scratch.chunk_shape == chunk_shape) return scratch;

  auto chunk_options = first.chunk.options();
  auto long_options = long_options_for(first.chunk);
  scratch.chunk_shape = std::move(chunk_shape);
  scratch.chunks = torch::empty({bucket, first.chunk.size(1), first.chunk.size(2)}, chunk_options);
  scratch.length = torch::empty({bucket}, long_options);
  scratch.cache_ch = torch::empty({first.cache_ch.size(0), bucket, first.cache_ch.size(2), first.cache_ch.size(3)},
                                  first.cache_ch.options());
  scratch.cache_t = torch::empty({first.cache_t.size(0), bucket, first.cache_t.size(2), first.cache_t.size(3)},
                                 first.cache_t.options());
  scratch.cache_ch_len = torch::empty({bucket}, long_options);
  scratch.row_indices.clear();
  scratch.length_scalars.clear();
  scratch.row_indices.reserve(static_cast<size_t>(bucket));
  scratch.length_scalars.reserve(static_cast<size_t>(bucket));
  for (int row = 0; row < bucket; ++row) {
    scratch.row_indices.push_back(torch::full({1}, row, long_options));
    scratch.length_scalars.push_back(torch::full({1}, first.chunk.size(2), long_options));
  }
  scratch.initialized = true;
  return scratch;
}

void BatchedSteadyScheduler::set_pending_exception_locked(std::vector<std::shared_ptr<QueueItem>>* pending) {
  int released = 0;
  while (!queue_.empty()) {
    pending->push_back(queue_.front());
    queue_.pop_front();
    ++released;
  }
  release_capacity_tokens_locked(released);
}

void BatchedSteadyScheduler::set_item_exception(const std::shared_ptr<QueueItem>& item, std::exception_ptr ep) {
  if (item->request.producer_event != nullptr) {
    cudaEventDestroy(item->request.producer_event);
    item->request.producer_event = nullptr;
  }
  try {
    item->promise.set_exception(ep);
  } catch (const std::future_error&) {
  }
}

void BatchedSteadyScheduler::apply_test_lane_hooks(DispatchLane& lane, const char* stage) {
  if ((test_slow_lane_ == -2 || test_slow_lane_ == lane.lane_id) && test_slow_us_ > 0) {
    std::printf("B2_SCHEDULER_TEST_SLOW lane=%d stage=%s attempt=%lld sleep_us=%d\n",
                lane.lane_id,
                stage,
                static_cast<long long>(lane.test_dispatch_attempts),
                test_slow_us_);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::microseconds(test_slow_us_));
  }
  if (test_fault_lane_ == lane.lane_id &&
      test_fault_after_ > 0 &&
      lane.test_dispatch_attempts >= test_fault_after_) {
    std::ostringstream oss;
    oss << "batch steady injected lane fault"
        << " lane=" << lane.lane_id
        << " stage=" << stage
        << " attempt=" << lane.test_dispatch_attempts;
    throw std::runtime_error(oss.str());
  }
}

void BatchedSteadyScheduler::add_dispatch_telemetry(int lane_id,
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
                                                    double dispatch_cpu_end_us) {
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  ++telemetry_.dispatch_cycles;
  telemetry_.completed += k;
  auto& lane_telem = lane_telemetry_locked(lane_id);
  ++lane_telem.dispatch_cycles;
  lane_telem.completed += k;
  if (queue_depth_at_form >= 0.0) lane_telem.queue_depth.push_back(queue_depth_at_form);
  if (bucket == 1) ++telemetry_.bucket_b1;
  if (bucket == 2) ++telemetry_.bucket_b2;
  if (bucket == 4) ++telemetry_.bucket_b4;
  if (bucket == 8) ++telemetry_.bucket_b8;
  if (bucket == 16) ++telemetry_.bucket_b16;
  if (k == 2 && bucket == 4) ++telemetry_.k2_padded_to_b4;
  if (k == 3 && bucket == 4) ++telemetry_.k3_padded_to_b4;
  if (k == 4) ++telemetry_.k4;
  if (k == 5 && bucket == 8) ++telemetry_.k5_padded_to_b8;
  if (k == 6 && bucket == 8) ++telemetry_.k6_padded_to_b8;
  if (k == 7 && bucket == 8) ++telemetry_.k7_padded_to_b8;
  if (k == 8) ++telemetry_.k8;
  if (k == 9 && bucket == 16) ++telemetry_.k9_padded_to_b16;
  if (k == 10 && bucket == 16) ++telemetry_.k10_padded_to_b16;
  if (k == 11 && bucket == 16) ++telemetry_.k11_padded_to_b16;
  if (k == 12 && bucket == 16) ++telemetry_.k12_padded_to_b16;
  if (k == 13 && bucket == 16) ++telemetry_.k13_padded_to_b16;
  if (k == 14 && bucket == 16) ++telemetry_.k14_padded_to_b16;
  if (k == 15 && bucket == 16) ++telemetry_.k15_padded_to_b16;
  if (k == 16) ++telemetry_.k16;
  if (backlog) ++telemetry_.backlog_gt_bmax;
  if (!gather_wait_us.empty()) {
    telemetry_.age_at_dispatch_us.push_back(*std::max_element(gather_wait_us.begin(), gather_wait_us.end()));
  }
  telemetry_.gather_wait_us.insert(telemetry_.gather_wait_us.end(), gather_wait_us.begin(), gather_wait_us.end());
  telemetry_.service_wait_us.insert(telemetry_.service_wait_us.end(), service_wait_us.begin(), service_wait_us.end());
  if (wakeup_jitter_us > 0.0) telemetry_.window_wakeup_jitter_us.push_back(wakeup_jitter_us);
  if (!dispatcher_measurement_started_) {
    dispatcher_measurement_started_ = true;
    dispatcher_measurement_wall_start_ = dispatch_wall_start;
    dispatcher_measurement_cpu_start_us_ = dispatch_cpu_start_us;
  }
  telemetry_.dispatcher_wall_us = elapsed_us(dispatcher_measurement_wall_start_, dispatch_wall_end);
  if (dispatcher_measurement_cpu_start_us_ >= 0.0 && dispatch_cpu_end_us >= dispatcher_measurement_cpu_start_us_) {
    telemetry_.dispatcher_cpu_us = dispatch_cpu_end_us - dispatcher_measurement_cpu_start_us_;
  }
  if (lane_id >= 0 && lane_id < static_cast<int>(dispatch_lanes_.size())) {
    auto& lane = dispatch_lanes_[static_cast<size_t>(lane_id)];
    if (!lane.telemetry_measurement_started) {
      lane.telemetry_measurement_started = true;
      lane.telemetry_measurement_wall_start = dispatch_wall_start;
      lane.telemetry_measurement_cpu_start_us = dispatch_cpu_start_us;
    }
    lane_telem.dispatcher_wall_us =
        elapsed_us(lane.telemetry_measurement_wall_start, dispatch_wall_end);
    if (lane.telemetry_measurement_cpu_start_us >= 0.0 &&
        dispatch_cpu_end_us >= lane.telemetry_measurement_cpu_start_us) {
      lane_telem.dispatcher_cpu_us = dispatch_cpu_end_us - lane.telemetry_measurement_cpu_start_us;
    }
  }
  worker_wait_expected_by_cycle_[cycle_id] = std::max(worker_wait_expected_by_cycle_[cycle_id], k);
  if (cuda_run_us >= 0.0) add_dispatch_timing_telemetry_locked(lane_id, k, cuda_run_us, dispatch_wall_end);
}

void BatchedSteadyScheduler::add_dispatch_timing_telemetry_locked(int lane_id,
                                                                  int k,
                                                                  double cuda_run_us,
                                                                  Clock::time_point timing_wall_end) {
  for (int i = 0; i < k; ++i) telemetry_.cuda_run_us.push_back(cuda_run_us);
  telemetry_.dispatcher_stream_run_us += cuda_run_us;
  if (lane_id >= 0) {
    auto& lane_telem = lane_telemetry_locked(lane_id);
    for (int i = 0; i < k; ++i) lane_telem.cuda_run_us.push_back(cuda_run_us);
    lane_telem.dispatcher_stream_run_us += cuda_run_us;
    if (lane_id < static_cast<int>(dispatch_lanes_.size()) &&
        dispatch_lanes_[static_cast<size_t>(lane_id)].telemetry_measurement_started) {
      lane_telem.dispatcher_wall_us =
          std::max(lane_telem.dispatcher_wall_us,
                   elapsed_us(dispatch_lanes_[static_cast<size_t>(lane_id)].telemetry_measurement_wall_start,
                              timing_wall_end));
    }
  }
  if (dispatcher_measurement_started_) {
    telemetry_.dispatcher_wall_us =
        std::max(telemetry_.dispatcher_wall_us, elapsed_us(dispatcher_measurement_wall_start_, timing_wall_end));
  }
}

BatchedSteadySchedulerTelemetry::Lane& BatchedSteadyScheduler::lane_telemetry_locked(int lane_id) {
  if (lane_id < 0) {
    throw std::runtime_error("batch steady lane telemetry requested with negative lane id");
  }
  if (static_cast<size_t>(lane_id) >= telemetry_.lanes.size()) {
    telemetry_.lanes.resize(static_cast<size_t>(lane_id) + 1);
  }
  auto& lane = telemetry_.lanes[static_cast<size_t>(lane_id)];
  lane.lane_id = lane_id;
  return lane;
}

BatchedSteadyScheduler::DispatchLane& BatchedSteadyScheduler::graph_lane() {
  if (dispatch_lanes_.empty()) throw std::runtime_error("batch steady graph requested with no dispatch lane");
  return dispatch_lanes_.front();
}

const BatchedSteadyScheduler::DispatchLane& BatchedSteadyScheduler::graph_lane() const {
  if (dispatch_lanes_.empty()) throw std::runtime_error("batch steady graph requested with no dispatch lane");
  return dispatch_lanes_.front();
}

#undef B2_CUDA_CHECK
