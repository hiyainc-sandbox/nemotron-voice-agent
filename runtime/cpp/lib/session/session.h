#pragma once

#include "lib/session/first_encoder.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;
namespace fs = std::filesystem;

class BatchedSteadyScheduler;

#include "lib/session/model_constants.h"

static constexpr double ARGMAX_MARGIN_WARNING_THRESHOLD = 1.0e-2;
static constexpr double ARGMAX_MARGIN_UNSAFE_THRESHOLD = 1.0e-3;

enum class SessionMode { STREAMING, PENDING_FINALIZE, FINALIZED };
enum class FinalizeFinish { SPECULATIVE_KEEP, TRUE_BOUNDARY_COLD_RESET };

static constexpr int64_t EVENT_INTERIM = 0;
static constexpr int64_t EVENT_FINAL = 1;
static constexpr int64_t EVENT_SUPPRESSED = 2;

struct EmittedEvent {
  int64_t kind = -1;
  std::vector<int64_t> tokens;
  std::vector<int64_t> collector_tokens;
  std::string text;
  std::string collector_text;
};

struct TokenMargin {
  int64_t token_index = -1;
  int64_t token_id = -1;
  double margin = std::numeric_limits<double>::quiet_NaN();
  std::string label = "NA";
  int64_t frame = -1;
  int64_t symbol = -1;
};

struct Tokenizer {
  std::vector<std::string> pieces;

  std::string ids_to_text(const std::vector<int64_t>& ids) const;
};

// Language -> prompt-index table shipped in the session bundle (prompted
// profile only; empty with default_index=-1 for the en profile).
struct PromptTable {
  std::unordered_map<std::string, int64_t> lang_to_index;
  int64_t num_prompts = 0;
  int64_t default_index = -1;
};

struct SessionState {
  std::atomic<uint64_t> generation{0};
  torch::Tensor clc;
  torch::Tensor clt;
  torch::Tensor clcl;
  torch::Tensor g;
  torch::Tensor h;
  torch::Tensor c;
  torch::Tensor ring;
  int64_t emitted = 0;
  std::vector<int64_t> hyp;
  std::vector<int64_t> last_interim_tokens;
  std::vector<int64_t> continuous_emitted_tokens;
  std::vector<float> pending_audio;
  std::vector<float> raw_audio_ring;
  std::vector<float> post_stop_audio;
  std::string last_interim_text;
  std::string continuous_emitted_text;
  SessionMode mode = SessionMode::STREAMING;
  int64_t total_audio_samples = 0;
  int64_t synthetic_prefix_samples = 0;
  // Language-ID prompt conditioning (prompted profile only): device-resident
  // [1, NUM_PROMPTS] one-hot applied to every encoder output before decode.
  // Persists across finalize forks and cold resets.
  torch::Tensor prompt;
  int64_t prompt_index = -1;
  std::string language;       // resolved locale for this session ("auto", "es-ES", ...)
  std::string last_language;  // last complete <xx-XX> tag observed in the hypothesis

  SessionState() = default;

  SessionState(const SessionState& other) {
    *this = other;
  }

  SessionState& operator=(const SessionState& other) {
    if (this == &other) return *this;
    generation.store(other.generation.load(std::memory_order_acquire), std::memory_order_release);
    clc = other.clc;
    clt = other.clt;
    clcl = other.clcl;
    g = other.g;
    h = other.h;
    c = other.c;
    ring = other.ring;
    emitted = other.emitted;
    hyp = other.hyp;
    last_interim_tokens = other.last_interim_tokens;
    continuous_emitted_tokens = other.continuous_emitted_tokens;
    pending_audio = other.pending_audio;
    raw_audio_ring = other.raw_audio_ring;
    post_stop_audio = other.post_stop_audio;
    last_interim_text = other.last_interim_text;
    continuous_emitted_text = other.continuous_emitted_text;
    mode = other.mode;
    total_audio_samples = other.total_audio_samples;
    synthetic_prefix_samples = other.synthetic_prefix_samples;
    prompt = other.prompt;
    prompt_index = other.prompt_index;
    language = other.language;
    last_language = other.last_language;
    return *this;
  }

  SessionState(SessionState&& other) noexcept {
    *this = std::move(other);
  }

  SessionState& operator=(SessionState&& other) noexcept {
    if (this == &other) return *this;
    generation.store(other.generation.load(std::memory_order_acquire), std::memory_order_release);
    clc = std::move(other.clc);
    clt = std::move(other.clt);
    clcl = std::move(other.clcl);
    g = std::move(other.g);
    h = std::move(other.h);
    c = std::move(other.c);
    ring = std::move(other.ring);
    emitted = other.emitted;
    hyp = std::move(other.hyp);
    last_interim_tokens = std::move(other.last_interim_tokens);
    continuous_emitted_tokens = std::move(other.continuous_emitted_tokens);
    pending_audio = std::move(other.pending_audio);
    raw_audio_ring = std::move(other.raw_audio_ring);
    post_stop_audio = std::move(other.post_stop_audio);
    last_interim_text = std::move(other.last_interim_text);
    continuous_emitted_text = std::move(other.continuous_emitted_text);
    mode = other.mode;
    total_audio_samples = other.total_audio_samples;
    synthetic_prefix_samples = other.synthetic_prefix_samples;
    prompt = std::move(other.prompt);
    prompt_index = other.prompt_index;
    language = std::move(other.language);
    last_language = std::move(other.last_language);
    return *this;
  }
};

struct AsrSnapshot {
  torch::Tensor clc;
  torch::Tensor clt;
  torch::Tensor clcl;
  torch::Tensor g;
  torch::Tensor h;
  torch::Tensor c;
  torch::Tensor ring;
  int64_t emitted = 0;
  std::vector<int64_t> hyp;
  std::vector<float> pending_audio;
  std::vector<float> raw_audio_ring;
  int64_t total_audio_samples = 0;
  int64_t synthetic_prefix_samples = 0;
  SessionMode mode = SessionMode::STREAMING;
};

struct AudioGeometry {
  int64_t shift_frames = -1;
  int64_t pre_encode_cache_size = -1;
  int64_t drop_extra = -1;
  int64_t final_padding_frames = -1;
  int64_t right_context = -1;
  int64_t first_preprocess_mel_frame = -1;
  int64_t hop_samples = -1;
  int64_t raw_audio_ring_samples = -1;
  int64_t preprocess_align_pad_samples = -1;
  int64_t preprocess_new_audio_samples = -1;
  int64_t stream_preprocess_valid_samples = -1;
  int64_t constant_preprocess_frames = -1;
  int64_t constant_preprocess_samples = -1;
};

struct ManifestContract {
  std::string model_id;
  std::vector<int64_t> att_context;
  int64_t right_context = -1;
  int64_t shift = -1;
  int64_t pre_encode_cache = -1;
  int64_t drop_extra = -1;
  int64_t final_padding_frames = -1;
  int64_t blank = -1;
  int64_t max_symbols = -1;
  std::string weights_sha256;
};

struct ManifestBucket {
  int64_t drop = -1;
  int64_t T = -1;
  std::string pkg;
  std::string pkg_sha256;
};

struct BucketManifest {
  ManifestContract contract;
  std::vector<ManifestBucket> buckets;
};

enum class ManifestShaVerifyMode { StartupCheap, Full };

struct BucketConstants {
  std::unordered_map<std::string, at::Tensor> values;
  size_t direct_matches = 0;
  size_t alias_fallbacks = 0;
};

struct FinalizeOutcome {
  bool token_ok = false;
  bool fork_ok = false;
  bool stale_dropped = false;
  size_t emitted_tokens = 0;
  std::vector<int64_t> final_tokens;
  std::string final_text;
};

struct RuntimeAudioFrontend;
using RuntimeAudioFrontendDeleter = void (*)(RuntimeAudioFrontend*);
using RuntimeAudioFrontendPtr = std::unique_ptr<RuntimeAudioFrontend, RuntimeAudioFrontendDeleter>;

struct ExecutionContext {
  c10::cuda::CUDAStream stream;
  torch::jit::Module& joint;
  torch::jit::Module& predict;
  torch::jit::Module& preproc;
};

struct RuntimeSteadyTiming {
  double preproc_ms = 0.0;
  uint64_t preproc_count = 0;
  double scheduler_enqueue_wait_ms = 0.0;
  uint64_t scheduler_enqueue_wait_count = 0;
  double scheduler_future_wait_ms = 0.0;
  uint64_t scheduler_future_wait_count = 0;
  double scheduler_completion_wait_ms = 0.0;
  uint64_t scheduler_completion_wait_count = 0;
  double decode_ms = 0.0;
  uint64_t decode_count = 0;

  bool has_any() const {
    return preproc_count > 0 ||
           scheduler_enqueue_wait_count > 0 ||
           scheduler_future_wait_count > 0 ||
           scheduler_completion_wait_count > 0 ||
           decode_count > 0;
  }

  void merge(const RuntimeSteadyTiming& other) {
    preproc_ms += other.preproc_ms;
    preproc_count += other.preproc_count;
    scheduler_enqueue_wait_ms += other.scheduler_enqueue_wait_ms;
    scheduler_enqueue_wait_count += other.scheduler_enqueue_wait_count;
    scheduler_future_wait_ms += other.scheduler_future_wait_ms;
    scheduler_future_wait_count += other.scheduler_future_wait_count;
    scheduler_completion_wait_ms += other.scheduler_completion_wait_ms;
    scheduler_completion_wait_count += other.scheduler_completion_wait_count;
    decode_ms += other.decode_ms;
    decode_count += other.decode_count;
  }
};

class FinalizeBucketLoaderProvider {
 public:
  virtual ~FinalizeBucketLoaderProvider() = default;
  virtual AOTIModelPackageLoader& get(int64_t drop, int64_t T) = 0;
};

int session_main_entrypoint(int argc, char** argv);

bool file_exists(const std::string& path);
bool directory_exists(const std::string& path);
torch::Tensor attr_tensor(torch::jit::Module& module, const std::string& name);
torch::Tensor utt_tensor(torch::jit::Module& bundle, int utt, const char* name);
torch::Tensor prefix_tensor(torch::jit::Module& bundle, const std::string& prefix, const char* name);
torch::Tensor prefix_chunk_tensor(torch::jit::Module& bundle,
                                  const std::string& prefix,
                                  int chunk,
                                  const char* name);
int64_t scalar_i64(torch::Tensor tensor);
double scalar_f64(torch::Tensor tensor);
std::vector<int64_t> tensor_to_vec(torch::Tensor tensor);
const char* event_kind_name(int64_t kind);
std::string vec_to_string(const std::vector<int64_t>& values);
std::string escaped_text(const std::string& text);
std::string json_quote(const std::string& text);
std::vector<int64_t> append_only_delta_tokens(const std::vector<int64_t>& final_tokens,
                                              const std::vector<int64_t>& emitted_tokens);
std::string append_only_delta_text(const std::string& final_text,
                                   const std::string& emitted_text);
std::string append_delta_to_collector(const std::string& collector, const std::string& delta);
Tokenizer tokenizer_from_bundle(torch::jit::Module& bundle);
void verify_tokenizer_selftest(torch::jit::Module& bundle, const Tokenizer& tokenizer);
// Language-ID prompt conditioning (prompted profile; all no-ops/identity for en).
torch::Tensor make_prompt_onehot(int64_t index, torch::Device device);
PromptTable prompt_table_from_bundle(torch::jit::Module& bundle);
void initialize_prompt_runtime(const std::string& artifact_dir,
                               torch::jit::Module& bundle,
                               torch::Device device);
bool prompt_runtime_initialized();
const PromptTable& prompt_runtime_table();
torch::Tensor condition_encoder_output(const torch::Tensor& enc_out, const SessionState& state);
// Wire-layer language-tag handling (mirrors server.py LANG_TAG_RE and friends).
std::string strip_lang_tags_text(const std::string& text);
std::string last_lang_tag(const std::string& text);
std::vector<EmittedEvent> gold_events_from_bundle(torch::jit::Module& bundle, int utt);
void emit_event(std::vector<EmittedEvent>& events,
                int64_t kind,
                const std::vector<int64_t>& tokens,
                const std::vector<int64_t>& collector_tokens,
                const std::string& text,
                const std::string& collector_text);
bool equal_tokens(const std::vector<int64_t>& got,
                  const std::vector<int64_t>& gold,
                  const char* label,
                  const std::string& row_label);
bool tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected);
bool tensor_close(const char* name,
                  const torch::Tensor& actual,
                  const torch::Tensor& expected,
                  double atol,
                  const std::string& label);
SessionState clone_session(const SessionState& state);
AsrSnapshot snapshot_asr(const SessionState& state);
bool fork_assert_parent_unchanged(const SessionState& parent, const AsrSnapshot& snapshot);
void reset_session(SessionState& state, torch::jit::Module& bundle, torch::Device device);
void finish_speculative_finalize(SessionState& state);
void cold_reset_after_finalize(SessionState& state,
                               torch::jit::Module& bundle,
                               torch::Device device,
                               const AudioGeometry* audio_geometry);
std::string sha256_file(const std::string& path);
std::map<std::pair<int64_t, int64_t>, std::string> discover_finalize_buckets(const std::string& buckets_dir);
BucketManifest load_bucket_manifest(const std::string& manifest_path);
ManifestShaVerifyMode manifest_sha_verify_mode_from_env();
const char* manifest_sha_verify_mode_name(ManifestShaVerifyMode mode);
void verify_bucket_manifest(const BucketManifest& manifest,
                            const std::map<std::pair<int64_t, int64_t>, std::string>& discovered,
                            const std::string& buckets_dir,
                            const std::string& shared_weights_pt,
                            ManifestShaVerifyMode sha_mode = manifest_sha_verify_mode_from_env());
std::unordered_map<std::string, at::Tensor> load_shared_constants(const std::string& weights_path,
                                                                  torch::Device device);
BucketConstants constants_for_bucket(const std::unordered_map<std::string, at::Tensor>& shared_constants,
                                     AOTIModelPackageLoader& loader,
                                     const std::string& pkg);
std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                          const torch::Tensor& chunk,
                                          SessionState& state);
std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                          const torch::Tensor& chunk,
                                          SessionState& state,
                                          c10::cuda::CUDAStream stream);
std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                          const torch::Tensor& chunk,
                                          SessionState& state,
                                          const ExecutionContext& ctx);
void vad_stop(SessionState& state);
void verify_session_bundle_meta(torch::jit::Module& bundle, bool multiturn);
AudioGeometry session_runtime_audio_geometry_from_bundle(torch::jit::Module& bundle);
std::string session_runtime_verify_preproc_manifest(const std::string& dir,
                                                    const std::string& preproc_path,
                                                    const AudioGeometry& audio_geometry);
RuntimeAudioFrontendPtr make_session_runtime_audio_frontend(torch::jit::Module& bundle,
                                                            torch::jit::Module& preproc,
                                                            torch::Device device);
void reset_session_runtime_audio_front(SessionState& state, RuntimeAudioFrontend& audio);
int session_runtime_append_pcm_and_drain(SessionState& state,
                                         const std::vector<float>& pcm,
                                         RuntimeAudioFrontend& audio,
                                         FirstEncoder& enc_first,
                                         AOTIModelPackageLoader* enc_steady,
                                         torch::jit::Module& joint,
                                         torch::jit::Module& predict,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label);
int session_runtime_append_pcm_and_drain(SessionState& state,
                                         const std::vector<float>& pcm,
                                         RuntimeAudioFrontend& audio,
                                         FirstEncoder& enc_first,
                                         AOTIModelPackageLoader* enc_steady,
                                         const ExecutionContext& ctx,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label);
int session_runtime_append_pcm_and_drain(SessionState& state,
                                         const std::vector<float>& pcm,
                                         RuntimeAudioFrontend& audio,
                                         FirstEncoder& enc_first,
                                         AOTIModelPackageLoader* enc_steady,
                                         const ExecutionContext& ctx,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label,
                                         BatchedSteadyScheduler* steady_scheduler,
                                         bool steady_shadow_enabled,
                                         RuntimeSteadyTiming* steady_timing = nullptr);
int session_runtime_vad_start(SessionState& state,
                              RuntimeAudioFrontend& audio,
                              FirstEncoder& enc_first,
                              AOTIModelPackageLoader* enc_steady,
                              torch::jit::Module& joint,
                              torch::jit::Module& predict,
                              torch::Device device,
                              const Tokenizer& tokenizer,
                              std::vector<EmittedEvent>& events,
                              const std::string& label);
int session_runtime_vad_start(SessionState& state,
                              RuntimeAudioFrontend& audio,
                              FirstEncoder& enc_first,
                              AOTIModelPackageLoader* enc_steady,
                              const ExecutionContext& ctx,
                              torch::Device device,
                              const Tokenizer& tokenizer,
                              std::vector<EmittedEvent>& events,
                              const std::string& label);
int session_runtime_vad_start(SessionState& state,
                              RuntimeAudioFrontend& audio,
                              FirstEncoder& enc_first,
                              AOTIModelPackageLoader* enc_steady,
                              const ExecutionContext& ctx,
                              torch::Device device,
                              const Tokenizer& tokenizer,
                              std::vector<EmittedEvent>& events,
                              const std::string& label,
                              BatchedSteadyScheduler* steady_scheduler,
                              bool steady_shadow_enabled,
                              RuntimeSteadyTiming* steady_timing = nullptr);
void session_runtime_print_steady_shadow_report();
FinalizeOutcome session_runtime_finalize(SessionState& state,
                                         torch::jit::Module& bundle,
                                         RuntimeAudioFrontend& audio,
                                         std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
                                         torch::jit::Module& joint,
                                         torch::jit::Module& predict,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         FinalizeFinish finish,
                                         const std::string& label);
FinalizeOutcome session_runtime_finalize(SessionState& state,
                                         torch::jit::Module& bundle,
                                         RuntimeAudioFrontend& audio,
                                         std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
                                         const ExecutionContext& ctx,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         FinalizeFinish finish,
                                         const std::string& label);
FinalizeOutcome session_runtime_finalize(SessionState& state,
                                         torch::jit::Module& bundle,
                                         RuntimeAudioFrontend& audio,
                                         FinalizeBucketLoaderProvider& finalize_loaders,
                                         const ExecutionContext& ctx,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         FinalizeFinish finish,
                                         const std::string& label);
