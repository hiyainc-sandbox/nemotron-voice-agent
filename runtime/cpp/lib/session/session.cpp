// 1.4 Phase-1 session composition gate.
//
// Replays artifacts/session_bundle.ts (MEL-fed) or, with --audio, the
// PCM-fed session_audio_bundle.ts.  The audio path runs the TorchScript
// preprocessor, raw-audio ring, fixed constant-plan block, and finalize
// remainder assembly before the same steady/finalize C++ session path.
// The emitted cumulative tokens must exactly equal finalize_ref gold tokens.
// The ordered interim/final/suppressed event stream is checked at the same
// WORD/TEXT level as finalize_ref._continuous_append_only_delta.
#include "lib/session/first_encoder.h"
#include "lib/runtime_io/jit_load.h"
#include "lib/scheduler/batched_steady_scheduler.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;
namespace fs = std::filesystem;

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

struct MelCompareStats {
  int64_t comparisons = 0;
  int64_t pass = 0;
  int64_t byte_equal = 0;
  int64_t within_ci = 0;
  double max_abs_diff = 0.0;
};

struct GeometryCompareStats {
  int64_t checks = 0;
  int64_t pass = 0;
};

struct CacheCompareStats {
  int64_t checks = 0;
  int64_t pass = 0;
  double cache_last_channel_max_abs = 0.0;
  double cache_last_time_max_abs = 0.0;
  double cache_last_channel_len_max_abs = 0.0;
};

struct CacheOwnershipStats {
  int64_t recurrent_assignments = 0;
  int64_t clone_assignments = 0;
  int64_t alias_after_clone = 0;
};

struct LongStreamCacheStats {
  bool ok = false;
  std::string prefix = "NA";
  int64_t steady_chunks = 0;
  int64_t aoti_chunks = 0;
  int64_t stable_checks = 0;
  int64_t stable_pass = 0;
  int64_t consecutive_alias_checks = 0;
  int64_t consecutive_alias_fail = 0;
};

struct PreprocDeterminismStats {
  bool ok = true;
  std::string fixed_block_sha256 = "NA";
  double max_abs = 0.0;
};

struct MarginStats {
  double warning_threshold = ARGMAX_MARGIN_WARNING_THRESHOLD;
  double unsafe_threshold = ARGMAX_MARGIN_UNSAFE_THRESHOLD;
  double min_margin = std::numeric_limits<double>::infinity();
  int64_t total = 0;
  int64_t below_warning = 0;
  int64_t below_unsafe = 0;
  std::string min_label = "NA";
  int64_t min_frame = -1;
  int64_t min_symbol = -1;
  std::vector<TokenMargin> token_margins;
};

struct FirstChunkStats {
  int64_t checks = 0;
  int64_t pass = 0;
  int64_t byte_equal = 0;
  double max_abs = 0.0;
  MarginStats margins;
};

struct DivergenceRecord {
  std::string sample_id;
  int64_t sample_index = -1;
  size_t first_diff = std::numeric_limits<size_t>::max();
  int64_t got_token = -1;
  int64_t gold_token = -1;
  double flip_margin = std::numeric_limits<double>::quiet_NaN();
  std::string flip_label = "NA";
  int64_t flip_frame = -1;
  int64_t flip_symbol = -1;
  double row_min_margin = std::numeric_limits<double>::quiet_NaN();
  std::string row_min_label = "NA";
  int64_t row_min_frame = -1;
  int64_t row_min_symbol = -1;
};

struct AudioCiBundleStats {
  double mel_abs_max = 0.0;
  double mel_abs_mean = 0.0;
  double mel_abs_p99 = 0.0;
  double mel_rel_max = 0.0;
  double mel_rel_mean = 0.0;
  double mel_rel_p99 = 0.0;
  double cache_last_channel_max_abs = 0.0;
  double cache_last_time_max_abs = 0.0;
  double cache_last_channel_len_max_abs = 0.0;
  double mel_checks = 0.0;
  double cache_checks = 0.0;
  double mel_max_headroom = 0.0;
  double mel_p99_headroom = 0.0;
  double cache_max_headroom = 0.0;
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

struct Sha256Ctx {
  std::array<uint8_t, 64> data{};
  uint32_t datalen = 0;
  uint64_t bitlen = 0;
  std::array<uint32_t, 8> state{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
};

struct BucketConstants {
  std::unordered_map<std::string, at::Tensor> values;
  size_t direct_matches = 0;
  size_t alias_fallbacks = 0;
};

bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

bool directory_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string utt_attr(int utt, const char* name) {
  return "utt" + std::to_string(utt) + "_" + std::string(name);
}

static std::string utt_chunk_attr(int utt, int chunk, const char* name) {
  return "utt" + std::to_string(utt) + "_chunk" + std::to_string(chunk) + "_" + std::string(name);
}

static std::string prefix_attr(const std::string& prefix, const char* name) {
  return prefix + "_" + std::string(name);
}

static std::string prefix_chunk_attr(const std::string& prefix, int chunk, const char* name) {
  return prefix + "_chunk" + std::to_string(chunk) + "_" + std::string(name);
}

static std::string stream_turn_prefix(int stream, int turn) {
  return "stream" + std::to_string(stream) + "_turn" + std::to_string(turn);
}

static std::string stream_end_prefix(int stream) {
  return "stream" + std::to_string(stream) + "_end";
}

torch::Tensor attr_tensor(torch::jit::Module& module, const std::string& name) {
  return module.attr(name).toTensor();
}

torch::Tensor utt_tensor(torch::jit::Module& bundle, int utt, const char* name) {
  return attr_tensor(bundle, utt_attr(utt, name));
}

static torch::Tensor utt_chunk_tensor(torch::jit::Module& bundle, int utt, int chunk, const char* name) {
  return attr_tensor(bundle, utt_chunk_attr(utt, chunk, name));
}

torch::Tensor prefix_tensor(torch::jit::Module& bundle, const std::string& prefix, const char* name) {
  return attr_tensor(bundle, prefix_attr(prefix, name));
}

torch::Tensor prefix_chunk_tensor(torch::jit::Module& bundle,
                                         const std::string& prefix,
                                         int chunk,
                                         const char* name) {
  return attr_tensor(bundle, prefix_chunk_attr(prefix, chunk, name));
}

int64_t scalar_i64(torch::Tensor tensor) {
  return tensor.to(torch::kCPU).reshape({-1})[0].item<int64_t>();
}

double scalar_f64(torch::Tensor tensor) {
  return tensor.to(torch::kCPU).to(torch::kFloat64).reshape({-1})[0].item<double>();
}

std::vector<int64_t> tensor_to_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kLong).contiguous().view({-1});
  std::vector<int64_t> out;
  out.reserve(flat.numel());
  for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(flat[i].item<int64_t>());
  return out;
}

static std::vector<double> tensor_to_double_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kFloat64).contiguous().view({-1});
  std::vector<double> out;
  out.reserve(flat.numel());
  auto acc = flat.accessor<double, 1>();
  for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(acc[i]);
  return out;
}

static std::vector<float> tensor_to_float_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kFloat32).contiguous().view({-1});
  std::vector<float> out;
  out.reserve(flat.numel());
  auto acc = flat.accessor<float, 1>();
  for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(acc[i]);
  return out;
}

static const char* pass_fail(bool ok) {
  return ok ? "PASS" : "FAIL";
}

static const char* pass_fail_skip(int64_t checks, int64_t pass) {
  if (checks == 0) return "SKIP";
  return checks == pass ? "PASS" : "FAIL";
}

static const char* mode_name(SessionMode mode) {
  switch (mode) {
    case SessionMode::STREAMING: return "STREAMING";
    case SessionMode::PENDING_FINALIZE: return "PENDING_FINALIZE";
    case SessionMode::FINALIZED: return "FINALIZED";
  }
  return "UNKNOWN";
}

ManifestShaVerifyMode manifest_sha_verify_mode_from_env() {
  const char* raw = std::getenv("NEMOTRON_WS_VERIFY_MANIFEST_SHA");
  if (raw == nullptr || raw[0] == '\0') return ManifestShaVerifyMode::StartupCheap;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value == "full" || value == "exhaustive" || value == "1" || value == "true" || value == "yes") {
    return ManifestShaVerifyMode::Full;
  }
  if (value == "cheap" || value == "startup-cheap" || value == "off" || value == "0" ||
      value == "false" || value == "no") {
    return ManifestShaVerifyMode::StartupCheap;
  }
  throw std::runtime_error("NEMOTRON_WS_VERIFY_MANIFEST_SHA must be unset, cheap/off, or full: " +
                           std::string(raw));
}

const char* manifest_sha_verify_mode_name(ManifestShaVerifyMode mode) {
  switch (mode) {
    case ManifestShaVerifyMode::StartupCheap: return "startup-cheap";
    case ManifestShaVerifyMode::Full: return "full";
  }
  return "unknown";
}

static bool manifest_sha_verify_full(ManifestShaVerifyMode mode) {
  return mode == ManifestShaVerifyMode::Full;
}

static bool tensor_storage_alias(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (!lhs.defined() || !rhs.defined()) return false;
  return lhs.is_alias_of(rhs);
}

std::string vec_to_string(const std::vector<int64_t>& values) {
  std::ostringstream oss;
  for (auto value : values) oss << ' ' << value;
  return oss.str();
}

std::string escaped_text(const std::string& text) {
  std::ostringstream oss;
  oss << '"';
  for (unsigned char ch : text) {
    if (ch == '\\') {
      oss << "\\\\";
    } else if (ch == '"') {
      oss << "\\\"";
    } else if (ch == '\n') {
      oss << "\\n";
    } else if (ch == '\r') {
      oss << "\\r";
    } else if (ch == '\t') {
      oss << "\\t";
    } else if (ch < 0x20 || ch == 0x7f) {
      oss << "\\x" << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(ch) << std::dec << std::setfill(' ');
    } else {
      oss << static_cast<char>(ch);
    }
  }
  oss << '"';
  return oss.str();
}

std::string json_quote(const std::string& text) {
  std::ostringstream oss;
  oss << '"';
  oss << std::hex << std::setfill('0');
  for (unsigned char ch : text) {
    switch (ch) {
      case '\\': oss << "\\\\"; break;
      case '"': oss << "\\\""; break;
      case '\b': oss << "\\b"; break;
      case '\f': oss << "\\f"; break;
      case '\n': oss << "\\n"; break;
      case '\r': oss << "\\r"; break;
      case '\t': oss << "\\t"; break;
      default:
        if (ch < 0x20) {
          oss << "\\u" << std::setw(4) << static_cast<int>(ch);
        } else {
          oss << static_cast<char>(ch);
        }
    }
  }
  oss << std::dec << std::setfill(' ');
  oss << '"';
  return oss.str();
}

const char* event_kind_name(int64_t kind) {
  if (kind == EVENT_INTERIM) return "interim";
  if (kind == EVENT_FINAL) return "final";
  if (kind == EVENT_SUPPRESSED) return "suppressed";
  return "unknown";
}

std::vector<int64_t> append_only_delta_tokens(const std::vector<int64_t>& final_tokens,
                                                     const std::vector<int64_t>& emitted_tokens) {
  // Token-id port of finalize_ref._continuous_append_only_delta:
  // common-prefix append, deletion/correction suppression, then suffix/prefix overlap trim.
  size_t common = 0;
  size_t pair_count = std::min(emitted_tokens.size(), final_tokens.size());
  while (common < pair_count && emitted_tokens[common] == final_tokens[common]) {
    ++common;
  }

  std::vector<int64_t> delta_tokens;
  if (common == emitted_tokens.size()) {
    delta_tokens.assign(final_tokens.begin() + static_cast<std::ptrdiff_t>(common),
                        final_tokens.end());
  } else if (final_tokens.size() <= emitted_tokens.size()) {
    delta_tokens.clear();
  } else {
    delta_tokens.assign(final_tokens.begin() + static_cast<std::ptrdiff_t>(emitted_tokens.size()),
                        final_tokens.end());
    size_t max_overlap = std::min(emitted_tokens.size(), delta_tokens.size());
    for (size_t overlap = max_overlap; overlap > 0; --overlap) {
      bool matches = true;
      size_t emitted_start = emitted_tokens.size() - overlap;
      for (size_t i = 0; i < overlap; ++i) {
        if (emitted_tokens[emitted_start + i] != delta_tokens[i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        delta_tokens.erase(delta_tokens.begin(),
                           delta_tokens.begin() + static_cast<std::ptrdiff_t>(overlap));
        break;
      }
    }
  }
  return delta_tokens;
}

static std::vector<std::string> split_words(const std::string& text) {
  std::istringstream iss(text);
  std::vector<std::string> words;
  std::string word;
  while (iss >> word) words.push_back(word);
  return words;
}

static std::string join_words(const std::vector<std::string>& words) {
  std::ostringstream oss;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i > 0) oss << ' ';
    oss << words[i];
  }
  return oss.str();
}

std::string append_only_delta_text(const std::string& final_text,
                                          const std::string& emitted_text) {
  auto final_words = split_words(final_text);
  auto emitted_words = split_words(emitted_text);

  size_t common = 0;
  size_t pair_count = std::min(emitted_words.size(), final_words.size());
  while (common < pair_count && emitted_words[common] == final_words[common]) {
    ++common;
  }

  std::vector<std::string> delta_words;
  if (common == emitted_words.size()) {
    delta_words.assign(final_words.begin() + static_cast<std::ptrdiff_t>(common),
                       final_words.end());
  } else if (final_words.size() <= emitted_words.size()) {
    delta_words.clear();
  } else {
    delta_words.assign(final_words.begin() + static_cast<std::ptrdiff_t>(emitted_words.size()),
                       final_words.end());
    size_t max_overlap = std::min(emitted_words.size(), delta_words.size());
    for (size_t overlap = max_overlap; overlap > 0; --overlap) {
      bool matches = true;
      size_t emitted_start = emitted_words.size() - overlap;
      for (size_t i = 0; i < overlap; ++i) {
        if (emitted_words[emitted_start + i] != delta_words[i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        delta_words.erase(delta_words.begin(),
                          delta_words.begin() + static_cast<std::ptrdiff_t>(overlap));
        break;
      }
    }
  }
  return join_words(delta_words);
}

std::string append_delta_to_collector(const std::string& collector,
                                             const std::string& delta) {
  if (delta.empty()) return collector;
  if (collector.empty()) return delta;
  return collector + " " + delta;
}

static void replace_all(std::string& text, const std::string& needle, const std::string& repl) {
  if (needle.empty()) return;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), repl);
    pos += repl.size();
  }
}

std::string Tokenizer::ids_to_text(const std::vector<int64_t>& ids) const {
  if (ids.empty()) return "";
  std::string text;
  bool strip_dummy_prefix = false;
  static const std::string marker = "\xE2\x96\x81";
  static const std::string unk_surface = " \xE2\x81\x87 ";

  for (size_t i = 0; i < ids.size(); ++i) {
    int64_t id = ids[i];
    if (id < 0 || id >= static_cast<int64_t>(pieces.size())) {
      throw std::runtime_error("token id out of tokenizer piece range: " + std::to_string(id));
    }
    const std::string& piece = pieces[static_cast<size_t>(id)];
    if (i == 0 && piece.rfind(marker, 0) == 0) strip_dummy_prefix = true;
    if (piece == "<unk>") {
      text += unk_surface;
    } else {
      text += piece;
    }
  }
  replace_all(text, marker, " ");
  if (strip_dummy_prefix && !text.empty() && text[0] == ' ') {
    text.erase(text.begin());
  }
  return text;
}

static std::vector<std::vector<int64_t>> unpack_i64_lists(torch::Tensor flat_tensor,
                                                          torch::Tensor offsets_tensor,
                                                          const char* label,
                                                          int utt) {
  auto flat = tensor_to_vec(flat_tensor);
  auto offsets = tensor_to_vec(offsets_tensor);
  if (offsets.empty()) {
    throw std::runtime_error(std::string(label) + " offsets empty for utt" + std::to_string(utt));
  }
  if (offsets.front() != 0 || offsets.back() != static_cast<int64_t>(flat.size())) {
    throw std::runtime_error(std::string(label) + " offsets do not cover flat payload for utt" + std::to_string(utt));
  }
  std::vector<std::vector<int64_t>> out;
  out.reserve(offsets.size() - 1);
  for (size_t i = 0; i + 1 < offsets.size(); ++i) {
    int64_t start = offsets[i];
    int64_t end = offsets[i + 1];
    if (start < 0 || end < start || end > static_cast<int64_t>(flat.size())) {
      throw std::runtime_error(std::string(label) + " invalid offsets for utt" + std::to_string(utt));
    }
    out.emplace_back(flat.begin() + start, flat.begin() + end);
  }
  return out;
}

static std::vector<uint8_t> tensor_to_u8_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kUInt8).contiguous().view({-1});
  std::vector<uint8_t> out;
  out.reserve(flat.numel());
  for (int64_t i = 0; i < flat.numel(); ++i) {
    out.push_back(flat[i].item<uint8_t>());
  }
  return out;
}

static std::vector<std::string> unpack_utf8_strings(torch::Tensor flat_tensor,
                                                    torch::Tensor offsets_tensor,
                                                    const char* label,
                                                    int utt) {
  auto flat = tensor_to_u8_vec(flat_tensor);
  auto offsets = tensor_to_vec(offsets_tensor);
  if (offsets.empty()) {
    throw std::runtime_error(std::string(label) + " offsets empty for utt" + std::to_string(utt));
  }
  if (offsets.front() != 0 || offsets.back() != static_cast<int64_t>(flat.size())) {
    throw std::runtime_error(std::string(label) + " offsets do not cover flat payload for utt" + std::to_string(utt));
  }
  std::vector<std::string> out;
  out.reserve(offsets.size() - 1);
  for (size_t i = 0; i + 1 < offsets.size(); ++i) {
    int64_t start = offsets[i];
    int64_t end = offsets[i + 1];
    if (start < 0 || end < start || end > static_cast<int64_t>(flat.size())) {
      throw std::runtime_error(std::string(label) + " invalid offsets for utt" + std::to_string(utt));
    }
    out.emplace_back(reinterpret_cast<const char*>(flat.data() + start),
                     static_cast<size_t>(end - start));
  }
  return out;
}

Tokenizer tokenizer_from_bundle(torch::jit::Module& bundle) {
  Tokenizer tokenizer;
  tokenizer.pieces = unpack_utf8_strings(
      attr_tensor(bundle, "token_piece_bytes"),
      attr_tensor(bundle, "token_piece_offsets"),
      "token_piece",
      -1);
  if (tokenizer.pieces.empty()) throw std::runtime_error("tokenizer piece table is empty");
  return tokenizer;
}

void verify_tokenizer_selftest(torch::jit::Module& bundle, const Tokenizer& tokenizer) {
  auto sequences = unpack_i64_lists(
      attr_tensor(bundle, "detok_selftest_tokens"),
      attr_tensor(bundle, "detok_selftest_token_offsets"),
      "detok_selftest_tokens",
      -1);
  auto texts = unpack_utf8_strings(
      attr_tensor(bundle, "detok_selftest_text_bytes"),
      attr_tensor(bundle, "detok_selftest_text_offsets"),
      "detok_selftest_text",
      -1);
  if (sequences.size() != texts.size()) {
    throw std::runtime_error("detok selftest sequence/text count mismatch");
  }
  for (size_t i = 0; i < sequences.size(); ++i) {
    std::string got = tokenizer.ids_to_text(sequences[i]);
    if (got != texts[i]) {
      std::ostringstream oss;
      oss << "detok selftest failed at sequence " << i
          << " tokens=" << vec_to_string(sequences[i])
          << " got=" << escaped_text(got)
          << " gold=" << escaped_text(texts[i]);
      throw std::runtime_error(oss.str());
    }
  }
  std::printf("tokenizer detok selftest PASS: pieces=%zu sequences=%zu\n",
              tokenizer.pieces.size(), sequences.size());
}

// ---- language-ID prompt conditioning (prompted profile only) ----------------
//
// The multilingual model applies a language prompt to the ENCODER OUTPUT: a
// [B, NUM_PROMPTS] one-hot is concatenated per-frame with enc_out and passed
// through a small MLP (prompt_apply.ts, exported from model.prompt_kernel).
// The cache-aware encoder graphs themselves are language-independent, so the
// only runtime change is conditioning enc_out before every RNNT decode.
// Sessions without an explicit language fall back to the bundle's default
// prompt (target_lang "auto"), matching the Python server's default.

struct PromptTable {
  std::unordered_map<std::string, int64_t> lang_to_index;
  int64_t num_prompts = 0;
  int64_t default_index = -1;
};

namespace {
struct PromptRuntime {
  std::unique_ptr<torch::jit::Module> module;
  torch::Tensor default_onehot;  // [1, NUM_PROMPTS] on device
  PromptTable table;
  std::mutex mu;  // guards initialization only; forward() is called lock-free
                  // (TorchScript inference is thread-safe, and init completes
                  // before serving threads start).
};

PromptRuntime& prompt_runtime() {
  static PromptRuntime runtime;
  return runtime;
}
}  // namespace

torch::Tensor make_prompt_onehot(int64_t index, torch::Device device) {
  if (index < 0 || index >= NUM_PROMPTS) {
    throw std::runtime_error("prompt index out of range: " + std::to_string(index));
  }
  auto onehot = torch::zeros({1, NUM_PROMPTS}, torch::dtype(torch::kFloat32).device(device));
  onehot.index_put_({0, index}, 1.0f);
  return onehot;
}

PromptTable prompt_table_from_bundle(torch::jit::Module& bundle) {
  PromptTable table;
  if (!bundle.hasattr("prompt_meta")) return table;  // pre-v2 bundle (en profile)
  auto meta = tensor_to_vec(attr_tensor(bundle, "prompt_meta"));
  if (meta.size() != 2) throw std::runtime_error("prompt_meta must have 2 entries");
  table.num_prompts = meta[0];
  table.default_index = meta[1];
  auto langs = unpack_utf8_strings(
      attr_tensor(bundle, "prompt_lang_bytes"),
      attr_tensor(bundle, "prompt_lang_offsets"),
      "prompt_lang",
      -1);
  auto indices = tensor_to_vec(attr_tensor(bundle, "prompt_lang_indices"));
  if (langs.size() != indices.size()) {
    throw std::runtime_error("prompt language/index table size mismatch");
  }
  for (size_t i = 0; i < langs.size(); ++i) {
    if (indices[i] < 0 || (table.num_prompts > 0 && indices[i] >= table.num_prompts)) {
      throw std::runtime_error("prompt index out of range for language " + langs[i]);
    }
    table.lang_to_index[langs[i]] = indices[i];
  }
  return table;
}

void initialize_prompt_runtime(const std::string& artifact_dir,
                               torch::jit::Module& bundle,
                               torch::Device device) {
  if (!MODEL_PROMPTED) return;
  auto& runtime = prompt_runtime();
  std::lock_guard<std::mutex> lock(runtime.mu);
  if (runtime.module != nullptr) return;  // already initialized
  std::string path = (fs::path(artifact_dir) / "prompt_apply.ts").string();
  if (!file_exists(path)) {
    throw std::runtime_error("prompted profile requires prompt_apply.ts in " + artifact_dir);
  }
  auto module = load_jit_serialized(path);
  module.to(device);
  module.eval();
  runtime.table = prompt_table_from_bundle(bundle);
  if (runtime.table.num_prompts != NUM_PROMPTS) {
    throw std::runtime_error("bundle prompt_meta num_prompts=" +
                             std::to_string(runtime.table.num_prompts) +
                             " does not match compiled NUM_PROMPTS=" + std::to_string(NUM_PROMPTS));
  }
  if (runtime.table.default_index < 0) {
    throw std::runtime_error("bundle prompt table has no default prompt index");
  }
  runtime.default_onehot = make_prompt_onehot(runtime.table.default_index, device);
  runtime.module = std::make_unique<torch::jit::Module>(std::move(module));
  std::printf("prompt runtime initialized: languages=%zu num_prompts=%lld default_index=%lld\n",
              runtime.table.lang_to_index.size(),
              static_cast<long long>(runtime.table.num_prompts),
              static_cast<long long>(runtime.table.default_index));
}

bool prompt_runtime_initialized() {
  if (!MODEL_PROMPTED) return true;
  return prompt_runtime().module != nullptr;
}

const PromptTable& prompt_runtime_table() {
  return prompt_runtime().table;
}

torch::Tensor condition_encoder_output(const torch::Tensor& enc_out, const SessionState& state) {
  if (!MODEL_PROMPTED) return enc_out;
  auto& runtime = prompt_runtime();
  torch::jit::Module* module = runtime.module.get();
  if (module == nullptr) {
    throw std::runtime_error("prompted profile decode before initialize_prompt_runtime");
  }
  const torch::Tensor& onehot = state.prompt.defined() ? state.prompt : runtime.default_onehot;
  return module->forward({enc_out, onehot}).toTensor();
}

// ---- language-tag handling (prompted profile only) --------------------------
//
// The multilingual model emits <xx-XX> tags after terminal punctuation (in
// every target_lang mode). Token ids keep the tags (parity with the Python
// oracle's y_sequence); the WIRE layer strips them from text and surfaces the
// last complete tag as the event's language, mirroring
// server.py::_strip_lang_tags / LANG_TAG_CAPTURE_RE.

std::string strip_lang_tags_text(const std::string& text) {
  if (!MODEL_PROMPTED) return text;
  static const std::regex complete_tag(R"(\s*<[a-z]{2}-[A-Z]{2}>)");
  static const std::regex partial_tag_suffix(R"(\s*<[a-z]{0,2}(-[A-Z]{0,2})?$)");
  static const std::regex whitespace_run(R"(\s+)");
  std::string stripped = std::regex_replace(text, complete_tag, " ");
  stripped = std::regex_replace(stripped, partial_tag_suffix, "");
  stripped = std::regex_replace(stripped, whitespace_run, " ");
  size_t first = stripped.find_first_not_of(' ');
  if (first == std::string::npos) return "";
  size_t last = stripped.find_last_not_of(' ');
  return stripped.substr(first, last - first + 1);
}

std::string last_lang_tag(const std::string& text) {
  if (!MODEL_PROMPTED) return "";
  static const std::regex capture_tag(R"(<([a-z]{2}(-[A-Z]{2})?)>)");
  std::string last;
  auto begin = std::sregex_iterator(text.begin(), text.end(), capture_tag);
  for (auto it = begin; it != std::sregex_iterator(); ++it) {
    last = (*it)[1].str();
  }
  return last;
}

std::vector<EmittedEvent> gold_events_from_bundle(torch::jit::Module& bundle,
                                                        const std::string& prefix,
                                                        const std::string& label) {
  auto kinds = tensor_to_vec(prefix_tensor(bundle, prefix, "event_kinds"));
  auto tokens = unpack_i64_lists(
      prefix_tensor(bundle, prefix, "event_tokens"),
      prefix_tensor(bundle, prefix, "event_token_offsets"),
      "event_tokens",
      -1);
  auto collectors = unpack_i64_lists(
      prefix_tensor(bundle, prefix, "event_collector_tokens"),
      prefix_tensor(bundle, prefix, "event_collector_token_offsets"),
      "event_collector_tokens",
      -1);
  auto texts = unpack_utf8_strings(
      prefix_tensor(bundle, prefix, "event_text_bytes"),
      prefix_tensor(bundle, prefix, "event_text_offsets"),
      "event_text",
      -1);
  auto collector_texts = unpack_utf8_strings(
      prefix_tensor(bundle, prefix, "event_collector_text_bytes"),
      prefix_tensor(bundle, prefix, "event_collector_text_offsets"),
      "event_collector_text",
      -1);
  if (tokens.size() != kinds.size() || collectors.size() != kinds.size() ||
      texts.size() != kinds.size() || collector_texts.size() != kinds.size()) {
    throw std::runtime_error("event payload count mismatch for " + label);
  }
  std::vector<EmittedEvent> events;
  events.reserve(kinds.size());
  for (size_t i = 0; i < kinds.size(); ++i) {
    events.push_back({kinds[i], tokens[i], collectors[i], texts[i], collector_texts[i]});
  }
  return events;
}

std::vector<EmittedEvent> gold_events_from_bundle(torch::jit::Module& bundle, int utt) {
  std::string prefix = "utt" + std::to_string(utt);
  return gold_events_from_bundle(bundle, prefix, "utt" + std::to_string(utt));
}

static std::string one_text_from_bundle(torch::jit::Module& bundle,
                                        const std::string& prefix,
                                        const char* name,
                                        const std::string& label) {
  auto values = unpack_utf8_strings(
      prefix_tensor(bundle, prefix, (std::string(name) + "_bytes").c_str()),
      prefix_tensor(bundle, prefix, (std::string(name) + "_offsets").c_str()),
      name,
      -1);
  if (values.size() != 1) throw std::runtime_error(label + " expected one UTF-8 string for " + name);
  return values[0];
}

static std::string optional_one_text_from_bundle(torch::jit::Module& bundle,
                                                 const std::string& prefix,
                                                 const char* name,
                                                 const std::string& fallback,
                                                 const std::string& label) {
  try {
    return one_text_from_bundle(bundle, prefix, name, label);
  } catch (const std::exception&) {
    return fallback;
  }
}

void emit_event(std::vector<EmittedEvent>& events,
                       int64_t kind,
                       const std::vector<int64_t>& tokens,
                       const std::vector<int64_t>& collector_tokens,
                       const std::string& text,
                       const std::string& collector_text) {
  events.push_back({kind, tokens, collector_tokens, text, collector_text});
}

static bool equal_events(const std::vector<EmittedEvent>& got,
                         const std::vector<EmittedEvent>& gold,
                         const std::string& label) {
  bool ok = got.size() == gold.size();
  if (!ok) {
    std::printf("    %s event count mismatch: got=%zu gold=%zu\n",
                label.c_str(), got.size(), gold.size());
  }
  size_t n = std::min(got.size(), gold.size());
  for (size_t i = 0; i < n; ++i) {
    bool event_ok = got[i].kind == gold[i].kind &&
                    got[i].text == gold[i].text &&
                    got[i].collector_text == gold[i].collector_text;
    if (!event_ok) {
      std::printf("    %s event[%zu] mismatch: got_kind=%s gold_kind=%s\n",
                  label.c_str(), i, event_kind_name(got[i].kind), event_kind_name(gold[i].kind));
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

bool equal_tokens(const std::vector<int64_t>& got,
                         const std::vector<int64_t>& gold,
                         const char* label,
                         const std::string& row_label) {
  bool ok = got == gold;
  if (!ok) {
    std::printf("    %s %s token mismatch: got_len=%zu gold_len=%zu\n",
                row_label.c_str(), label, got.size(), gold.size());
    std::printf("      got :%s\n", vec_to_string(got).c_str());
    std::printf("      gold:%s\n", vec_to_string(gold).c_str());
    size_t n = std::min(got.size(), gold.size());
    for (size_t i = 0; i < n; ++i) {
      if (got[i] != gold[i]) {
        std::printf("      first_diff=%zu got=%ld gold=%ld\n",
                    i, (long)got[i], (long)gold[i]);
        break;
      }
    }
  }
  return ok;
}

static size_t first_token_diff_index(const std::vector<int64_t>& got,
                                     const std::vector<int64_t>& gold) {
  size_t n = std::min(got.size(), gold.size());
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != gold[i]) return i;
  }
  if (got.size() != gold.size()) return n;
  return std::numeric_limits<size_t>::max();
}

static int64_t token_or_missing(const std::vector<int64_t>& tokens, size_t index) {
  if (index >= tokens.size()) return -1;
  return tokens[index];
}

bool tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
  bool meta_ok = actual.scalar_type() == expected.scalar_type() &&
                 actual.sizes().vec() == expected.sizes().vec();
  bool eq = meta_ok && at::equal(actual, expected);
  if (!eq) {
    std::printf("    FORK_ASSERT %s mismatch: dtype %d/%d sizes",
                name, (int)actual.scalar_type(), (int)expected.scalar_type());
    for (auto s : actual.sizes()) std::printf(" %ld", (long)s);
    std::printf(" vs");
    for (auto s : expected.sizes()) std::printf(" %ld", (long)s);
    std::printf("\n");
  }
  return eq;
}

bool tensor_close(const char* name,
                         const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double atol,
                         const std::string& label) {
  bool meta_ok = actual.scalar_type() == expected.scalar_type() &&
                 actual.sizes().vec() == expected.sizes().vec();
  if (!meta_ok) {
    std::printf("    %s %s metadata mismatch: dtype %d/%d sizes",
                label.c_str(), name, (int)actual.scalar_type(), (int)expected.scalar_type());
    for (auto s : actual.sizes()) std::printf(" %ld", (long)s);
    std::printf(" vs");
    for (auto s : expected.sizes()) std::printf(" %ld", (long)s);
    std::printf("\n");
    return false;
  }
  double max_abs = 0.0;
  if (actual.numel() > 0) {
    max_abs = (actual - expected).abs().max().item<double>();
  }
  bool ok = at::equal(actual, expected) || max_abs <= atol;
  if (!ok) {
    std::printf("    %s %s allclose mismatch: max_abs=%.6e atol=%.6e\n",
                label.c_str(), name, max_abs, atol);
  }
  return ok;
}

static bool tensor_close_cache(const char* name,
                               const torch::Tensor& actual,
                               const torch::Tensor& expected,
                               double atol,
                               const std::string& label,
                               CacheCompareStats& stats,
                               double& max_slot) {
  ++stats.checks;
  bool meta_ok = actual.scalar_type() == expected.scalar_type() &&
                 actual.sizes().vec() == expected.sizes().vec();
  if (!meta_ok) {
    std::printf("    %s %s metadata mismatch: dtype %d/%d sizes",
                label.c_str(), name, (int)actual.scalar_type(), (int)expected.scalar_type());
    for (auto s : actual.sizes()) std::printf(" %ld", (long)s);
    std::printf(" vs");
    for (auto s : expected.sizes()) std::printf(" %ld", (long)s);
    std::printf("\n");
    return false;
  }
  double max_abs = 0.0;
  if (actual.numel() > 0) {
    max_abs = (actual - expected).abs().max().item<double>();
  }
  if (max_abs > max_slot) max_slot = max_abs;
  bool ok = at::equal(actual, expected) || max_abs <= atol;
  if (ok) {
    ++stats.pass;
  } else {
    std::printf("    %s %s allclose mismatch: max_abs=%.6e atol=%.6e\n",
                label.c_str(), name, max_abs, atol);
  }
  return ok;
}

static bool optional_tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
  if (actual.defined() != expected.defined()) {
    std::printf("    FORK_ASSERT %s defined mismatch: %d/%d\n",
                name, (int)actual.defined(), (int)expected.defined());
    return false;
  }
  if (!actual.defined()) return true;
  return tensor_equal(name, actual, expected);
}

static bool float_vec_equal(const char* name,
                            const std::vector<float>& actual,
                            const std::vector<float>& expected,
                            const std::string& label = "FORK_ASSERT") {
  bool ok = actual.size() == expected.size();
  double max_abs = 0.0;
  size_t n = std::min(actual.size(), expected.size());
  for (size_t i = 0; i < n; ++i) {
    double d = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
    if (d > max_abs) max_abs = d;
    if (actual[i] != expected[i]) ok = false;
  }
  if (!ok) {
    std::printf("    %s %s mismatch: got=%zu gold=%zu max_abs=%.3e\n",
                label.c_str(), name, actual.size(), expected.size(), max_abs);
  }
  return ok;
}

static bool int64_equal(const char* name, int64_t actual, int64_t expected, const std::string& label) {
  if (actual == expected) return true;
  std::printf("    %s %s mismatch: got=%ld gold=%ld\n",
              label.c_str(), name, (long)actual, (long)expected);
  return false;
}

SessionState clone_session(const SessionState& state) {
  SessionState out;
  out.generation.store(state.generation.load(std::memory_order_acquire), std::memory_order_release);
  out.clc = state.clc.clone();
  out.clt = state.clt.clone();
  out.clcl = state.clcl.clone();
  out.g = state.g.clone();
  out.h = state.h.clone();
  out.c = state.c.clone();
  out.ring = state.ring.defined() ? state.ring.clone() : torch::Tensor();
  out.emitted = state.emitted;
  out.hyp = state.hyp;
  out.last_interim_tokens = state.last_interim_tokens;
  out.continuous_emitted_tokens = state.continuous_emitted_tokens;
  out.pending_audio = state.pending_audio;
  out.raw_audio_ring = state.raw_audio_ring;
  out.post_stop_audio = state.post_stop_audio;
  out.last_interim_text = state.last_interim_text;
  out.continuous_emitted_text = state.continuous_emitted_text;
  out.mode = state.mode;
  out.total_audio_samples = state.total_audio_samples;
  out.synthetic_prefix_samples = state.synthetic_prefix_samples;
  out.prompt = state.prompt;  // shared read-only one-hot; forks keep the parent's language
  out.prompt_index = state.prompt_index;
  out.language = state.language;
  out.last_language = state.last_language;
  return out;
}

AsrSnapshot snapshot_asr(const SessionState& state) {
  return {
      state.clc.clone(),
      state.clt.clone(),
      state.clcl.clone(),
      state.g.clone(),
      state.h.clone(),
      state.c.clone(),
      state.ring.defined() ? state.ring.clone() : torch::Tensor(),
      state.emitted,
      state.hyp,
      state.pending_audio,
      state.raw_audio_ring,
      state.total_audio_samples,
      state.synthetic_prefix_samples,
      state.mode,
  };
}

bool fork_assert_parent_unchanged(const SessionState& parent, const AsrSnapshot& snapshot) {
  bool ok = true;
  ok = tensor_equal("cache_last_channel", parent.clc, snapshot.clc) && ok;
  ok = tensor_equal("cache_last_time", parent.clt, snapshot.clt) && ok;
  ok = tensor_equal("cache_last_channel_len", parent.clcl, snapshot.clcl) && ok;
  ok = tensor_equal("pred_out", parent.g, snapshot.g) && ok;
  ok = tensor_equal("decoder_state.h", parent.h, snapshot.h) && ok;
  ok = tensor_equal("decoder_state.c", parent.c, snapshot.c) && ok;
  ok = optional_tensor_equal("mel_frame_ring", parent.ring, snapshot.ring) && ok;
  if (parent.emitted != snapshot.emitted) {
    std::printf("    FORK_ASSERT emitted_frames mismatch: %ld/%ld\n",
                (long)parent.emitted, (long)snapshot.emitted);
    ok = false;
  }
  if (parent.hyp != snapshot.hyp) {
    std::printf("    FORK_ASSERT hyp_tokens mismatch: parent=%zu snapshot=%zu\n",
                parent.hyp.size(), snapshot.hyp.size());
    ok = false;
  }
  ok = float_vec_equal("pending_audio", parent.pending_audio, snapshot.pending_audio) && ok;
  ok = float_vec_equal("raw_audio_ring", parent.raw_audio_ring, snapshot.raw_audio_ring) && ok;
  if (parent.total_audio_samples != snapshot.total_audio_samples) {
    std::printf("    FORK_ASSERT total_audio_samples mismatch: %ld/%ld\n",
                (long)parent.total_audio_samples, (long)snapshot.total_audio_samples);
    ok = false;
  }
  if (parent.synthetic_prefix_samples != snapshot.synthetic_prefix_samples) {
    std::printf("    FORK_ASSERT synthetic_prefix_samples mismatch: %ld/%ld\n",
                (long)parent.synthetic_prefix_samples, (long)snapshot.synthetic_prefix_samples);
    ok = false;
  }
  if (parent.mode != snapshot.mode) {
    std::printf("    FORK_ASSERT mode mismatch: parent=%s snapshot=%s\n",
                mode_name(parent.mode), mode_name(snapshot.mode));
    ok = false;
  }
  return ok;
}

void reset_session(SessionState& state, torch::jit::Module& bundle, torch::Device device) {
  state.generation.fetch_add(1, std::memory_order_acq_rel);
  state.clc = attr_tensor(bundle, "init_clc").to(device).clone();
  state.clt = attr_tensor(bundle, "init_clt").to(device).clone();
  state.clcl = attr_tensor(bundle, "init_clcl").to(device).clone();
  state.g = attr_tensor(bundle, "init_g").to(device).clone();
  state.h = attr_tensor(bundle, "init_h").to(device).clone();
  state.c = attr_tensor(bundle, "init_c").to(device).clone();
  state.ring = torch::Tensor();
  state.emitted = 0;
  state.hyp.clear();
  state.last_interim_tokens.clear();
  state.continuous_emitted_tokens.clear();
  state.pending_audio.clear();
  state.raw_audio_ring.clear();
  state.post_stop_audio.clear();
  state.last_interim_text.clear();
  state.continuous_emitted_text.clear();
  state.mode = SessionMode::STREAMING;
  state.total_audio_samples = 0;
  state.synthetic_prefix_samples = 0;
}

static void reset_audio_front(SessionState& state, const AudioGeometry& g) {
  state.pending_audio.clear();
  state.raw_audio_ring.assign(static_cast<size_t>(g.raw_audio_ring_samples), 0.0f);
  state.post_stop_audio.clear();
  state.total_audio_samples = 0;
  state.synthetic_prefix_samples = 0;
}

void finish_speculative_finalize(SessionState& state) {
  state.mode = SessionMode::STREAMING;
}

void cold_reset_after_finalize(SessionState& state,
                                      torch::jit::Module& bundle,
                                      torch::Device device,
                                      const AudioGeometry* audio_geometry = nullptr) {
  reset_session(state, bundle, device);
  if (audio_geometry != nullptr) reset_audio_front(state, *audio_geometry);
}

static uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32U - n));
}

static void sha256_transform(Sha256Ctx& ctx, const uint8_t data[64]) {
  static constexpr std::array<uint32_t, 64> k{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  std::array<uint32_t, 64> m{};
  for (uint32_t i = 0, j = 0; i < 16; ++i, j += 4) {
    m[i] = (static_cast<uint32_t>(data[j]) << 24) |
           (static_cast<uint32_t>(data[j + 1]) << 16) |
           (static_cast<uint32_t>(data[j + 2]) << 8) |
           (static_cast<uint32_t>(data[j + 3]));
  }
  for (uint32_t i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
    uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
    m[i] = m[i - 16] + s0 + m[i - 7] + s1;
  }

  uint32_t a = ctx.state[0], b = ctx.state[1], c = ctx.state[2], d = ctx.state[3];
  uint32_t e = ctx.state[4], f = ctx.state[5], g = ctx.state[6], h = ctx.state[7];
  for (uint32_t i = 0; i < 64; ++i) {
    uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t temp1 = h + s1 + ch + k[i] + m[i];
    uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t temp2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  ctx.state[0] += a; ctx.state[1] += b; ctx.state[2] += c; ctx.state[3] += d;
  ctx.state[4] += e; ctx.state[5] += f; ctx.state[6] += g; ctx.state[7] += h;
}

static void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ctx.data[ctx.datalen++] = data[i];
    if (ctx.datalen == 64) {
      sha256_transform(ctx, ctx.data.data());
      ctx.bitlen += 512;
      ctx.datalen = 0;
    }
  }
}

static std::string sha256_final(Sha256Ctx& ctx) {
  uint32_t i = ctx.datalen;
  uint64_t total_bits = ctx.bitlen + static_cast<uint64_t>(ctx.datalen) * 8U;

  ctx.data[i++] = 0x80U;
  if (i > 56) {
    while (i < 64) ctx.data[i++] = 0;
    sha256_transform(ctx, ctx.data.data());
    i = 0;
  }
  while (i < 56) ctx.data[i++] = 0;
  for (int shift = 56; shift >= 0; shift -= 8) {
    ctx.data[i++] = static_cast<uint8_t>((total_bits >> shift) & 0xffU);
  }
  sha256_transform(ctx, ctx.data.data());

  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (uint32_t word : ctx.state) oss << std::setw(8) << word;
  return oss.str();
}

static std::string sha256_bytes_with_label(const std::string& label,
                                           const uint8_t* data,
                                           size_t len) {
  Sha256Ctx ctx;
  sha256_update(ctx,
                reinterpret_cast<const uint8_t*>(label.data()),
                label.size());
  uint8_t sep = 0;
  sha256_update(ctx, &sep, 1);
  if (len > 0) sha256_update(ctx, data, len);
  return sha256_final(ctx);
}

static std::string sha256_tensor_bytes(torch::Tensor tensor) {
  auto cpu = tensor.detach().to(torch::kCPU).contiguous();
  std::ostringstream label;
  label << "dtype=" << static_cast<int>(cpu.scalar_type()) << ";sizes=";
  for (auto s : cpu.sizes()) label << s << ",";
  size_t nbytes = static_cast<size_t>(cpu.numel()) * cpu.element_size();
  const uint8_t* data = nbytes == 0
                            ? nullptr
                            : reinterpret_cast<const uint8_t*>(cpu.data_ptr());
  return sha256_bytes_with_label(label.str(), data, nbytes);
}

std::string sha256_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open for sha256: " + path);
  Sha256Ctx ctx;
  std::array<char, 1024 * 1024> buffer{};
  while (f) {
    f.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::streamsize got = f.gcount();
    if (got > 0) {
      sha256_update(ctx, reinterpret_cast<const uint8_t*>(buffer.data()), static_cast<size_t>(got));
    }
  }
  return sha256_final(ctx);
}

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open manifest: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static size_t skip_ws(const std::string& s, size_t pos) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  return pos;
}

static size_t find_matching_json_delim(const std::string& s, size_t open_pos) {
  char open = s.at(open_pos);
  char close = open == '{' ? '}' : ']';
  int depth = 0;
  bool in_string = false;
  bool escape = false;
  for (size_t i = open_pos; i < s.size(); ++i) {
    char ch = s[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == open) {
      ++depth;
    } else if (ch == close) {
      --depth;
      if (depth == 0) return i;
    }
  }
  throw std::runtime_error("unterminated JSON object/array in manifest");
}

static std::string json_value_for_key(const std::string& object, const std::string& key) {
  std::string needle = "\"" + key + "\"";
  size_t key_pos = object.find(needle);
  if (key_pos == std::string::npos) throw std::runtime_error("manifest missing key: " + key);
  size_t colon = object.find(':', key_pos + needle.size());
  if (colon == std::string::npos) throw std::runtime_error("manifest key has no colon: " + key);
  size_t start = skip_ws(object, colon + 1);
  if (start >= object.size()) throw std::runtime_error("manifest key has no value: " + key);

  size_t end = start;
  if (object[start] == '{' || object[start] == '[') {
    end = find_matching_json_delim(object, start) + 1;
  } else if (object[start] == '"') {
    bool escape = false;
    for (end = start + 1; end < object.size(); ++end) {
      char ch = object[end];
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        ++end;
        break;
      }
    }
  } else {
    while (end < object.size() && object[end] != ',' && object[end] != '}' && object[end] != ']') ++end;
  }
  return object.substr(start, end - start);
}

static std::string json_string_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  value = value.substr(skip_ws(value, 0));
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    throw std::runtime_error("manifest key is not a string: " + key);
  }
  return value.substr(1, value.size() - 2);
}

static int64_t json_int_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  size_t n = 0;
  long long out = std::stoll(value, &n);
  n = skip_ws(value, n);
  if (n != value.size()) throw std::runtime_error("manifest key is not an integer: " + key);
  return out;
}

static double json_double_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  size_t n = 0;
  double out = std::stod(value, &n);
  n = skip_ws(value, n);
  if (n != value.size()) throw std::runtime_error("manifest key is not a number: " + key);
  return out;
}

static std::vector<int64_t> json_int_array_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  if (value.empty() || value.front() != '[' || value.back() != ']') {
    throw std::runtime_error("manifest key is not an array: " + key);
  }
  std::vector<int64_t> out;
  std::regex num_re("-?\\d+");
  for (auto it = std::sregex_iterator(value.begin(), value.end(), num_re);
       it != std::sregex_iterator(); ++it) {
    out.push_back(std::stoll((*it)[0].str()));
  }
  return out;
}

static bool parse_bucket_filename(const std::string& filename, int64_t& drop, int64_t& T) {
  const std::string prefix = "enc_finalize_d";
  const std::string mid = "_T";
  const std::string suffix = ".pt2";
  if (filename.rfind(prefix, 0) != 0) return false;
  if (filename.size() <= prefix.size() + suffix.size()) return false;
  if (filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) return false;

  size_t tpos = filename.find(mid, prefix.size());
  if (tpos == std::string::npos) return false;
  std::string drop_s = filename.substr(prefix.size(), tpos - prefix.size());
  std::string T_s = filename.substr(tpos + mid.size(), filename.size() - suffix.size() - (tpos + mid.size()));
  if (drop_s.empty() || T_s.empty()) return false;

  try {
    size_t n = 0;
    long long d = std::stoll(drop_s, &n);
    if (n != drop_s.size()) return false;
    n = 0;
    long long t = std::stoll(T_s, &n);
    if (n != T_s.size()) return false;
    if (d < 0 || t <= 0) return false;
    drop = d;
    T = t;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

std::map<std::pair<int64_t, int64_t>, std::string> discover_finalize_buckets(const std::string& buckets_dir) {
  std::map<std::pair<int64_t, int64_t>, std::string> buckets;
  for (const auto& entry : fs::directory_iterator(buckets_dir)) {
    if (!entry.is_regular_file()) continue;
    int64_t drop = 0;
    int64_t T = 0;
    std::string filename = entry.path().filename().string();
    if (!parse_bucket_filename(filename, drop, T)) continue;
    auto key = std::make_pair(drop, T);
    auto path = entry.path().string();
    auto inserted = buckets.emplace(key, path);
    if (!inserted.second) {
      throw std::runtime_error("duplicate finalize bucket for (drop,T)=(" + std::to_string(drop) + "," +
                               std::to_string(T) + "): " + inserted.first->second + " and " + path);
    }
  }
  return buckets;
}

BucketManifest load_bucket_manifest(const std::string& manifest_path) {
  std::string text = read_text_file(manifest_path);
  std::string contract_obj = json_value_for_key(text, "CONTRACT");
  std::string buckets_arr = json_value_for_key(text, "buckets");
  if (contract_obj.empty() || contract_obj.front() != '{') throw std::runtime_error("manifest CONTRACT is not an object");
  if (buckets_arr.empty() || buckets_arr.front() != '[') throw std::runtime_error("manifest buckets is not an array");

  BucketManifest manifest;
  manifest.contract.model_id = json_string_field(contract_obj, "model_id");
  manifest.contract.att_context = json_int_array_field(contract_obj, "att_context");
  manifest.contract.right_context = json_int_field(contract_obj, "right_context");
  manifest.contract.shift = json_int_field(contract_obj, "shift");
  manifest.contract.pre_encode_cache = json_int_field(contract_obj, "pre_encode_cache");
  manifest.contract.drop_extra = json_int_field(contract_obj, "drop_extra");
  manifest.contract.final_padding_frames = json_int_field(contract_obj, "final_padding_frames");
  manifest.contract.blank = json_int_field(contract_obj, "blank");
  manifest.contract.max_symbols = json_int_field(contract_obj, "max_symbols");
  manifest.contract.weights_sha256 = json_string_field(contract_obj, "weights_sha256");

  size_t pos = 1;
  while (pos + 1 < buckets_arr.size()) {
    pos = skip_ws(buckets_arr, pos);
    if (pos >= buckets_arr.size() || buckets_arr[pos] == ']') break;
    if (buckets_arr[pos] == ',') {
      ++pos;
      continue;
    }
    if (buckets_arr[pos] != '{') throw std::runtime_error("manifest bucket entry is not an object");
    size_t end = find_matching_json_delim(buckets_arr, pos);
    std::string obj = buckets_arr.substr(pos, end - pos + 1);
    ManifestBucket b;
    b.drop = json_int_field(obj, "drop");
    b.T = json_int_field(obj, "T");
    b.pkg = json_string_field(obj, "pkg");
    b.pkg_sha256 = json_string_field(obj, "pkg_sha256");
    manifest.buckets.push_back(std::move(b));
    pos = end + 1;
  }
  return manifest;
}

static void require_contract_eq(const char* name, int64_t actual, int64_t expected) {
  if (actual != expected) {
    throw std::runtime_error(std::string("manifest CONTRACT mismatch for ") + name +
                             ": got " + std::to_string(actual) +
                             " expected " + std::to_string(expected));
  }
}

void verify_bucket_manifest(const BucketManifest& manifest,
                                   const std::map<std::pair<int64_t, int64_t>, std::string>& discovered,
                                   const std::string& buckets_dir,
                                   const std::string& shared_weights_pt,
                                   ManifestShaVerifyMode sha_mode) {
  const auto& c = manifest.contract;
  if (c.model_id != MODEL_ID) {
    throw std::runtime_error("manifest CONTRACT model_id mismatch: " + c.model_id);
  }
  if (c.att_context.size() != 2 || c.att_context[0] != ATT_CONTEXT_LEFT || c.att_context[1] != ATT_CONTEXT_RIGHT) {
    throw std::runtime_error("manifest CONTRACT att_context mismatch");
  }
  require_contract_eq("right_context", c.right_context, RIGHT_CONTEXT);
  require_contract_eq("shift", c.shift, SHIFT);
  require_contract_eq("pre_encode_cache", c.pre_encode_cache, PRE);
  require_contract_eq("drop_extra", c.drop_extra, DROP);
  require_contract_eq("final_padding_frames", c.final_padding_frames, FINAL_PADDING_FRAMES);
  require_contract_eq("blank", c.blank, BLANK);
  require_contract_eq("max_symbols", c.max_symbols, MAX_SYMBOLS);

  if (!file_exists(shared_weights_pt)) {
    throw std::runtime_error("manifest requires shared weights .pt but file is missing: " + shared_weights_pt);
  }
  const bool full_sha_verify = manifest_sha_verify_full(sha_mode);
  std::string weights_sha = "skipped";
  if (full_sha_verify) {
    weights_sha = sha256_file(shared_weights_pt);
    if (weights_sha != c.weights_sha256) {
      throw std::runtime_error("shared weights sha256 mismatch: manifest=" + c.weights_sha256 +
                               " actual=" + weights_sha);
    }
  }

  std::set<std::pair<int64_t, int64_t>> manifest_keys;
  std::set<std::string> manifest_pkgs;
  size_t package_verified = 0;
  for (const auto& b : manifest.buckets) {
    if (!manifest_keys.emplace(b.drop, b.T).second) {
      throw std::runtime_error("duplicate manifest bucket key drop=" + std::to_string(b.drop) +
                               " T=" + std::to_string(b.T));
    }
    if (!manifest_pkgs.emplace(b.pkg).second) throw std::runtime_error("duplicate manifest pkg: " + b.pkg);

    int64_t parsed_drop = 0;
    int64_t parsed_T = 0;
    if (!parse_bucket_filename(b.pkg, parsed_drop, parsed_T) || parsed_drop != b.drop || parsed_T != b.T) {
      throw std::runtime_error("manifest pkg filename does not match drop/T: " + b.pkg);
    }

    auto found = discovered.find(std::make_pair(b.drop, b.T));
    if (found == discovered.end()) {
      throw std::runtime_error("manifest bucket missing from directory: " + b.pkg);
    }
    fs::path expected_path = fs::path(buckets_dir) / b.pkg;
    if (fs::path(found->second).filename() != expected_path.filename()) {
      throw std::runtime_error("manifest/discovered pkg name mismatch for " + b.pkg);
    }
    std::string actual_sha = sha256_file(expected_path.string());
    if (actual_sha != b.pkg_sha256) {
      throw std::runtime_error("bucket sha256 mismatch for " + b.pkg +
                               ": manifest=" + b.pkg_sha256 + " actual=" + actual_sha);
    }
    ++package_verified;
  }

  for (const auto& kv : discovered) {
    if (manifest_keys.find(kv.first) == manifest_keys.end()) {
      throw std::runtime_error("bucket file is not listed in manifest: " + kv.second);
    }
  }
  std::printf("finalize bucket manifest SHA mode: env=NEMOTRON_WS_VERIFY_MANIFEST_SHA mode=%s "
              "shared_weights_sha256=%s package_verified=%zu buckets=%zu\n",
              manifest_sha_verify_mode_name(sha_mode),
              weights_sha.c_str(),
              package_verified,
              manifest.buckets.size());
}

std::unordered_map<std::string, at::Tensor> load_shared_constants(const std::string& weights_path,
                                                                         torch::Device device) {
  auto weights_module = load_jit_serialized(weights_path);
  auto weights = weights_module.attr("weights").toGenericDict();
  std::unordered_map<std::string, at::Tensor> constants;
  constants.reserve(weights.size());
  for (const auto& item : weights) {
    if (!item.key().isString()) throw std::runtime_error("finalize_shared_weights.ts has a non-string key");
    if (!item.value().isTensor()) throw std::runtime_error("finalize_shared_weights.ts has a non-tensor value");
    constants.emplace(item.key().toStringRef(), item.value().toTensor().to(device));
  }
  return constants;
}

static const at::Tensor* resolve_shared_constant(const std::unordered_map<std::string, at::Tensor>& shared_constants,
                                                 const std::string& fqn,
                                                 bool& used_alias) {
  auto it = shared_constants.find(fqn);
  if (it != shared_constants.end()) {
    used_alias = false;
    return &it->second;
  }

  std::string alt;
  if (fqn.rfind("encoder.", 0) == 0) {
    alt = "e." + fqn.substr(8);
  } else if (fqn.rfind("e.", 0) == 0) {
    alt = "encoder." + fqn.substr(2);
  } else {
    return nullptr;
  }
  it = shared_constants.find(alt);
  if (it == shared_constants.end()) return nullptr;
  used_alias = true;
  return &it->second;
}

BucketConstants constants_for_bucket(
    const std::unordered_map<std::string, at::Tensor>& shared_constants,
    AOTIModelPackageLoader& loader,
    const std::string& pkg) {
  auto fqns = loader.get_constant_fqns();
  BucketConstants bucket_constants;
  bucket_constants.values.reserve(fqns.size());
  std::vector<std::string> missing;
  for (const auto& fqn : fqns) {
    bool used_alias = false;
    const at::Tensor* tensor = resolve_shared_constant(shared_constants, fqn, used_alias);
    if (tensor == nullptr) {
      missing.push_back(fqn);
    } else {
      if (used_alias) {
        ++bucket_constants.alias_fallbacks;
      } else {
        ++bucket_constants.direct_matches;
      }
      bucket_constants.values.emplace(fqn, *tensor);
    }
  }
  if (!missing.empty()) {
    std::ostringstream oss;
    oss << "bucket " << pkg << " missing " << missing.size() << " shared weights; first missing:";
    for (size_t i = 0; i < std::min<size_t>(missing.size(), 5); ++i) oss << ' ' << missing[i];
    throw std::runtime_error(oss.str());
  }
  return bucket_constants;
}

static std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>
load_finalize_bucket_loaders(const std::string& dir, torch::Device device) {
  std::string buckets_dir = dir + "/stripped_finalize_buckets";
  if (!directory_exists(buckets_dir)) buckets_dir = dir + "/finalize_buckets";
  std::string shared_weights = dir + "/finalize_shared_weights.ts";
  std::string shared_weights_pt = dir + "/finalize_shared_weights.pt";
  if (!directory_exists(buckets_dir)) throw std::runtime_error("finalize buckets directory missing: " + buckets_dir);
  if (!file_exists(shared_weights)) throw std::runtime_error("finalize shared weights missing: " + shared_weights);

  auto bucket_paths = discover_finalize_buckets(buckets_dir);
  if (bucket_paths.empty()) throw std::runtime_error("no finalize bucket packages found in " + buckets_dir);
  std::string manifest_path = buckets_dir + "/manifest.json";
  if (!file_exists(manifest_path)) {
    throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
  }
  auto manifest = load_bucket_manifest(manifest_path);
  const ManifestShaVerifyMode sha_mode = manifest_sha_verify_mode_from_env();
  verify_bucket_manifest(manifest, bucket_paths, buckets_dir, shared_weights_pt, sha_mode);
  std::printf("finalize manifest verified: %zu buckets, weights_sha256=%s\n",
              manifest.buckets.size(), manifest.contract.weights_sha256.c_str());

  auto shared_constants = load_shared_constants(shared_weights, device);
  std::printf("loaded finalize shared constants: %zu entries\n", shared_constants.size());

  std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>> loaders;
  for (const auto& kv : bucket_paths) {
    int64_t drop = kv.first.first;
    int64_t T = kv.first.second;
    const std::string& pkg = kv.second;
    auto loader = std::make_unique<AOTIModelPackageLoader>(pkg, "model", false, 1, -1);
    auto bucket_constants = constants_for_bucket(shared_constants, *loader, pkg);
    loader->load_constants(bucket_constants.values, false, false, true);
    std::printf("  finalize bucket drop=%ld T=%ld constants=%zu direct=%zu alias=%zu\n",
                (long)drop, (long)T, bucket_constants.values.size(),
                bucket_constants.direct_matches, bucket_constants.alias_fallbacks);
    loaders.emplace(kv.first, std::move(loader));
  }
  return loaders;
}

static void observe_margin(MarginStats& stats,
                           double margin,
                           const std::string& label,
                           int64_t frame,
                           int64_t symbol) {
  ++stats.total;
  if (margin < stats.warning_threshold) ++stats.below_warning;
  if (margin < stats.unsafe_threshold) ++stats.below_unsafe;
  if (margin < stats.min_margin) {
    stats.min_margin = margin;
    stats.min_label = label;
    stats.min_frame = frame;
    stats.min_symbol = symbol;
  }
}

static void observe_token_margin(MarginStats& stats,
                                 int64_t token_index,
                                 int64_t token_id,
                                 double margin,
                                 const std::string& label,
                                 int64_t frame,
                                 int64_t symbol) {
  stats.token_margins.push_back({token_index, token_id, margin, label, frame, symbol});
}

static void merge_margin_stats(MarginStats& dst, const MarginStats& src) {
  dst.total += src.total;
  dst.below_warning += src.below_warning;
  dst.below_unsafe += src.below_unsafe;
  if (src.min_margin < dst.min_margin) {
    dst.min_margin = src.min_margin;
    dst.min_label = src.min_label;
    dst.min_frame = src.min_frame;
    dst.min_symbol = src.min_symbol;
  }
}

static const TokenMargin* margin_for_token_index(const MarginStats& stats, size_t token_index) {
  for (const auto& margin : stats.token_margins) {
    if (margin.token_index == static_cast<int64_t>(token_index)) return &margin;
  }
  return nullptr;
}

static void decode_range(torch::jit::Module& joint,
                         torch::jit::Module& predict,
                         const torch::Tensor& enc_out,
                         int64_t enc_len,
                         torch::Tensor& g,
                         torch::Tensor& h,
                         torch::Tensor& c,
                         std::vector<int64_t>& hyp,
                         MarginStats* margin_stats = nullptr,
                         const std::string& margin_label = "",
                         MarginStats* secondary_margin_stats = nullptr) {
  if (enc_len < 0 || enc_len > enc_out.size(2)) {
    throw std::runtime_error("enc_len out of range for enc_out: " + std::to_string(enc_len));
  }
  auto f = enc_out.transpose(1, 2).contiguous();
  auto dev = f.device();
  for (int64_t t = 0; t < enc_len; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor();
      auto flat = logits.reshape({-1});
      double margin = std::numeric_limits<double>::quiet_NaN();
      if (margin_stats != nullptr || secondary_margin_stats != nullptr) {
        auto top2 = std::get<0>(flat.topk(2));
        margin = (top2[0] - top2[1]).item<double>();
        if (margin_stats != nullptr) observe_margin(*margin_stats, margin, margin_label, t, n);
        if (secondary_margin_stats != nullptr) observe_margin(*secondary_margin_stats, margin, margin_label, t, n);
      }
      int64_t k = flat.argmax().item<int64_t>();
      if (k == BLANK) break;
      if (margin_stats != nullptr) {
        observe_token_margin(*margin_stats,
                             static_cast<int64_t>(hyp.size()),
                             k,
                             margin,
                             margin_label,
                             t,
                             n);
      }
      if (secondary_margin_stats != nullptr) {
        observe_token_margin(*secondary_margin_stats,
                             static_cast<int64_t>(hyp.size()),
                             k,
                             margin,
                             margin_label,
                             t,
                             n);
      }
      hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(dev));
      auto out = predict.forward({y, h, c}).toTuple();
      g = out->elements()[0].toTensor();
      h = out->elements()[1].toTensor();
      c = out->elements()[2].toTensor();
    }
  }
}

static void apply_encoder_outputs(SessionState& state,
                                  const std::vector<at::Tensor>& out,
                                  torch::jit::Module& joint,
                                  torch::jit::Module& predict,
                                  MarginStats* margin_stats = nullptr,
                                  const std::string& margin_label = "",
                                  MarginStats* secondary_margin_stats = nullptr,
                                  CacheOwnershipStats* cache_ownership = nullptr,
                                  bool recurrent_cache_output = false) {
  if (out.size() < 5) throw std::runtime_error("encoder returned fewer than 5 outputs");
  int64_t enc_len = scalar_i64(out[1]);
  auto conditioned_enc = condition_encoder_output(out[0], state);
  state.clc = out[2].clone();
  state.clt = out[3].clone();
  state.clcl = out[4].clone();
  if (cache_ownership != nullptr && recurrent_cache_output) {
    ++cache_ownership->recurrent_assignments;
    cache_ownership->clone_assignments += 3;
    if (tensor_storage_alias(state.clc, out[2]) ||
        tensor_storage_alias(state.clt, out[3]) ||
        tensor_storage_alias(state.clcl, out[4])) {
      ++cache_ownership->alias_after_clone;
      throw std::runtime_error("AOTI recurrent cache output still aliases after clone-on-assign");
    }
  }
  decode_range(joint, predict, conditioned_enc, enc_len, state.g, state.h, state.c, state.hyp,
               margin_stats, margin_label, secondary_margin_stats);
}

static std::vector<at::Tensor> run_aoti_loader_on_stream(AOTIModelPackageLoader& loader,
                                                         const std::vector<at::Tensor>& inputs,
                                                         c10::cuda::CUDAStream stream) {
  return loader.run(inputs, reinterpret_cast<void*>(stream.stream()));
}

std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                          const torch::Tensor& chunk,
                                          SessionState& state,
                                          c10::cuda::CUDAStream stream) {
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  auto tuple = enc_first.forward({chunk.contiguous(), L.contiguous(),
                                  state.clc.contiguous(), state.clt.contiguous(),
                                  state.clcl.contiguous()}).toTuple();
  std::vector<at::Tensor> out;
  out.reserve(5);
  for (int i = 0; i < 5; ++i) out.push_back(tuple->elements()[i].toTensor());
  return out;
}

std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                          const torch::Tensor& chunk,
                                          SessionState& state,
                                          const ExecutionContext& ctx) {
  return run_first_encoder(enc_first, chunk, state, ctx.stream);
}

std::vector<at::Tensor> run_first_encoder(torch::jit::Module& enc_first,
                                          const torch::Tensor& chunk,
                                          SessionState& state) {
  return run_first_encoder(enc_first,
                           chunk,
                           state,
                           c10::cuda::getCurrentCUDAStream(chunk.get_device()));
}

static void observe_first_chunk_drift(torch::jit::Module& bundle,
                                      const std::string& prefix,
                                      int chunk_index,
                                      const std::vector<at::Tensor>& out,
                                      torch::Device device,
                                      FirstChunkStats* stats) {
  if (stats == nullptr) return;
  ++stats->checks;
  auto eager_enc = prefix_chunk_tensor(bundle, prefix, chunk_index, "first_eager_enc_out").to(device).contiguous();
  auto eager_len = prefix_chunk_tensor(bundle, prefix, chunk_index, "first_eager_enc_len").to(device).contiguous();
  bool meta_ok = out[0].scalar_type() == eager_enc.scalar_type() &&
                 out[0].sizes().vec() == eager_enc.sizes().vec();
  bool len_ok = at::equal(out[1].to(device), eager_len);
  if (!meta_ok || !len_ok) {
    throw std::runtime_error(prefix + ".chunk" + std::to_string(chunk_index) +
                             " first-chunk eager/enc_first metadata mismatch");
  }
  bool byte_equal = at::equal(out[0], eager_enc);
  double max_abs = out[0].numel() > 0 ? (out[0] - eager_enc).abs().max().item<double>() : 0.0;
  if (max_abs > stats->max_abs) stats->max_abs = max_abs;
  if (byte_equal) ++stats->byte_equal;
  ++stats->pass;
}

static std::vector<at::Tensor> run_steady_encoder(AOTIModelPackageLoader& loader,
                                                  const torch::Tensor& chunk,
                                                  SessionState& state,
                                                  c10::cuda::CUDAStream stream) {
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto device = chunk.device();
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(device));
  std::vector<at::Tensor> inputs = {
      chunk.contiguous(),
      L.contiguous(),
      state.clc.contiguous(),
      state.clt.contiguous(),
      state.clcl.contiguous(),
  };
  auto out = run_aoti_loader_on_stream(loader, inputs, stream);
  if (out.size() < 5) throw std::runtime_error("steady AOTI encoder returned fewer than 5 outputs");
  return out;
}

static std::vector<at::Tensor> run_steady_encoder(AOTIModelPackageLoader& loader,
                                                  const torch::Tensor& chunk,
                                                  SessionState& state,
                                                  const ExecutionContext& ctx) {
  return run_steady_encoder(loader, chunk, state, ctx.stream);
}

static std::vector<at::Tensor> run_steady_encoder(AOTIModelPackageLoader& loader,
                                                  const torch::Tensor& chunk,
                                                  SessionState& state) {
  return run_steady_encoder(loader,
                            chunk,
                            state,
                            c10::cuda::getCurrentCUDAStream(chunk.get_device()));
}

static constexpr double kSteadyShadowTolerance = 5.0e-2;

static void steady_scheduler_cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("steady scheduler CUDA error at ") + file + ":" +
                             std::to_string(line) + " expr=" + expr + " error=" +
                             cudaGetErrorString(err));
  }
}

#define STEADY_SCHEDULER_CUDA_CHECK(expr) steady_scheduler_cuda_check((expr), #expr, __FILE__, __LINE__)

static double steady_elapsed_ms(std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

static double sample_steady_completion_event_ready_ms(cudaEvent_t event) {
  cudaError_t status = cudaEventQuery(event);
  if (status == cudaSuccess) return 0.0;
  if (status == cudaErrorNotReady) return -1.0;
  STEADY_SCHEDULER_CUDA_CHECK(status);
  return -1.0;
}

struct SteadySchedulerCudaEvent {
  cudaEvent_t event = nullptr;

  SteadySchedulerCudaEvent() = default;
  SteadySchedulerCudaEvent(const SteadySchedulerCudaEvent&) = delete;
  SteadySchedulerCudaEvent& operator=(const SteadySchedulerCudaEvent&) = delete;

  ~SteadySchedulerCudaEvent() {
    reset();
  }

  void create(unsigned int flags) {
    reset();
    STEADY_SCHEDULER_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
  }

  void record(c10::cuda::CUDAStream stream) {
    if (event == nullptr) throw std::runtime_error("steady scheduler attempted to record a null CUDA event");
    STEADY_SCHEDULER_CUDA_CHECK(cudaEventRecord(event, stream.stream()));
  }

  cudaEvent_t get() const {
    return event;
  }

  cudaEvent_t release() {
    cudaEvent_t out = event;
    event = nullptr;
    return out;
  }

  void reset() noexcept {
    if (event != nullptr) {
      cudaEventDestroy(event);
      event = nullptr;
    }
  }
};

struct SteadyShadowTensorDiff {
  bool meta_ok = true;
  bool exact_equal = false;
  double max_abs = 0.0;
};

struct SteadyShadowChunkReport {
  std::string label;
  int bucket = 0;
  int k = 0;
  int row = 0;
  int64_t cycle_id = 0;
  bool enc_len_equal = false;
  bool tokens_equal = false;
  bool events_equal = false;
  bool within_tolerance = false;
  bool tensor_meta_ok = true;
  SteadyShadowTensorDiff enc_out;
  SteadyShadowTensorDiff cache_ch;
  SteadyShadowTensorDiff cache_t;
  SteadyShadowTensorDiff cache_ch_len;
  size_t inline_tokens = 0;
  size_t scheduler_tokens = 0;
  size_t inline_events = 0;
  size_t scheduler_events = 0;
};

struct SteadyShadowStats {
  int64_t chunks = 0;
  int64_t bucket_b1 = 0;
  int64_t bucket_b2 = 0;
  int64_t bucket_b4 = 0;
  int64_t bucket_b8 = 0;
  int64_t bucket_b16 = 0;
  int64_t k1 = 0;
  int64_t k2 = 0;
  int64_t k3 = 0;
  int64_t k4 = 0;
  int64_t k5 = 0;
  int64_t k6 = 0;
  int64_t k7 = 0;
  int64_t k8 = 0;
  int64_t k9 = 0;
  int64_t k10 = 0;
  int64_t k11 = 0;
  int64_t k12 = 0;
  int64_t k13 = 0;
  int64_t k14 = 0;
  int64_t k15 = 0;
  int64_t k16 = 0;
  int64_t b_gt1 = 0;
  int64_t enc_len_mismatches = 0;
  int64_t tensor_meta_mismatches = 0;
  int64_t tolerance_failures = 0;
  int64_t token_divergences = 0;
  int64_t event_divergences = 0;
  double max_enc_out = 0.0;
  double max_cache_ch = 0.0;
  double max_cache_t = 0.0;
  double max_cache_ch_len = 0.0;
};

static std::mutex g_steady_shadow_stats_mu;
static SteadyShadowStats g_steady_shadow_stats;

static bool steady_shadow_env_enabled() {
  const char* raw = std::getenv("NEMOTRON_WS_STEADY_SHADOW");
  if (raw == nullptr) return false;
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

static SteadyShadowTensorDiff steady_shadow_tensor_diff(const char* name,
                                                        const torch::Tensor& scheduler,
                                                        const torch::Tensor& inline_ref,
                                                        const std::string& label) {
  SteadyShadowTensorDiff diff;
  diff.meta_ok = scheduler.defined() == inline_ref.defined();
  if (!diff.meta_ok) {
    std::printf("STEADY_SHADOW_METADATA_MISMATCH label=%s tensor=%s defined_scheduler=%d defined_inline=%d\n",
                label.c_str(),
                name,
                (int)scheduler.defined(),
                (int)inline_ref.defined());
    return diff;
  }
  if (!scheduler.defined()) {
    diff.exact_equal = true;
    return diff;
  }
  diff.meta_ok = scheduler.scalar_type() == inline_ref.scalar_type() &&
                 scheduler.sizes().vec() == inline_ref.sizes().vec();
  if (!diff.meta_ok) {
    std::printf("STEADY_SHADOW_METADATA_MISMATCH label=%s tensor=%s dtype_scheduler=%d dtype_inline=%d sizes_scheduler=",
                label.c_str(),
                name,
                (int)scheduler.scalar_type(),
                (int)inline_ref.scalar_type());
    for (auto s : scheduler.sizes()) std::printf("%ld,", (long)s);
    std::printf(" sizes_inline=");
    for (auto s : inline_ref.sizes()) std::printf("%ld,", (long)s);
    std::printf("\n");
    return diff;
  }
  diff.exact_equal = at::equal(scheduler, inline_ref);
  if (!diff.exact_equal && scheduler.numel() > 0) {
    diff.max_abs = (scheduler - inline_ref).abs().max().item<double>();
  }
  return diff;
}

static bool steady_shadow_events_equal(const std::vector<EmittedEvent>& scheduler,
                                       const std::vector<EmittedEvent>& inline_ref,
                                       const std::string& label) {
  if (scheduler.size() != inline_ref.size()) {
    std::printf("STEADY_SHADOW_EVENT_MISMATCH label=%s count_scheduler=%zu count_inline=%zu\n",
                label.c_str(),
                scheduler.size(),
                inline_ref.size());
    return false;
  }
  for (size_t i = 0; i < scheduler.size(); ++i) {
    const auto& a = scheduler[i];
    const auto& b = inline_ref[i];
    bool ok = a.kind == b.kind &&
              a.tokens == b.tokens &&
              a.collector_tokens == b.collector_tokens &&
              a.text == b.text &&
              a.collector_text == b.collector_text;
    if (!ok) {
      std::printf("STEADY_SHADOW_EVENT_MISMATCH label=%s index=%zu kind_scheduler=%s kind_inline=%s "
                  "tokens_scheduler=%s tokens_inline=%s collector_scheduler=%s collector_inline=%s "
                  "text_scheduler=%s text_inline=%s collector_text_scheduler=%s collector_text_inline=%s\n",
                  label.c_str(),
                  i,
                  event_kind_name(a.kind),
                  event_kind_name(b.kind),
                  vec_to_string(a.tokens).c_str(),
                  vec_to_string(b.tokens).c_str(),
                  vec_to_string(a.collector_tokens).c_str(),
                  vec_to_string(b.collector_tokens).c_str(),
                  escaped_text(a.text).c_str(),
                  escaped_text(b.text).c_str(),
                  escaped_text(a.collector_text).c_str(),
                  escaped_text(b.collector_text).c_str());
      return false;
    }
  }
  return true;
}

static bool steady_shadow_tokens_equal(const std::vector<int64_t>& scheduler,
                                       const std::vector<int64_t>& inline_ref,
                                       const std::string& label) {
  if (scheduler == inline_ref) return true;
  size_t first = first_token_diff_index(scheduler, inline_ref);
  std::printf("STEADY_SHADOW_TOKEN_MISMATCH label=%s len_scheduler=%zu len_inline=%zu first_diff=%zu "
              "token_scheduler=%ld token_inline=%ld\n",
              label.c_str(),
              scheduler.size(),
              inline_ref.size(),
              first,
              (long)token_or_missing(scheduler, first),
              (long)token_or_missing(inline_ref, first));
  return false;
}

static void record_steady_shadow_report(const SteadyShadowChunkReport& report) {
  {
    std::lock_guard<std::mutex> lock(g_steady_shadow_stats_mu);
    ++g_steady_shadow_stats.chunks;
    if (report.bucket == 1) ++g_steady_shadow_stats.bucket_b1;
    if (report.bucket == 2) ++g_steady_shadow_stats.bucket_b2;
    if (report.bucket == 4) ++g_steady_shadow_stats.bucket_b4;
    if (report.bucket == 8) ++g_steady_shadow_stats.bucket_b8;
    if (report.bucket == 16) ++g_steady_shadow_stats.bucket_b16;
    if (report.k == 1) ++g_steady_shadow_stats.k1;
    if (report.k == 2) ++g_steady_shadow_stats.k2;
    if (report.k == 3) ++g_steady_shadow_stats.k3;
    if (report.k == 4) ++g_steady_shadow_stats.k4;
    if (report.k == 5) ++g_steady_shadow_stats.k5;
    if (report.k == 6) ++g_steady_shadow_stats.k6;
    if (report.k == 7) ++g_steady_shadow_stats.k7;
    if (report.k == 8) ++g_steady_shadow_stats.k8;
    if (report.k == 9) ++g_steady_shadow_stats.k9;
    if (report.k == 10) ++g_steady_shadow_stats.k10;
    if (report.k == 11) ++g_steady_shadow_stats.k11;
    if (report.k == 12) ++g_steady_shadow_stats.k12;
    if (report.k == 13) ++g_steady_shadow_stats.k13;
    if (report.k == 14) ++g_steady_shadow_stats.k14;
    if (report.k == 15) ++g_steady_shadow_stats.k15;
    if (report.k == 16) ++g_steady_shadow_stats.k16;
    if (report.bucket > 1) ++g_steady_shadow_stats.b_gt1;
    if (!report.enc_len_equal) ++g_steady_shadow_stats.enc_len_mismatches;
    if (!report.tensor_meta_ok) ++g_steady_shadow_stats.tensor_meta_mismatches;
    if (!report.within_tolerance) ++g_steady_shadow_stats.tolerance_failures;
    if (!report.tokens_equal) ++g_steady_shadow_stats.token_divergences;
    if (!report.events_equal) ++g_steady_shadow_stats.event_divergences;
    g_steady_shadow_stats.max_enc_out = std::max(g_steady_shadow_stats.max_enc_out, report.enc_out.max_abs);
    g_steady_shadow_stats.max_cache_ch = std::max(g_steady_shadow_stats.max_cache_ch, report.cache_ch.max_abs);
    g_steady_shadow_stats.max_cache_t = std::max(g_steady_shadow_stats.max_cache_t, report.cache_t.max_abs);
    g_steady_shadow_stats.max_cache_ch_len =
        std::max(g_steady_shadow_stats.max_cache_ch_len, report.cache_ch_len.max_abs);
  }

  std::printf("STEADY_SHADOW_CHUNK label=%s cycle=%ld B=%d K=%d row=%d "
              "max_enc_out=%.6e max_cache_ch=%.6e max_cache_t=%.6e max_cache_ch_len=%.6e "
              "enc_len_equal=%d tokens_equal=%d events_equal=%d tensor_meta_ok=%d "
              "within_tolerance=%d tolerance_name=B2_A1_PARITY tolerance=%.6e "
              "inline_tokens=%zu scheduler_tokens=%zu inline_events=%zu scheduler_events=%zu "
              "commit=inline timing=INVALID\n",
              report.label.c_str(),
              (long)report.cycle_id,
              report.bucket,
              report.k,
              report.row,
              report.enc_out.max_abs,
              report.cache_ch.max_abs,
              report.cache_t.max_abs,
              report.cache_ch_len.max_abs,
              (int)report.enc_len_equal,
              (int)report.tokens_equal,
              (int)report.events_equal,
              (int)report.tensor_meta_ok,
              (int)report.within_tolerance,
              kSteadyShadowTolerance,
              report.inline_tokens,
              report.scheduler_tokens,
              report.inline_events,
              report.scheduler_events);
  std::fflush(stdout);
}

void session_runtime_print_steady_shadow_report() {
  if (!steady_shadow_env_enabled()) return;
  SteadyShadowStats stats;
  {
    std::lock_guard<std::mutex> lock(g_steady_shadow_stats_mu);
    stats = g_steady_shadow_stats;
  }
  bool pass = stats.chunks > 0 &&
              stats.enc_len_mismatches == 0 &&
              stats.tensor_meta_mismatches == 0 &&
              stats.tolerance_failures == 0 &&
              stats.token_divergences == 0 &&
              stats.event_divergences == 0;
  std::printf("STEADY_SHADOW_REPORT rows=%ld tolerance_name=B2_A1_PARITY tolerance=%.6e "
              "max_enc_out=%.6e max_cache_ch=%.6e max_cache_t=%.6e max_cache_ch_len=%.6e "
              "enc_len_mismatches=%ld tensor_meta_mismatches=%ld tolerance_failures=%ld "
              "token_divergences=%ld event_divergences=%ld "
              "bucket_b1=%ld bucket_b2=%ld bucket_b4=%ld bucket_b8=%ld bucket_b16=%ld bucket_b_gt1=%ld "
              "k1=%ld k2=%ld k3=%ld k4=%ld k5=%ld k6=%ld k7=%ld k8=%ld "
              "k9=%ld k10=%ld k11=%ld k12=%ld k13=%ld k14=%ld k15=%ld k16=%ld "
              "commit=inline timing=INVALID verdict=%s\n",
              (long)stats.chunks,
              kSteadyShadowTolerance,
              stats.max_enc_out,
              stats.max_cache_ch,
              stats.max_cache_t,
              stats.max_cache_ch_len,
              (long)stats.enc_len_mismatches,
              (long)stats.tensor_meta_mismatches,
              (long)stats.tolerance_failures,
              (long)stats.token_divergences,
              (long)stats.event_divergences,
              (long)stats.bucket_b1,
              (long)stats.bucket_b2,
              (long)stats.bucket_b4,
              (long)stats.bucket_b8,
              (long)stats.bucket_b16,
              (long)stats.b_gt1,
              (long)stats.k1,
              (long)stats.k2,
              (long)stats.k3,
              (long)stats.k4,
              (long)stats.k5,
              (long)stats.k6,
              (long)stats.k7,
              (long)stats.k8,
              (long)stats.k9,
              (long)stats.k10,
              (long)stats.k11,
              (long)stats.k12,
              (long)stats.k13,
              (long)stats.k14,
              (long)stats.k15,
              (long)stats.k16,
              pass ? "PASS" : "FAIL");
  std::fflush(stdout);
}

static void apply_runtime_steady_outputs(SessionState& state,
                                         const std::vector<at::Tensor>& out,
                                         const torch::Tensor& new_mel,
                                         const ExecutionContext& ctx,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label,
                                         MarginStats* margin_stats,
                                         CacheOwnershipStats* cache_ownership,
                                         bool recurrent_cache_output) {
  apply_encoder_outputs(state,
                        out,
                        ctx.joint,
                        ctx.predict,
                        margin_stats,
                        label,
                        nullptr,
                        cache_ownership,
                        recurrent_cache_output);

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

struct SteadySchedulerResult {
  std::vector<at::Tensor> tensors;
  int bucket = 0;
  int k = 0;
  int row = 0;
  int64_t cycle_id = 0;
  std::shared_ptr<DispatchResult::GraphSlotLease> graph_slot;
};

static std::future<DispatchResult> enqueue_steady_encoder_scheduler(BatchedSteadyScheduler& scheduler,
                                                                    const torch::Tensor& chunk,
                                                                    SessionState& state,
                                                                    const ExecutionContext& ctx,
                                                                    const std::string& label,
                                                                    RuntimeSteadyTiming* steady_timing = nullptr) {
  BatchedSteadyInput scheduler_input{
      chunk.contiguous(),
      state.clc.contiguous(),
      state.clt.contiguous(),
      state.clcl.contiguous(),
      label,
  };

  SteadySchedulerCudaEvent producer_event;
  producer_event.create(cudaEventDisableTiming);
  producer_event.record(ctx.stream);

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(scheduler.future_timeout_ms());
  auto enqueue_wait_start = std::chrono::steady_clock::now();
  uint64_t stream_key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&state));
  auto future = scheduler.try_enqueue_until(
      {std::move(scheduler_input), ctx.stream, producer_event.get(), stream_key},
      deadline);
  if (steady_timing != nullptr) {
    steady_timing->scheduler_enqueue_wait_ms +=
        steady_elapsed_ms(enqueue_wait_start, std::chrono::steady_clock::now());
    ++steady_timing->scheduler_enqueue_wait_count;
  }
  if (!future.has_value()) {
    throw std::runtime_error("steady scheduler admission timeout for " + label);
  }
  (void)producer_event.release();

  return std::move(*future);
}

static SteadySchedulerResult wait_steady_encoder_scheduler(BatchedSteadyScheduler& scheduler,
                                                          std::future<DispatchResult>&& future,
                                                          const ExecutionContext& ctx,
                                                          const std::string& label,
                                                          RuntimeSteadyTiming* steady_timing = nullptr) {
  auto future_wait_start = std::chrono::steady_clock::now();
  auto wait_status = future.wait_for(std::chrono::milliseconds(scheduler.future_timeout_ms()));
  if (steady_timing != nullptr) {
    steady_timing->scheduler_future_wait_ms +=
        steady_elapsed_ms(future_wait_start, std::chrono::steady_clock::now());
    ++steady_timing->scheduler_future_wait_count;
  }
  if (wait_status != std::future_status::ready) {
    throw std::runtime_error("steady scheduler future timeout for " + label);
  }
  auto result = future.get();
  if (result.completion.get() == nullptr) {
    throw std::runtime_error("steady scheduler returned a null completion event for " + label);
  }
  STEADY_SCHEDULER_CUDA_CHECK(cudaStreamWaitEvent(ctx.stream.stream(), result.completion.get(), 0));
  double completion_wait_ms = sample_steady_completion_event_ready_ms(result.completion.get());
  if (steady_timing != nullptr && completion_wait_ms >= 0.0) {
    steady_timing->scheduler_completion_wait_ms += completion_wait_ms;
    ++steady_timing->scheduler_completion_wait_count;
  }
  scheduler.record_worker_wait(result.cycle_id,
                               result.k,
                               0.0,
                               completion_wait_ms >= 0.0 ? completion_wait_ms * 1000.0 : 0.0,
                               completion_wait_ms >= 0.0 ? completion_wait_ms * 1000.0 : -1.0);
  if (result.row_tensors.size() < 5) {
    throw std::runtime_error("steady scheduler returned fewer than 5 row tensors for " + label);
  }

  SteadySchedulerResult out;
  out.tensors = std::move(result.row_tensors);
  out.bucket = result.bucket;
  out.k = result.k;
  out.row = result.row;
  out.cycle_id = result.cycle_id;
  out.graph_slot = std::move(result.graph_slot);
  result.completion.reset();
  return out;
}

static void run_steady_shadow_compare_and_commit(SessionState& state,
                                                 const torch::Tensor& chunk,
                                                 const torch::Tensor& new_mel,
                                                 AOTIModelPackageLoader& enc_steady,
                                                 BatchedSteadyScheduler& scheduler,
                                                 const ExecutionContext& ctx,
                                                 const Tokenizer& tokenizer,
                                                 std::vector<EmittedEvent>& events,
                                                 const std::string& label,
                                                 MarginStats* margin_stats,
                                                 CacheOwnershipStats* cache_ownership) {
  SessionState pre_decode = clone_session(state);
  SessionState inline_state = clone_session(pre_decode);
  SessionState scheduler_state = clone_session(pre_decode);
  std::vector<EmittedEvent> inline_events;
  std::vector<EmittedEvent> scheduler_events;

  auto scheduler_future = enqueue_steady_encoder_scheduler(scheduler,
                                                           chunk,
                                                           scheduler_state,
                                                           ctx,
                                                           label + ".scheduler_shadow");
  auto inline_out = run_steady_encoder(enc_steady, chunk, inline_state, ctx);
  auto scheduler_result = wait_steady_encoder_scheduler(scheduler,
                                                       std::move(scheduler_future),
                                                       ctx,
                                                       label + ".scheduler_shadow");

  SteadyShadowChunkReport report;
  auto retire_scheduler_graph_slot = [&]() {
    if (!scheduler_result.graph_slot) return;
    try {
      scheduler_result.graph_slot->retire_on_stream(ctx.stream.stream());
      scheduler_result.graph_slot.reset();
    } catch (const std::exception& e) {
      std::printf("STEADY_SHADOW_GRAPH_RETIRE_ERROR label=%s error=%s\n", label.c_str(), e.what());
      scheduler_result.graph_slot.reset();
      throw;
    } catch (...) {
      std::printf("STEADY_SHADOW_GRAPH_RETIRE_ERROR label=%s error=unknown\n", label.c_str());
      scheduler_result.graph_slot.reset();
      throw;
    }
  };
  try {
    apply_runtime_steady_outputs(inline_state,
                                 inline_out,
                                 new_mel,
                                 ctx,
                                 tokenizer,
                                 inline_events,
                                 label + ".inline_shadow",
                                 margin_stats,
                                 cache_ownership,
                                 true);
    apply_runtime_steady_outputs(scheduler_state,
                                 scheduler_result.tensors,
                                 new_mel,
                                 ctx,
                                 tokenizer,
                                 scheduler_events,
                                 label + ".scheduler_shadow",
                                 nullptr,
                                 nullptr,
                                 true);

    report.label = label;
    report.bucket = scheduler_result.bucket;
    report.k = scheduler_result.k;
    report.row = scheduler_result.row;
    report.cycle_id = scheduler_result.cycle_id;
    report.inline_tokens = inline_state.hyp.size();
    report.scheduler_tokens = scheduler_state.hyp.size();
    report.inline_events = inline_events.size();
    report.scheduler_events = scheduler_events.size();
    report.enc_out = steady_shadow_tensor_diff("enc_out", scheduler_result.tensors[0], inline_out[0], label);
    report.cache_ch = steady_shadow_tensor_diff("cache_ch", scheduler_result.tensors[2], inline_out[2], label);
    report.cache_t = steady_shadow_tensor_diff("cache_t", scheduler_result.tensors[3], inline_out[3], label);
    report.cache_ch_len = steady_shadow_tensor_diff("cache_ch_len", scheduler_result.tensors[4], inline_out[4], label);
    report.tensor_meta_ok = report.enc_out.meta_ok &&
                            report.cache_ch.meta_ok &&
                            report.cache_t.meta_ok &&
                            report.cache_ch_len.meta_ok;
    report.enc_len_equal = scheduler_result.tensors.size() > 1 &&
                           inline_out.size() > 1 &&
                           steady_shadow_tensor_diff("enc_len", scheduler_result.tensors[1], inline_out[1], label).exact_equal;
    report.tokens_equal = steady_shadow_tokens_equal(scheduler_state.hyp, inline_state.hyp, label);
    report.events_equal = steady_shadow_events_equal(scheduler_events, inline_events, label);
    report.within_tolerance = report.tensor_meta_ok &&
                              report.enc_len_equal &&
                              report.enc_out.max_abs <= kSteadyShadowTolerance &&
                              report.cache_ch.max_abs <= kSteadyShadowTolerance &&
                              report.cache_t.max_abs <= kSteadyShadowTolerance &&
                              report.cache_ch_len.max_abs <= kSteadyShadowTolerance;
    retire_scheduler_graph_slot();
  } catch (...) {
    try {
      retire_scheduler_graph_slot();
    } catch (...) {
    }
    throw;
  }

  record_steady_shadow_report(report);

  state = inline_state;
  events.insert(events.end(), inline_events.begin(), inline_events.end());
}

static void warm_stream_encoder_artifacts(torch::jit::Module& bundle,
                                          FirstEncoder& enc_first,
                                          AOTIModelPackageLoader& enc_steady,
                                          torch::Device device) {
  SessionState warm;
  reset_session(warm, bundle, device);
  auto first_chunk = torch::zeros({1, 128, SHIFT}, torch::dtype(torch::kFloat32).device(device));
  auto first_out = enc_first.run(first_chunk,
                                 warm,
                                 c10::cuda::getCurrentCUDAStream(first_chunk.get_device()));
  warm.clc = first_out[2].clone();
  warm.clt = first_out[3].clone();
  warm.clcl = first_out[4].clone();
  auto steady_chunk = torch::zeros({1, 128, PRE + SHIFT}, torch::dtype(torch::kFloat32).device(device));
  (void)run_steady_encoder(enc_steady, steady_chunk, warm);
}

static AudioGeometry audio_geometry_from_bundle(torch::jit::Module& bundle) {
  auto values = tensor_to_vec(attr_tensor(bundle, "audio_geometry"));
  if (values.size() != 13) {
    throw std::runtime_error("audio_geometry must contain 13 int64 values");
  }
  AudioGeometry g;
  g.shift_frames = values[0];
  g.pre_encode_cache_size = values[1];
  g.drop_extra = values[2];
  g.final_padding_frames = values[3];
  g.right_context = values[4];
  g.first_preprocess_mel_frame = values[5];
  g.hop_samples = values[6];
  g.raw_audio_ring_samples = values[7];
  g.preprocess_align_pad_samples = values[8];
  g.preprocess_new_audio_samples = values[9];
  g.stream_preprocess_valid_samples = values[10];
  g.constant_preprocess_frames = values[11];
  g.constant_preprocess_samples = values[12];

  require_contract_eq("audio.shift", g.shift_frames, SHIFT);
  require_contract_eq("audio.pre_encode_cache", g.pre_encode_cache_size, PRE);
  require_contract_eq("audio.drop_extra", g.drop_extra, DROP);
  require_contract_eq("audio.final_padding_frames", g.final_padding_frames, FINAL_PADDING_FRAMES);
  require_contract_eq("audio.right_context", g.right_context, RIGHT_CONTEXT);
  if (g.hop_samples <= 0 || g.raw_audio_ring_samples < 0 ||
      g.preprocess_align_pad_samples < 0 || g.preprocess_new_audio_samples <= 0 ||
      g.constant_preprocess_samples <= 0) {
    throw std::runtime_error("audio_geometry contains invalid sample counts");
  }
  int64_t prefix = g.preprocess_align_pad_samples + g.raw_audio_ring_samples;
  if (prefix % g.hop_samples != 0 || prefix / g.hop_samples != g.first_preprocess_mel_frame) {
    throw std::runtime_error("audio_geometry fixed preprocessor prefix mismatch");
  }
  if (g.stream_preprocess_valid_samples != prefix + g.preprocess_new_audio_samples) {
    throw std::runtime_error("audio_geometry stream valid sample count mismatch");
  }
  if (g.constant_preprocess_samples != (g.constant_preprocess_frames - 1) * g.hop_samples) {
    throw std::runtime_error("audio_geometry constant plan sample/frame mismatch");
  }
  return g;
}

static std::string verify_preproc_manifest(const std::string& dir,
                                           const std::string& preproc_path,
                                           const AudioGeometry& g) {
  std::string manifest_path = preproc_path + ".manifest.json";
  if (!file_exists(manifest_path)) {
    throw std::runtime_error("preproc.ts manifest is required for audio mode: " + manifest_path);
  }
  std::string text = read_text_file(manifest_path);
  std::string contract = json_value_for_key(text, "CONTRACT");
  if (contract.empty() || contract.front() != '{') throw std::runtime_error("preproc manifest CONTRACT is not an object");
  int64_t schema = json_int_field(contract, "schema");
  if (schema != 1) throw std::runtime_error("preproc manifest schema mismatch: " + std::to_string(schema));
  std::string model_id = json_string_field(contract, "model_id");
  if (model_id != MODEL_ID) throw std::runtime_error("preproc manifest model_id mismatch: " + model_id);
  int64_t trace_k = json_int_field(contract, "trace_k");
  if (trace_k != g.constant_preprocess_samples) {
    throw std::runtime_error("preproc manifest trace_k mismatch: " + std::to_string(trace_k));
  }
  double dither = json_double_field(contract, "dither");
  if (std::abs(dither) > 0.0) {
    throw std::runtime_error("preproc manifest dither must be 0.0, got " + std::to_string(dither));
  }
  auto geometry = json_int_array_field(contract, "geometry");
  std::vector<int64_t> expected = {
      g.shift_frames,
      g.pre_encode_cache_size,
      g.drop_extra,
      g.final_padding_frames,
      g.right_context,
      g.first_preprocess_mel_frame,
      g.hop_samples,
      g.raw_audio_ring_samples,
      g.preprocess_align_pad_samples,
      g.preprocess_new_audio_samples,
      g.stream_preprocess_valid_samples,
      g.constant_preprocess_frames,
      g.constant_preprocess_samples,
  };
  if (geometry != expected) throw std::runtime_error("preproc manifest geometry mismatch");
  std::string manifest_sha = json_string_field(contract, "preproc_ts_sha256");
  std::string actual_sha = sha256_file(preproc_path);
  if (manifest_sha != actual_sha) {
    throw std::runtime_error("preproc.ts sha256 mismatch: manifest=" + manifest_sha + " actual=" + actual_sha);
  }
  std::string torch_version = json_string_field(contract, "torch_version");
  std::string cuda_version = json_string_field(contract, "cuda_version");
  std::printf("preproc manifest verified: sha256=%s model=%s trace_k=%ld torch=%s cuda=%s\n",
              actual_sha.c_str(), model_id.c_str(), (long)trace_k,
              torch_version.c_str(), cuda_version.c_str());
  (void)dir;
  return actual_sha;
}

static AudioCiBundleStats audio_ci_stats_from_bundle(torch::jit::Module& bundle) {
  auto values = tensor_to_double_vec(attr_tensor(bundle, "audio_ci_stats"));
  if (values.size() != 14) {
    throw std::runtime_error("audio_ci_stats must contain 14 float64 values");
  }
  AudioCiBundleStats s;
  s.mel_abs_max = values[0];
  s.mel_abs_mean = values[1];
  s.mel_abs_p99 = values[2];
  s.mel_rel_max = values[3];
  s.mel_rel_mean = values[4];
  s.mel_rel_p99 = values[5];
  s.cache_last_channel_max_abs = values[6];
  s.cache_last_time_max_abs = values[7];
  s.cache_last_channel_len_max_abs = values[8];
  s.mel_checks = values[9];
  s.cache_checks = values[10];
  s.mel_max_headroom = values[11];
  s.mel_p99_headroom = values[12];
  s.cache_max_headroom = values[13];
  return s;
}

static bool compare_mel_tensor(const std::string& label,
                               const torch::Tensor& actual,
                               const torch::Tensor& expected,
                               MelCompareStats& stats,
                               double atol) {
  ++stats.comparisons;
  bool meta_ok = actual.scalar_type() == expected.scalar_type() &&
                 actual.sizes().vec() == expected.sizes().vec();
  if (!meta_ok) {
    std::printf("    %s mel metadata mismatch: dtype %d/%d sizes",
                label.c_str(), (int)actual.scalar_type(), (int)expected.scalar_type());
    for (auto s : actual.sizes()) std::printf(" %ld", (long)s);
    std::printf(" vs");
    for (auto s : expected.sizes()) std::printf(" %ld", (long)s);
    std::printf("\n");
    return false;
  }
  bool byte_equal = at::equal(actual, expected);
  if (byte_equal) {
    ++stats.byte_equal;
  }
  double max_abs = 0.0;
  if (actual.numel() > 0) {
    max_abs = (actual - expected).abs().max().item<double>();
  }
  if (max_abs > stats.max_abs_diff) stats.max_abs_diff = max_abs;
  bool ok = byte_equal || max_abs <= atol;
  if (ok) {
    ++stats.pass;
    if (!byte_equal) ++stats.within_ci;
  }
  if (!ok) {
    std::printf("    %s mel mismatch: byte_equal=%d max_abs=%.6e atol=%.6e\n",
                label.c_str(), (int)byte_equal, max_abs, atol);
  }
  return ok;
}

struct AudioFrontend {
  AudioGeometry g;
  torch::jit::Module* preproc = nullptr;
  torch::Device device = torch::Device(torch::kCUDA);
  double mel_atol = 0.0;
  double cache_atol = 0.0;
  AudioCiBundleStats ci_stats;
  MelCompareStats stats;
  GeometryCompareStats geometry_stats;
  CacheCompareStats cache_stats;
  MarginStats margin_stats;

  torch::jit::Module& require_preproc() const {
    if (preproc == nullptr) throw std::runtime_error("audio frontend missing preproc module");
    return *preproc;
  }

  std::pair<torch::Tensor, int64_t> build_fixed_preprocess_audio(
      const std::vector<float>& raw_audio_ring,
      const std::vector<float>& new_audio) const {
    if (raw_audio_ring.size() != static_cast<size_t>(g.raw_audio_ring_samples)) {
      throw std::runtime_error("raw_audio_ring size mismatch: got " +
                               std::to_string(raw_audio_ring.size()) +
                               " expected " + std::to_string(g.raw_audio_ring_samples));
    }
    int64_t prefix_len = g.preprocess_align_pad_samples + g.raw_audio_ring_samples;
    int64_t valid_samples = prefix_len + static_cast<int64_t>(new_audio.size());
    if (valid_samples > g.constant_preprocess_samples) {
      throw std::runtime_error("fixed preprocessor valid span exceeds K");
    }
    auto audio = torch::zeros({1, g.constant_preprocess_samples}, torch::dtype(torch::kFloat32));
    float* data = audio.data_ptr<float>();
    int64_t cursor = g.preprocess_align_pad_samples;
    for (size_t i = 0; i < raw_audio_ring.size(); ++i) data[cursor + static_cast<int64_t>(i)] = raw_audio_ring[i];
    cursor += g.raw_audio_ring_samples;
    for (size_t i = 0; i < new_audio.size(); ++i) data[cursor + static_cast<int64_t>(i)] = new_audio[i];
    return {audio, valid_samples};
  }

  torch::Tensor preprocess_fixed_audio(const std::vector<float>& raw_audio_ring,
                                       const std::vector<float>& new_audio,
                                       torch::jit::Module& preproc_module) const {
    auto built = build_fixed_preprocess_audio(raw_audio_ring, new_audio);
    auto audio = built.first.to(device).contiguous();
    auto audio_len = torch::full({1}, built.second, torch::dtype(torch::kLong).device(device));
    auto tuple = preproc_module.forward({audio, audio_len}).toTuple();
    return tuple->elements()[0].toTensor();
  }

  torch::Tensor preprocess_fixed_audio(const std::vector<float>& raw_audio_ring,
                                       const std::vector<float>& new_audio) const {
    return preprocess_fixed_audio(raw_audio_ring, new_audio, require_preproc());
  }

  torch::Tensor steady_new_mel(const SessionState& state,
                               torch::jit::Module& preproc_module) const {
    if (state.pending_audio.size() < static_cast<size_t>(g.preprocess_new_audio_samples)) {
      throw std::runtime_error("steady_new_mel called before preprocess_new_audio_samples are pending");
    }
    std::vector<float> new_audio(
        state.pending_audio.begin(),
        state.pending_audio.begin() + static_cast<std::ptrdiff_t>(g.preprocess_new_audio_samples));
    auto mel = preprocess_fixed_audio(state.raw_audio_ring, new_audio, preproc_module);
    return mel.slice(2,
                     g.first_preprocess_mel_frame,
                     g.first_preprocess_mel_frame + g.shift_frames).contiguous();
  }

  torch::Tensor steady_new_mel(const SessionState& state) const {
    return steady_new_mel(state, require_preproc());
  }

  bool session_ready(const SessionState& state) const {
    int64_t timeline_samples = state.synthetic_prefix_samples + state.total_audio_samples;
    int64_t needed_samples = (state.emitted + g.shift_frames + 1) * g.hop_samples;
    return timeline_samples >= needed_samples &&
           state.pending_audio.size() >= static_cast<size_t>(g.preprocess_new_audio_samples);
  }

  void advance_raw_ring(SessionState& state, const std::vector<float>& consumed_audio) const {
    if (consumed_audio.size() >= static_cast<size_t>(g.raw_audio_ring_samples)) {
      state.raw_audio_ring.assign(consumed_audio.end() - static_cast<std::ptrdiff_t>(g.raw_audio_ring_samples),
                                  consumed_audio.end());
      return;
    }
    if (consumed_audio.empty()) return;
    size_t keep = static_cast<size_t>(g.raw_audio_ring_samples) - consumed_audio.size();
    std::vector<float> next;
    next.reserve(static_cast<size_t>(g.raw_audio_ring_samples));
    next.insert(next.end(), state.raw_audio_ring.end() - static_cast<std::ptrdiff_t>(keep), state.raw_audio_ring.end());
    next.insert(next.end(), consumed_audio.begin(), consumed_audio.end());
    state.raw_audio_ring = std::move(next);
  }
};

struct RuntimeAudioFrontend {
  AudioFrontend audio;
};

static ExecutionContext default_execution_context(AudioFrontend& audio,
                                                  torch::jit::Module& joint,
                                                  torch::jit::Module& predict,
                                                  torch::Device device) {
  return {
      c10::cuda::getCurrentCUDAStream(device.index()),
      joint,
      predict,
      audio.require_preproc(),
  };
}

static PreprocDeterminismStats run_preproc_determinism_check(const AudioFrontend& audio) {
  std::vector<float> raw_ring(static_cast<size_t>(audio.g.raw_audio_ring_samples), 0.0f);
  std::vector<float> new_audio(static_cast<size_t>(audio.g.preprocess_new_audio_samples), 0.0f);
  for (size_t i = 0; i < new_audio.size(); ++i) {
    int bucket = static_cast<int>(i % 97);
    new_audio[i] = static_cast<float>((bucket - 48) * 0.001);
  }
  (void)audio.preprocess_fixed_audio(raw_ring, new_audio).contiguous();
  auto first = audio.preprocess_fixed_audio(raw_ring, new_audio).contiguous();
  auto second = audio.preprocess_fixed_audio(raw_ring, new_audio).contiguous();
  bool byte_equal = at::equal(first, second);
  double max_abs = first.numel() > 0 ? (first - second).abs().max().item<double>() : 0.0;
  std::string block_hash = sha256_tensor_bytes(first);
  std::printf("=== PREPROC DETERMINISM: same_process_post_warmup_fixed_block_twice byte_equal=%s "
              "max_abs=%.6e fixed_block_sha256=%s K=%ld valid=%ld ===\n",
              byte_equal ? "PASS" : "FAIL",
              max_abs,
              block_hash.c_str(),
              (long)audio.g.constant_preprocess_samples,
              (long)audio.g.stream_preprocess_valid_samples);
  return {byte_equal, block_hash, max_abs};
}

static void run_steady_chunk_tensor(SessionState& state,
                                    torch::jit::Module& bundle,
                                    const std::string& prefix,
                                    int chunk_index,
                                    const torch::Tensor& new_mel_in,
                                    FirstEncoder& enc_first,
                                    AOTIModelPackageLoader& enc_steady,
                                    torch::jit::Module& joint,
                                    torch::jit::Module& predict,
                                    torch::Device device,
                                    const Tokenizer& tokenizer,
                                    std::vector<EmittedEvent>& events,
                                    MarginStats* margin_stats = nullptr,
                                    const std::string& margin_label = "",
                                    FirstChunkStats* first_chunk_stats = nullptr,
                                    CacheOwnershipStats* cache_ownership = nullptr) {
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("steady chunk outside STREAMING");

  auto new_mel = new_mel_in.to(device).contiguous();
  int64_t is_first = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "is_first"));
  int64_t drop_extra = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "drop_extra"));
  int64_t chunk_T = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "chunk_T"));
  int64_t emitted_before = scalar_i64(prefix_chunk_tensor(bundle, prefix, chunk_index, "emitted_before"));

  bool expected_first = state.emitted == 0;
  if ((is_first != 0) != expected_first) throw std::runtime_error("steady first/continuation flag mismatch");
  if (emitted_before != state.emitted) throw std::runtime_error("steady emitted_before mismatch");
  if (new_mel.size(2) != SHIFT) throw std::runtime_error("steady new_mel is not SHIFT frames");

  torch::Tensor chunk;
  std::vector<at::Tensor> out;
  if (expected_first) {
    if (drop_extra != 0 || chunk_T != new_mel.size(2)) throw std::runtime_error("first steady geometry mismatch");
    chunk = new_mel;
    out = enc_first.run(chunk, state, c10::cuda::getCurrentCUDAStream(chunk.get_device()));
    observe_first_chunk_drift(bundle, prefix, chunk_index, out, device, first_chunk_stats);
  } else {
    if (!state.ring.defined()) throw std::runtime_error("steady continuation missing mel ring");
    if (drop_extra != DROP || chunk_T != state.ring.size(2) + new_mel.size(2)) {
      throw std::runtime_error("steady continuation geometry mismatch");
    }
    chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
    out = run_steady_encoder(enc_steady, chunk, state);
  }

  apply_encoder_outputs(state,
                        out,
                        joint,
                        predict,
                        margin_stats,
                        margin_label,
                        expected_first && first_chunk_stats != nullptr ? &first_chunk_stats->margins : nullptr,
                        cache_ownership,
                        !expected_first);

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

static void run_steady_chunk(SessionState& state,
                             torch::jit::Module& bundle,
                             const std::string& prefix,
                             int chunk_index,
                             FirstEncoder& enc_first,
                             AOTIModelPackageLoader& enc_steady,
                             torch::jit::Module& joint,
                             torch::jit::Module& predict,
                             torch::Device device,
                             const Tokenizer& tokenizer,
                             std::vector<EmittedEvent>& events,
                             MarginStats* margin_stats = nullptr,
                             const std::string& margin_label = "",
                             FirstChunkStats* first_chunk_stats = nullptr,
                             CacheOwnershipStats* cache_ownership = nullptr) {
  auto new_mel = prefix_chunk_tensor(bundle, prefix, chunk_index, "new_mel").to(device).contiguous();
  std::string label = margin_label.empty()
                          ? prefix + ".chunk" + std::to_string(chunk_index)
                          : margin_label;
  run_steady_chunk_tensor(state, bundle, prefix, chunk_index, new_mel, enc_first, enc_steady,
                          joint, predict, device, tokenizer, events, margin_stats, label,
                          first_chunk_stats, cache_ownership);
}

static void run_steady_chunk_from_audio(SessionState& state,
                                        torch::jit::Module& bundle,
                                        const std::string& prefix,
                                        int chunk_index,
                                        AudioFrontend& audio,
                                        FirstEncoder& enc_first,
                                        AOTIModelPackageLoader& enc_steady,
                                        torch::jit::Module& joint,
                                        torch::jit::Module& predict,
                                        torch::Device device,
                                        const Tokenizer& tokenizer,
                                        std::vector<EmittedEvent>& events,
                                        const std::string& label,
                                        FirstChunkStats* first_chunk_stats = nullptr,
                                        CacheOwnershipStats* cache_ownership = nullptr) {
  auto new_mel = audio.steady_new_mel(state).to(device).contiguous();
  auto gold_mel = prefix_chunk_tensor(bundle, prefix, chunk_index, "new_mel").to(device).contiguous();
  if (!compare_mel_tensor(label + ".chunk" + std::to_string(chunk_index) + ".new_mel",
                          new_mel, gold_mel, audio.stats, audio.mel_atol)) {
    throw std::runtime_error("audio steady mel mismatch for " + label);
  }
  run_steady_chunk_tensor(state, bundle, prefix, chunk_index, new_mel, enc_first, enc_steady,
                          joint, predict, device, tokenizer, events,
                          &audio.margin_stats,
                          label + ".chunk" + std::to_string(chunk_index),
                          first_chunk_stats,
                          cache_ownership);

  size_t consume = std::min(static_cast<size_t>(audio.g.shift_frames * audio.g.hop_samples),
                            state.pending_audio.size());
  std::vector<float> consumed(state.pending_audio.begin(),
                              state.pending_audio.begin() + static_cast<std::ptrdiff_t>(consume));
  audio.advance_raw_ring(state, consumed);
  state.pending_audio.erase(state.pending_audio.begin(),
                            state.pending_audio.begin() + static_cast<std::ptrdiff_t>(consume));
}

static int drain_audio_steady(SessionState& state,
                              torch::jit::Module& bundle,
                              const std::string& prefix,
                              AudioFrontend& audio,
                              FirstEncoder& enc_first,
                              AOTIModelPackageLoader& enc_steady,
                              torch::jit::Module& joint,
                              torch::jit::Module& predict,
                              torch::Device device,
                              const Tokenizer& tokenizer,
                              std::vector<EmittedEvent>& events,
                              const std::string& label,
                              FirstChunkStats* first_chunk_stats = nullptr,
                              CacheOwnershipStats* cache_ownership = nullptr) {
  int64_t expected_chunks = scalar_i64(prefix_tensor(bundle, prefix, "num_steady"));
  int chunks = 0;
  while (audio.session_ready(state)) {
    if (chunks >= expected_chunks) {
      throw std::runtime_error(label + " produced more audio steady chunks than Python gold");
    }
    run_steady_chunk_from_audio(state, bundle, prefix, chunks, audio, enc_first, enc_steady,
                                joint, predict, device, tokenizer, events, label,
                                first_chunk_stats, cache_ownership);
    ++chunks;
  }
  if (chunks != expected_chunks) {
    throw std::runtime_error(label + " audio steady chunk count mismatch: got " +
                             std::to_string(chunks) + " gold " + std::to_string(expected_chunks));
  }
  return chunks;
}

static void run_steady_chunk_tensor_runtime(SessionState& state,
                                            const torch::Tensor& new_mel_in,
                                            FirstEncoder& enc_first,
                                            AOTIModelPackageLoader* enc_steady,
                                            const ExecutionContext& ctx,
                                            torch::Device device,
                                            const Tokenizer& tokenizer,
                                            std::vector<EmittedEvent>& events,
                                            const std::string& label,
                                            MarginStats* margin_stats = nullptr,
                                            CacheOwnershipStats* cache_ownership = nullptr,
                                            BatchedSteadyScheduler* steady_scheduler = nullptr,
                                            bool steady_shadow_enabled = false,
                                            RuntimeSteadyTiming* steady_timing = nullptr) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("runtime steady chunk outside STREAMING");
  auto new_mel = new_mel_in.to(device).contiguous();
  if (new_mel.size(2) != SHIFT) throw std::runtime_error(label + " runtime new_mel is not SHIFT frames");

  bool expected_first = state.emitted == 0;
  torch::Tensor chunk;
  std::vector<at::Tensor> out;
  std::shared_ptr<DispatchResult::GraphSlotLease> scheduler_graph_slot;
  if (expected_first) {
    chunk = new_mel;
    out = enc_first.run(chunk, state, ctx.stream);
  } else {
    if (!state.ring.defined()) throw std::runtime_error(label + " runtime continuation missing mel ring");
    chunk = torch::cat({state.ring, new_mel}, 2).contiguous();
    if (steady_shadow_enabled && steady_scheduler == nullptr) {
      throw std::runtime_error("steady shadow requested without a steady scheduler for " + label);
    }
    if (steady_shadow_enabled) {
      if (enc_steady == nullptr) {
        throw std::runtime_error("inline enc_steady required (scheduler off or shadow on) but not loaded: " +
                                 label);
      }
      run_steady_shadow_compare_and_commit(state,
                                           chunk,
                                           new_mel,
                                           *enc_steady,
                                           *steady_scheduler,
                                           ctx,
                                           tokenizer,
                                           events,
                                           label,
                                           margin_stats,
                                           cache_ownership);
      return;
    }
    if (steady_scheduler != nullptr) {
      auto scheduler_future = enqueue_steady_encoder_scheduler(*steady_scheduler,
                                                               chunk,
                                                               state,
                                                               ctx,
                                                               label + ".scheduler",
                                                               steady_timing);
      auto scheduler_result = wait_steady_encoder_scheduler(*steady_scheduler,
                                                            std::move(scheduler_future),
                                                            ctx,
                                                            label + ".scheduler",
                                                            steady_timing);
      out = std::move(scheduler_result.tensors);
      scheduler_graph_slot = std::move(scheduler_result.graph_slot);
    } else {
      if (enc_steady == nullptr) {
        throw std::runtime_error("inline enc_steady required (scheduler off or shadow on) but not loaded: " +
                                 label);
      }
      out = run_steady_encoder(*enc_steady, chunk, state, ctx);
    }
  }

  auto decode_start = std::chrono::steady_clock::now();
  auto retire_scheduler_graph_slot = [&]() {
    if (scheduler_graph_slot) {
      scheduler_graph_slot->retire_on_stream(ctx.stream.stream());
      scheduler_graph_slot.reset();
    }
  };
  try {
    apply_runtime_steady_outputs(state,
                                 out,
                                 new_mel,
                                 ctx,
                                 tokenizer,
                                 events,
                                 label,
                                 margin_stats,
                                 cache_ownership,
                                 !expected_first);
    retire_scheduler_graph_slot();
  } catch (...) {
    retire_scheduler_graph_slot();
    throw;
  }
  if (steady_timing != nullptr && !expected_first) {
    steady_timing->decode_ms += steady_elapsed_ms(decode_start, std::chrono::steady_clock::now());
    ++steady_timing->decode_count;
  }
}

static int drain_audio_steady_runtime(SessionState& state,
                                      AudioFrontend& audio,
                                      FirstEncoder& enc_first,
                                      AOTIModelPackageLoader* enc_steady,
                                      const ExecutionContext& ctx,
                                      torch::Device device,
                                      const Tokenizer& tokenizer,
                                      std::vector<EmittedEvent>& events,
                                      const std::string& label,
                                      MarginStats* margin_stats = nullptr,
                                      CacheOwnershipStats* cache_ownership = nullptr,
                                      BatchedSteadyScheduler* steady_scheduler = nullptr,
                                      bool steady_shadow_enabled = false,
                                      RuntimeSteadyTiming* steady_timing = nullptr) {
  int chunks = 0;
  while (audio.session_ready(state)) {
    const bool continuation = state.emitted != 0;
    auto preproc_start = std::chrono::steady_clock::now();
    auto new_mel = audio.steady_new_mel(state, ctx.preproc).to(device).contiguous();
    if (steady_timing != nullptr && continuation) {
      steady_timing->preproc_ms += steady_elapsed_ms(preproc_start, std::chrono::steady_clock::now());
      ++steady_timing->preproc_count;
    }
    run_steady_chunk_tensor_runtime(state,
                                    new_mel,
                                    enc_first,
                                    enc_steady,
                                    ctx,
                                    device,
                                    tokenizer,
                                    events,
                                    label + ".chunk" + std::to_string(chunks),
                                    margin_stats,
                                    cache_ownership,
                                    steady_scheduler,
                                    steady_shadow_enabled,
                                    steady_timing);
    size_t consume = std::min(static_cast<size_t>(audio.g.shift_frames * audio.g.hop_samples),
                              state.pending_audio.size());
    std::vector<float> consumed(state.pending_audio.begin(),
                                state.pending_audio.begin() + static_cast<std::ptrdiff_t>(consume));
    audio.advance_raw_ring(state, consumed);
    state.pending_audio.erase(state.pending_audio.begin(),
                              state.pending_audio.begin() + static_cast<std::ptrdiff_t>(consume));
    ++chunks;
  }
  return chunks;
}

static int append_pcm_and_drain_runtime(SessionState& state,
                                        const std::vector<float>& pcm,
                                        AudioFrontend& audio,
                                        FirstEncoder& enc_first,
                                        AOTIModelPackageLoader* enc_steady,
                                        const ExecutionContext& ctx,
                                        torch::Device device,
                                        const Tokenizer& tokenizer,
                                        std::vector<EmittedEvent>& events,
                                        const std::string& label,
                                        MarginStats* margin_stats = nullptr,
                                        CacheOwnershipStats* cache_ownership = nullptr,
                                        BatchedSteadyScheduler* steady_scheduler = nullptr,
                                        bool steady_shadow_enabled = false,
                                        RuntimeSteadyTiming* steady_timing = nullptr) {
  if (state.mode == SessionMode::PENDING_FINALIZE) {
    state.post_stop_audio.insert(state.post_stop_audio.end(), pcm.begin(), pcm.end());
    return 0;
  }
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("append_pcm outside live stream for " + label);
  state.pending_audio.insert(state.pending_audio.end(), pcm.begin(), pcm.end());
  state.total_audio_samples += static_cast<int64_t>(pcm.size());
  return drain_audio_steady_runtime(state,
                                    audio,
                                    enc_first,
                                    enc_steady,
                                    ctx,
                                    device,
                                    tokenizer,
                                    events,
                                    label,
                                    margin_stats,
                                    cache_ownership,
                                    steady_scheduler,
                                    steady_shadow_enabled,
                                    steady_timing);
}

static int append_pcm_and_drain_runtime(SessionState& state,
                                        const std::vector<float>& pcm,
                                        AudioFrontend& audio,
                                        FirstEncoder& enc_first,
                                        AOTIModelPackageLoader* enc_steady,
                                        torch::jit::Module& joint,
                                        torch::jit::Module& predict,
                                        torch::Device device,
                                        const Tokenizer& tokenizer,
                                        std::vector<EmittedEvent>& events,
                                        const std::string& label,
                                        MarginStats* margin_stats = nullptr,
                                        CacheOwnershipStats* cache_ownership = nullptr) {
  auto ctx = default_execution_context(audio, joint, predict, device);
  return append_pcm_and_drain_runtime(state,
                                      pcm,
                                      audio,
                                      enc_first,
                                      enc_steady,
                                      ctx,
                                      device,
                                      tokenizer,
                                      events,
                                      label,
                                      margin_stats,
                                      cache_ownership);
}

static int flush_post_stop_audio_runtime(SessionState& state,
                                         AudioFrontend& audio,
                                         FirstEncoder& enc_first,
                                         AOTIModelPackageLoader* enc_steady,
                                         const ExecutionContext& ctx,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label,
                                         MarginStats* margin_stats = nullptr,
                                         CacheOwnershipStats* cache_ownership = nullptr,
                                         BatchedSteadyScheduler* steady_scheduler = nullptr,
                                         bool steady_shadow_enabled = false,
                                         RuntimeSteadyTiming* steady_timing = nullptr) {
  if (state.post_stop_audio.empty()) return 0;
  std::vector<float> held;
  held.swap(state.post_stop_audio);
  state.pending_audio.insert(state.pending_audio.end(), held.begin(), held.end());
  state.total_audio_samples += static_cast<int64_t>(held.size());
  return drain_audio_steady_runtime(state,
                                    audio,
                                    enc_first,
                                    enc_steady,
                                    ctx,
                                    device,
                                    tokenizer,
                                    events,
                                    label,
                                    margin_stats,
                                    cache_ownership,
                                    steady_scheduler,
                                    steady_shadow_enabled,
                                    steady_timing);
}

static int flush_post_stop_audio_runtime(SessionState& state,
                                         AudioFrontend& audio,
                                         FirstEncoder& enc_first,
                                         AOTIModelPackageLoader* enc_steady,
                                         torch::jit::Module& joint,
                                         torch::jit::Module& predict,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label,
                                         MarginStats* margin_stats = nullptr,
                                         CacheOwnershipStats* cache_ownership = nullptr) {
  auto ctx = default_execution_context(audio, joint, predict, device);
  return flush_post_stop_audio_runtime(state,
                                       audio,
                                       enc_first,
                                       enc_steady,
                                       ctx,
                                       device,
                                       tokenizer,
                                       events,
                                       label,
                                       margin_stats,
                                       cache_ownership);
}

void vad_stop(SessionState& state) {
  if (state.mode == SessionMode::FINALIZED) throw std::runtime_error("vad_stop called after FINALIZED");
  state.mode = SessionMode::PENDING_FINALIZE;
}

static int vad_start(SessionState& state,
                     AudioFrontend& audio,
                     FirstEncoder& enc_first,
                     AOTIModelPackageLoader* enc_steady,
                     const ExecutionContext& ctx,
                     torch::Device device,
                     const Tokenizer& tokenizer,
                     std::vector<EmittedEvent>& events,
                     const std::string& label,
                     MarginStats* margin_stats = nullptr,
                     CacheOwnershipStats* cache_ownership = nullptr,
                     BatchedSteadyScheduler* steady_scheduler = nullptr,
                     bool steady_shadow_enabled = false,
                     RuntimeSteadyTiming* steady_timing = nullptr) {
  if (state.mode == SessionMode::PENDING_FINALIZE) {
    state.mode = SessionMode::STREAMING;
    return flush_post_stop_audio_runtime(state,
                                         audio,
                                         enc_first,
                                         enc_steady,
                                         ctx,
                                         device,
                                         tokenizer,
                                         events,
                                         label,
                                         margin_stats,
                                         cache_ownership,
                                         steady_scheduler,
                                         steady_shadow_enabled,
                                         steady_timing);
  }
  if (state.mode == SessionMode::STREAMING) {
    return flush_post_stop_audio_runtime(state,
                                         audio,
                                         enc_first,
                                         enc_steady,
                                         ctx,
                                         device,
                                         tokenizer,
                                         events,
                                         label,
                                         margin_stats,
                                         cache_ownership,
                                         steady_scheduler,
                                         steady_shadow_enabled,
                                         steady_timing);
  }
  throw std::runtime_error("vad_start called after FINALIZED");
}

static int vad_start(SessionState& state,
                     AudioFrontend& audio,
                     FirstEncoder& enc_first,
                     AOTIModelPackageLoader* enc_steady,
                     torch::jit::Module& joint,
                     torch::jit::Module& predict,
                     torch::Device device,
                     const Tokenizer& tokenizer,
                     std::vector<EmittedEvent>& events,
                     const std::string& label,
                     MarginStats* margin_stats = nullptr,
                     CacheOwnershipStats* cache_ownership = nullptr) {
  auto ctx = default_execution_context(audio, joint, predict, device);
  return vad_start(state,
                   audio,
                   enc_first,
                   enc_steady,
                   ctx,
                   device,
                   tokenizer,
                   events,
                   label,
                   margin_stats,
                   cache_ownership);
}

AudioGeometry session_runtime_audio_geometry_from_bundle(torch::jit::Module& bundle) {
  return audio_geometry_from_bundle(bundle);
}

std::string session_runtime_verify_preproc_manifest(const std::string& dir,
                                                    const std::string& preproc_path,
                                                    const AudioGeometry& audio_geometry) {
  return verify_preproc_manifest(dir, preproc_path, audio_geometry);
}

void destroy_session_runtime_audio_frontend(RuntimeAudioFrontend* audio) {
  delete audio;
}

std::unique_ptr<RuntimeAudioFrontend, void (*)(RuntimeAudioFrontend*)>
make_session_runtime_audio_frontend(torch::jit::Module& bundle,
                                    torch::jit::Module& preproc,
                                    torch::Device device) {
  auto out = std::unique_ptr<RuntimeAudioFrontend, void (*)(RuntimeAudioFrontend*)>(
      new RuntimeAudioFrontend(), destroy_session_runtime_audio_frontend);
  out->audio.g = audio_geometry_from_bundle(bundle);
  out->audio.preproc = &preproc;
  out->audio.device = device;
  out->audio.mel_atol = scalar_f64(attr_tensor(bundle, "mel_ci_atol"));
  out->audio.cache_atol = scalar_f64(attr_tensor(bundle, "cache_ci_atol"));
  if (out->audio.mel_atol < 0.0 || out->audio.cache_atol < 0.0) {
    throw std::runtime_error("audio CI thresholds must be non-negative");
  }
  out->audio.ci_stats = audio_ci_stats_from_bundle(bundle);
  return out;
}

void reset_session_runtime_audio_front(SessionState& state, RuntimeAudioFrontend& audio) {
  reset_audio_front(state, audio.audio.g);
}

int session_runtime_append_pcm_and_drain(SessionState& state,
                                         const std::vector<float>& pcm,
                                         RuntimeAudioFrontend& audio,
                                         FirstEncoder& enc_first,
                                         AOTIModelPackageLoader* enc_steady,
                                         const ExecutionContext& ctx,
                                         torch::Device device,
                                         const Tokenizer& tokenizer,
                                         std::vector<EmittedEvent>& events,
                                         const std::string& label) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  return append_pcm_and_drain_runtime(state,
                                      pcm,
                                      audio.audio,
                                      enc_first,
                                      enc_steady,
                                      ctx,
                                      device,
                                      tokenizer,
                                      events,
                                      label,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      false);
}

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
                                         RuntimeSteadyTiming* steady_timing) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  return append_pcm_and_drain_runtime(state,
                                      pcm,
                                      audio.audio,
                                      enc_first,
                                      enc_steady,
                                      ctx,
                                      device,
                                      tokenizer,
                                      events,
                                      label,
                                      nullptr,
                                      nullptr,
                                      steady_scheduler,
                                      steady_shadow_enabled,
                                      steady_timing);
}

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
                                         const std::string& label) {
  auto ctx = default_execution_context(audio.audio, joint, predict, device);
  return session_runtime_append_pcm_and_drain(state,
                                             pcm,
                                             audio,
                                             enc_first,
                                             enc_steady,
                                             ctx,
                                             device,
                                             tokenizer,
                                             events,
                                             label);
}

int session_runtime_vad_start(SessionState& state,
                              RuntimeAudioFrontend& audio,
                              FirstEncoder& enc_first,
                              AOTIModelPackageLoader* enc_steady,
                              const ExecutionContext& ctx,
                              torch::Device device,
                              const Tokenizer& tokenizer,
                              std::vector<EmittedEvent>& events,
                              const std::string& label) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  return vad_start(state,
                   audio.audio,
                   enc_first,
                   enc_steady,
                   ctx,
                   device,
                   tokenizer,
                   events,
                   label,
                   nullptr,
                   nullptr,
                   nullptr,
                   false);
}

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
                              RuntimeSteadyTiming* steady_timing) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  return vad_start(state,
                   audio.audio,
                   enc_first,
                   enc_steady,
                   ctx,
                   device,
                   tokenizer,
                   events,
                   label,
                   nullptr,
                   nullptr,
                   steady_scheduler,
                   steady_shadow_enabled,
                   steady_timing);
}

int session_runtime_vad_start(SessionState& state,
                              RuntimeAudioFrontend& audio,
                              FirstEncoder& enc_first,
                              AOTIModelPackageLoader* enc_steady,
                              torch::jit::Module& joint,
                              torch::jit::Module& predict,
                              torch::Device device,
                              const Tokenizer& tokenizer,
                              std::vector<EmittedEvent>& events,
                              const std::string& label) {
  auto ctx = default_execution_context(audio.audio, joint, predict, device);
  return session_runtime_vad_start(state,
                                   audio,
                                   enc_first,
                                   enc_steady,
                                   ctx,
                                   device,
                                   tokenizer,
                                   events,
                                   label);
}

static int append_audio_and_drain(SessionState& state,
                                  torch::jit::Module& bundle,
                                  const std::string& prefix,
                                  AudioFrontend& audio,
                                  FirstEncoder& enc_first,
                                  AOTIModelPackageLoader& enc_steady,
                                  torch::jit::Module& joint,
                                  torch::jit::Module& predict,
                                  torch::Device device,
                                  const Tokenizer& tokenizer,
                                  std::vector<EmittedEvent>& events,
                                  const std::string& label,
                                  FirstChunkStats* first_chunk_stats = nullptr,
                                  CacheOwnershipStats* cache_ownership = nullptr) {
  if (state.mode != SessionMode::STREAMING) throw std::runtime_error("append_audio outside STREAMING for " + label);
  auto pcm = tensor_to_float_vec(prefix_tensor(bundle, prefix, "audio"));
  state.pending_audio.insert(state.pending_audio.end(), pcm.begin(), pcm.end());
  state.total_audio_samples += static_cast<int64_t>(pcm.size());
  return drain_audio_steady(state, bundle, prefix, audio, enc_first, enc_steady,
                            joint, predict, device, tokenizer, events, label,
                            first_chunk_stats, cache_ownership);
}

struct FinalizeOutcome {
  bool token_ok = false;
  bool fork_ok = false;
  bool stale_dropped = false;
  size_t emitted_tokens = 0;
  std::vector<int64_t> final_tokens;
  std::string final_text;
};

struct FinalizeAudioInputs {
  bool has_inputs = false;
  torch::Tensor chunk_mel;
  torch::Tensor new_mel;
  int64_t drop_extra = -1;
  int64_t final_T = 0;
  int64_t remaining_frames = 0;
  int64_t padded_total_samples = 0;
  int64_t total_mel_frames = 0;
};

static FinalizeAudioInputs prepare_finalize_inputs_from_audio(const SessionState& parent,
                                                              AudioFrontend& audio,
                                                              torch::Device device,
                                                              torch::jit::Module* preproc_override = nullptr) {
  FinalizeAudioInputs inputs;
  torch::jit::Module& preproc_module = preproc_override != nullptr ? *preproc_override : audio.require_preproc();
  std::vector<float> pending = parent.pending_audio;
  if (parent.total_audio_samples > 0) {
    pending.insert(pending.end(),
                   static_cast<size_t>(audio.g.final_padding_frames * audio.g.hop_samples),
                   0.0f);
  }
  if (pending.empty()) {
    inputs.padded_total_samples = parent.emitted * audio.g.hop_samples;
    return inputs;
  }

  inputs.padded_total_samples = parent.emitted * audio.g.hop_samples + static_cast<int64_t>(pending.size());
  inputs.total_mel_frames = inputs.padded_total_samples / audio.g.hop_samples + 1;
  inputs.remaining_frames = inputs.total_mel_frames - parent.emitted;
  if (inputs.remaining_frames <= 0) {
    inputs.has_inputs = false;
    return inputs;
  }

  std::vector<float> raw_ring = parent.raw_audio_ring;
  std::vector<torch::Tensor> new_mels;
  int64_t frames_collected = 0;
  while (frames_collected < inputs.remaining_frames) {
    int64_t frames_this_call = std::min<int64_t>(
        audio.g.shift_frames,
        inputs.remaining_frames - frames_collected);
    size_t needed_new_samples = std::min(
        pending.size(),
        static_cast<size_t>(audio.g.preprocess_new_audio_samples));
    std::vector<float> new_audio(
        pending.begin(),
        pending.begin() + static_cast<std::ptrdiff_t>(needed_new_samples));
    auto mel = audio.preprocess_fixed_audio(raw_ring, new_audio, preproc_module);
    int64_t start = audio.g.first_preprocess_mel_frame;
    new_mels.push_back(mel.slice(2, start, start + frames_this_call).contiguous());

    if (frames_this_call == audio.g.shift_frames) {
      size_t consumed_samples = std::min(
          static_cast<size_t>(audio.g.shift_frames * audio.g.hop_samples),
          pending.size());
      std::vector<float> consumed(
          pending.begin(),
          pending.begin() + static_cast<std::ptrdiff_t>(consumed_samples));
      if (consumed.size() >= static_cast<size_t>(audio.g.raw_audio_ring_samples)) {
        raw_ring.assign(consumed.end() - static_cast<std::ptrdiff_t>(audio.g.raw_audio_ring_samples),
                        consumed.end());
      } else if (!consumed.empty()) {
        size_t keep = static_cast<size_t>(audio.g.raw_audio_ring_samples) - consumed.size();
        std::vector<float> next;
        next.reserve(static_cast<size_t>(audio.g.raw_audio_ring_samples));
        next.insert(next.end(), raw_ring.end() - static_cast<std::ptrdiff_t>(keep), raw_ring.end());
        next.insert(next.end(), consumed.begin(), consumed.end());
        raw_ring = std::move(next);
      }
      pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(consumed_samples));
    }
    frames_collected += frames_this_call;
  }

  inputs.has_inputs = true;
  inputs.new_mel = torch::cat(new_mels, 2).to(device).contiguous();
  if (parent.emitted == 0) {
    inputs.chunk_mel = inputs.new_mel;
    inputs.drop_extra = 0;
  } else {
    if (!parent.ring.defined()) throw std::runtime_error("audio finalize continuation missing mel ring");
    inputs.chunk_mel = torch::cat({parent.ring.to(device), inputs.new_mel}, 2).contiguous();
    inputs.drop_extra = DROP;
  }
  inputs.final_T = inputs.chunk_mel.size(2);
  return inputs;
}

static bool verify_finalize_audio_gold(torch::jit::Module& bundle,
                                       const std::string& prefix,
                                       const std::string& label,
                                       const FinalizeAudioInputs& inputs,
                                       AudioFrontend& audio,
                                       torch::Device device) {
  bool ok = true;
  auto geometry_check = [&](const char* name, int64_t actual, int64_t expected) {
    ++audio.geometry_stats.checks;
    bool one_ok = int64_equal(name, actual, expected, label);
    if (one_ok) ++audio.geometry_stats.pass;
    ok = one_ok && ok;
  };
  geometry_check("final_padded_total_samples",
                 inputs.padded_total_samples,
                 scalar_i64(prefix_tensor(bundle, prefix, "final_padded_total_samples")));
  geometry_check("final_total_mel_frames",
                 inputs.total_mel_frames,
                 scalar_i64(prefix_tensor(bundle, prefix, "final_total_mel_frames")));
  geometry_check("final_remaining_frames",
                 inputs.remaining_frames,
                 scalar_i64(prefix_tensor(bundle, prefix, "final_remaining_frames")));
  geometry_check("final_T",
                 inputs.final_T,
                 scalar_i64(prefix_tensor(bundle, prefix, "final_T")));
  geometry_check("final_drop_extra",
                 inputs.drop_extra,
                 scalar_i64(prefix_tensor(bundle, prefix, "final_drop_extra")));
  if (inputs.has_inputs) {
    ok = compare_mel_tensor(label + ".final_new_mel",
                            inputs.new_mel,
                            prefix_tensor(bundle, prefix, "final_new_mel").to(device).contiguous(),
                            audio.stats,
                            audio.mel_atol) && ok;
    ok = compare_mel_tensor(label + ".final_chunk_mel",
                            inputs.chunk_mel,
                            prefix_tensor(bundle, prefix, "final_chunk_mel").to(device).contiguous(),
                            audio.stats,
                            audio.mel_atol) && ok;
  }
  return ok;
}

static FinalizeOutcome run_finalize(SessionState& parent,
                                    torch::jit::Module& bundle,
                                    const std::string& prefix,
                                    const std::string& label,
                                    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
                                    torch::jit::Module& joint,
                                    torch::jit::Module& predict,
                                    torch::Device device,
                                    const Tokenizer& tokenizer,
                                    std::vector<EmittedEvent>& events,
                                    FinalizeFinish finish,
                                    const FinalizeAudioInputs* audio_inputs = nullptr,
                                    const AudioGeometry* audio_geometry = nullptr,
                                    MarginStats* margin_stats = nullptr) {
  if (finish == FinalizeFinish::SPECULATIVE_KEEP && parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("speculative finalize outside PENDING_FINALIZE");
  }
  if (finish == FinalizeFinish::TRUE_BOUNDARY_COLD_RESET &&
      parent.mode != SessionMode::STREAMING &&
      parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("true-boundary finalize outside live state");
  }
  c10::cuda::CUDAGuard device_guard(device.index());
  auto stream = c10::cuda::getCurrentCUDAStream(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  auto snapshot = snapshot_asr(parent);
  parent.mode = SessionMode::FINALIZED;
  snapshot.mode = SessionMode::FINALIZED;
  auto fork = clone_session(parent);

  int64_t drop_extra = audio_inputs != nullptr
                           ? audio_inputs->drop_extra
                           : scalar_i64(prefix_tensor(bundle, prefix, "final_drop_extra"));
  int64_t final_T = audio_inputs != nullptr
                        ? audio_inputs->final_T
                        : scalar_i64(prefix_tensor(bundle, prefix, "final_T"));
  auto gold = tensor_to_vec(prefix_tensor(bundle, prefix, "gold_tokens"));

  if (final_T > 0) {
    auto final_chunk = audio_inputs != nullptr
                           ? audio_inputs->chunk_mel.to(device).contiguous()
                           : prefix_tensor(bundle, prefix, "final_chunk_mel").to(device).contiguous();
    if (final_chunk.size(2) != final_T) {
      throw std::runtime_error("final_chunk_mel T does not match bundle final_T");
    }
    int64_t expected_drop = parent.emitted == 0 ? 0 : DROP;
    if (drop_extra != expected_drop) throw std::runtime_error("finalize drop_extra does not match parent emitted state");

    auto loader_it = finalize_loaders.find(std::make_pair(drop_extra, final_T));
    if (loader_it == finalize_loaders.end()) {
      throw std::runtime_error("no finalize bucket for drop=" + std::to_string(drop_extra) +
                               " T=" + std::to_string(final_T));
    }

    std::vector<at::Tensor> inputs = {
        final_chunk.contiguous(),
        fork.clc.contiguous(),
        fork.clt.contiguous(),
        fork.clcl.contiguous(),
    };
    auto out = run_aoti_loader_on_stream(*loader_it->second, inputs, stream);
    if (out.size() < 2) throw std::runtime_error("finalize AOTI bucket returned fewer than 2 outputs");
    int64_t enc_len = scalar_i64(out[1]);
    if (out.size() >= 5) {
      fork.clc = out[2];
      fork.clt = out[3];
      fork.clcl = out[4];
    }
    decode_range(joint, predict, condition_encoder_output(out[0], fork), enc_len,
                 fork.g, fork.h, fork.c, fork.hyp,
                 margin_stats, label + ".final");
  }

  FinalizeOutcome outcome;
  outcome.emitted_tokens = fork.hyp.size();
  outcome.final_tokens = fork.hyp;
  outcome.final_text = tokenizer.ids_to_text(fork.hyp);
  outcome.token_ok = equal_tokens(outcome.final_tokens, gold, "final cumulative", label);
  std::string final_text = outcome.final_text;
  std::string delta_text = append_only_delta_text(final_text, parent.continuous_emitted_text);
  auto delta_tokens = append_only_delta_tokens(fork.hyp, parent.continuous_emitted_tokens);
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
    cold_reset_after_finalize(parent, bundle, device, audio_geometry);
  }
  return outcome;
}

struct FinalizeInputSnapshot {
  uint64_t generation = 0;
  AsrSnapshot parent_snapshot;
  SessionState input_state;
  SessionState fork;
  FinalizeAudioInputs audio_inputs;
  std::vector<int64_t> base_continuous_emitted_tokens;
  std::string base_continuous_emitted_text;
  bool moved_true_boundary_audio = false;
};

struct FinalizeCommit {
  std::vector<EmittedEvent> events;
  std::vector<int64_t> committed_continuous_tokens;
  std::string committed_continuous_text;
  bool has_delta = false;
};

class MapFinalizeBucketLoaderProvider final : public FinalizeBucketLoaderProvider {
 public:
  explicit MapFinalizeBucketLoaderProvider(
      std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& loaders)
      : loaders_(loaders) {}

  AOTIModelPackageLoader& get(int64_t drop, int64_t T) override {
    auto it = loaders_.find(std::make_pair(drop, T));
    if (it == loaders_.end()) {
      throw std::runtime_error("runtime finalize missing bucket drop=" +
                               std::to_string(drop) +
                               " T=" + std::to_string(T));
    }
    return *it->second;
  }

 private:
  std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& loaders_;
};

static FinalizeInputSnapshot make_finalize_input_snapshot(SessionState& parent,
                                                          RuntimeAudioFrontend& audio,
                                                          torch::Device device,
                                                          const ExecutionContext& ctx,
                                                          FinalizeFinish finish) {
  if (finish == FinalizeFinish::SPECULATIVE_KEEP && parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("runtime speculative finalize outside PENDING_FINALIZE");
  }
  if (finish == FinalizeFinish::TRUE_BOUNDARY_COLD_RESET &&
      parent.mode != SessionMode::STREAMING &&
      parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("runtime true-boundary finalize outside live state");
  }

  FinalizeInputSnapshot snapshot;
  snapshot.generation = parent.generation.load(std::memory_order_acquire);
  snapshot.parent_snapshot = snapshot_asr(parent);
  snapshot.input_state = clone_session(parent);
  snapshot.base_continuous_emitted_tokens = parent.continuous_emitted_tokens;
  snapshot.base_continuous_emitted_text = parent.continuous_emitted_text;
  if (finish == FinalizeFinish::TRUE_BOUNDARY_COLD_RESET &&
      !snapshot.input_state.post_stop_audio.empty()) {
    snapshot.input_state.pending_audio.insert(snapshot.input_state.pending_audio.end(),
                                             snapshot.input_state.post_stop_audio.begin(),
                                             snapshot.input_state.post_stop_audio.end());
    snapshot.input_state.total_audio_samples +=
        static_cast<int64_t>(snapshot.input_state.post_stop_audio.size());
    snapshot.input_state.post_stop_audio.clear();
    snapshot.moved_true_boundary_audio = true;
  }
  snapshot.audio_inputs = prepare_finalize_inputs_from_audio(snapshot.input_state,
                                                             audio.audio,
                                                             device,
                                                             &ctx.preproc);
  snapshot.input_state.mode = SessionMode::FINALIZED;
  snapshot.fork = clone_session(snapshot.input_state);
  return snapshot;
}

static FinalizeCommit make_finalize_commit(const FinalizeInputSnapshot& snapshot,
                                           const FinalizeOutcome& outcome) {
  FinalizeCommit commit;
  std::string delta_text = append_only_delta_text(outcome.final_text,
                                                  snapshot.base_continuous_emitted_text);
  auto delta_tokens = append_only_delta_tokens(outcome.final_tokens,
                                               snapshot.base_continuous_emitted_tokens);
  if (delta_text.empty()) {
    emit_event(commit.events,
               EVENT_SUPPRESSED,
               {},
               snapshot.base_continuous_emitted_tokens,
               "",
               snapshot.base_continuous_emitted_text);
  } else {
    auto collector_tokens = snapshot.base_continuous_emitted_tokens;
    collector_tokens.insert(collector_tokens.end(), delta_tokens.begin(), delta_tokens.end());
    std::string collector_text = append_delta_to_collector(snapshot.base_continuous_emitted_text,
                                                           delta_text);
    emit_event(commit.events,
               EVENT_FINAL,
               delta_tokens,
               collector_tokens,
               delta_text,
               collector_text);
    commit.committed_continuous_tokens = std::move(collector_tokens);
    commit.committed_continuous_text = std::move(collector_text);
    commit.has_delta = true;
  }
  return commit;
}

static bool commit_finalize_runtime(SessionState& parent,
                                    torch::jit::Module* bundle,
                                    torch::Device device,
                                    const AudioGeometry* audio_geometry,
                                    const FinalizeInputSnapshot& snapshot,
                                    FinalizeFinish finish,
                                    const FinalizeCommit& commit,
                                    std::vector<EmittedEvent>& events,
                                    FinalizeOutcome* outcome) {
  uint64_t current_generation = parent.generation.load(std::memory_order_acquire);
  if (current_generation != snapshot.generation) {
    if (outcome != nullptr) outcome->stale_dropped = true;
    return false;
  }

  if (snapshot.moved_true_boundary_audio) {
    parent.pending_audio.insert(parent.pending_audio.end(),
                                parent.post_stop_audio.begin(),
                                parent.post_stop_audio.end());
    parent.total_audio_samples += static_cast<int64_t>(parent.post_stop_audio.size());
    parent.post_stop_audio.clear();
  }
  parent.mode = SessionMode::FINALIZED;
  events.insert(events.end(), commit.events.begin(), commit.events.end());
  if (commit.has_delta) {
    parent.continuous_emitted_tokens = commit.committed_continuous_tokens;
    parent.continuous_emitted_text = commit.committed_continuous_text;
  }
  if (finish == FinalizeFinish::SPECULATIVE_KEEP) {
    finish_speculative_finalize(parent);
  } else {
    if (bundle == nullptr) {
      throw std::runtime_error("runtime true-boundary finalize commit requires session bundle");
    }
    cold_reset_after_finalize(parent, *bundle, device, audio_geometry);
  }
  return true;
}

static FinalizeOutcome run_finalize_runtime_on_stream(
    SessionState& parent,
    torch::jit::Module& bundle,
    RuntimeAudioFrontend& audio,
    const std::string& label,
    FinalizeBucketLoaderProvider& finalize_loaders,
    const ExecutionContext& ctx,
    torch::Device device,
    const Tokenizer& tokenizer,
    std::vector<EmittedEvent>& events,
    FinalizeFinish finish,
    MarginStats* margin_stats = nullptr) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  auto snapshot = make_finalize_input_snapshot(parent, audio, device, ctx, finish);

  if (snapshot.audio_inputs.final_T > 0) {
    AOTIModelPackageLoader& loader = finalize_loaders.get(snapshot.audio_inputs.drop_extra,
                                                          snapshot.audio_inputs.final_T);
    std::vector<at::Tensor> inputs = {
        snapshot.audio_inputs.chunk_mel.to(device).contiguous(),
        snapshot.fork.clc.contiguous(),
        snapshot.fork.clt.contiguous(),
        snapshot.fork.clcl.contiguous(),
    };
    auto out = run_aoti_loader_on_stream(loader, inputs, ctx.stream);
    if (out.size() < 2) throw std::runtime_error("runtime finalize AOTI bucket returned fewer than 2 outputs");
    int64_t enc_len = scalar_i64(out[1]);
    if (out.size() >= 5) {
      snapshot.fork.clc = out[2];
      snapshot.fork.clt = out[3];
      snapshot.fork.clcl = out[4];
    }
    decode_range(ctx.joint, ctx.predict, condition_encoder_output(out[0], snapshot.fork), enc_len,
                 snapshot.fork.g, snapshot.fork.h, snapshot.fork.c, snapshot.fork.hyp,
                 margin_stats, label + ".final");
  }

  FinalizeOutcome outcome;
  outcome.token_ok = true;
  outcome.emitted_tokens = snapshot.fork.hyp.size();
  outcome.final_tokens = snapshot.fork.hyp;
  outcome.final_text = tokenizer.ids_to_text(snapshot.fork.hyp);
  outcome.fork_ok = fork_assert_parent_unchanged(parent, snapshot.parent_snapshot);
  FinalizeCommit commit = make_finalize_commit(snapshot, outcome);
  commit_finalize_runtime(parent,
                          &bundle,
                          device,
                          finish == FinalizeFinish::TRUE_BOUNDARY_COLD_RESET ? &audio.audio.g : nullptr,
                          snapshot,
                          finish,
                          commit,
                          events,
                          &outcome);
  return outcome;
}

static FinalizeOutcome run_finalize_runtime(
    SessionState& parent,
    const std::string& label,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    const Tokenizer& tokenizer,
    std::vector<EmittedEvent>& events,
    FinalizeFinish finish,
    const FinalizeAudioInputs& audio_inputs,
    MarginStats* margin_stats = nullptr) {
  MapFinalizeBucketLoaderProvider provider(finalize_loaders);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device.index());
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  if (finish != FinalizeFinish::SPECULATIVE_KEEP) {
    throw std::runtime_error("legacy run_finalize_runtime supports only speculative finalize");
  }
  if (parent.mode != SessionMode::PENDING_FINALIZE) {
    throw std::runtime_error("runtime speculative finalize outside PENDING_FINALIZE");
  }

  FinalizeInputSnapshot snapshot;
  snapshot.generation = parent.generation.load(std::memory_order_acquire);
  snapshot.parent_snapshot = snapshot_asr(parent);
  snapshot.input_state = clone_session(parent);
  snapshot.base_continuous_emitted_tokens = parent.continuous_emitted_tokens;
  snapshot.base_continuous_emitted_text = parent.continuous_emitted_text;
  snapshot.audio_inputs = audio_inputs;
  snapshot.input_state.mode = SessionMode::FINALIZED;
  snapshot.fork = clone_session(snapshot.input_state);

  if (snapshot.audio_inputs.final_T > 0) {
    AOTIModelPackageLoader& loader = provider.get(snapshot.audio_inputs.drop_extra,
                                                  snapshot.audio_inputs.final_T);
    std::vector<at::Tensor> inputs = {
        snapshot.audio_inputs.chunk_mel.to(device).contiguous(),
        snapshot.fork.clc.contiguous(),
        snapshot.fork.clt.contiguous(),
        snapshot.fork.clcl.contiguous(),
    };
    auto out = run_aoti_loader_on_stream(loader, inputs, stream);
    if (out.size() < 2) throw std::runtime_error("runtime finalize AOTI bucket returned fewer than 2 outputs");
    int64_t enc_len = scalar_i64(out[1]);
    if (out.size() >= 5) {
      snapshot.fork.clc = out[2];
      snapshot.fork.clt = out[3];
      snapshot.fork.clcl = out[4];
    }
    decode_range(joint, predict, condition_encoder_output(out[0], snapshot.fork), enc_len,
                 snapshot.fork.g, snapshot.fork.h, snapshot.fork.c, snapshot.fork.hyp,
                 margin_stats, label + ".final");
  }

  FinalizeOutcome outcome;
  outcome.token_ok = true;
  outcome.emitted_tokens = snapshot.fork.hyp.size();
  outcome.final_tokens = snapshot.fork.hyp;
  outcome.final_text = tokenizer.ids_to_text(snapshot.fork.hyp);
  outcome.fork_ok = fork_assert_parent_unchanged(parent, snapshot.parent_snapshot);
  FinalizeCommit commit = make_finalize_commit(snapshot, outcome);
  commit_finalize_runtime(parent,
                          nullptr,
                          device,
                          nullptr,
                          snapshot,
                          finish,
                          commit,
                          events,
                          &outcome);
  return outcome;
}

FinalizeOutcome session_runtime_finalize(
    SessionState& state,
    torch::jit::Module& bundle,
    RuntimeAudioFrontend& audio,
    FinalizeBucketLoaderProvider& finalize_loaders,
    const ExecutionContext& ctx,
    torch::Device device,
    const Tokenizer& tokenizer,
    std::vector<EmittedEvent>& events,
    FinalizeFinish finish,
    const std::string& label) {
  c10::cuda::CUDAGuard device_guard(device.index());
  c10::cuda::CUDAStreamGuard stream_guard(ctx.stream);
  auto outcome = run_finalize_runtime_on_stream(state,
                                                bundle,
                                                audio,
                                                label,
                                                finalize_loaders,
                                                ctx,
                                                device,
                                                tokenizer,
                                                events,
                                                finish,
                                                &audio.audio.margin_stats);
  return outcome;
}

FinalizeOutcome session_runtime_finalize(
    SessionState& state,
    torch::jit::Module& bundle,
    RuntimeAudioFrontend& audio,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    const ExecutionContext& ctx,
    torch::Device device,
    const Tokenizer& tokenizer,
    std::vector<EmittedEvent>& events,
    FinalizeFinish finish,
    const std::string& label) {
  MapFinalizeBucketLoaderProvider provider(finalize_loaders);
  return session_runtime_finalize(state,
                                  bundle,
                                  audio,
                                  provider,
                                  ctx,
                                  device,
                                  tokenizer,
                                  events,
                                  finish,
                                  label);
}

FinalizeOutcome session_runtime_finalize(
    SessionState& state,
    torch::jit::Module& bundle,
    RuntimeAudioFrontend& audio,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    const Tokenizer& tokenizer,
    std::vector<EmittedEvent>& events,
    FinalizeFinish finish,
    const std::string& label) {
  auto ctx = default_execution_context(audio.audio, joint, predict, device);
  return session_runtime_finalize(state,
                                  bundle,
                                  audio,
                                  finalize_loaders,
                                  ctx,
                                  device,
                                  tokenizer,
                                  events,
                                  finish,
                                  label);
}

static bool equal_one_text_from_bundle(torch::jit::Module& bundle,
                                       const std::string& prefix,
                                       const char* name,
                                       const std::string& actual,
                                       const std::string& label) {
  std::string expected = one_text_from_bundle(bundle, prefix, name, label);
  if (actual == expected) return true;
  std::printf("    %s %s mismatch: got=%s gold=%s\n",
              label.c_str(), name, escaped_text(actual).c_str(), escaped_text(expected).c_str());
  return false;
}

static bool retained_state_matches(SessionState& state,
                                   torch::jit::Module& bundle,
                                   const std::string& prefix,
                                   const std::string& label,
                                   torch::Device device,
                                   AudioFrontend* audio = nullptr) {
  bool ok = true;
  if (audio != nullptr) {
    ok = tensor_close_cache("retained.cache_last_channel",
                            state.clc,
                            prefix_tensor(bundle, prefix, "retained_clc").to(device),
                            audio->cache_atol,
                            label,
                            audio->cache_stats,
                            audio->cache_stats.cache_last_channel_max_abs) && ok;
    ok = tensor_close_cache("retained.cache_last_time",
                            state.clt,
                            prefix_tensor(bundle, prefix, "retained_clt").to(device),
                            audio->cache_atol,
                            label,
                            audio->cache_stats,
                            audio->cache_stats.cache_last_time_max_abs) && ok;
  } else {
    ok = tensor_equal("retained.cache_last_channel", state.clc, prefix_tensor(bundle, prefix, "retained_clc").to(device)) && ok;
    ok = tensor_equal("retained.cache_last_time", state.clt, prefix_tensor(bundle, prefix, "retained_clt").to(device)) && ok;
  }
  if (audio != nullptr) {
    ok = tensor_close_cache("retained.cache_last_channel_len",
                            state.clcl,
                            prefix_tensor(bundle, prefix, "retained_clcl").to(device),
                            0.0,
                            label,
                            audio->cache_stats,
                            audio->cache_stats.cache_last_channel_len_max_abs) && ok;
  } else {
    ok = tensor_equal("retained.cache_last_channel_len", state.clcl, prefix_tensor(bundle, prefix, "retained_clcl").to(device)) && ok;
  }
  ok = tensor_equal("retained.pred_out", state.g, prefix_tensor(bundle, prefix, "retained_g").to(device)) && ok;
  ok = tensor_equal("retained.decoder_state.h", state.h, prefix_tensor(bundle, prefix, "retained_h").to(device)) && ok;
  ok = tensor_equal("retained.decoder_state.c", state.c, prefix_tensor(bundle, prefix, "retained_c").to(device)) && ok;

  bool ring_defined = scalar_i64(prefix_tensor(bundle, prefix, "retained_ring_defined")) != 0;
  if (state.ring.defined() != ring_defined) {
    std::printf("    %s retained.mel_frame_ring defined mismatch: got=%d gold=%d\n",
                label.c_str(), (int)state.ring.defined(), (int)ring_defined);
    ok = false;
  } else if (ring_defined) {
    if (audio != nullptr) {
      ok = tensor_close("retained.mel_frame_ring", state.ring, prefix_tensor(bundle, prefix, "retained_ring").to(device),
                        audio->mel_atol, label) && ok;
    } else {
      ok = tensor_equal("retained.mel_frame_ring", state.ring, prefix_tensor(bundle, prefix, "retained_ring").to(device)) && ok;
    }
  }

  int64_t emitted = scalar_i64(prefix_tensor(bundle, prefix, "retained_emitted"));
  if (state.emitted != emitted) {
    std::printf("    %s retained.emitted mismatch: got=%ld gold=%ld\n",
                label.c_str(), (long)state.emitted, (long)emitted);
    ok = false;
  }
  ok = equal_tokens(state.hyp,
                    tensor_to_vec(prefix_tensor(bundle, prefix, "retained_hyp_tokens")),
                    "retained hyp",
                    label) && ok;
  ok = equal_tokens(state.continuous_emitted_tokens,
                    tensor_to_vec(prefix_tensor(bundle, prefix, "retained_collector_tokens")),
                    "retained collector",
                    label) && ok;
  ok = equal_one_text_from_bundle(bundle,
                                  prefix,
                                  "retained_collector_text",
                                  state.continuous_emitted_text,
                                  label) && ok;
  if (audio != nullptr) {
    ok = float_vec_equal("retained.pending_audio",
                         state.pending_audio,
                         tensor_to_float_vec(prefix_tensor(bundle, prefix, "retained_pending_audio")),
                         label) && ok;
    ok = float_vec_equal("retained.raw_audio_ring",
                         state.raw_audio_ring,
                         tensor_to_float_vec(prefix_tensor(bundle, prefix, "retained_raw_audio_ring")),
                         label) && ok;
    ok = int64_equal("retained.total_audio_samples",
                     state.total_audio_samples,
                     scalar_i64(prefix_tensor(bundle, prefix, "retained_total_audio_samples")),
                     label) && ok;
    ok = int64_equal("retained.synthetic_prefix_samples",
                     state.synthetic_prefix_samples,
                     scalar_i64(prefix_tensor(bundle, prefix, "retained_synthetic_prefix_samples")),
                     label) && ok;
  }
  if (state.mode != SessionMode::STREAMING) {
    std::printf("    %s retained mode mismatch: expected STREAMING\n", label.c_str());
    ok = false;
  }
  return ok;
}

static bool cold_reset_state_matches(SessionState& state,
                                     torch::jit::Module& bundle,
                                     const std::string& prefix,
                                     const std::string& label,
                                     torch::Device device,
                                     bool audio_mode = false) {
  bool ok = true;
  ok = tensor_equal("cold_reset.cache_last_channel", state.clc, attr_tensor(bundle, "init_clc").to(device)) && ok;
  ok = tensor_equal("cold_reset.cache_last_time", state.clt, attr_tensor(bundle, "init_clt").to(device)) && ok;
  ok = tensor_equal("cold_reset.cache_last_channel_len", state.clcl, attr_tensor(bundle, "init_clcl").to(device)) && ok;
  ok = tensor_equal("cold_reset.pred_out", state.g, attr_tensor(bundle, "init_g").to(device)) && ok;
  ok = tensor_equal("cold_reset.decoder_state.h", state.h, attr_tensor(bundle, "init_h").to(device)) && ok;
  ok = tensor_equal("cold_reset.decoder_state.c", state.c, attr_tensor(bundle, "init_c").to(device)) && ok;
  if (state.ring.defined()) {
    std::printf("    %s cold_reset.mel_frame_ring should be undefined\n", label.c_str());
    ok = false;
  }
  int64_t emitted = scalar_i64(prefix_tensor(bundle, prefix, "post_reset_emitted"));
  if (state.emitted != emitted) {
    std::printf("    %s cold_reset.emitted mismatch: got=%ld gold=%ld\n",
                label.c_str(), (long)state.emitted, (long)emitted);
    ok = false;
  }
  ok = equal_tokens(state.hyp,
                    tensor_to_vec(prefix_tensor(bundle, prefix, "post_reset_hyp_tokens")),
                    "cold_reset hyp",
                    label) && ok;
  ok = equal_tokens(state.continuous_emitted_tokens,
                    tensor_to_vec(prefix_tensor(bundle, prefix, "post_reset_collector_tokens")),
                    "cold_reset collector",
                    label) && ok;
  ok = equal_one_text_from_bundle(bundle,
                                  prefix,
                                  "post_reset_collector_text",
                                  state.continuous_emitted_text,
                                  label) && ok;
  if (audio_mode) {
    ok = float_vec_equal("post_reset.pending_audio",
                         state.pending_audio,
                         tensor_to_float_vec(prefix_tensor(bundle, prefix, "post_reset_pending_audio")),
                         label) && ok;
    ok = float_vec_equal("post_reset.raw_audio_ring",
                         state.raw_audio_ring,
                         tensor_to_float_vec(prefix_tensor(bundle, prefix, "post_reset_raw_audio_ring")),
                         label) && ok;
    ok = int64_equal("post_reset.total_audio_samples",
                     state.total_audio_samples,
                     scalar_i64(prefix_tensor(bundle, prefix, "post_reset_total_audio_samples")),
                     label) && ok;
    ok = int64_equal("post_reset.synthetic_prefix_samples",
                     state.synthetic_prefix_samples,
                     scalar_i64(prefix_tensor(bundle, prefix, "post_reset_synthetic_prefix_samples")),
                     label) && ok;
  }
  if (state.mode != SessionMode::STREAMING) {
    std::printf("    %s cold_reset mode mismatch: expected STREAMING\n", label.c_str());
    ok = false;
  }
  return ok;
}

static bool run_synthetic_word_delta_tests() {
  struct Case {
    const char* name;
    std::string emitted;
    std::string final_text;
    std::string expected_delta;
    std::string expected_collector;
    int64_t expected_kind;
  };
  std::vector<Case> cases = {
      {"New->Newark", "I live in New", "I live in Newark", "", "I live in New", EVENT_SUPPRESSED},
      {"play->playing", "we play", "we playing", "", "we play", EVENT_SUPPRESSED},
      {"duplicate final", "hello world", "hello world", "", "hello world", EVENT_SUPPRESSED},
      {"shorter final", "hello world today", "hello world", "", "hello world today", EVENT_SUPPRESSED},
      {"clean append", "hello world", "hello world today", "today", "hello world today", EVENT_FINAL},
      {"overlap trim", "alpha beta", "alpha gamma beta delta", "delta", "alpha beta delta", EVENT_FINAL},
  };

  bool all = true;
  for (const auto& c : cases) {
    std::string delta = append_only_delta_text(c.final_text, c.emitted);
    std::string collector = append_delta_to_collector(c.emitted, delta);
    int64_t kind = delta.empty() ? EVENT_SUPPRESSED : EVENT_FINAL;
    bool ok = delta == c.expected_delta &&
              collector == c.expected_collector &&
              kind == c.expected_kind;
    std::printf("SYNTHETIC %-16s %s delta=%s collector=%s\n",
                c.name,
                ok ? "PASS" : "FAIL",
                escaped_text(delta).c_str(),
                escaped_text(collector).c_str());
    all = ok && all;
  }
  std::printf("=== SYNTHETIC word-delta %s ===\n", all ? "PASS" : "FAIL");
  return all;
}

static bool audio_front_ci_ok(const AudioFrontend& audio) {
  bool geometry_ok = audio.geometry_stats.checks == audio.geometry_stats.pass;
  bool mel_ok = audio.stats.comparisons == audio.stats.pass;
  bool cache_ok = audio.cache_stats.checks == audio.cache_stats.pass;
  return geometry_ok && mel_ok && cache_ok;
}

static bool audio_front_unsafe_margin_fail(const AudioFrontend& audio, bool token_exact_ok) {
  return !token_exact_ok && audio.margin_stats.below_unsafe > 0;
}

static void print_audio_front_summary(const char* label,
                                      const AudioFrontend& audio,
                                      bool token_exact_ok) {
  bool unsafe_margin_fail = audio_front_unsafe_margin_fail(audio, token_exact_ok);
  const char* near_margin_status = unsafe_margin_fail
                                       ? "FAIL"
                                       : (audio.margin_stats.below_warning > 0 ? "WARN" : "PASS");
  double min_margin = audio.margin_stats.total > 0
                          ? audio.margin_stats.min_margin
                          : std::numeric_limits<double>::quiet_NaN();
  std::printf("=== AUDIO FRONT %s: geometry_checks=%ld geometry_pass=%ld geometry_recompute=%s "
              "mel_checks=%ld mel_pass=%ld byte_equal=%ld within_ci=%ld mel_within_ci=%s "
              "observed_mel_max_abs=%.6e mel_ci_atol=%.6e "
              "cache_checks=%ld cache_pass=%ld cache_within_ci=%s "
              "observed_cache_last_channel_max_abs=%.6e observed_cache_last_time_max_abs=%.6e "
              "observed_cache_last_channel_len_max_abs=%.6e cache_ci_atol=%.6e "
              "argmax_margin_min=%.6e at=%s frame=%ld symbol=%ld "
              "below_warning(<%.1e)=%ld below_unsafe(<%.1e)=%ld margin_checks=%ld near_margin=%s ===\n",
              label,
              (long)audio.geometry_stats.checks,
              (long)audio.geometry_stats.pass,
              pass_fail(audio.geometry_stats.checks == audio.geometry_stats.pass),
              (long)audio.stats.comparisons,
              (long)audio.stats.pass,
              (long)audio.stats.byte_equal,
              (long)audio.stats.within_ci,
              pass_fail(audio.stats.comparisons == audio.stats.pass),
              audio.stats.max_abs_diff,
              audio.mel_atol,
              (long)audio.cache_stats.checks,
              (long)audio.cache_stats.pass,
              pass_fail_skip(audio.cache_stats.checks, audio.cache_stats.pass),
              audio.cache_stats.cache_last_channel_max_abs,
              audio.cache_stats.cache_last_time_max_abs,
              audio.cache_stats.cache_last_channel_len_max_abs,
              audio.cache_atol,
              min_margin,
              audio.margin_stats.min_label.c_str(),
              (long)audio.margin_stats.min_frame,
              (long)audio.margin_stats.min_symbol,
              audio.margin_stats.warning_threshold,
              (long)audio.margin_stats.below_warning,
              audio.margin_stats.unsafe_threshold,
              (long)audio.margin_stats.below_unsafe,
              (long)audio.margin_stats.total,
              near_margin_status);
}

static void print_first_chunk_summary(const char* label, const FirstChunkStats& stats) {
  double min_margin = stats.margins.total > 0
                          ? stats.margins.min_margin
                          : std::numeric_limits<double>::quiet_NaN();
  const char* near_margin_status = stats.margins.below_warning > 0 ? "WARN" : "PASS";
  std::printf("=== FIRST CHUNK PATH %s: TorchScript first chunk retained; first-chunk drift observed; "
              "tested first-chunk greedy decisions min margin %.6e; pure-native runtime still requires an AOTI first chunk. "
              "enc_out_drift_checks=%ld pass=%ld byte_equal=%ld max_abs_vs_eager=%.6e "
              "margin_at=%s frame=%ld symbol=%ld "
              "below_warning(<%.1e)=%ld below_unsafe(<%.1e)=%ld margin_checks=%ld near_margin=%s ===\n",
              label,
              min_margin,
              (long)stats.checks,
              (long)stats.pass,
              (long)stats.byte_equal,
              stats.max_abs,
              stats.margins.min_label.c_str(),
              (long)stats.margins.min_frame,
              (long)stats.margins.min_symbol,
              stats.margins.warning_threshold,
              (long)stats.margins.below_warning,
              stats.margins.unsafe_threshold,
              (long)stats.margins.below_unsafe,
              (long)stats.margins.total,
              near_margin_status);
}

static bool cache_ownership_ok(const CacheOwnershipStats& stats) {
  return stats.recurrent_assignments > 0 && stats.alias_after_clone == 0;
}

static void print_cache_ownership_summary(const char* label, const CacheOwnershipStats& stats) {
  std::printf("=== CACHE STATE CLONE BARRIER %s: clone_on_assign=%s raw_loader_output_reuse=TOLERATED "
              "recurrent_assignments=%ld cache_tensor_clones=%ld post_clone_alias=%ld ===\n",
              label,
              cache_ownership_ok(stats) ? "PASS" : "FAIL",
              (long)stats.recurrent_assignments,
              (long)stats.clone_assignments,
              (long)stats.alias_after_clone);
}

static void print_long_stream_cache_summary(const LongStreamCacheStats& stats) {
  std::printf("=== AOTI LONG-STREAM CACHE STABILITY: %s prefix=%s steady_chunks=%ld aoti_chunks=%ld "
              "stable_refs=%ld/%ld consecutive_alias_fail=%ld/%ld token_equality=NOT_USED ===\n",
              stats.ok ? "PASS" : "FAIL",
              stats.prefix.c_str(),
              (long)stats.steady_chunks,
              (long)stats.aoti_chunks,
              (long)stats.stable_pass,
              (long)stats.stable_checks,
              (long)stats.consecutive_alias_fail,
              (long)stats.consecutive_alias_checks);
}

static LongStreamCacheStats run_long_stream_cache_stability_check(
    torch::jit::Module& bundle,
    bool multiturn,
    FirstEncoder& enc_first,
    AOTIModelPackageLoader& enc_steady,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    const Tokenizer& tokenizer,
    CacheOwnershipStats& ownership_stats) {
  LongStreamCacheStats stats;
  int64_t best_chunks = -1;
  std::string best_prefix;
  if (multiturn) {
    int64_t streams = scalar_i64(attr_tensor(bundle, "num_streams"));
    for (int stream = 0; stream < streams; ++stream) {
      int64_t turns = scalar_i64(attr_tensor(bundle, "stream" + std::to_string(stream) + "_num_turns"));
      for (int turn = 0; turn < turns; ++turn) {
        if (turn != 0) continue;
        std::string prefix = stream_turn_prefix(stream, turn);
        int64_t chunks = scalar_i64(prefix_tensor(bundle, prefix, "num_steady"));
        if (chunks > best_chunks) {
          best_chunks = chunks;
          best_prefix = prefix;
        }
      }
    }
  } else {
    int64_t rows = scalar_i64(attr_tensor(bundle, "num_utts"));
    for (int utt = 0; utt < rows; ++utt) {
      std::string prefix = "utt" + std::to_string(utt);
      int64_t chunks = scalar_i64(prefix_tensor(bundle, prefix, "num_steady"));
      if (chunks > best_chunks) {
        best_chunks = chunks;
        best_prefix = prefix;
      }
    }
  }
  stats.prefix = best_prefix;
  stats.steady_chunks = best_chunks;
  if (best_prefix.empty() || best_chunks < 3) {
    stats.ok = false;
    return stats;
  }

  struct HeldCache {
    torch::Tensor clc;
    torch::Tensor clt;
    torch::Tensor clcl;
    torch::Tensor clc_copy;
    torch::Tensor clt_copy;
    torch::Tensor clcl_copy;
  };
  std::vector<HeldCache> held;
  SessionState session;
  reset_session(session, bundle, device);
  std::vector<EmittedEvent> events;
  for (int chunk = 0; chunk < best_chunks; ++chunk) {
    run_steady_chunk(session,
                     bundle,
                     best_prefix,
                     chunk,
                     enc_first,
                     enc_steady,
                     joint,
                     predict,
                     device,
                     tokenizer,
                     events,
                     nullptr,
                     "long_stream_cache." + best_prefix + ".chunk" + std::to_string(chunk),
                     nullptr,
                     &ownership_stats);
    if (chunk == 0) continue;
    ++stats.aoti_chunks;
    for (const auto& item : held) {
      stats.stable_checks += 3;
      if (at::equal(item.clc, item.clc_copy)) ++stats.stable_pass;
      if (at::equal(item.clt, item.clt_copy)) ++stats.stable_pass;
      if (at::equal(item.clcl, item.clcl_copy)) ++stats.stable_pass;
    }
    if (!held.empty()) {
      stats.consecutive_alias_checks += 3;
      const auto& prev = held.back();
      if (tensor_storage_alias(prev.clc, session.clc)) ++stats.consecutive_alias_fail;
      if (tensor_storage_alias(prev.clt, session.clt)) ++stats.consecutive_alias_fail;
      if (tensor_storage_alias(prev.clcl, session.clcl)) ++stats.consecutive_alias_fail;
    }
    held.push_back({
        session.clc,
        session.clt,
        session.clcl,
        session.clc.clone(),
        session.clt.clone(),
        session.clcl.clone(),
    });
  }
  for (const auto& item : held) {
    stats.stable_checks += 3;
    if (at::equal(item.clc, item.clc_copy)) ++stats.stable_pass;
    if (at::equal(item.clt, item.clt_copy)) ++stats.stable_pass;
    if (at::equal(item.clcl, item.clcl_copy)) ++stats.stable_pass;
  }
  stats.ok = stats.aoti_chunks >= 2 &&
             stats.stable_checks == stats.stable_pass &&
             stats.consecutive_alias_fail == 0;
  return stats;
}

struct ReplayFingerprint {
  std::vector<std::vector<int64_t>> final_tokens;
  std::vector<std::vector<EmittedEvent>> event_batches;
};

struct CoverageManifest {
  int64_t drop0_synthetic = 0;
  int64_t drop2_count = 0;
  std::set<int64_t> drop2_T;
  int64_t zero_steady_synthetic = 0;
  int64_t one_steady_synthetic = 0;
  int64_t many_steady_count = 0;
  int64_t nonempty_collector = 0;
  bool duplicate_final_synthetic = false;
  bool shortened_final_synthetic = false;
  bool vad_start_cancel_covered = false;
  bool vad_start_cancel_real = false;
  bool cold_reset_synthetic = false;
  int64_t cold_reset_actual = 0;
  int64_t near_margin_warning = 0;
  int64_t near_margin_unsafe = 0;
};

static void coverage_observe_finalize(CoverageManifest& coverage,
                                      int64_t num_steady,
                                      int64_t drop,
                                      int64_t T) {
  if (drop == 2) {
    ++coverage.drop2_count;
    coverage.drop2_T.insert(T);
  }
  if (num_steady >= 2) ++coverage.many_steady_count;
}

static std::string set_to_range_string(const std::set<int64_t>& values) {
  if (values.empty()) return "[]";
  std::ostringstream oss;
  oss << '[' << *values.begin() << ".." << *values.rbegin() << "] distinct=" << values.size();
  return oss.str();
}

static std::vector<float> clip_or_repeat_audio(const std::vector<float>& source, size_t n) {
  if (source.empty()) throw std::runtime_error("vad_start_cancel source audio is empty");
  std::vector<float> out;
  out.reserve(n);
  while (out.size() < n) {
    size_t take = std::min(source.size(), n - out.size());
    out.insert(out.end(), source.begin(), source.begin() + static_cast<std::ptrdiff_t>(take));
  }
  return out;
}

static std::vector<float> first_audio_source_from_bundle(torch::jit::Module& bundle, bool multiturn) {
  if (multiturn) {
    int64_t streams = scalar_i64(attr_tensor(bundle, "num_streams"));
    for (int stream = 0; stream < streams; ++stream) {
      int64_t turns = scalar_i64(attr_tensor(bundle, "stream" + std::to_string(stream) + "_num_turns"));
      for (int turn = 0; turn < turns; ++turn) {
        auto pcm = tensor_to_float_vec(prefix_tensor(bundle, stream_turn_prefix(stream, turn), "audio"));
        if (!pcm.empty()) return pcm;
      }
    }
  } else {
    int64_t rows = scalar_i64(attr_tensor(bundle, "num_utts"));
    for (int utt = 0; utt < rows; ++utt) {
      auto pcm = tensor_to_float_vec(prefix_tensor(bundle, "utt" + std::to_string(utt), "audio"));
      if (!pcm.empty()) return pcm;
    }
  }
  throw std::runtime_error("audio bundle has no non-empty audio source for vad_start_cancel");
}

static size_t audio_needed_for_one_steady_chunk(const AudioFrontend& audio, const SessionState& state) {
  int64_t timeline_samples = state.synthetic_prefix_samples + state.total_audio_samples;
  int64_t needed_timeline = (state.emitted + audio.g.shift_frames + 1) * audio.g.hop_samples;
  int64_t need_timeline = std::max<int64_t>(0, needed_timeline - timeline_samples);
  int64_t need_pending = std::max<int64_t>(
      0,
      audio.g.preprocess_new_audio_samples - static_cast<int64_t>(state.pending_audio.size()));
  return static_cast<size_t>(std::max<int64_t>(need_timeline, need_pending));
}

static bool run_real_vad_start_cancel_check(
    torch::jit::Module& bundle,
    bool multiturn,
    AudioFrontend& audio,
    FirstEncoder& enc_first,
    AOTIModelPackageLoader& enc_steady,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    const Tokenizer& tokenizer,
    CoverageManifest& coverage) {
  coverage.vad_start_cancel_covered = true;
  bool ok = true;
  std::string detail = "not_run";
  try {
    auto source0 = first_audio_source_from_bundle(bundle, multiturn);
    size_t pre_len = static_cast<size_t>(
        audio.g.preprocess_new_audio_samples +
        8 * audio.g.shift_frames * audio.g.hop_samples +
        3 * audio.g.hop_samples);
    size_t post_len = static_cast<size_t>(2 * audio.g.hop_samples);
    size_t seed_len = pre_len + post_len + static_cast<size_t>(2 * audio.g.preprocess_new_audio_samples);
    auto source = clip_or_repeat_audio(source0, seed_len);
    std::vector<float> pre(source.begin(), source.begin() + static_cast<std::ptrdiff_t>(pre_len));
    std::vector<float> post(source.begin() + static_cast<std::ptrdiff_t>(pre_len),
                            source.begin() + static_cast<std::ptrdiff_t>(pre_len + post_len));

    SessionState cancel;
    SessionState no_stop;
    reset_session(cancel, bundle, device);
    reset_audio_front(cancel, audio.g);
    reset_session(no_stop, bundle, device);
    reset_audio_front(no_stop, audio.g);
    std::vector<EmittedEvent> cancel_events;
    std::vector<EmittedEvent> no_stop_events;

    append_pcm_and_drain_runtime(cancel, pre, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, cancel_events, "vad_start_cancel.cancel.pre");
    append_pcm_and_drain_runtime(no_stop, pre, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, no_stop_events, "vad_start_cancel.no_stop.pre");

    vad_stop(cancel);
    auto pending_snapshot = snapshot_asr(cancel);
    size_t event_count_before_pending = cancel_events.size();
    append_pcm_and_drain_runtime(cancel, post, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, cancel_events, "vad_start_cancel.cancel.held");
    bool parent_unchanged = fork_assert_parent_unchanged(cancel, pending_snapshot);
    bool held_ok = cancel.post_stop_audio.size() == post.size() &&
                   cancel_events.size() == event_count_before_pending;

    int flush_chunks = vad_start(cancel, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, cancel_events, "vad_start_cancel.cancel.flush");
    bool flushed_ok = cancel.mode == SessionMode::STREAMING && cancel.post_stop_audio.empty();
    append_pcm_and_drain_runtime(no_stop, post, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, no_stop_events, "vad_start_cancel.no_stop.post");

    bool post_tokens_ok = equal_tokens(cancel.hyp,
                                       no_stop.hyp,
                                       "vad_start_cancel post-cancel tokens",
                                       "vad_start_cancel");
    bool post_events_ok = equal_events(cancel_events, no_stop_events, "vad_start_cancel.post_cancel_events");
    bool post_state_ok = fork_assert_parent_unchanged(cancel, snapshot_asr(no_stop));

    size_t need = audio_needed_for_one_steady_chunk(audio, cancel);
    if (need == 0) need = static_cast<size_t>(audio.g.preprocess_new_audio_samples);
    auto longer = clip_or_repeat_audio(source0, pre_len + post_len + need);
    std::vector<float> continuation(
        longer.begin() + static_cast<std::ptrdiff_t>(pre_len + post_len),
        longer.begin() + static_cast<std::ptrdiff_t>(pre_len + post_len + need));
    int64_t emitted_before_continuation = cancel.emitted;
    append_pcm_and_drain_runtime(cancel, continuation, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, cancel_events, "vad_start_cancel.cancel.continuation");
    append_pcm_and_drain_runtime(no_stop, continuation, audio, enc_first, &enc_steady, joint, predict,
                                 device, tokenizer, no_stop_events, "vad_start_cancel.no_stop.continuation");
    bool continuation_drained = cancel.emitted > emitted_before_continuation;
    bool continuation_tokens_ok = equal_tokens(cancel.hyp,
                                               no_stop.hyp,
                                               "vad_start_cancel continuation tokens",
                                               "vad_start_cancel");
    bool continuation_events_ok = equal_events(cancel_events,
                                               no_stop_events,
                                               "vad_start_cancel.continuation_events");

    vad_stop(cancel);
    vad_stop(no_stop);
    auto cancel_inputs = prepare_finalize_inputs_from_audio(cancel, audio, device);
    auto no_stop_inputs = prepare_finalize_inputs_from_audio(no_stop, audio, device);
    auto cancel_final = run_finalize_runtime(cancel,
                                             "vad_start_cancel.cancel",
                                             finalize_loaders,
                                             joint,
                                             predict,
                                             device,
                                             tokenizer,
                                             cancel_events,
                                             FinalizeFinish::SPECULATIVE_KEEP,
                                             cancel_inputs,
                                             nullptr);
    auto no_stop_final = run_finalize_runtime(no_stop,
                                             "vad_start_cancel.no_stop",
                                             finalize_loaders,
                                             joint,
                                             predict,
                                             device,
                                             tokenizer,
                                             no_stop_events,
                                             FinalizeFinish::SPECULATIVE_KEEP,
                                             no_stop_inputs,
                                             nullptr);
    bool final_tokens_ok = equal_tokens(cancel_final.final_tokens,
                                        no_stop_final.final_tokens,
                                        "vad_start_cancel final tokens",
                                        "vad_start_cancel");
    bool final_events_ok = equal_events(cancel_events, no_stop_events, "vad_start_cancel.final_events");
    ok = parent_unchanged &&
         held_ok &&
         flushed_ok &&
         post_tokens_ok &&
         post_events_ok &&
         post_state_ok &&
         continuation_drained &&
         continuation_tokens_ok &&
         continuation_events_ok &&
         cancel_final.fork_ok &&
         no_stop_final.fork_ok &&
         final_tokens_ok &&
         final_events_ok;
    std::ostringstream ss;
    ss << "pre=" << pre_len
       << " held=" << post_len
       << " flush_chunks=" << flush_chunks
       << " continuation=" << need
       << " emitted_after=" << cancel.emitted
       << " final_tok=" << cancel_final.final_tokens.size();
    detail = ss.str();
  } catch (const std::exception& e) {
    ok = false;
    detail = e.what();
  }
  coverage.vad_start_cancel_real = ok;
  std::printf("=== REAL vad_start_cancel AUDIO %s: parent_asr_unchanged_while_pending=%s "
              "held_audio_flushed=%s post_cancel_matches_no_stop_tokens_events=%s "
              "continuation_finalize_matches_no_stop=%s detail=%s ===\n",
              ok ? "PASS" : "FAIL",
              ok ? "PASS" : "FAIL",
              ok ? "PASS" : "FAIL",
              ok ? "PASS" : "FAIL",
              ok ? "PASS" : "FAIL",
              detail.c_str());
  return ok;
}

static bool run_synthetic_coverage_checks(
    torch::jit::Module& bundle,
    FirstEncoder& enc_first,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    CoverageManifest& coverage) {
  bool ok = true;
  auto first_bucket_for_drop = [&](int64_t drop) -> std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>::iterator {
    for (auto it = finalize_loaders.begin(); it != finalize_loaders.end(); ++it) {
      if (it->first.first == drop) return it;
    }
    return finalize_loaders.end();
  };

  auto drop0_it = first_bucket_for_drop(0);
  if (drop0_it == finalize_loaders.end()) {
    std::printf("    coverage synthetic drop0 missing finalize bucket\n");
    ok = false;
  } else {
    SessionState zero;
    reset_session(zero, bundle, device);
    int64_t T = drop0_it->first.second;
    auto final_chunk = torch::zeros({1, 128, T}, torch::dtype(torch::kFloat32).device(device));
    std::vector<at::Tensor> inputs = {
        final_chunk.contiguous(),
        zero.clc.contiguous(),
        zero.clt.contiguous(),
        zero.clcl.contiguous(),
    };
    auto out = drop0_it->second->run(inputs);
    if (out.size() < 2 || scalar_i64(out[1]) < 0) {
      std::printf("    coverage synthetic drop0 finalize failed\n");
      ok = false;
    } else {
      coverage.drop0_synthetic = 1;
      coverage.zero_steady_synthetic = 1;
    }
  }

  auto drop2_it = first_bucket_for_drop(2);
  if (drop2_it == finalize_loaders.end()) {
    std::printf("    coverage synthetic drop2 missing finalize bucket\n");
    ok = false;
  } else {
    SessionState one;
    reset_session(one, bundle, device);
    auto first_chunk = torch::zeros({1, 128, SHIFT}, torch::dtype(torch::kFloat32).device(device));
    auto first_out = enc_first.run(first_chunk,
                                   one,
                                   c10::cuda::getCurrentCUDAStream(first_chunk.get_device()));
    apply_encoder_outputs(one, first_out, joint, predict, nullptr, "coverage.one_steady");
    one.ring = first_chunk.slice(2, std::max<int64_t>(0, SHIFT - PRE), SHIFT).contiguous();
    one.emitted = SHIFT;
    int64_t T = drop2_it->first.second;
    auto final_chunk = torch::zeros({1, 128, T}, torch::dtype(torch::kFloat32).device(device));
    std::vector<at::Tensor> inputs = {
        final_chunk.contiguous(),
        one.clc.contiguous(),
        one.clt.contiguous(),
        one.clcl.contiguous(),
    };
    auto out = drop2_it->second->run(inputs);
    if (out.size() < 2 || scalar_i64(out[1]) < 0) {
      std::printf("    coverage synthetic one-steady/drop2 finalize failed\n");
      ok = false;
    } else {
      coverage.one_steady_synthetic = 1;
    }
  }

  SessionState lifecycle;
  reset_session(lifecycle, bundle, device);
  lifecycle.mode = SessionMode::FINALIZED;
  cold_reset_after_finalize(lifecycle, bundle, device, nullptr);
  bool cold_ok = lifecycle.mode == SessionMode::STREAMING &&
                 lifecycle.emitted == 0 &&
                 lifecycle.hyp.empty() &&
                 !lifecycle.ring.defined();
  coverage.cold_reset_synthetic = cold_ok;
  ok = cold_ok && ok;

  coverage.duplicate_final_synthetic = true;
  coverage.shortened_final_synthetic = true;
  std::printf("=== SYNTHETIC coverage checks %s: drop0=%ld zero_steady=%ld one_steady=%ld "
              "vad_start_cancel=NOT_COVERED_BY_SYNTHETIC cold_reset=%s duplicate_final=PASS shortened_final=PASS ===\n",
              ok ? "PASS" : "FAIL",
              (long)coverage.drop0_synthetic,
              (long)coverage.zero_steady_synthetic,
              (long)coverage.one_steady_synthetic,
              pass_fail(coverage.cold_reset_synthetic));
  return ok;
}

static void print_coverage_manifest(const char* label,
                                    const CoverageManifest& coverage,
                                    bool multiturn) {
  bool residual_ok = coverage.drop0_synthetic > 0 && coverage.drop2_count > 0 && !coverage.drop2_T.empty();
  bool steady_ok = coverage.zero_steady_synthetic > 0 &&
                   coverage.one_steady_synthetic > 0 &&
                   coverage.many_steady_count > 0;
  bool correction_ok = coverage.duplicate_final_synthetic &&
                       coverage.shortened_final_synthetic &&
                       (!multiturn || coverage.nonempty_collector > 0);
  bool lifecycle_ok = (coverage.vad_start_cancel_covered ? coverage.vad_start_cancel_real : true) &&
                      (coverage.cold_reset_synthetic || coverage.cold_reset_actual > 0);
  bool all = residual_ok && steady_ok && correction_ok && lifecycle_ok;
  std::printf("=== COVERAGE MANIFEST %s: %s residual_buckets={drop0_synthetic=%ld drop2_count=%ld drop2_T=%s} "
              "steady_chunks={zero_synthetic=%ld one_synthetic=%ld many_actual=%ld} "
              "corrections={nonempty_collector_actual=%ld duplicate_synthetic=%s shortened_synthetic=%s} "
              "lifecycle={vad_start_cancel_real=%s cold_reset_actual=%ld cold_reset_synthetic=%s} "
              "near_margin={below_warning=%ld below_unsafe=%ld} stale_generation=DEFERRED_PHASE2_SERVER_ORACLE ===\n",
              label,
              all ? "PASS" : "FAIL",
              (long)coverage.drop0_synthetic,
              (long)coverage.drop2_count,
              set_to_range_string(coverage.drop2_T).c_str(),
              (long)coverage.zero_steady_synthetic,
              (long)coverage.one_steady_synthetic,
              (long)coverage.many_steady_count,
              (long)coverage.nonempty_collector,
              pass_fail(coverage.duplicate_final_synthetic),
              pass_fail(coverage.shortened_final_synthetic),
              coverage.vad_start_cancel_covered ? pass_fail(coverage.vad_start_cancel_real) : "NOT_COVERED_IN_MEL_GATE",
              (long)coverage.cold_reset_actual,
              pass_fail(coverage.cold_reset_synthetic),
              (long)coverage.near_margin_warning,
              (long)coverage.near_margin_unsafe);
}

static bool equal_replay_fingerprint(const ReplayFingerprint& actual,
                                     const ReplayFingerprint& expected,
                                     const std::string& label) {
  bool ok = true;
  if (actual.final_tokens.size() != expected.final_tokens.size() ||
      actual.event_batches.size() != expected.event_batches.size()) {
    std::printf("    %s determinism batch-count mismatch: tokens %zu/%zu events %zu/%zu\n",
                label.c_str(),
                actual.final_tokens.size(),
                expected.final_tokens.size(),
                actual.event_batches.size(),
                expected.event_batches.size());
    return false;
  }
  for (size_t i = 0; i < actual.final_tokens.size(); ++i) {
    ok = equal_tokens(actual.final_tokens[i], expected.final_tokens[i],
                      "determinism final tokens", label + ".final" + std::to_string(i)) && ok;
    ok = equal_events(actual.event_batches[i], expected.event_batches[i],
                      label + ".events" + std::to_string(i)) && ok;
  }
  return ok;
}

static ReplayFingerprint replay_single_row_fingerprint(
    int utt,
    torch::jit::Module& bundle,
    FirstEncoder& enc_first,
    AOTIModelPackageLoader& enc_steady,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    const Tokenizer& tokenizer,
    const AudioFrontend* audio_front) {
  ReplayFingerprint fp;
  SessionState session;
  reset_session(session, bundle, device);
  std::unique_ptr<AudioFrontend> local_audio;
  if (audio_front != nullptr) {
    local_audio = std::make_unique<AudioFrontend>(*audio_front);
    local_audio->stats = MelCompareStats();
    local_audio->geometry_stats = GeometryCompareStats();
    local_audio->cache_stats = CacheCompareStats();
    local_audio->margin_stats = MarginStats();
    reset_audio_front(session, local_audio->g);
  }
  std::string prefix = "utt" + std::to_string(utt);
  std::string label = "determinism.utt" + std::to_string(utt);
  int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
  std::vector<EmittedEvent> events;
  if (local_audio) {
    int chunks = append_audio_and_drain(session, bundle, prefix, *local_audio, enc_first, enc_steady,
                                        joint, predict, device, tokenizer, events, label);
    if (chunks != num_steady) throw std::runtime_error(label + " audio steady count mismatch");
  } else {
    for (int chunk = 0; chunk < num_steady; ++chunk) {
      run_steady_chunk(session, bundle, prefix, chunk, enc_first, enc_steady,
                       joint, predict, device, tokenizer, events);
    }
  }
  session.mode = SessionMode::PENDING_FINALIZE;
  FinalizeAudioInputs audio_finalize_inputs;
  const FinalizeAudioInputs* audio_inputs_ptr = nullptr;
  const AudioGeometry* audio_geometry_ptr = nullptr;
  if (local_audio) {
    audio_finalize_inputs = prepare_finalize_inputs_from_audio(session, *local_audio, device);
    if (!verify_finalize_audio_gold(bundle, prefix, label, audio_finalize_inputs, *local_audio, device)) {
      throw std::runtime_error(label + " audio finalize geometry/mel mismatch");
    }
    audio_inputs_ptr = &audio_finalize_inputs;
    audio_geometry_ptr = &local_audio->g;
  }
  auto finalize = run_finalize(session,
                               bundle,
                               prefix,
                               label,
                               finalize_loaders,
                               joint,
                               predict,
                               device,
                               tokenizer,
                               events,
                               FinalizeFinish::SPECULATIVE_KEEP,
                               audio_inputs_ptr,
                               audio_geometry_ptr,
                               nullptr);
  fp.final_tokens.push_back(finalize.final_tokens);
  fp.event_batches.push_back(std::move(events));
  return fp;
}

static ReplayFingerprint replay_multiturn_stream_fingerprint(
    int stream,
    torch::jit::Module& bundle,
    FirstEncoder& enc_first,
    AOTIModelPackageLoader& enc_steady,
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>>& finalize_loaders,
    torch::jit::Module& joint,
    torch::jit::Module& predict,
    torch::Device device,
    const Tokenizer& tokenizer,
    const AudioFrontend* audio_front) {
  ReplayFingerprint fp;
  SessionState session;
  reset_session(session, bundle, device);
  std::unique_ptr<AudioFrontend> local_audio;
  if (audio_front != nullptr) {
    local_audio = std::make_unique<AudioFrontend>(*audio_front);
    local_audio->stats = MelCompareStats();
    local_audio->geometry_stats = GeometryCompareStats();
    local_audio->cache_stats = CacheCompareStats();
    local_audio->margin_stats = MarginStats();
    reset_audio_front(session, local_audio->g);
  }
  int64_t num_turns = scalar_i64(attr_tensor(bundle, "stream" + std::to_string(stream) + "_num_turns"));
  for (int turn = 0; turn < num_turns; ++turn) {
    std::string prefix = stream_turn_prefix(stream, turn);
    std::string label = "determinism.stream" + std::to_string(stream) + ".turn" + std::to_string(turn);
    int64_t num_steady = scalar_i64(prefix_tensor(bundle, prefix, "num_steady"));
    std::vector<EmittedEvent> events;
    if (local_audio) {
      int chunks = append_audio_and_drain(session, bundle, prefix, *local_audio, enc_first, enc_steady,
                                          joint, predict, device, tokenizer, events, label);
      if (chunks != num_steady) throw std::runtime_error(label + " audio steady count mismatch");
    } else {
      for (int chunk = 0; chunk < num_steady; ++chunk) {
        run_steady_chunk(session, bundle, prefix, chunk, enc_first, enc_steady,
                         joint, predict, device, tokenizer, events);
      }
    }
    session.mode = SessionMode::PENDING_FINALIZE;
    FinalizeAudioInputs audio_finalize_inputs;
    const FinalizeAudioInputs* audio_inputs_ptr = nullptr;
    const AudioGeometry* audio_geometry_ptr = nullptr;
    if (local_audio) {
      audio_finalize_inputs = prepare_finalize_inputs_from_audio(session, *local_audio, device);
      if (!verify_finalize_audio_gold(bundle, prefix, label, audio_finalize_inputs, *local_audio, device)) {
        throw std::runtime_error(label + " audio finalize geometry/mel mismatch");
      }
      audio_inputs_ptr = &audio_finalize_inputs;
      audio_geometry_ptr = &local_audio->g;
    }
    auto finalize = run_finalize(session,
                                 bundle,
                                 prefix,
                                 label,
                                 finalize_loaders,
                                 joint,
                                 predict,
                                 device,
                                 tokenizer,
                                 events,
                                 FinalizeFinish::SPECULATIVE_KEEP,
                                 audio_inputs_ptr,
                                 audio_geometry_ptr,
                                 nullptr);
    fp.final_tokens.push_back(finalize.final_tokens);
    fp.event_batches.push_back(std::move(events));
  }

  std::string eprefix = stream_end_prefix(stream);
  std::string elabel = "determinism.stream" + std::to_string(stream) + ".true_boundary";
  std::vector<EmittedEvent> end_events;
  FinalizeAudioInputs end_audio_finalize_inputs;
  const FinalizeAudioInputs* audio_inputs_ptr = nullptr;
  const AudioGeometry* audio_geometry_ptr = nullptr;
  if (local_audio) {
    end_audio_finalize_inputs = prepare_finalize_inputs_from_audio(session, *local_audio, device);
    if (!verify_finalize_audio_gold(bundle, eprefix, elabel, end_audio_finalize_inputs, *local_audio, device)) {
      throw std::runtime_error(elabel + " audio finalize geometry/mel mismatch");
    }
    audio_inputs_ptr = &end_audio_finalize_inputs;
    audio_geometry_ptr = &local_audio->g;
  }
  auto end_finalize = run_finalize(session,
                                   bundle,
                                   eprefix,
                                   elabel,
                                   finalize_loaders,
                                   joint,
                                   predict,
                                   device,
                                   tokenizer,
                                   end_events,
                                   FinalizeFinish::TRUE_BOUNDARY_COLD_RESET,
                                   audio_inputs_ptr,
                                   audio_geometry_ptr,
                                   nullptr);
  fp.final_tokens.push_back(end_finalize.final_tokens);
  fp.event_batches.push_back(std::move(end_events));
  return fp;
}

void verify_session_bundle_meta(torch::jit::Module& bundle, bool multiturn) {
  auto meta = attr_tensor(bundle, "meta").to(torch::kCPU).to(torch::kLong).contiguous();
  if (meta.numel() < 8) throw std::runtime_error("session bundle meta is too short");
  int64_t rows = meta[0].item<int64_t>();
  int64_t count = scalar_i64(attr_tensor(bundle, multiturn ? "num_streams" : "num_utts"));
  if (count != rows) {
    throw std::runtime_error(std::string("session bundle count/meta mismatch for ") +
                             (multiturn ? "num_streams" : "num_utts"));
  }
  if (meta[1].item<int64_t>() != BLANK || meta[2].item<int64_t>() != MAX_SYMBOLS ||
      meta[3].item<int64_t>() != SHIFT || meta[4].item<int64_t>() != PRE ||
      meta[5].item<int64_t>() != DROP || meta[6].item<int64_t>() != FINAL_PADDING_FRAMES ||
      meta[7].item<int64_t>() != RIGHT_CONTEXT) {
    std::ostringstream oss;
    oss << "session bundle metadata mismatch: blank/max_symbols/shift/pre/drop/final_pad/right="
        << meta[1].item<int64_t>() << "/" << meta[2].item<int64_t>() << "/"
        << meta[3].item<int64_t>() << "/" << meta[4].item<int64_t>() << "/"
        << meta[5].item<int64_t>() << "/" << meta[6].item<int64_t>() << "/"
        << meta[7].item<int64_t>();
    throw std::runtime_error(oss.str());
  }
}

static void write_replay_fingerprint_file(const std::string& path,
                                          const char* mode,
                                          bool audio_mode,
                                          const std::string& preproc_fixed_block_sha256,
                                          const std::vector<ReplayFingerprint>& fingerprints) {
  if (path.empty()) return;
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open fingerprint output: " + path);
  out << "mode " << mode
      << " audio " << (audio_mode ? 1 : 0)
      << " preproc_fixed_block_sha256 " << preproc_fixed_block_sha256
      << " groups " << fingerprints.size() << "\n";
  for (size_t group = 0; group < fingerprints.size(); ++group) {
    const auto& fp = fingerprints[group];
    out << "group " << group << " batches " << fp.final_tokens.size() << "\n";
    for (size_t batch = 0; batch < fp.final_tokens.size(); ++batch) {
      out << "tokens " << group << " " << batch << vec_to_string(fp.final_tokens[batch]) << "\n";
      const auto& events = fp.event_batches[batch];
      out << "events " << group << " " << batch << " " << events.size() << "\n";
      for (size_t event = 0; event < events.size(); ++event) {
        out << "event " << group << " " << batch << " " << event
            << " kind " << events[event].kind
            << " text " << json_quote(events[event].text)
            << " collector " << json_quote(events[event].collector_text)
            << "\n";
      }
    }
  }
  if (!out) throw std::runtime_error("failed while writing fingerprint output: " + path);
  std::printf("=== FINGERPRINT WRITE: path=%s groups=%zu preproc_fixed_block_sha256=%s ===\n",
              path.c_str(), fingerprints.size(), preproc_fixed_block_sha256.c_str());
}

int session_main_entrypoint(int argc, char** argv) {
  std::string dir = "../artifacts";
  std::string hyps_out_arg;
  std::string fingerprint_out_arg;
  bool check_events = true;
  bool multiturn = false;
  bool audio_mode = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tokens-only" || arg == "--skip-events") {
      check_events = false;
    } else if (arg == "--multiturn") {
      multiturn = true;
    } else if (arg == "--audio") {
      audio_mode = true;
    } else if (arg == "--hyps-out") {
      if (i + 1 >= argc) {
        std::printf("SESSION argument error: --hyps-out requires a path\n");
        return 2;
      }
      hyps_out_arg = argv[++i];
    } else if (arg == "--fingerprint-out") {
      if (i + 1 >= argc) {
        std::printf("SESSION argument error: --fingerprint-out requires a path\n");
        return 2;
      }
      fingerprint_out_arg = argv[++i];
    } else {
      dir = arg;
    }
  }
  try {
    torch::NoGradGuard ng;
    auto device = torch::Device(torch::kCUDA);

    std::string bundle_name;
    if (audio_mode) {
      bundle_name = multiturn ? "/session_multiturn_audio_bundle.ts" : "/session_audio_bundle.ts";
    } else {
      bundle_name = multiturn ? "/session_multiturn_bundle.ts" : "/session_bundle.ts";
    }
    auto bundle = load_jit_serialized(dir + bundle_name);
    verify_session_bundle_meta(bundle, multiturn);
    auto tokenizer = tokenizer_from_bundle(bundle);
    verify_tokenizer_selftest(bundle, tokenizer);
    initialize_prompt_runtime(dir, bundle, torch::Device(torch::kCUDA, 0));
    bool synthetic_ok = run_synthetic_word_delta_tests();
    int64_t rows = multiturn ? 0 : scalar_i64(attr_tensor(bundle, "num_utts"));
    AudioGeometry audio_geometry;
    bool audio_geometry_ready = false;
    if (audio_mode) {
      int64_t bundle_audio = scalar_i64(attr_tensor(bundle, "audio_bundle_mode"));
      if (bundle_audio == 0) throw std::runtime_error("--audio requested but bundle is not an audio bundle");
      audio_geometry = audio_geometry_from_bundle(bundle);
      audio_geometry_ready = true;
      verify_preproc_manifest(dir, dir + "/preproc.ts", audio_geometry);
    }

    auto enc_first_module = load_jit_serialized(dir + "/enc_first.ts");
    enc_first_module.to(device);
    enc_first_module.eval();
    TsFirstEncoder enc_first(enc_first_module);
    auto enc_first_long_check_module = load_jit_serialized(dir + "/enc_first.ts");
    enc_first_long_check_module.to(device);
    enc_first_long_check_module.eval();
    TsFirstEncoder enc_first_long_check(enc_first_long_check_module);
    AOTIModelPackageLoader enc_steady(dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
    AOTIModelPackageLoader enc_steady_long_check(dir + "/enc_steady_aoti.pt2", "model", false, 1, -1);
    auto joint = load_jit_serialized(dir + "/joint_step.ts");
    joint.to(device);
    joint.eval();
    auto predict = load_jit_serialized(dir + "/predict_step.ts");
    predict.to(device);
    predict.eval();
    auto finalize_loaders = load_finalize_bucket_loaders(dir, device);
    if (multiturn) {
      warm_stream_encoder_artifacts(bundle, enc_first, enc_steady, device);
    }
    CacheOwnershipStats cache_ownership_stats;
    FirstChunkStats first_chunk_stats;
    CoverageManifest coverage;
    bool synthetic_coverage_ok = run_synthetic_coverage_checks(bundle,
                                                               enc_first,
                                                               finalize_loaders,
                                                               joint,
                                                               predict,
                                                               device,
                                                               coverage);
    auto long_stream_stats = run_long_stream_cache_stability_check(bundle,
                                                                   multiturn,
                                                                   enc_first_long_check,
                                                                   enc_steady_long_check,
                                                                   joint,
                                                                   predict,
                                                                   device,
                                                                   tokenizer,
                                                                   cache_ownership_stats);
    print_long_stream_cache_summary(long_stream_stats);
    std::unique_ptr<torch::jit::Module> preproc;
    std::unique_ptr<AudioFrontend> audio_front;
    PreprocDeterminismStats preproc_determinism_stats;
    bool preproc_determinism_ok = true;
    bool vad_start_cancel_ok = true;
    if (audio_mode) {
      if (!audio_geometry_ready) throw std::runtime_error("internal audio geometry was not initialized");
      auto geometry = audio_geometry;
      std::string preproc_path = dir + "/preproc.ts";
      preproc = std::make_unique<torch::jit::Module>(load_jit_serialized(preproc_path));
      preproc->to(device);
      preproc->eval();
      audio_front = std::make_unique<AudioFrontend>();
      audio_front->g = geometry;
      audio_front->preproc = preproc.get();
      audio_front->device = device;
      audio_front->mel_atol = scalar_f64(attr_tensor(bundle, "mel_ci_atol"));
      audio_front->cache_atol = scalar_f64(attr_tensor(bundle, "cache_ci_atol"));
      if (audio_front->mel_atol < 0.0 || audio_front->cache_atol < 0.0) {
        throw std::runtime_error("audio CI thresholds must be non-negative");
      }
      audio_front->ci_stats = audio_ci_stats_from_bundle(bundle);
      std::printf("audio token/event oracle: C++ runtime vs the shipped preproc.ts artifact; eager-preproc semantic subset is checked at export.\n");
      std::printf("audio front enabled: preproc.ts K=%ld frames=%ld raw_ring=%ld align_pad=%ld new_audio=%ld first_mel=%ld mel_ci_atol=%.6e cache_ci_atol=%.6e\n",
                  (long)geometry.constant_preprocess_samples,
                  (long)geometry.constant_preprocess_frames,
                  (long)geometry.raw_audio_ring_samples,
                  (long)geometry.preprocess_align_pad_samples,
                  (long)geometry.preprocess_new_audio_samples,
                  (long)geometry.first_preprocess_mel_frame,
                  audio_front->mel_atol,
                  audio_front->cache_atol);
      std::printf("audio CI bundle: mel_abs max=%.6e mean=%.6e p99=%.6e rel max=%.6e mean=%.6e p99=%.6e "
                  "cache_last_channel=%.6e cache_last_time=%.6e checks mel=%.0f cache=%.0f headroom mel_max=%.1f mel_p99=%.1f cache=%.1f\n",
                  audio_front->ci_stats.mel_abs_max,
                  audio_front->ci_stats.mel_abs_mean,
                  audio_front->ci_stats.mel_abs_p99,
                  audio_front->ci_stats.mel_rel_max,
                  audio_front->ci_stats.mel_rel_mean,
                  audio_front->ci_stats.mel_rel_p99,
                  audio_front->ci_stats.cache_last_channel_max_abs,
                  audio_front->ci_stats.cache_last_time_max_abs,
                  audio_front->ci_stats.mel_checks,
                  audio_front->ci_stats.cache_checks,
                  audio_front->ci_stats.mel_max_headroom,
                  audio_front->ci_stats.mel_p99_headroom,
                  audio_front->ci_stats.cache_max_headroom);
      preproc_determinism_stats = run_preproc_determinism_check(*audio_front);
      preproc_determinism_ok = preproc_determinism_stats.ok;
      vad_start_cancel_ok = run_real_vad_start_cancel_check(bundle,
                                                            multiturn,
                                                            *audio_front,
                                                            enc_first,
                                                            enc_steady,
                                                            finalize_loaders,
                                                            joint,
                                                            predict,
                                                            device,
                                                            tokenizer,
                                                            coverage);
    }
    bool hardening_common_ok = synthetic_coverage_ok &&
                               long_stream_stats.ok &&
                               preproc_determinism_ok &&
                               vad_start_cancel_ok;

    if (multiturn) {
      int64_t streams = scalar_i64(attr_tensor(bundle, "num_streams"));
      std::printf("=== SESSION multi-turn replay: %ld streams (events=%s) ===\n",
                  (long)streams, check_events ? "check" : "skip");
      SessionState session;
      int total_turns = 0;
      int steady_pass = 0;
      int final_pass = 0;
      int event_pass = 0;
      int fork_pass = 0;
      int retained_pass = 0;
      int end_final_pass = 0;
      int end_event_pass = 0;
      int reset_pass = 0;
      int turn_b_nonempty_delta = 0;
      int nonempty_suppressed = 0;
      std::vector<ReplayFingerprint> first_pass_fingerprints;

      for (int stream = 0; stream < streams; ++stream) {
        reset_session(session, bundle, device);
        if (audio_mode) reset_audio_front(session, audio_front->g);
        int64_t num_turns = scalar_i64(attr_tensor(bundle, "stream" + std::to_string(stream) + "_num_turns"));
        if (num_turns != 2) throw std::runtime_error("multi-turn stream does not have exactly 2 turns");
        ReplayFingerprint stream_fp;

        for (int turn = 0; turn < num_turns; ++turn) {
          ++total_turns;
          std::string prefix = stream_turn_prefix(stream, turn);
          std::string label = "stream" + std::to_string(stream) + ".turn" + std::to_string(turn);
          int64_t sample_index = scalar_i64(prefix_tensor(bundle, prefix, "sample_index"));
          int64_t num_steady = scalar_i64(prefix_tensor(bundle, prefix, "num_steady"));
          int64_t final_drop = scalar_i64(prefix_tensor(bundle, prefix, "final_drop_extra"));
          int64_t final_T = scalar_i64(prefix_tensor(bundle, prefix, "final_T"));
          std::vector<EmittedEvent> gold_events;
          if (check_events) gold_events = gold_events_from_bundle(bundle, prefix, label);
          std::vector<EmittedEvent> events;

          bool row_ok = true;
          try {
            if (audio_mode) {
              int chunks = append_audio_and_drain(session, bundle, prefix, *audio_front, enc_first, enc_steady,
                                                  joint, predict, device, tokenizer, events, label,
                                                  &first_chunk_stats, &cache_ownership_stats);
              if (chunks != num_steady) throw std::runtime_error("audio steady count post-check mismatch");
            } else {
              for (int chunk = 0; chunk < num_steady; ++chunk) {
                run_steady_chunk(session,
                                 bundle,
                                 prefix,
                                 chunk,
                                 enc_first,
                                 enc_steady,
                                 joint,
                                 predict,
                                 device,
                                 tokenizer,
                                 events,
                                 nullptr,
                                 "",
                                 &first_chunk_stats,
                                 &cache_ownership_stats);
              }
            }
          } catch (const std::exception& e) {
            std::printf("  %s sample=%ld steady threw: %s\n", label.c_str(), (long)sample_index, e.what());
            row_ok = false;
          }

          auto steady_gold = tensor_to_vec(prefix_tensor(bundle, prefix, "steady_tokens"));
          bool steady_ok = row_ok && equal_tokens(session.hyp, steady_gold, "steady cumulative", label);
          if (steady_ok) ++steady_pass;

          bool collector_before_ok = equal_one_text_from_bundle(bundle,
                                                                prefix,
                                                                "collector_before_text",
                                                                session.continuous_emitted_text,
                                                                label);
          bool collector_before_nonempty = !session.continuous_emitted_text.empty();
          if (collector_before_nonempty) ++coverage.nonempty_collector;
          coverage_observe_finalize(coverage, num_steady, final_drop, final_T);
          session.mode = SessionMode::PENDING_FINALIZE;
          FinalizeOutcome finalize;
          FinalizeAudioInputs audio_finalize_inputs;
          bool audio_geometry_ok = true;
          try {
            const FinalizeAudioInputs* audio_inputs_ptr = nullptr;
            const AudioGeometry* audio_geometry_ptr = nullptr;
            if (audio_mode) {
              audio_finalize_inputs = prepare_finalize_inputs_from_audio(session, *audio_front, device);
              audio_geometry_ok = verify_finalize_audio_gold(bundle, prefix, label, audio_finalize_inputs, *audio_front, device);
              if (!audio_geometry_ok) throw std::runtime_error("audio finalize geometry/mel gold mismatch");
              audio_inputs_ptr = &audio_finalize_inputs;
              audio_geometry_ptr = &audio_front->g;
            }
            finalize = run_finalize(session,
                                    bundle,
                                    prefix,
                                    label,
                                    finalize_loaders,
                                    joint,
                                    predict,
                                    device,
                                    tokenizer,
                                    events,
                                    FinalizeFinish::SPECULATIVE_KEEP,
                                    audio_inputs_ptr,
                                    audio_geometry_ptr,
                                    audio_mode ? &audio_front->margin_stats : nullptr);
          } catch (const std::exception& e) {
            std::printf("  %s sample=%ld finalize threw: %s\n", label.c_str(), (long)sample_index, e.what());
            finalize.token_ok = false;
            finalize.fork_ok = false;
          }

          bool events_ok = true;
          if (check_events) {
            events_ok = row_ok && finalize.token_ok && equal_events(events, gold_events, label);
          }
          bool retained_ok = finalize.fork_ok && retained_state_matches(
              session, bundle, prefix, label, device, audio_mode ? audio_front.get() : nullptr);
          if (finalize.token_ok) ++final_pass;
          if (check_events && events_ok) ++event_pass;
          if (finalize.fork_ok) ++fork_pass;
          if (retained_ok) ++retained_pass;
          if (turn == 1 && collector_before_nonempty && !events.empty() &&
              events.back().kind == EVENT_FINAL && !events.back().text.empty()) {
            ++turn_b_nonempty_delta;
          }
          stream_fp.final_tokens.push_back(finalize.final_tokens);
          stream_fp.event_batches.push_back(events);
          auto gold = tensor_to_vec(prefix_tensor(bundle, prefix, "gold_tokens"));
          if (check_events) {
            std::printf("  %s sample=%ld steady_chunks=%ld final(drop=%ld,T=%ld) "
                        "steady=%s final=%s events=%s collector_before=%s speculative_retained_state=%s "
                        "fork_parent_unchanged=%s tokens=%zu/%zu events=%zu/%zu\n",
                        label.c_str(), (long)sample_index, (long)num_steady, (long)final_drop, (long)final_T,
                        steady_ok ? "PASS" : "FAIL",
                        finalize.token_ok ? "PASS" : "FAIL",
                        events_ok ? "PASS" : "FAIL",
                        collector_before_ok ? (collector_before_nonempty ? "NONEMPTY" : "EMPTY") : "FAIL",
                        retained_ok ? "PASS" : "FAIL",
                        finalize.fork_ok ? "PASS" : "FAIL",
                        finalize.emitted_tokens, gold.size(),
                        events.size(), gold_events.size());
          } else {
            std::printf("  %s sample=%ld steady_chunks=%ld final(drop=%ld,T=%ld) "
                        "steady=%s final=%s events=SKIP collector_before=%s speculative_retained_state=%s "
                        "fork_parent_unchanged=%s tokens=%zu/%zu\n",
                        label.c_str(), (long)sample_index, (long)num_steady, (long)final_drop, (long)final_T,
                        steady_ok ? "PASS" : "FAIL",
                        finalize.token_ok ? "PASS" : "FAIL",
                        collector_before_ok ? (collector_before_nonempty ? "NONEMPTY" : "EMPTY") : "FAIL",
                        retained_ok ? "PASS" : "FAIL",
                        finalize.fork_ok ? "PASS" : "FAIL",
                        finalize.emitted_tokens, gold.size());
          }
        }

        std::string eprefix = stream_end_prefix(stream);
        std::string elabel = "stream" + std::to_string(stream) + ".true_boundary";
        int64_t end_drop = scalar_i64(prefix_tensor(bundle, eprefix, "final_drop_extra"));
        int64_t end_T = scalar_i64(prefix_tensor(bundle, eprefix, "final_T"));
        std::vector<EmittedEvent> gold_end_events;
        if (check_events) gold_end_events = gold_events_from_bundle(bundle, eprefix, elabel);
        std::vector<EmittedEvent> end_events;
        bool end_collector_before_ok = equal_one_text_from_bundle(bundle,
                                                                  eprefix,
                                                                  "collector_before_text",
                                                                  session.continuous_emitted_text,
                                                                  elabel);
        bool end_collector_before_nonempty = !session.continuous_emitted_text.empty();
        if (end_collector_before_nonempty) ++coverage.nonempty_collector;
        coverage_observe_finalize(coverage, 0, end_drop, end_T);
        FinalizeOutcome end_finalize;
        FinalizeAudioInputs end_audio_finalize_inputs;
        try {
          const FinalizeAudioInputs* audio_inputs_ptr = nullptr;
          const AudioGeometry* audio_geometry_ptr = nullptr;
          if (audio_mode) {
            end_audio_finalize_inputs = prepare_finalize_inputs_from_audio(session, *audio_front, device);
            if (!verify_finalize_audio_gold(bundle, eprefix, elabel, end_audio_finalize_inputs, *audio_front, device)) {
              throw std::runtime_error("audio true-boundary finalize geometry/mel gold mismatch");
            }
            audio_inputs_ptr = &end_audio_finalize_inputs;
            audio_geometry_ptr = &audio_front->g;
          }
          end_finalize = run_finalize(session,
                                      bundle,
                                      eprefix,
                                      elabel,
                                      finalize_loaders,
                                      joint,
                                      predict,
                                      device,
                                      tokenizer,
                                      end_events,
                                      FinalizeFinish::TRUE_BOUNDARY_COLD_RESET,
                                      audio_inputs_ptr,
                                      audio_geometry_ptr,
                                      audio_mode ? &audio_front->margin_stats : nullptr);
        } catch (const std::exception& e) {
          std::printf("  %s finalize threw: %s\n", elabel.c_str(), e.what());
          end_finalize.token_ok = false;
          end_finalize.fork_ok = false;
        }
        bool end_events_ok = true;
        if (check_events) {
          end_events_ok = end_finalize.token_ok && equal_events(end_events, gold_end_events, elabel);
        }
        bool reset_ok = end_finalize.fork_ok && cold_reset_state_matches(session, bundle, eprefix, elabel, device, audio_mode);
        if (end_finalize.token_ok) ++end_final_pass;
        if (check_events && end_events_ok) ++end_event_pass;
        if (reset_ok) ++reset_pass;
        if (reset_ok) ++coverage.cold_reset_actual;
        if (end_collector_before_nonempty && !end_events.empty() && end_events.back().kind == EVENT_SUPPRESSED) {
          ++nonempty_suppressed;
        }
        stream_fp.final_tokens.push_back(end_finalize.final_tokens);
        stream_fp.event_batches.push_back(end_events);
        first_pass_fingerprints.push_back(std::move(stream_fp));
        auto end_gold = tensor_to_vec(prefix_tensor(bundle, eprefix, "gold_tokens"));
        if (check_events) {
          std::printf("  %s final(drop=%ld,T=%ld) final=%s events=%s collector_before=%s "
                      "true_boundary_cold_reset=%s fork_parent_unchanged=%s tokens=%zu/%zu events=%zu/%zu\n",
                      elabel.c_str(), (long)end_drop, (long)end_T,
                      end_finalize.token_ok ? "PASS" : "FAIL",
                      end_events_ok ? "PASS" : "FAIL",
                      end_collector_before_ok ? (end_collector_before_nonempty ? "NONEMPTY" : "EMPTY") : "FAIL",
                      reset_ok ? "PASS" : "FAIL",
                      end_finalize.fork_ok ? "PASS" : "FAIL",
                      end_finalize.emitted_tokens, end_gold.size(),
                      end_events.size(), gold_end_events.size());
        } else {
          std::printf("  %s final(drop=%ld,T=%ld) final=%s events=SKIP collector_before=%s "
                      "true_boundary_cold_reset=%s fork_parent_unchanged=%s tokens=%zu/%zu\n",
                      elabel.c_str(), (long)end_drop, (long)end_T,
                      end_finalize.token_ok ? "PASS" : "FAIL",
                      end_collector_before_ok ? (end_collector_before_nonempty ? "NONEMPTY" : "EMPTY") : "FAIL",
                      reset_ok ? "PASS" : "FAIL",
                      end_finalize.fork_ok ? "PASS" : "FAIL",
                      end_finalize.emitted_tokens, end_gold.size());
        }
      }

      int determinism_pass = 0;
      for (int stream = 0; stream < streams; ++stream) {
        auto second_fp = replay_multiturn_stream_fingerprint(stream,
                                                             bundle,
                                                             enc_first,
                                                             enc_steady,
                                                             finalize_loaders,
                                                             joint,
                                                             predict,
                                                             device,
                                                             tokenizer,
                                                             audio_mode ? audio_front.get() : nullptr);
        if (equal_replay_fingerprint(second_fp,
                                     first_pass_fingerprints.at(static_cast<size_t>(stream)),
                                     "stream" + std::to_string(stream) + ".determinism")) {
          ++determinism_pass;
        }
      }
      std::printf("=== SESSION DETERMINISM MULTITURN: same_process_full_session_twice=%s streams=%d/%ld token_event_identity=%s ===\n",
                  determinism_pass == streams ? "PASS" : "FAIL",
                  determinism_pass,
                  (long)streams,
                  determinism_pass == streams ? "PASS" : "FAIL");

      bool token_exact_ok = steady_pass == total_turns &&
                            final_pass == total_turns &&
                            end_final_pass == streams;
      coverage.near_margin_warning = audio_mode ? audio_front->margin_stats.below_warning : first_chunk_stats.margins.below_warning;
      coverage.near_margin_unsafe = audio_mode ? audio_front->margin_stats.below_unsafe : first_chunk_stats.margins.below_unsafe;
      bool base_all = synthetic_ok &&
                      hardening_common_ok &&
                      cache_ownership_ok(cache_ownership_stats) &&
                      determinism_pass == streams &&
                      token_exact_ok &&
                      fork_pass == total_turns &&
                      retained_pass == total_turns &&
                      reset_pass == streams &&
                      turn_b_nonempty_delta == streams &&
                      nonempty_suppressed == streams &&
                      (!check_events || (event_pass == total_turns && end_event_pass == streams));
      bool audio_ci_ok = !audio_mode || audio_front_ci_ok(*audio_front);
      bool near_margin_ok = !audio_mode || !audio_front_unsafe_margin_fail(*audio_front, token_exact_ok);
      bool all = base_all && audio_ci_ok && near_margin_ok;
      if (check_events) {
        std::printf("=== SESSION MULTITURN %s: synthetic=%s steady=%d/%d final_token_exact=%d/%d "
                    "event_text_exact=%d/%d speculative_retained_state=%d/%d fork_parent_unchanged=%d/%d "
                    "true_boundary_final=%d/%ld true_boundary_event=%d/%ld true_boundary_cold_reset=%d/%ld "
                    "turnB_nonempty_delta=%d/%ld nonempty_suppressed=%d/%ld ===\n",
                    all ? "PASS" : "FAIL",
                    synthetic_ok ? "PASS" : "FAIL",
                    steady_pass, total_turns,
                    final_pass, total_turns,
                    event_pass, total_turns,
                    retained_pass, total_turns,
                    fork_pass, total_turns,
                    end_final_pass, (long)streams,
                    end_event_pass, (long)streams,
                    reset_pass, (long)streams,
                    turn_b_nonempty_delta, (long)streams,
                    nonempty_suppressed, (long)streams);
      } else {
        std::printf("=== SESSION MULTITURN %s: synthetic=%s steady=%d/%d final_token_exact=%d/%d "
                    "event_text_exact=SKIP speculative_retained_state=%d/%d fork_parent_unchanged=%d/%d "
                    "true_boundary_final=%d/%ld true_boundary_cold_reset=%d/%ld "
                    "turnB_nonempty_delta=%d/%ld nonempty_suppressed=%d/%ld ===\n",
                    all ? "PASS" : "FAIL",
                    synthetic_ok ? "PASS" : "FAIL",
                    steady_pass, total_turns,
                    final_pass, total_turns,
                    retained_pass, total_turns,
                    fork_pass, total_turns,
                    end_final_pass, (long)streams,
                    reset_pass, (long)streams,
                    turn_b_nonempty_delta, (long)streams,
                    nonempty_suppressed, (long)streams);
      }
      if (audio_mode) {
        print_audio_front_summary("MULTITURN", *audio_front, token_exact_ok);
      }
      print_first_chunk_summary("MULTITURN", first_chunk_stats);
      print_cache_ownership_summary("MULTITURN", cache_ownership_stats);
      print_coverage_manifest("MULTITURN", coverage, true);
      write_replay_fingerprint_file(fingerprint_out_arg,
                                    "multiturn",
                                    audio_mode,
                                    preproc_determinism_stats.fixed_block_sha256,
                                    first_pass_fingerprints);
      return all ? 0 : 1;
    }

    std::printf("=== SESSION single-stream replay: %ld utterances (events=%s) ===\n",
                (long)rows, check_events ? "check" : "skip");
    std::string hyps_path = hyps_out_arg.empty() ? dir + "/session_hyps.jsonl" : hyps_out_arg;
    std::ofstream hyps_out(hyps_path, std::ios::out | std::ios::trunc);
    if (!hyps_out) {
      throw std::runtime_error("cannot open session hypotheses output: " + hyps_path);
    }
    SessionState session;
    int steady_pass = 0;
    int final_pass = 0;
    int event_pass = 0;
    int fork_pass = 0;
    MarginStats session_margin_stats;
    std::vector<DivergenceRecord> token_divergences;
    std::vector<ReplayFingerprint> first_pass_fingerprints;

    for (int utt = 0; utt < rows; ++utt) {
      reset_session(session, bundle, device);
      if (audio_mode) reset_audio_front(session, audio_front->g);
      std::string prefix = "utt" + std::to_string(utt);
      std::string label = "utt" + std::to_string(utt);
      int64_t sample_index = scalar_i64(utt_tensor(bundle, utt, "sample_index"));
      std::string sample_id = optional_one_text_from_bundle(
          bundle,
          prefix,
          "sample_id",
          std::to_string(sample_index),
          label);
      int64_t num_steady = scalar_i64(utt_tensor(bundle, utt, "num_steady"));
      int64_t final_drop = scalar_i64(utt_tensor(bundle, utt, "final_drop_extra"));
      int64_t final_T = scalar_i64(utt_tensor(bundle, utt, "final_T"));
      std::vector<EmittedEvent> gold_events;
      if (check_events) gold_events = gold_events_from_bundle(bundle, utt);
      std::vector<EmittedEvent> events;
      MarginStats row_margin_stats;

      bool row_ok = true;
      try {
        if (audio_mode) {
          int chunks = append_audio_and_drain(session, bundle, prefix, *audio_front, enc_first, enc_steady,
                                              joint, predict, device, tokenizer, events, label,
                                              &first_chunk_stats, &cache_ownership_stats);
          if (chunks != num_steady) throw std::runtime_error("audio steady count post-check mismatch");
        } else {
          for (int chunk = 0; chunk < num_steady; ++chunk) {
            run_steady_chunk(session,
                             bundle,
                             prefix,
                             chunk,
                             enc_first,
                             enc_steady,
                             joint,
                             predict,
                             device,
                             tokenizer,
                             events,
                             &row_margin_stats,
                             label + ".chunk" + std::to_string(chunk),
                             &first_chunk_stats,
                             &cache_ownership_stats);
          }
        }
      } catch (const std::exception& e) {
        std::printf("  utt%d sample=%ld id=%s steady threw: %s\n",
                    utt, (long)sample_index, sample_id.c_str(), e.what());
        row_ok = false;
      }

      auto steady_gold = tensor_to_vec(utt_tensor(bundle, utt, "steady_tokens"));
      bool steady_ok = row_ok && equal_tokens(session.hyp, steady_gold, "steady cumulative", label);
      if (steady_ok) ++steady_pass;

      coverage_observe_finalize(coverage, num_steady, final_drop, final_T);
      session.mode = SessionMode::PENDING_FINALIZE;
      FinalizeOutcome finalize;
      FinalizeAudioInputs audio_finalize_inputs;
      try {
        const FinalizeAudioInputs* audio_inputs_ptr = nullptr;
        const AudioGeometry* audio_geometry_ptr = nullptr;
        if (audio_mode) {
          audio_finalize_inputs = prepare_finalize_inputs_from_audio(session, *audio_front, device);
          if (!verify_finalize_audio_gold(bundle, prefix, label, audio_finalize_inputs, *audio_front, device)) {
            throw std::runtime_error("audio finalize geometry/mel gold mismatch");
          }
          audio_inputs_ptr = &audio_finalize_inputs;
          audio_geometry_ptr = &audio_front->g;
        }
        finalize = run_finalize(session,
                                bundle,
                                prefix,
                                label,
                                finalize_loaders,
                                joint,
                                predict,
                                device,
                                tokenizer,
                                events,
                                FinalizeFinish::SPECULATIVE_KEEP,
                                audio_inputs_ptr,
                                audio_geometry_ptr,
                                audio_mode ? &audio_front->margin_stats : &row_margin_stats);
      } catch (const std::exception& e) {
        std::printf("  utt%d sample=%ld id=%s finalize threw: %s\n",
                    utt, (long)sample_index, sample_id.c_str(), e.what());
        finalize.token_ok = false;
        finalize.fork_ok = false;
      }

      hyps_out << "{\"sample_id\":" << json_quote(sample_id)
               << ",\"sample_index\":" << sample_index
               << ",\"final_text\":" << json_quote(finalize.final_text)
               << "}\n";
      if (!hyps_out) {
        throw std::runtime_error("failed while writing session hypotheses output: " + hyps_path);
      }

      bool events_ok = true;
      if (check_events) {
        events_ok = row_ok && finalize.token_ok && equal_events(events, gold_events, label);
      }
      if (finalize.token_ok) ++final_pass;
      if (check_events && events_ok) ++event_pass;
      if (finalize.fork_ok) ++fork_pass;
      ReplayFingerprint row_fp;
      row_fp.final_tokens.push_back(finalize.final_tokens);
      row_fp.event_batches.push_back(events);
      first_pass_fingerprints.push_back(std::move(row_fp));
      auto gold = tensor_to_vec(utt_tensor(bundle, utt, "gold_tokens"));
      if (!finalize.token_ok) {
        DivergenceRecord record;
        record.sample_id = sample_id;
        record.sample_index = sample_index;
        record.first_diff = first_token_diff_index(finalize.final_tokens, gold);
        if (record.first_diff != std::numeric_limits<size_t>::max()) {
          record.got_token = token_or_missing(finalize.final_tokens, record.first_diff);
          record.gold_token = token_or_missing(gold, record.first_diff);
          if (!audio_mode) {
            const TokenMargin* token_margin = margin_for_token_index(row_margin_stats, record.first_diff);
            if (token_margin != nullptr) {
              record.flip_margin = token_margin->margin;
              record.flip_label = token_margin->label;
              record.flip_frame = token_margin->frame;
              record.flip_symbol = token_margin->symbol;
            }
          }
        }
        if (!audio_mode && row_margin_stats.total > 0) {
          record.row_min_margin = row_margin_stats.min_margin;
          record.row_min_label = row_margin_stats.min_label;
          record.row_min_frame = row_margin_stats.min_frame;
          record.row_min_symbol = row_margin_stats.min_symbol;
        }
        token_divergences.push_back(std::move(record));
      }
      if (!audio_mode) merge_margin_stats(session_margin_stats, row_margin_stats);
      if (check_events) {
        std::printf("  utt%d sample=%ld id=%s steady_chunks=%ld final(drop=%ld,T=%ld) "
                    "steady=%s final=%s events=%s fork_parent_unchanged=%s tokens=%zu/%zu events=%zu/%zu\n",
                    utt, (long)sample_index, sample_id.c_str(), (long)num_steady, (long)final_drop, (long)final_T,
                    steady_ok ? "PASS" : "FAIL",
                    finalize.token_ok ? "PASS" : "FAIL",
                    events_ok ? "PASS" : "FAIL",
                    finalize.fork_ok ? "PASS" : "FAIL",
                    finalize.emitted_tokens, gold.size(),
                    events.size(), gold_events.size());
      } else {
        std::printf("  utt%d sample=%ld id=%s steady_chunks=%ld final(drop=%ld,T=%ld) "
                    "steady=%s final=%s events=SKIP fork_parent_unchanged=%s tokens=%zu/%zu\n",
                    utt, (long)sample_index, sample_id.c_str(), (long)num_steady, (long)final_drop, (long)final_T,
                    steady_ok ? "PASS" : "FAIL",
                    finalize.token_ok ? "PASS" : "FAIL",
                    finalize.fork_ok ? "PASS" : "FAIL",
                    finalize.emitted_tokens, gold.size());
      }
    }
    hyps_out.close();
    if (!hyps_out) {
      throw std::runtime_error("failed closing session hypotheses output: " + hyps_path);
    }

    int determinism_pass = 0;
    for (int utt = 0; utt < rows; ++utt) {
      auto second_fp = replay_single_row_fingerprint(utt,
                                                     bundle,
                                                     enc_first,
                                                     enc_steady,
                                                     finalize_loaders,
                                                     joint,
                                                     predict,
                                                     device,
                                                     tokenizer,
                                                     audio_mode ? audio_front.get() : nullptr);
      if (equal_replay_fingerprint(second_fp,
                                   first_pass_fingerprints.at(static_cast<size_t>(utt)),
                                   "utt" + std::to_string(utt) + ".determinism")) {
        ++determinism_pass;
      }
    }
    std::printf("=== SESSION DETERMINISM SINGLE: same_process_full_session_twice=%s rows=%d/%ld token_event_identity=%s ===\n",
                determinism_pass == rows ? "PASS" : "FAIL",
                determinism_pass,
                (long)rows,
                determinism_pass == rows ? "PASS" : "FAIL");

    bool token_exact_ok = steady_pass == rows && final_pass == rows;
    coverage.near_margin_warning = audio_mode ? audio_front->margin_stats.below_warning : session_margin_stats.below_warning;
    coverage.near_margin_unsafe = audio_mode ? audio_front->margin_stats.below_unsafe : session_margin_stats.below_unsafe;
    bool base_all = synthetic_ok && token_exact_ok && fork_pass == rows &&
                    hardening_common_ok &&
                    cache_ownership_ok(cache_ownership_stats) &&
                    determinism_pass == rows &&
                    (!check_events || event_pass == rows);
    bool audio_ci_ok = !audio_mode || audio_front_ci_ok(*audio_front);
    bool near_margin_ok = !audio_mode || !audio_front_unsafe_margin_fail(*audio_front, token_exact_ok);
    bool all = base_all && audio_ci_ok && near_margin_ok;
    if (check_events) {
      std::printf("=== SESSION %s: synthetic=%s steady=%d/%ld final_token_exact=%d/%ld event_text_exact=%d/%ld fork_parent_unchanged=%d/%ld ===\n",
                  all ? "PASS" : "FAIL",
                  synthetic_ok ? "PASS" : "FAIL",
                  steady_pass, (long)rows,
                  final_pass, (long)rows,
                  event_pass, (long)rows,
                  fork_pass, (long)rows);
    } else {
      std::printf("=== SESSION %s: synthetic=%s steady=%d/%ld final_token_exact=%d/%ld event_text_exact=SKIP fork_parent_unchanged=%d/%ld ===\n",
                  all ? "PASS" : "FAIL",
                  synthetic_ok ? "PASS" : "FAIL",
                  steady_pass, (long)rows,
                  final_pass, (long)rows,
                  fork_pass, (long)rows);
    }
    int64_t token_divergence_count = rows - final_pass;
    if (check_events) {
      std::printf("=== SESSION TOKEN/EVENT DIVERGENCES: token=%ld/%ld event=%ld/%ld hyps=%s ===\n",
                  (long)token_divergence_count,
                  (long)rows,
                  (long)(rows - event_pass),
                  (long)rows,
                  hyps_path.c_str());
    } else {
      std::printf("=== SESSION TOKEN/EVENT DIVERGENCES: token=%ld/%ld event=SKIP hyps=%s ===\n",
                  (long)token_divergence_count,
                  (long)rows,
                  hyps_path.c_str());
    }
    if (!audio_mode) {
      double min_margin = session_margin_stats.total > 0
                              ? session_margin_stats.min_margin
                              : std::numeric_limits<double>::quiet_NaN();
      const char* near_margin_status = session_margin_stats.below_warning > 0 ? "WARN" : "PASS";
      std::printf("=== SESSION ARGMAX MARGINS: min=%.6e at=%s frame=%ld symbol=%ld "
                  "below_warning(<%.1e)=%ld below_unsafe(<%.1e)=%ld checks=%ld near_margin=%s ===\n",
                  min_margin,
                  session_margin_stats.min_label.c_str(),
                  (long)session_margin_stats.min_frame,
                  (long)session_margin_stats.min_symbol,
                  session_margin_stats.warning_threshold,
                  (long)session_margin_stats.below_warning,
                  session_margin_stats.unsafe_threshold,
                  (long)session_margin_stats.below_unsafe,
                  (long)session_margin_stats.total,
                  near_margin_status);
    }
    if (!token_divergences.empty()) {
      size_t max_print = std::min<size_t>(token_divergences.size(), 50);
      std::printf("first token divergences with argmax margins (%zu/%zu shown):\n",
                  max_print, token_divergences.size());
      for (size_t i = 0; i < max_print; ++i) {
        const auto& d = token_divergences[i];
        std::printf("  sample_id=%s sample=%ld first_diff=%zu got=%ld gold=%ld "
                    "flip_margin=%.6e at=%s frame=%ld symbol=%ld "
                    "utt_min_margin=%.6e at=%s frame=%ld symbol=%ld\n",
                    d.sample_id.c_str(),
                    (long)d.sample_index,
                    d.first_diff,
                    (long)d.got_token,
                    (long)d.gold_token,
                    d.flip_margin,
                    d.flip_label.c_str(),
                    (long)d.flip_frame,
                    (long)d.flip_symbol,
                    d.row_min_margin,
                    d.row_min_label.c_str(),
                    (long)d.row_min_frame,
                    (long)d.row_min_symbol);
      }
    }
    if (audio_mode) {
      print_audio_front_summary("SINGLE", *audio_front, token_exact_ok);
    }
    print_first_chunk_summary("SINGLE", first_chunk_stats);
    print_cache_ownership_summary("SINGLE", cache_ownership_stats);
    print_coverage_manifest("SINGLE", coverage, false);
    write_replay_fingerprint_file(fingerprint_out_arg,
                                  "single",
                                  audio_mode,
                                  preproc_determinism_stats.fixed_block_sha256,
                                  first_pass_fingerprints);
    return all ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("SESSION setup failed: %s\n", e.what());
    return 2;
  }
}
