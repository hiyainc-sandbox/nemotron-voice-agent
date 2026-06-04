// 1.3b — C++ continuous-finalize compute path.
//
// Phase A: load finalize_bundle.ts, fork/clone parent encoder + decoder state, decode-continuation over the eager
// finalize enc_out, and assert token-exactness vs finalize_ref and NeMo stream+finalize oracle. Also proves the parent
// state is byte-identical after the fork decode.
//
// Phase B: if per-exact-T finalize AOTI buckets + shared weights exist, wire one CUDA weight set into every bucket,
// route each row by (drop_extra, chunk T), run on contiguous cloned inputs, then decode and assert the same gold tokens.
#include "lib/runtime_io/jit_load.h"

#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

using torch::inductor::AOTIModelPackageLoader;
namespace fs = std::filesystem;

static constexpr int BLANK = 1024;
static constexpr int MAX_SYMBOLS = 10;
static constexpr int SHIFT = 16;
static constexpr int PRE = 9;
static constexpr int DROP = 2;
static constexpr int RIGHT_CONTEXT = 1;
static constexpr int FINAL_PADDING_FRAMES = 32;
static constexpr int ATT_CONTEXT_LEFT = 70;
static constexpr int ATT_CONTEXT_RIGHT = 1;
static constexpr const char* MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b";

struct ParentState {
  torch::Tensor clc;
  torch::Tensor clt;
  torch::Tensor clcl;
  torch::Tensor g;
  torch::Tensor h;
  torch::Tensor c;
};

static bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static bool directory_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string row_attr(int row, const char* name) {
  return "row" + std::to_string(row) + "_" + std::string(name);
}

static torch::Tensor attr_tensor(torch::jit::Module& module, const std::string& name) {
  return module.attr(name).toTensor();
}

static torch::Tensor row_tensor(torch::jit::Module& bundle, int row, const char* name) {
  return attr_tensor(bundle, row_attr(row, name));
}

static int64_t scalar_i64(torch::Tensor tensor) {
  return tensor.to(torch::kCPU).reshape({-1})[0].item<int64_t>();
}

static std::vector<int64_t> tensor_to_vec(torch::Tensor tensor) {
  auto flat = tensor.to(torch::kCPU).to(torch::kLong).contiguous().view({-1});
  std::vector<int64_t> out;
  out.reserve(flat.numel());
  for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(flat[i].item<int64_t>());
  return out;
}

static std::string vec_to_string(const std::vector<int64_t>& values) {
  std::ostringstream oss;
  for (auto value : values) oss << ' ' << value;
  return oss.str();
}

static bool equal_tokens(const std::vector<int64_t>& got,
                         const std::vector<int64_t>& gold,
                         const char* label,
                         int row) {
  bool ok = got == gold;
  if (!ok) {
    std::printf("    row%d %s token mismatch: got_len=%zu gold_len=%zu\n", row, label, got.size(), gold.size());
    std::printf("      got :%s\n", vec_to_string(got).c_str());
    std::printf("      gold:%s\n", vec_to_string(gold).c_str());
  }
  return ok;
}

static bool tensor_equal(const char* name, const torch::Tensor& actual, const torch::Tensor& expected) {
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

static ParentState clone_state(const ParentState& state) {
  return {
      state.clc.clone(),
      state.clt.clone(),
      state.clcl.clone(),
      state.g.clone(),
      state.h.clone(),
      state.c.clone(),
  };
}

static bool fork_assert_parent_unchanged(const ParentState& parent, const ParentState& snapshot) {
  bool ok = true;
  ok = tensor_equal("cache_last_channel", parent.clc, snapshot.clc) && ok;
  ok = tensor_equal("cache_last_time", parent.clt, snapshot.clt) && ok;
  ok = tensor_equal("cache_last_channel_len", parent.clcl, snapshot.clcl) && ok;
  ok = tensor_equal("pred_out", parent.g, snapshot.g) && ok;
  ok = tensor_equal("decoder_state.h", parent.h, snapshot.h) && ok;
  ok = tensor_equal("decoder_state.c", parent.c, snapshot.c) && ok;
  return ok;
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

static std::map<std::pair<int64_t, int64_t>, std::string> discover_finalize_buckets(const std::string& buckets_dir) {
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

struct Sha256Ctx {
  std::array<uint8_t, 64> data{};
  uint32_t datalen = 0;
  uint64_t bitlen = 0;
  std::array<uint32_t, 8> state{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
};

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

static std::string sha256_file(const std::string& path) {
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

static BucketManifest load_bucket_manifest(const std::string& manifest_path) {
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

static void verify_bucket_manifest(const BucketManifest& manifest,
                                   const std::map<std::pair<int64_t, int64_t>, std::string>& discovered,
                                   const std::string& buckets_dir,
                                   const std::string& shared_weights_pt) {
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
  std::string weights_sha = sha256_file(shared_weights_pt);
  if (weights_sha != c.weights_sha256) {
    throw std::runtime_error("shared weights sha256 mismatch: manifest=" + c.weights_sha256 + " actual=" + weights_sha);
  }

  std::set<std::pair<int64_t, int64_t>> manifest_keys;
  std::set<std::string> manifest_pkgs;
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
  }

  for (const auto& kv : discovered) {
    if (manifest_keys.find(kv.first) == manifest_keys.end()) {
      throw std::runtime_error("bucket file is not listed in manifest: " + kv.second);
    }
  }
}

static std::unordered_map<std::string, at::Tensor> load_shared_constants(const std::string& weights_path,
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

struct BucketConstants {
  std::unordered_map<std::string, at::Tensor> values;
  size_t direct_matches = 0;
  size_t alias_fallbacks = 0;
};

static const at::Tensor* resolve_shared_constant(const std::unordered_map<std::string, at::Tensor>& shared_constants,
                                                 const std::string& fqn,
                                                 bool& used_alias) {
  auto it = shared_constants.find(fqn);
  if (it != shared_constants.end()) {
    used_alias = false;
    return &it->second;
  }

  // Compatibility fallback only: old shared-weight files used e.* while current buckets use encoder.*.
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

static BucketConstants constants_for_bucket(
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

static void decode_range(torch::jit::Module& joint,
                         torch::jit::Module& predict,
                         const torch::Tensor& enc_out,
                         int64_t enc_len,
                         torch::Tensor& g,
                         torch::Tensor& h,
                         torch::Tensor& c,
                         std::vector<int64_t>& hyp) {
  if (enc_len < 0 || enc_len > enc_out.size(2)) {
    throw std::runtime_error("enc_len out of range for enc_out");
  }
  auto f = enc_out.transpose(1, 2).contiguous();  // [1,T,1024]
  auto dev = f.device();
  for (int64_t t = 0; t < enc_len; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor();
      int64_t k = logits.reshape({-1}).argmax().item<int64_t>();
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

static ParentState load_parent(torch::jit::Module& bundle, int row, torch::Device device) {
  return {
      row_tensor(bundle, row, "cache_last_channel").to(device),
      row_tensor(bundle, row, "cache_last_time").to(device),
      row_tensor(bundle, row, "cache_last_channel_len").to(device),
      row_tensor(bundle, row, "pre_final_pred_out").to(device),
      row_tensor(bundle, row, "pre_final_h").to(device),
      row_tensor(bundle, row, "pre_final_c").to(device),
  };
}

static std::vector<int64_t> run_decode_from_eager(torch::jit::Module& bundle,
                                                  int row,
                                                  ParentState& fork,
                                                  torch::jit::Module& joint,
                                                  torch::jit::Module& predict,
                                                  torch::Device device) {
  std::vector<int64_t> hyp = tensor_to_vec(row_tensor(bundle, row, "pre_final_tokens"));
  auto enc_out = row_tensor(bundle, row, "eager_enc_out").to(device);
  int64_t enc_len = scalar_i64(row_tensor(bundle, row, "eager_enc_len"));
  decode_range(joint, predict, enc_out, enc_len, fork.g, fork.h, fork.c, hyp);
  return hyp;
}

static std::vector<int64_t> run_decode_from_aoti(AOTIModelPackageLoader& loader,
                                                 torch::jit::Module& bundle,
                                                 int row,
                                                 ParentState& fork,
                                                 torch::jit::Module& joint,
                                                 torch::jit::Module& predict,
                                                 torch::Device device) {
  std::vector<at::Tensor> inputs = {
      row_tensor(bundle, row, "chunk_mel").to(device).contiguous(),
      fork.clc.contiguous(),
      fork.clt.contiguous(),
      fork.clcl.contiguous(),
  };
  auto out = loader.run(inputs);
  if (out.size() < 2) throw std::runtime_error("finalize AOTI bucket returned fewer than 2 outputs");

  std::vector<int64_t> hyp = tensor_to_vec(row_tensor(bundle, row, "pre_final_tokens"));
  int64_t enc_len = scalar_i64(out[1]);
  decode_range(joint, predict, out[0], enc_len, fork.g, fork.h, fork.c, hyp);
  return hyp;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng;
  auto device = torch::Device(torch::kCUDA);

  auto joint = load_jit_serialized(dir + "/joint_step.ts");
  joint.to(device);
  joint.eval();
  auto predict = load_jit_serialized(dir + "/predict_step.ts");
  predict.to(device);
  predict.eval();
  auto bundle = load_jit_serialized(dir + "/finalize_bundle.ts");

  auto meta = attr_tensor(bundle, "meta").to(torch::kCPU).to(torch::kLong).contiguous();
  int64_t rows = meta[0].item<int64_t>();
  if (meta[1].item<int64_t>() != BLANK || meta[2].item<int64_t>() != MAX_SYMBOLS ||
      meta[3].item<int64_t>() != SHIFT || meta[4].item<int64_t>() != PRE ||
      meta[5].item<int64_t>() != DROP) {
    std::printf("METADATA MISMATCH: bundle blank/msym/shift/pre/drop = %ld/%ld/%ld/%ld/%ld\n",
                meta[1].item<int64_t>(), meta[2].item<int64_t>(), meta[3].item<int64_t>(),
                meta[4].item<int64_t>(), meta[5].item<int64_t>());
    return 2;
  }
  int64_t num_rows = scalar_i64(attr_tensor(bundle, "num_rows"));
  if (num_rows != rows) {
    std::printf("num_rows mismatch: meta=%ld buffer=%ld\n", (long)rows, (long)num_rows);
    return 2;
  }

  std::printf("=== Phase A: eager finalize enc_out decode-continuation (%ld rows) ===\n", (long)rows);
  bool phase_a_ok = true;
  for (int row = 0; row < rows; ++row) {
    int64_t drop_extra = scalar_i64(row_tensor(bundle, row, "drop_extra"));
    bool emitted_gt0 = row_tensor(bundle, row, "emitted_gt0").to(torch::kCPU).item<bool>();
    int64_t enc_len = scalar_i64(row_tensor(bundle, row, "eager_enc_len"));
    auto parent = load_parent(bundle, row, device);
    auto snapshot = clone_state(parent);
    auto fork = clone_state(parent);

    std::vector<int64_t> got;
    bool row_ok = true;
    try {
      got = run_decode_from_eager(bundle, row, fork, joint, predict, device);
    } catch (const std::exception& e) {
      std::printf("  row%d Phase A decode threw: %s\n", row, e.what());
      row_ok = false;
    }
    auto gold_ref = tensor_to_vec(row_tensor(bundle, row, "finalize_ref_final_tokens"));
    auto gold_nemo = tensor_to_vec(row_tensor(bundle, row, "nemo_stream_finalize_tokens"));
    bool ref_ok = row_ok && equal_tokens(got, gold_ref, "finalize_ref", row);
    bool nemo_ok = row_ok && equal_tokens(got, gold_nemo, "nemo_stream_finalize", row);
    bool fork_ok = fork_assert_parent_unchanged(parent, snapshot);
    bool meta_ok = ((drop_extra != 0) == emitted_gt0);
    if (!meta_ok) {
      std::printf("    row%d metadata mismatch: drop_extra=%ld emitted_gt0=%d\n",
                  row, (long)drop_extra, (int)emitted_gt0);
    }
    bool all = ref_ok && nemo_ok && fork_ok && meta_ok;
    phase_a_ok = phase_a_ok && all;
    std::printf("  row%d drop=%ld emitted_gt0=%d enc_len=%ld tokens=%zu: "
                "finalize_ref=%s nemo=%s FORK_ASSERT=%s\n",
                row, (long)drop_extra, (int)emitted_gt0, (long)enc_len, got.size(),
                ref_ok ? "PASS" : "FAIL", nemo_ok ? "PASS" : "FAIL",
                fork_ok ? "PASS" : "FAIL");
  }

  // Prefer the stripped buckets (the deployable form: tiny wrapper .so, weights supplied via load_constants); fall back
  // to the unstripped finalize_buckets/ if present. (1.3b-enc-scale moved the buckets to stripped_finalize_buckets/.)
  std::string buckets_dir = dir + "/stripped_finalize_buckets";
  if (!directory_exists(buckets_dir)) buckets_dir = dir + "/finalize_buckets";
  std::string shared_weights = dir + "/finalize_shared_weights.ts";
  std::string shared_weights_pt = dir + "/finalize_shared_weights.pt";
  bool phase_b_present = false;
  bool phase_b_ok = true;
  if (!directory_exists(buckets_dir) || !file_exists(shared_weights)) {
    std::printf("PHASE B skipped (finalize buckets not present)\n");
  } else {
    auto bucket_paths = discover_finalize_buckets(buckets_dir);
    if (bucket_paths.empty()) {
      std::printf("PHASE B skipped (finalize buckets not present)\n");
    } else {
      phase_b_present = true;
      std::printf("=== Phase B: AOTI finalize encoder buckets + decode-continuation (%zu buckets) ===\n",
                  bucket_paths.size());
      try {
        std::string manifest_path = buckets_dir + "/manifest.json";
        if (!file_exists(manifest_path)) {
          throw std::runtime_error("finalize bucket manifest is required when buckets are present: " + manifest_path);
        }
        auto manifest = load_bucket_manifest(manifest_path);
        verify_bucket_manifest(manifest, bucket_paths, buckets_dir, shared_weights_pt);
        std::printf("  manifest verified: %zu buckets, weights_sha256=%s\n",
                    manifest.buckets.size(), manifest.contract.weights_sha256.c_str());

        auto shared_constants = load_shared_constants(shared_weights, device);
        std::printf("  loaded shared constants: %zu primary FQN entries from %s\n",
                    shared_constants.size(), shared_weights.c_str());

        std::map<std::pair<int64_t, int64_t>, std::unique_ptr<AOTIModelPackageLoader>> loaders;
        for (const auto& kv : bucket_paths) {
          int64_t drop = kv.first.first;
          int64_t T = kv.first.second;
          const std::string& pkg = kv.second;
          auto loader = std::make_unique<AOTIModelPackageLoader>(pkg, "model", /*run_single_threaded=*/false,
                                                                 /*num_runners=*/1, /*device_index=*/-1);
          auto bucket_constants = constants_for_bucket(shared_constants, *loader, pkg);
          // C++ signature: (constants_map, use_inactive, check_full_update, user_managed).
          loader->load_constants(bucket_constants.values, /*use_inactive=*/false,
                                 /*check_full_update=*/false, /*user_managed=*/true);
          std::printf("  bucket drop=%ld T=%ld: loaded %zu constants (direct=%zu alias_fallback=%zu) -> %s\n",
                      (long)drop, (long)T, bucket_constants.values.size(),
                      bucket_constants.direct_matches, bucket_constants.alias_fallbacks, pkg.c_str());
          loaders.emplace(kv.first, std::move(loader));
        }

        for (int row = 0; row < rows; ++row) {
          int64_t drop_extra = scalar_i64(row_tensor(bundle, row, "drop_extra"));
          auto chunk_mel = row_tensor(bundle, row, "chunk_mel");
          int64_t chunk_T = chunk_mel.size(chunk_mel.dim() - 1);
          auto loader_it = loaders.find(std::make_pair(drop_extra, chunk_T));
          if (loader_it == loaders.end()) {
            // FAIL-CLOSED (enc-scale review B2): no bucket -> no finalize encoder -> dropped final transcript.
            // Until a validated eager fallback exists, treat as a hard failure, not a skip.
            std::printf("  row%d drop=%ld T=%ld: no bucket for (drop,T) -> FAIL (no validated fallback)\n",
                        row, (long)drop_extra, (long)chunk_T);
            phase_b_ok = false;
            continue;
          }

          auto parent = load_parent(bundle, row, device);
          auto snapshot = clone_state(parent);
          auto fork = clone_state(parent);
          std::vector<int64_t> got;
          bool row_ok = true;
          try {
            got = run_decode_from_aoti(*loader_it->second, bundle, row, fork, joint, predict, device);
          } catch (const std::exception& e) {
            std::printf("  row%d Phase B threw: %s\n", row, e.what());
            row_ok = false;
          }
          auto gold_ref = tensor_to_vec(row_tensor(bundle, row, "finalize_ref_final_tokens"));
          auto gold_nemo = tensor_to_vec(row_tensor(bundle, row, "nemo_stream_finalize_tokens"));
          bool ref_ok = row_ok && equal_tokens(got, gold_ref, "finalize_ref", row);
          bool nemo_ok = row_ok && equal_tokens(got, gold_nemo, "nemo_stream_finalize", row);
          bool fork_ok = fork_assert_parent_unchanged(parent, snapshot);
          bool all = ref_ok && nemo_ok && fork_ok;
          phase_b_ok = phase_b_ok && all;
          std::printf("  row%d drop=%ld T=%ld tokens=%zu: finalize_ref=%s nemo=%s FORK_ASSERT=%s\n",
                      row, (long)drop_extra, (long)chunk_T, got.size(),
                      ref_ok ? "PASS" : "FAIL", nemo_ok ? "PASS" : "FAIL",
                      fork_ok ? "PASS" : "FAIL");
        }
      } catch (const std::exception& e) {
        std::printf("  Phase B setup threw: %s\n", e.what());
        phase_b_ok = false;
      }
    }
  }

  std::printf("=== FINALIZE %s: PhaseA=%s PhaseB=%s ===\n",
              (phase_a_ok && phase_b_ok) ? "PASS" : "FAIL",
              phase_a_ok ? "PASS" : "FAIL",
              phase_b_present ? (phase_b_ok ? "PASS" : "FAIL") : "SKIPPED");
  return (phase_a_ok && phase_b_ok) ? 0 : 1;
}
