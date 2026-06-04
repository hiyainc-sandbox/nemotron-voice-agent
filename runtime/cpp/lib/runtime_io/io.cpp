#include "lib/runtime_io/io.h"

#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace runtime_io {
namespace {

struct Sha256Ctx {
  std::array<uint8_t, 64> data{};
  uint32_t datalen = 0;
  uint64_t bitlen = 0;
  std::array<uint32_t, 8> state{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
};

uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32U - n));
}

void sha256_transform(Sha256Ctx& ctx, const uint8_t data[64]) {
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

void sha256_update(Sha256Ctx& ctx, const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ctx.data[ctx.datalen++] = data[i];
    if (ctx.datalen == 64) {
      sha256_transform(ctx, ctx.data.data());
      ctx.bitlen += 512;
      ctx.datalen = 0;
    }
  }
}

std::string sha256_final(Sha256Ctx& ctx) {
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

}  // namespace

bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

bool directory_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
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

std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open manifest: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t skip_ws(const std::string& s, size_t pos) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
  return pos;
}

size_t find_matching_json_delim(const std::string& s, size_t open_pos) {
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

std::string json_value_for_key(const std::string& object,
                               const std::string& key,
                               bool required) {
  std::string needle = "\"" + key + "\"";
  size_t key_pos = object.find(needle);
  if (key_pos == std::string::npos) {
    if (required) throw std::runtime_error("manifest missing key: " + key);
    return "";
  }
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

std::string json_string_field(const std::string& object,
                              const std::string& key,
                              bool required) {
  std::string value = json_value_for_key(object, key, required);
  if (value.empty() && !required) return "";
  value = value.substr(skip_ws(value, 0));
  if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
    throw std::runtime_error("manifest key is not a string: " + key);
  }
  return value.substr(1, value.size() - 2);
}

int64_t json_int_field(const std::string& object, const std::string& key) {
  std::string value = json_value_for_key(object, key);
  size_t n = 0;
  long long out = std::stoll(value, &n);
  n = skip_ws(value, n);
  if (n != value.size()) throw std::runtime_error("manifest key is not an integer: " + key);
  return out;
}

}  // namespace runtime_io
