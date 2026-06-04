#include "lib/ws/handshake.h"

#include "lib/runtime_io/picohttpparser.h"

#include <openssl/evp.h>
#include <openssl/sha.h>

#include <array>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace ws_handshake {
namespace {

constexpr const char* kWebSocketGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

std::string ascii_lower(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (unsigned char ch : value) {
    out.push_back(static_cast<char>(std::tolower(ch)));
  }
  return out;
}

std::string trim_ascii(std::string_view value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

std::map<std::string, std::string>::const_iterator find_header(
    const HttpRequest& req,
    const char* name) {
  return req.headers.find(name);
}

bool header_equals_ci(const HttpRequest& req, const char* name, const char* expected) {
  auto it = find_header(req, name);
  if (it == req.headers.end()) return false;
  return ascii_lower(trim_ascii(it->second)) == expected;
}

bool header_has_comma_token_ci(const HttpRequest& req,
                               const char* name,
                               const char* expected) {
  auto it = find_header(req, name);
  if (it == req.headers.end()) return false;
  std::string_view value = it->second;
  size_t pos = 0;
  while (pos <= value.size()) {
    size_t comma = value.find(',', pos);
    std::string token = trim_ascii(value.substr(
        pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos));
    if (ascii_lower(token) == expected) return true;
    if (comma == std::string_view::npos) break;
    pos = comma + 1;
  }
  return false;
}

bool has_get_body_headers(const HttpRequest& req) {
  auto content_length = find_header(req, "content-length");
  if (content_length != req.headers.end()) {
    std::string value = trim_ascii(content_length->second);
    if (value.empty()) return true;
    uint64_t parsed = 0;
    for (unsigned char ch : value) {
      if (!std::isdigit(ch)) return true;
      uint64_t digit = static_cast<uint64_t>(ch - '0');
      if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
        return true;
      }
      parsed = parsed * 10 + digit;
    }
    if (parsed != 0) return true;
  }
  auto transfer_encoding = find_header(req, "transfer-encoding");
  return transfer_encoding != req.headers.end() &&
         !trim_ascii(transfer_encoding->second).empty();
}

size_t header_end_pos(const std::string& buffer) {
  size_t crlf = buffer.find("\r\n\r\n");
  if (crlf != std::string::npos) return crlf + 4;
  size_t lf = buffer.find("\n\n");
  if (lf != std::string::npos) return lf + 2;
  return std::string::npos;
}

bool is_base64_char(char ch) {
  return (ch >= 'A' && ch <= 'Z') ||
         (ch >= 'a' && ch <= 'z') ||
         (ch >= '0' && ch <= '9') ||
         ch == '+' ||
         ch == '/';
}

bool valid_sec_websocket_key(const std::string& raw_key) {
  std::string key = trim_ascii(raw_key);
  if (key.size() != 24) return false;
  for (size_t i = 0; i < 22; ++i) {
    if (!is_base64_char(key[i])) return false;
  }
  if (key[22] != '=' || key[23] != '=') return false;

  std::array<unsigned char, 24> in{};
  std::copy(key.begin(), key.end(), in.begin());
  std::array<unsigned char, 24> decoded{};
  const int decoded_len = EVP_DecodeBlock(decoded.data(), in.data(), static_cast<int>(key.size()));
  return decoded_len >= 2 && decoded_len - 2 == 16;
}

const char* reason_phrase(int status) {
  switch (status) {
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 431:
      return "Request Header Fields Too Large";
    case 503:
      return "Service Unavailable";
    default:
      return "Error";
  }
}

}  // namespace

ParseResult parse_http_request(const std::string& buffer, HttpRequest& out) {
  const size_t header_end = header_end_pos(buffer);
  if (header_end == std::string::npos) {
    if (buffer.size() > kMaxHttpHeaderBytes) return ParseResult::OVERSIZE_HEADERS;
  } else if (header_end > kMaxHttpHeaderBytes) {
    return ParseResult::OVERSIZE_HEADERS;
  }

  constexpr size_t kMaxHeaders = 128;
  std::array<phr_header, kMaxHeaders> headers{};
  const char* method = nullptr;
  const char* path = nullptr;
  size_t method_len = 0;
  size_t path_len = 0;
  int minor_version = 0;
  size_t num_headers = headers.size();
  const size_t parse_len = header_end == std::string::npos ? buffer.size() : header_end;
  int parsed = phr_parse_request(buffer.data(),
                                 parse_len,
                                 &method,
                                 &method_len,
                                 &path,
                                 &path_len,
                                 &minor_version,
                                 headers.data(),
                                 &num_headers,
                                 0);
  if (parsed == -2) return ParseResult::NEED_MORE;
  if (parsed == -1) return ParseResult::MALFORMED;
  if (parsed <= 0 || static_cast<size_t>(parsed) > kMaxHttpHeaderBytes) {
    return ParseResult::OVERSIZE_HEADERS;
  }
  if (num_headers == kMaxHeaders && header_end != std::string::npos) {
    return ParseResult::MALFORMED;
  }

  HttpRequest req;
  req.method.assign(method, method_len);
  req.path.assign(path, path_len);

  std::string last_key;
  for (size_t i = 0; i < num_headers; ++i) {
    const phr_header& header = headers[i];
    if (header.name == nullptr) {
      if (last_key.empty()) return ParseResult::MALFORMED;
      std::string continuation = trim_ascii(std::string_view(header.value, header.value_len));
      if (!continuation.empty()) {
        req.headers[last_key] += " ";
        req.headers[last_key] += continuation;
      }
      continue;
    }

    std::string key = ascii_lower(std::string_view(header.name, header.name_len));
    std::string value = trim_ascii(std::string_view(header.value, header.value_len));
    auto inserted = req.headers.emplace(key, value);
    if (!inserted.second) {
      inserted.first->second += ",";
      inserted.first->second += value;
    }
    last_key = std::move(key);
  }

  if (req.method == "GET" && has_get_body_headers(req)) {
    return ParseResult::MALFORMED;
  }

  out = std::move(req);
  return ParseResult::OK;
}

bool is_websocket_upgrade(const HttpRequest& req) {
  if (req.method != "GET") return false;
  if (!header_equals_ci(req, "upgrade", "websocket")) return false;
  if (!header_has_comma_token_ci(req, "connection", "upgrade")) return false;
  if (!header_has_comma_token_ci(req, "sec-websocket-version", "13")) return false;
  auto key = find_header(req, "sec-websocket-key");
  return key != req.headers.end() && valid_sec_websocket_key(key->second);
}

std::string compute_accept_key(const std::string& sec_websocket_key) {
  const std::string joined = trim_ascii(sec_websocket_key) + kWebSocketGuid;
  unsigned char digest[SHA_DIGEST_LENGTH];
  SHA1(reinterpret_cast<const unsigned char*>(joined.data()), joined.size(), digest);

  std::vector<unsigned char> encoded(4 * ((SHA_DIGEST_LENGTH + 2) / 3) + 1);
  int encoded_len = EVP_EncodeBlock(encoded.data(), digest, SHA_DIGEST_LENGTH);
  return std::string(reinterpret_cast<char*>(encoded.data()), static_cast<size_t>(encoded_len));
}

std::string build_handshake_response(const std::string& accept_key) {
  std::ostringstream response;
  response << "HTTP/1.1 101 Switching Protocols\r\n"
           << "Upgrade: websocket\r\n"
           << "Connection: Upgrade\r\n"
           << "Sec-WebSocket-Accept: " << accept_key << "\r\n"
           << "\r\n";
  return response.str();
}

std::string build_http_error_response(int status, std::string_view body) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << " " << reason_phrase(status) << "\r\n"
           << "Content-Type: application/json\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n"
           << "\r\n"
           << body;
  return response.str();
}

}  // namespace ws_handshake
