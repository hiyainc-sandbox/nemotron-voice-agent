#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <string_view>

namespace ws_handshake {

inline constexpr size_t kMaxHttpHeaderBytes = 8 * 1024;
inline constexpr int kHttpReadTimeoutMs = 2000;

struct HttpRequest {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
};

enum class ParseResult {
  OK,
  NEED_MORE,
  MALFORMED,
  OVERSIZE_HEADERS,
};

ParseResult parse_http_request(const std::string& buffer, HttpRequest& out);

bool is_websocket_upgrade(const HttpRequest& req);

std::string compute_accept_key(const std::string& sec_websocket_key);

std::string build_handshake_response(const std::string& accept_key);

std::string build_http_error_response(int status, std::string_view body);

}  // namespace ws_handshake
