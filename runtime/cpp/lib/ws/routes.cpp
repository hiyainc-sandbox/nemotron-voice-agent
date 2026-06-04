#include "lib/ws/routes.h"

#include <string_view>
#include <utility>

namespace ws_routes {
namespace {

int hex_value(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  return -1;
}

std::string percent_decode(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    char ch = value[i];
    if (ch == '+') {
      out.push_back(' ');
    } else if (ch == '%' && i + 2 < value.size()) {
      int hi = hex_value(value[i + 1]);
      int lo = hex_value(value[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
      } else {
        out.push_back(ch);
      }
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

std::map<std::string, std::string> parse_query(std::string_view query) {
  std::map<std::string, std::string> params;
  size_t pos = 0;
  while (pos <= query.size()) {
    size_t amp = query.find('&', pos);
    std::string_view pair = query.substr(
        pos, amp == std::string_view::npos ? std::string_view::npos : amp - pos);
    if (!pair.empty()) {
      size_t eq = pair.find('=');
      std::string key = percent_decode(pair.substr(0, eq));
      std::string value = eq == std::string_view::npos
                              ? std::string()
                              : percent_decode(pair.substr(eq + 1));
      params[std::move(key)] = std::move(value);
    }
    if (amp == std::string_view::npos) break;
    pos = amp + 1;
  }
  return params;
}

}  // namespace

Route dispatch(const ws_handshake::HttpRequest& req) {
  std::string_view target = req.path;
  size_t query_pos = target.find('?');
  std::string path(target.substr(0, query_pos));
  std::map<std::string, std::string> query_params =
      query_pos == std::string_view::npos
          ? std::map<std::string, std::string>{}
          : parse_query(target.substr(query_pos + 1));

  RouteKind kind = RouteKind::NOT_FOUND;
  if (req.method == "GET") {
    if (path == "/health") {
      kind = RouteKind::HEALTH;
    } else if (path == "/stats") {
      kind = RouteKind::STATS;
    } else if (path == "/scheduler_telemetry") {
      kind = RouteKind::SCHEDULER_TELEMETRY;
    } else if (path == "/") {
      kind = ws_handshake::is_websocket_upgrade(req)
                 ? RouteKind::WEBSOCKET
                 : RouteKind::BAD_REQUEST;
    }
  }

  return Route{kind, std::move(path), std::move(query_params)};
}

}  // namespace ws_routes
