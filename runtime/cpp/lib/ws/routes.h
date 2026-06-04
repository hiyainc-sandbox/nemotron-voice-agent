#pragma once

#include "lib/ws/handshake.h"

#include <map>
#include <string>

namespace ws_routes {

enum class RouteKind {
  HEALTH,
  STATS,
  SCHEDULER_TELEMETRY,
  WEBSOCKET,
  NOT_FOUND,
  BAD_REQUEST,
};

struct Route {
  RouteKind kind;
  std::string path;
  std::map<std::string, std::string> query_params;
};

Route dispatch(const ws_handshake::HttpRequest& req);

}  // namespace ws_routes
