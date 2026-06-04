#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <openssl/sha.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kWsGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
constexpr const char* kWireMagic = "WSTAIL1";
// Linux steady_clock is backed by system-wide CLOCK_MONOTONIC, so the local
// client can stamp a frame and the server can compute send->recv on one host.
constexpr int64_t kMaxCrossProcessDeltaUs = 60LL * 60LL * 1000LL * 1000LL;

const std::vector<std::string>& stage_names() {
  static const std::vector<std::string> names = {
      "accept_to_ready_us",
      "send_to_recv_us",
      "recv_to_queue_us",
      "queue_to_scheduler_us",
      "serialize_and_send_us",
      "client_recv_us",
      "event_loop_lag_us",
  };
  return names;
}

struct Fd {
  int fd = -1;
  Fd() = default;
  explicit Fd(int value) : fd(value) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : fd(other.fd) { other.fd = -1; }
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) {
      reset();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }
  ~Fd() { reset(); }
  void reset(int value = -1) {
    if (fd >= 0) close(fd);
    fd = value;
  }
  int release() {
    const int out = fd;
    fd = -1;
    return out;
  }
};

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
}

std::string trim(std::string value) {
  while (!value.empty() &&
         (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' ||
          value.back() == '\t')) {
    value.pop_back();
  }
  size_t first = 0;
  while (first < value.size() &&
         (value[first] == ' ' || value[first] == '\t')) {
    ++first;
  }
  return value.substr(first);
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

int64_t steady_us_now() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             Clock::now().time_since_epoch())
      .count();
}

int64_t elapsed_us(Clock::time_point start, Clock::time_point end) {
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (ns <= 0) return 0;
  // Use ceil so completed sub-microsecond local stages remain visible.
  return (ns + 999) / 1000;
}

std::string base64_encode(const unsigned char* data, size_t len) {
  static constexpr char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((len + 2) / 3) * 4);
  for (size_t i = 0; i < len; i += 3) {
    const uint32_t b0 = data[i];
    const uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
    const uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;
    const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(table[(triple >> 18) & 0x3f]);
    out.push_back(table[(triple >> 12) & 0x3f]);
    out.push_back(i + 1 < len ? table[(triple >> 6) & 0x3f] : '=');
    out.push_back(i + 2 < len ? table[triple & 0x3f] : '=');
  }
  return out;
}

std::string websocket_accept_key(const std::string& key) {
  const std::string joined = key + kWsGuid;
  unsigned char digest[SHA_DIGEST_LENGTH];
  SHA1(reinterpret_cast<const unsigned char*>(joined.data()), joined.size(),
       digest);
  return base64_encode(digest, sizeof(digest));
}

bool write_all(int fd, const void* data, size_t len) {
  const char* ptr = static_cast<const char*>(data);
  while (len > 0) {
    const ssize_t n = send(fd, ptr, len, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) return false;
    ptr += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool read_exact(int fd, void* data, size_t len) {
  char* ptr = static_cast<char*>(data);
  while (len > 0) {
    const ssize_t n = recv(fd, ptr, len, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) return false;
    ptr += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

std::string stat_json(std::vector<int64_t> values) {
  std::ostringstream out;
  out << "{\"n\":" << values.size();
  if (values.empty()) {
    out << ",\"p50\":null,\"p95\":null,\"p99\":null}";
    return out.str();
  }

  std::sort(values.begin(), values.end());
  auto percentile = [&](double q) {
    size_t idx =
        static_cast<size_t>(std::ceil(q * static_cast<double>(values.size()))) -
        1;
    if (idx >= values.size()) idx = values.size() - 1;
    return values[idx];
  };
  out << ",\"p50\":" << percentile(0.50) << ",\"p95\":" << percentile(0.95)
      << ",\"p99\":" << percentile(0.99) << "}";
  return out.str();
}

std::unordered_map<std::string, std::string> parse_key_values(
    const std::string& message,
    const std::string& prefix) {
  std::unordered_map<std::string, std::string> out;
  if (!starts_with(message, prefix)) return out;

  std::istringstream in(message.substr(prefix.size()));
  std::string token;
  while (in >> token) {
    const auto pos = token.find('=');
    if (pos == std::string::npos) continue;
    out[token.substr(0, pos)] = token.substr(pos + 1);
  }
  return out;
}

int parse_int_or(const std::unordered_map<std::string, std::string>& values,
                 const std::string& key,
                 int fallback) {
  auto it = values.find(key);
  if (it == values.end()) return fallback;
  try {
    return std::stoi(it->second);
  } catch (...) {
    return fallback;
  }
}

std::vector<int64_t> parse_csv_us(const std::string& csv) {
  std::vector<int64_t> out;
  std::istringstream in(csv);
  std::string token;
  while (std::getline(in, token, ',')) {
    if (token.empty()) continue;
    try {
      const int64_t value = std::stoll(token);
      if (value >= 0) out.push_back(value);
    } catch (...) {
    }
  }
  return out;
}

struct ServerOptions {
  uint16_t port = 8765;
  int duration_ms = 6000;
  int loop_lag_period_ms = 10;
  bool stop_after_client_metrics = true;
  std::string output = "ws_tail_microbench.json";
  std::string events_output;
};

struct ClientConfig {
  int n_idle = -1;
  int m_streaming = -1;
  int duration_ms = -1;
  int chunk_period_ms = -1;
  int payload_bytes = -1;
};

struct EventTiming {
  uint64_t seq = 0;
  int64_t send_to_recv_us = -1;
  int64_t recv_to_queue_us = -1;
  int64_t queue_to_scheduler_us = -1;
  int64_t serialize_and_send_us = -1;
};

class Telemetry {
 public:
  explicit Telemetry(std::string events_output)
      : events_output_(std::move(events_output)) {}

  void add_sample(const std::string& stage, int64_t us) {
    if (us < 0) return;
    std::lock_guard<std::mutex> lock(mu_);
    samples_[stage].push_back(us);
  }

  void set_client_config(const ClientConfig& config) {
    std::lock_guard<std::mutex> lock(mu_);
    client_config_ = config;
  }

  void add_client_recv_samples(const std::vector<int64_t>& samples) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& dst = samples_["client_recv_us"];
    dst.insert(dst.end(), samples.begin(), samples.end());
    client_metrics_seen_ = true;
  }

  bool client_metrics_seen() const {
    std::lock_guard<std::mutex> lock(mu_);
    return client_metrics_seen_;
  }

  void add_event(const EventTiming& event) {
    if (events_output_.empty()) return;
    std::ostringstream out;
    auto field = [&](const char* name, int64_t value) {
      out << ",\"" << name << "\":";
      if (value < 0) {
        out << "null";
      } else {
        out << value;
      }
    };
    out << "{\"seq\":" << event.seq;
    field("send_to_recv_us", event.send_to_recv_us);
    field("recv_to_queue_us", event.recv_to_queue_us);
    field("queue_to_scheduler_us", event.queue_to_scheduler_us);
    field("serialize_and_send_us", event.serialize_and_send_us);
    out << "}";

    std::lock_guard<std::mutex> lock(mu_);
    event_lines_.push_back(out.str());
  }

  void write_summary(const std::string& path,
                     const ServerOptions& options) const {
    std::unordered_map<std::string, std::vector<int64_t>> samples;
    ClientConfig client_config;
    {
      std::lock_guard<std::mutex> lock(mu_);
      samples = samples_;
      client_config = client_config_;
    }

    std::ofstream out(path);
    if (!out) throw std::runtime_error("failed to open output: " + path);

    out << "{\n  \"ws_tail\": {\n";
    const auto& names = stage_names();
    for (size_t i = 0; i < names.size(); ++i) {
      auto it = samples.find(names[i]);
      out << "    \"" << names[i] << "\": "
          << stat_json(it == samples.end() ? std::vector<int64_t>{}
                                           : it->second);
      out << (i + 1 == names.size() ? "\n" : ",\n");
    }
    out << "  },\n  \"config\": {\n";
    write_config_int(out, "n_idle", client_config.n_idle, true);
    write_config_int(out, "m_streaming", client_config.m_streaming, true);
    write_config_int(out, "duration_ms", client_config.duration_ms, true);
    write_config_int(out, "chunk_period_ms", client_config.chunk_period_ms,
                     true);
    write_config_int(out, "payload_bytes", client_config.payload_bytes, true);
    out << "    \"server_duration_ms\": " << options.duration_ms << ",\n";
    out << "    \"port\": " << options.port << ",\n";
    out << "    \"host\": \"127.0.0.1\",";
    out << "\n    \"ws_impl\": \"raw-rfc6455-posix\",";
    out << "\n    \"events_output\": ";
    if (events_output_.empty()) {
      out << "null\n";
    } else {
      out << "\"" << json_escape(events_output_) << "\"\n";
    }
    out << "  }\n}\n";
  }

  void write_events() const {
    if (events_output_.empty()) return;

    std::vector<std::string> event_lines;
    {
      std::lock_guard<std::mutex> lock(mu_);
      event_lines = event_lines_;
    }

    std::ofstream out(events_output_);
    if (!out) {
      throw std::runtime_error("failed to open events output: " +
                               events_output_);
    }
    for (const auto& line : event_lines) out << line << '\n';
  }

 private:
  static void write_config_int(std::ostream& out,
                               const std::string& key,
                               int value,
                               bool comma) {
    out << "    \"" << key << "\": ";
    if (value < 0) {
      out << "null";
    } else {
      out << value;
    }
    out << (comma ? ",\n" : "\n");
  }

  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<int64_t>> samples_;
  ClientConfig client_config_;
  bool client_metrics_seen_ = false;
  std::string events_output_;
  std::vector<std::string> event_lines_;
};

struct ParsedPayload {
  bool ok = false;
  uint64_t seq = 0;
  int64_t client_send_us = 0;
};

ParsedPayload parse_payload(const std::string& payload) {
  ParsedPayload parsed;
  if (!starts_with(payload, kWireMagic)) return parsed;
  std::istringstream in(payload.substr(std::string(kWireMagic).size()));
  in >> parsed.seq >> parsed.client_send_us;
  parsed.ok = !in.fail();
  return parsed;
}

struct WsFrame {
  uint8_t opcode = 0;
  std::string payload;
};

bool read_ws_frame(int fd, WsFrame* frame) {
  uint8_t hdr[2];
  if (!read_exact(fd, hdr, sizeof(hdr))) return false;

  const uint8_t opcode = hdr[0] & 0x0f;
  const bool masked = (hdr[1] & 0x80) != 0;
  uint64_t len = hdr[1] & 0x7f;
  if (len == 126) {
    uint8_t ext[2];
    if (!read_exact(fd, ext, sizeof(ext))) return false;
    len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
  } else if (len == 127) {
    uint8_t ext[8];
    if (!read_exact(fd, ext, sizeof(ext))) return false;
    len = 0;
    for (uint8_t byte : ext) len = (len << 8) | byte;
  }

  uint8_t mask[4] = {0, 0, 0, 0};
  if (masked && !read_exact(fd, mask, sizeof(mask))) return false;
  if (len > 64ULL * 1024ULL * 1024ULL) return false;

  std::string payload(static_cast<size_t>(len), '\0');
  if (len > 0 && !read_exact(fd, payload.data(), payload.size())) return false;
  if (masked) {
    for (size_t i = 0; i < payload.size(); ++i) {
      payload[i] = static_cast<char>(payload[i] ^ mask[i % 4]);
    }
  }

  frame->opcode = opcode;
  frame->payload = std::move(payload);
  return true;
}

bool write_ws_frame(int fd, uint8_t opcode, const std::string& payload) {
  std::vector<uint8_t> header;
  header.reserve(10);
  header.push_back(static_cast<uint8_t>(0x80 | (opcode & 0x0f)));
  const uint64_t len = payload.size();
  if (len < 126) {
    header.push_back(static_cast<uint8_t>(len));
  } else if (len <= 0xffff) {
    header.push_back(126);
    header.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
    header.push_back(static_cast<uint8_t>(len & 0xff));
  } else {
    header.push_back(127);
    for (int shift = 56; shift >= 0; shift -= 8) {
      header.push_back(static_cast<uint8_t>((len >> shift) & 0xff));
    }
  }
  return write_all(fd, header.data(), header.size()) &&
         (payload.empty() || write_all(fd, payload.data(), payload.size()));
}

bool websocket_handshake(int fd, const std::atomic<bool>& stop) {
  std::string request;
  request.reserve(4096);
  char buf[512];
  const auto deadline = Clock::now() + std::chrono::seconds(5);
  while (request.find("\r\n\r\n") == std::string::npos &&
         request.size() < 64 * 1024) {
    if (stop.load(std::memory_order_acquire) || Clock::now() >= deadline) {
      return false;
    }
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 100);
    if (ready < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (ready == 0) continue;

    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) return false;
    request.append(buf, static_cast<size_t>(n));
  }

  std::string key;
  std::istringstream lines(request);
  std::string line;
  while (std::getline(lines, line)) {
    const std::string lower_prefix = "Sec-WebSocket-Key:";
    if (line.size() >= lower_prefix.size() &&
        line.compare(0, lower_prefix.size(), lower_prefix) == 0) {
      key = trim(line.substr(lower_prefix.size()));
      break;
    }
  }
  if (key.empty()) return false;

  std::ostringstream response;
  response << "HTTP/1.1 101 Switching Protocols\r\n"
           << "Upgrade: websocket\r\n"
           << "Connection: Upgrade\r\n"
           << "Sec-WebSocket-Accept: " << websocket_accept_key(key) << "\r\n"
           << "\r\n";
  const std::string text = response.str();
  return write_all(fd, text.data(), text.size());
}

class WorkerQueue {
 public:
  void post(std::function<void()> work) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back(std::move(work));
    }
    cv_.notify_one();
  }

  bool wait_pop(std::function<void()>* work) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&]() { return stopped_ || !queue_.empty(); });
    if (queue_.empty()) return false;
    *work = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  bool try_pop(std::function<void()>* work) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (queue_.empty()) return false;
      *work = std::move(queue_.front());
      queue_.pop_front();
    }
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopped_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> queue_;
  bool stopped_ = false;
};

struct FrameDone {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
};

void handle_control(const std::string& message,
                    Telemetry& telemetry,
                    std::function<void()> request_stop) {
  if (starts_with(message, "CONFIG ")) {
    const auto values = parse_key_values(message, "CONFIG ");
    ClientConfig config;
    config.n_idle = parse_int_or(values, "n_idle", -1);
    config.m_streaming = parse_int_or(values, "m_streaming", -1);
    config.duration_ms = parse_int_or(values, "duration_ms", -1);
    config.chunk_period_ms = parse_int_or(values, "chunk_period_ms", -1);
    config.payload_bytes = parse_int_or(values, "payload_bytes", -1);
    telemetry.set_client_config(config);
    return;
  }

  if (starts_with(message, "CLIENT_RECV_US ")) {
    telemetry.add_client_recv_samples(
        parse_csv_us(message.substr(std::string("CLIENT_RECV_US ").size())));
    request_stop();
  }
}

void connection_thread(Fd client,
                       Clock::time_point accept_done,
                       Telemetry& telemetry,
                       WorkerQueue& worker_queue,
                       std::function<void()> request_stop,
                       const std::atomic<bool>& stop) {
  if (!websocket_handshake(client.fd, stop)) return;
  telemetry.add_sample("accept_to_ready_us", elapsed_us(accept_done, Clock::now()));

  while (!stop.load(std::memory_order_acquire)) {
    pollfd pfd{client.fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 100);
    if (ready < 0) {
      if (errno == EINTR) continue;
      return;
    }
    if (ready == 0) continue;

    WsFrame frame;
    if (!read_ws_frame(client.fd, &frame)) return;
    const auto recv_done = Clock::now();

    if (frame.opcode == 0x8) {
      write_ws_frame(client.fd, 0x8, "");
      return;
    }
    if (frame.opcode == 0x1) {
      handle_control(frame.payload, telemetry, request_stop);
      continue;
    }
    if (frame.opcode != 0x2) continue;

    EventTiming event;
    const auto parsed = parse_payload(frame.payload);
    if (parsed.ok) {
      event.seq = parsed.seq;
      const int64_t delta_us = steady_us_now() - parsed.client_send_us;
      if (delta_us >= 0 && delta_us < kMaxCrossProcessDeltaUs) {
        event.send_to_recv_us = delta_us;
        telemetry.add_sample("send_to_recv_us", delta_us);
      }
    }

    const auto queued_at = Clock::now();
    event.recv_to_queue_us = elapsed_us(recv_done, queued_at);
    telemetry.add_sample("recv_to_queue_us", event.recv_to_queue_us);

    auto done = std::make_shared<FrameDone>();
    worker_queue.post([fd = client.fd, payload = std::move(frame.payload),
                       queued_at, event, &telemetry, done]() mutable {
      const auto pickup_at = Clock::now();
      event.queue_to_scheduler_us = elapsed_us(queued_at, pickup_at);
      telemetry.add_sample("queue_to_scheduler_us",
                           event.queue_to_scheduler_us);

      const auto send_start = Clock::now();
      write_ws_frame(fd, 0x2, payload);
      event.serialize_and_send_us = elapsed_us(send_start, Clock::now());
      telemetry.add_sample("serialize_and_send_us",
                           event.serialize_and_send_us);
      telemetry.add_event(event);
      {
        std::lock_guard<std::mutex> lock(done->mu);
        done->done = true;
      }
      done->cv.notify_one();
    });

    std::unique_lock<std::mutex> lock(done->mu);
    done->cv.wait(lock, [&]() { return done->done; });
  }
}

Fd make_listener(uint16_t port) {
  Fd fd(socket(AF_INET, SOCK_STREAM, 0));
  if (fd.fd < 0) throw std::runtime_error("socket() failed");

  int yes = 1;
  if (setsockopt(fd.fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) < 0) {
    throw std::runtime_error("setsockopt(SO_REUSEADDR) failed");
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (bind(fd.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    throw std::runtime_error("bind(127.0.0.1:" + std::to_string(port) +
                             ") failed: " + std::strerror(errno));
  }
  if (listen(fd.fd, SOMAXCONN) < 0) {
    throw std::runtime_error("listen() failed");
  }
  return fd;
}

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--port 8765] [--duration-ms 6000] [--output ws_tail.json]\n"
      << "          [--events-output ws_tail.events.jsonl]"
      << " [--loop-lag-period-ms 10] [--no-stop-after-client-metrics]\n";
}

ServerOptions parse_args(int argc, char** argv) {
  ServerOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
      return argv[++i];
    };

    if (arg == "--port") {
      const int port = std::stoi(require_value(arg));
      if (port <= 0 || port > 65535) {
        throw std::runtime_error("port must be in 1..65535");
      }
      options.port = static_cast<uint16_t>(port);
    } else if (arg == "--duration-ms") {
      options.duration_ms = std::stoi(require_value(arg));
    } else if (arg == "--output") {
      options.output = require_value(arg);
    } else if (arg == "--events-output") {
      options.events_output = require_value(arg);
    } else if (arg == "--loop-lag-period-ms") {
      options.loop_lag_period_ms = std::stoi(require_value(arg));
    } else if (arg == "--no-stop-after-client-metrics") {
      options.stop_after_client_metrics = false;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (options.duration_ms <= 0) {
    throw std::runtime_error("duration-ms must be positive");
  }
  if (options.loop_lag_period_ms <= 0) {
    throw std::runtime_error("loop-lag-period-ms must be positive");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ServerOptions options = parse_args(argc, argv);
    Telemetry telemetry(options.events_output);
    WorkerQueue worker_queue;
    Fd listener = make_listener(options.port);
    std::atomic<bool> stop{false};
    std::vector<std::thread> connection_threads;

    auto request_stop = [&]() {
      if (options.stop_after_client_metrics) {
        stop.store(true, std::memory_order_release);
      }
    };

    std::thread worker([&]() {
      std::function<void()> work;
      while (worker_queue.wait_pop(&work)) {
        work();
      }
      while (worker_queue.try_pop(&work)) {
        work();
      }
    });

    const auto start = Clock::now();
    auto next_lag_sample =
        start + std::chrono::milliseconds(options.loop_lag_period_ms);
    std::cerr << "ws_tail_microbench listening on 127.0.0.1:" << options.port
              << " for up to " << options.duration_ms << " ms\n";

    while (!stop.load(std::memory_order_acquire)) {
      const auto now = Clock::now();
      if (now - start >= std::chrono::milliseconds(options.duration_ms)) break;

      const auto timeout =
          std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::milliseconds>(
                                   next_lag_sample - now)
                                   .count());
      pollfd pfd{listener.fd, POLLIN, 0};
      const int rc = poll(&pfd, 1, static_cast<int>(timeout));
      const auto woke = Clock::now();
      if (woke >= next_lag_sample) {
        telemetry.add_sample("event_loop_lag_us",
                             elapsed_us(next_lag_sample, woke));
        do {
          next_lag_sample +=
              std::chrono::milliseconds(options.loop_lag_period_ms);
        } while (next_lag_sample <= woke);
      }
      if (rc < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("poll() failed");
      }
      if (rc > 0 && (pfd.revents & POLLIN)) {
        sockaddr_in peer{};
        socklen_t peer_len = sizeof(peer);
        const int fd =
            accept(listener.fd, reinterpret_cast<sockaddr*>(&peer), &peer_len);
        const auto accept_done = Clock::now();
        if (fd >= 0) {
          connection_threads.emplace_back(connection_thread, Fd(fd),
                                          accept_done, std::ref(telemetry),
                                          std::ref(worker_queue),
                                          request_stop, std::cref(stop));
        }
      }
    }

    listener.reset();
    stop.store(true, std::memory_order_release);
    for (auto& thread : connection_threads) {
      if (thread.joinable()) thread.join();
    }
    worker_queue.stop();
    if (worker.joinable()) worker.join();

    telemetry.write_summary(options.output, options);
    telemetry.write_events();
    std::cerr << "wrote " << options.output << "\n";
    if (!options.events_output.empty()) {
      std::cerr << "wrote " << options.events_output << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ws_tail_microbench: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
