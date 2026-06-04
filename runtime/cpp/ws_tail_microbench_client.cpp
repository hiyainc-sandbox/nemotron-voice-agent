#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/sha.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kWireMagic = "WSTAIL1";
constexpr const char* kWsGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

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
};

struct ClientOptions {
  std::string host = "127.0.0.1";
  std::string port = "8765";
  int n_idle = 0;
  int m_streaming = 1;
  int duration_ms = 5000;
  int chunk_period_ms = 160;
  int payload_bytes = 4096;
};

int64_t steady_us_now() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             Clock::now().time_since_epoch())
      .count();
}

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() &&
         text.compare(0, prefix.size(), prefix) == 0;
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

std::string random_key() {
  std::array<unsigned char, 16> bytes{};
  std::random_device rd;
  for (auto& byte : bytes) byte = static_cast<unsigned char>(rd());
  return base64_encode(bytes.data(), bytes.size());
}

Fd connect_tcp(const ClientOptions& options) {
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  addrinfo* results = nullptr;
  const int rc = getaddrinfo(options.host.c_str(), options.port.c_str(), &hints,
                             &results);
  if (rc != 0) throw std::runtime_error(gai_strerror(rc));

  Fd fd;
  for (addrinfo* ai = results; ai != nullptr; ai = ai->ai_next) {
    Fd candidate(socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol));
    if (candidate.fd < 0) continue;
    if (connect(candidate.fd, ai->ai_addr, ai->ai_addrlen) == 0) {
      int yes = 1;
      setsockopt(candidate.fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
      fd = std::move(candidate);
      break;
    }
  }
  freeaddrinfo(results);
  if (fd.fd < 0) throw std::runtime_error("connect failed");
  return fd;
}

Fd connect_ws(const ClientOptions& options) {
  Fd fd = connect_tcp(options);
  const std::string key = random_key();
  std::ostringstream req;
  req << "GET / HTTP/1.1\r\n"
      << "Host: " << options.host << ":" << options.port << "\r\n"
      << "Upgrade: websocket\r\n"
      << "Connection: Upgrade\r\n"
      << "Sec-WebSocket-Version: 13\r\n"
      << "Sec-WebSocket-Key: " << key << "\r\n"
      << "\r\n";
  const std::string text = req.str();
  if (!write_all(fd.fd, text.data(), text.size())) {
    throw std::runtime_error("handshake write failed");
  }

  std::string response;
  char buf[512];
  while (response.find("\r\n\r\n") == std::string::npos &&
         response.size() < 64 * 1024) {
    const ssize_t n = recv(fd.fd, buf, sizeof(buf), 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("handshake read failed");
    }
    if (n == 0) throw std::runtime_error("handshake closed");
    response.append(buf, static_cast<size_t>(n));
  }
  if (response.find("101 Switching Protocols") == std::string::npos ||
      response.find(websocket_accept_key(key)) == std::string::npos) {
    throw std::runtime_error("bad websocket handshake response");
  }
  return fd;
}

bool write_ws_frame(int fd, uint8_t opcode, const std::string& payload) {
  std::vector<uint8_t> header;
  header.reserve(14);
  header.push_back(static_cast<uint8_t>(0x80 | (opcode & 0x0f)));
  const uint64_t len = payload.size();
  if (len < 126) {
    header.push_back(static_cast<uint8_t>(0x80 | len));
  } else if (len <= 0xffff) {
    header.push_back(0x80 | 126);
    header.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
    header.push_back(static_cast<uint8_t>(len & 0xff));
  } else {
    header.push_back(0x80 | 127);
    for (int shift = 56; shift >= 0; shift -= 8) {
      header.push_back(static_cast<uint8_t>((len >> shift) & 0xff));
    }
  }

  std::array<uint8_t, 4> mask{};
  std::random_device rd;
  for (auto& byte : mask) byte = static_cast<uint8_t>(rd());
  header.insert(header.end(), mask.begin(), mask.end());

  std::string masked = payload;
  for (size_t i = 0; i < masked.size(); ++i) {
    masked[i] = static_cast<char>(masked[i] ^ mask[i % 4]);
  }

  return write_all(fd, header.data(), header.size()) &&
         (masked.empty() || write_all(fd, masked.data(), masked.size()));
}

bool read_ws_frame(int fd, uint8_t* opcode, std::string* payload) {
  uint8_t hdr[2];
  if (!read_exact(fd, hdr, sizeof(hdr))) return false;
  *opcode = hdr[0] & 0x0f;
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

  payload->assign(static_cast<size_t>(len), '\0');
  if (len > 0 && !read_exact(fd, payload->data(), payload->size())) return false;
  if (masked) {
    for (size_t i = 0; i < payload->size(); ++i) {
      (*payload)[i] = static_cast<char>((*payload)[i] ^ mask[i % 4]);
    }
  }
  return true;
}

void send_control(const ClientOptions& options,
                  const std::string& message,
                  bool retry) {
  const auto deadline = Clock::now() + std::chrono::seconds(5);
  std::string last_error = "connect failed";
  while (true) {
    try {
      Fd fd = connect_ws(options);
      if (!write_ws_frame(fd.fd, 0x1, message)) {
        throw std::runtime_error("control frame write failed");
      }
      write_ws_frame(fd.fd, 0x8, "");
      return;
    } catch (const std::exception& e) {
      last_error = e.what();
      if (!retry || Clock::now() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  throw std::runtime_error(last_error);
}

void parse_host_port(const std::string& server, ClientOptions& options) {
  const auto pos = server.rfind(':');
  if (pos == std::string::npos || pos == 0 || pos + 1 >= server.size()) {
    throw std::runtime_error("--server must be host:port");
  }
  options.host = server.substr(0, pos);
  options.port = server.substr(pos + 1);
}

std::string make_config_message(const ClientOptions& options) {
  std::ostringstream out;
  out << "CONFIG n_idle=" << options.n_idle
      << " m_streaming=" << options.m_streaming
      << " duration_ms=" << options.duration_ms
      << " chunk_period_ms=" << options.chunk_period_ms
      << " payload_bytes=" << options.payload_bytes;
  return out.str();
}

std::string make_metrics_message(const std::vector<int64_t>& samples) {
  std::ostringstream out;
  out << "CLIENT_RECV_US ";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i > 0) out << ',';
    out << samples[i];
  }
  return out.str();
}

std::string make_payload(uint64_t seq, int64_t send_us, int payload_bytes) {
  std::ostringstream header;
  header << kWireMagic << ' ' << seq << ' ' << send_us << ' ';
  std::string payload = header.str();
  if (payload_bytes > static_cast<int>(payload.size())) {
    payload.resize(static_cast<size_t>(payload_bytes), 'x');
  }
  return payload;
}

void idle_worker(const ClientOptions& options, std::atomic<int>& failures) {
  try {
    Fd fd = connect_ws(options);
    std::this_thread::sleep_for(std::chrono::milliseconds(options.duration_ms));
    write_ws_frame(fd.fd, 0x8, "");
  } catch (const std::exception& e) {
    failures.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "idle worker failed: " << e.what() << "\n";
  }
}

void streaming_worker(int worker_id,
                      const ClientOptions& options,
                      std::vector<int64_t>& client_recv_us,
                      std::mutex& samples_mu,
                      std::atomic<uint64_t>& sent,
                      std::atomic<uint64_t>& received,
                      std::atomic<int>& failures) {
  try {
    Fd fd = connect_ws(options);
    const auto deadline =
        Clock::now() + std::chrono::milliseconds(options.duration_ms);
    auto next_send = Clock::now();
    uint64_t seq = static_cast<uint64_t>(worker_id) << 48;

    while (Clock::now() < deadline) {
      const int64_t send_us = steady_us_now();
      const std::string payload =
          make_payload(seq++, send_us, options.payload_bytes);
      if (!write_ws_frame(fd.fd, 0x2, payload)) {
        throw std::runtime_error("binary frame write failed");
      }
      sent.fetch_add(1, std::memory_order_relaxed);

      uint8_t opcode = 0;
      std::string response;
      if (!read_ws_frame(fd.fd, &opcode, &response) || opcode != 0x2) {
        throw std::runtime_error("binary echo read failed");
      }
      const int64_t recv_us = steady_us_now();
      received.fetch_add(1, std::memory_order_relaxed);

      {
        std::lock_guard<std::mutex> lock(samples_mu);
        client_recv_us.push_back(recv_us - send_us);
      }

      next_send += std::chrono::milliseconds(options.chunk_period_ms);
      std::this_thread::sleep_until(next_send);
    }

    write_ws_frame(fd.fd, 0x8, "");
  } catch (const std::exception& e) {
    failures.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "streaming worker failed: " << e.what() << "\n";
  }
}

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--server 127.0.0.1:8765] [--n-idle 0]"
            << " [--m-streaming 1] [--duration-ms 5000]"
            << " [--chunk-period-ms 160] [--payload-bytes 4096]\n";
}

ClientOptions parse_args(int argc, char** argv) {
  ClientOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
      return argv[++i];
    };

    if (arg == "--server") {
      parse_host_port(require_value(arg), options);
    } else if (arg == "--n-idle") {
      options.n_idle = std::stoi(require_value(arg));
    } else if (arg == "--m-streaming") {
      options.m_streaming = std::stoi(require_value(arg));
    } else if (arg == "--duration-ms") {
      options.duration_ms = std::stoi(require_value(arg));
    } else if (arg == "--chunk-period-ms") {
      options.chunk_period_ms = std::stoi(require_value(arg));
    } else if (arg == "--payload-bytes") {
      options.payload_bytes = std::stoi(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (options.n_idle < 0 || options.m_streaming < 0) {
    throw std::runtime_error("socket counts must be non-negative");
  }
  if (options.duration_ms <= 0 || options.chunk_period_ms <= 0) {
    throw std::runtime_error("duration-ms and chunk-period-ms must be positive");
  }
  if (options.payload_bytes <= 0) {
    throw std::runtime_error("payload-bytes must be positive");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ClientOptions options = parse_args(argc, argv);

    send_control(options, make_config_message(options), true);

    std::vector<int64_t> client_recv_us;
    std::mutex samples_mu;
    std::atomic<uint64_t> sent{0};
    std::atomic<uint64_t> received{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(options.n_idle + options.m_streaming));

    for (int i = 0; i < options.n_idle; ++i) {
      threads.emplace_back(idle_worker, std::cref(options), std::ref(failures));
    }
    for (int i = 0; i < options.m_streaming; ++i) {
      threads.emplace_back(streaming_worker, i, std::cref(options),
                           std::ref(client_recv_us), std::ref(samples_mu),
                           std::ref(sent), std::ref(received),
                           std::ref(failures));
    }
    for (auto& thread : threads) thread.join();

    std::vector<int64_t> snapshot;
    {
      std::lock_guard<std::mutex> lock(samples_mu);
      snapshot = client_recv_us;
    }
    send_control(options, make_metrics_message(snapshot), true);

    std::cout << "{\"sent\":" << sent.load(std::memory_order_relaxed)
              << ",\"received\":" << received.load(std::memory_order_relaxed)
              << ",\"client_recv_samples\":" << snapshot.size()
              << ",\"failures\":" << failures.load(std::memory_order_relaxed)
              << "}\n";
    return failures.load(std::memory_order_relaxed) == 0 ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "ws_tail_microbench_client: " << e.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
