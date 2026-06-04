#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ws_framing {

inline constexpr size_t kMaxMessageSize = 10 * 1024 * 1024;

enum class Opcode : uint8_t {
  CONT = 0x0,
  TEXT = 0x1,
  BINARY = 0x2,
  CLOSE = 0x8,
  PING = 0x9,
  PONG = 0xA,
};

struct Frame {
  Opcode opcode;
  std::vector<uint8_t> payload;
  bool fin = true;
};

struct Message {
  Opcode opcode;
  std::vector<uint8_t> payload;
};

enum class ReadResult {
  OK,
  NEED_MORE,
  FRAME_TOO_LARGE,
  MALFORMED,
};

ReadResult read_frame(const std::string& buffer,
                      Frame& out,
                      size_t& consumed,
                      size_t max_payload_size = kMaxMessageSize);

class MessageAssembler {
 public:
  explicit MessageAssembler(size_t max_message_size = kMaxMessageSize);

  ReadResult push_frame(const Frame& frame, Message& out);
  bool has_partial_message() const noexcept;
  void reset();

 private:
  size_t max_message_size_;
  bool fragmented_ = false;
  Opcode fragmented_opcode_ = Opcode::CONT;
  std::vector<uint8_t> fragmented_payload_;
};

std::vector<uint8_t> write_frame(Opcode opcode,
                                 std::string_view payload,
                                 bool mask = false,
                                 bool fin = true);

std::vector<uint8_t> write_close_frame(uint16_t code, std::string_view reason);

}  // namespace ws_framing
