#include "lib/ws/framing.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ws_framing {
namespace {

bool is_known_opcode(uint8_t opcode) {
  switch (opcode) {
    case 0x0:
    case 0x1:
    case 0x2:
    case 0x8:
    case 0x9:
    case 0xA:
      return true;
    default:
      return false;
  }
}

bool is_control_opcode(uint8_t opcode) {
  return opcode >= 0x8;
}

bool is_control_opcode(Opcode opcode) {
  return is_control_opcode(static_cast<uint8_t>(opcode));
}

uint8_t byte_at(const std::string& buffer, size_t pos) {
  return static_cast<uint8_t>(buffer[pos]);
}

}  // namespace

ReadResult read_frame(const std::string& buffer,
                      Frame& out,
                      size_t& consumed,
                      size_t max_payload_size) {
  consumed = 0;
  if (buffer.size() < 2) return ReadResult::NEED_MORE;

  const uint8_t b0 = byte_at(buffer, 0);
  const uint8_t b1 = byte_at(buffer, 1);
  const bool fin = (b0 & 0x80) != 0;
  const bool rsv_set = (b0 & 0x70) != 0;
  const uint8_t opcode = b0 & 0x0f;
  const bool masked = (b1 & 0x80) != 0;
  uint64_t payload_len = b1 & 0x7f;
  size_t pos = 2;

  if (rsv_set || !is_known_opcode(opcode)) return ReadResult::MALFORMED;
  if (!masked) return ReadResult::MALFORMED;
  if (is_control_opcode(opcode) && !fin) return ReadResult::MALFORMED;

  if (payload_len == 126) {
    if (buffer.size() < pos + 2) return ReadResult::NEED_MORE;
    payload_len = (static_cast<uint64_t>(byte_at(buffer, pos)) << 8) |
                  static_cast<uint64_t>(byte_at(buffer, pos + 1));
    pos += 2;
    if (payload_len < 126) return ReadResult::MALFORMED;
  } else if (payload_len == 127) {
    if (buffer.size() < pos + 8) return ReadResult::NEED_MORE;
    payload_len = 0;
    for (int i = 0; i < 8; ++i) {
      payload_len = (payload_len << 8) | byte_at(buffer, pos + static_cast<size_t>(i));
    }
    pos += 8;
    if ((payload_len & (1ULL << 63)) != 0 || payload_len < 65536) {
      return ReadResult::MALFORMED;
    }
  }

  if (payload_len > static_cast<uint64_t>(max_payload_size)) {
    return ReadResult::FRAME_TOO_LARGE;
  }
  if (is_control_opcode(opcode) && payload_len > 125) return ReadResult::MALFORMED;
  if (payload_len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return ReadResult::FRAME_TOO_LARGE;
  }

  if (buffer.size() < pos + 4) return ReadResult::NEED_MORE;
  std::array<uint8_t, 4> mask = {
      byte_at(buffer, pos),
      byte_at(buffer, pos + 1),
      byte_at(buffer, pos + 2),
      byte_at(buffer, pos + 3),
  };
  pos += 4;

  const size_t len = static_cast<size_t>(payload_len);
  if (len > buffer.size() - pos) return ReadResult::NEED_MORE;

  Frame frame;
  frame.opcode = static_cast<Opcode>(opcode);
  frame.fin = fin;
  frame.payload.resize(len);
  for (size_t i = 0; i < len; ++i) {
    frame.payload[i] = static_cast<uint8_t>(byte_at(buffer, pos + i) ^ mask[i % 4]);
  }

  consumed = pos + len;
  out = std::move(frame);
  return ReadResult::OK;
}

MessageAssembler::MessageAssembler(size_t max_message_size)
    : max_message_size_(max_message_size) {}

ReadResult MessageAssembler::push_frame(const Frame& frame, Message& out) {
  out = Message{};

  if (is_control_opcode(frame.opcode)) {
    out.opcode = frame.opcode;
    out.payload = frame.payload;
    return ReadResult::OK;
  }

  if (frame.opcode == Opcode::CONT) {
    if (!fragmented_) return ReadResult::MALFORMED;
    if (fragmented_payload_.size() > max_message_size_ ||
        frame.payload.size() > max_message_size_ - fragmented_payload_.size()) {
      reset();
      return ReadResult::FRAME_TOO_LARGE;
    }
    fragmented_payload_.insert(fragmented_payload_.end(),
                               frame.payload.begin(),
                               frame.payload.end());
    if (!frame.fin) return ReadResult::NEED_MORE;

    out.opcode = fragmented_opcode_;
    out.payload = std::move(fragmented_payload_);
    reset();
    return ReadResult::OK;
  }

  if (frame.opcode != Opcode::TEXT && frame.opcode != Opcode::BINARY) {
    return ReadResult::MALFORMED;
  }
  if (fragmented_) return ReadResult::MALFORMED;
  if (frame.payload.size() > max_message_size_) return ReadResult::FRAME_TOO_LARGE;

  if (frame.fin) {
    out.opcode = frame.opcode;
    out.payload = frame.payload;
    return ReadResult::OK;
  }

  fragmented_ = true;
  fragmented_opcode_ = frame.opcode;
  fragmented_payload_ = frame.payload;
  return ReadResult::NEED_MORE;
}

bool MessageAssembler::has_partial_message() const noexcept {
  return fragmented_;
}

void MessageAssembler::reset() {
  fragmented_ = false;
  fragmented_opcode_ = Opcode::CONT;
  fragmented_payload_.clear();
}

std::vector<uint8_t> write_frame(Opcode opcode,
                                 std::string_view payload,
                                 bool mask,
                                 bool fin) {
  std::vector<uint8_t> out;
  out.reserve(14 + payload.size());
  out.push_back(static_cast<uint8_t>((fin ? 0x80 : 0x00) |
                                     (static_cast<uint8_t>(opcode) & 0x0f)));

  const uint64_t len = payload.size();
  const uint8_t mask_bit = mask ? 0x80 : 0x00;
  if (len < 126) {
    out.push_back(static_cast<uint8_t>(mask_bit | static_cast<uint8_t>(len)));
  } else if (len <= 0xffff) {
    out.push_back(static_cast<uint8_t>(mask_bit | 126));
    out.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>(len & 0xff));
  } else {
    out.push_back(static_cast<uint8_t>(mask_bit | 127));
    for (int shift = 56; shift >= 0; shift -= 8) {
      out.push_back(static_cast<uint8_t>((len >> shift) & 0xff));
    }
  }

  constexpr std::array<uint8_t, 4> kTestMask = {0x12, 0x34, 0x56, 0x78};
  if (mask) {
    out.insert(out.end(), kTestMask.begin(), kTestMask.end());
    for (size_t i = 0; i < payload.size(); ++i) {
      out.push_back(static_cast<uint8_t>(
          static_cast<uint8_t>(payload[i]) ^ kTestMask[i % kTestMask.size()]));
    }
  } else {
    out.insert(out.end(), payload.begin(), payload.end());
  }
  return out;
}

std::vector<uint8_t> write_close_frame(uint16_t code, std::string_view reason) {
  if (reason.size() > 123) {
    throw std::invalid_argument("WebSocket close reason exceeds 123 bytes");
  }
  std::string payload;
  payload.reserve(2 + reason.size());
  payload.push_back(static_cast<char>((code >> 8) & 0xff));
  payload.push_back(static_cast<char>(code & 0xff));
  payload.append(reason);
  return write_frame(Opcode::CLOSE, payload, false);
}

}  // namespace ws_framing
