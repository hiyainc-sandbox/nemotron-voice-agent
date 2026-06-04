#include "lib/ws/framing.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string to_string_bytes(const std::vector<uint8_t>& bytes) {
  return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::string payload_text(const ws_framing::Message& message) {
  return std::string(message.payload.begin(), message.payload.end());
}

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<ws_framing::Message> decode_stream(const std::string& stream,
                                               const std::vector<size_t>& chunk_sizes,
                                               size_t max_message_size = 1024 * 1024) {
  std::string buffer;
  ws_framing::MessageAssembler assembler(max_message_size);
  std::vector<ws_framing::Message> messages;
  size_t offset = 0;
  size_t chunk_index = 0;

  while (offset < stream.size()) {
    size_t chunk_size = stream.size() - offset;
    if (!chunk_sizes.empty()) {
      chunk_size = chunk_sizes[std::min(chunk_index, chunk_sizes.size() - 1)];
      if (chunk_size == 0) chunk_size = 1;
      ++chunk_index;
    }
    chunk_size = std::min(chunk_size, stream.size() - offset);
    buffer.append(stream.data() + offset, chunk_size);
    offset += chunk_size;

    for (;;) {
      ws_framing::Frame frame;
      size_t consumed = 0;
      ws_framing::ReadResult read = ws_framing::read_frame(buffer, frame, consumed, max_message_size);
      if (read == ws_framing::ReadResult::NEED_MORE) break;
      if (read != ws_framing::ReadResult::OK) throw std::runtime_error("frame decode failed");
      buffer.erase(0, consumed);

      ws_framing::Message message;
      ws_framing::ReadResult assembled = assembler.push_frame(frame, message);
      if (assembled == ws_framing::ReadResult::NEED_MORE) continue;
      if (assembled != ws_framing::ReadResult::OK) throw std::runtime_error("message assembly failed");
      messages.push_back(std::move(message));
    }
  }

  expect(buffer.empty(), "decoder left buffered bytes after complete stream");
  expect(!assembler.has_partial_message(), "assembler left a partial fragmented message");
  return messages;
}

std::string masked_frame(ws_framing::Opcode opcode,
                         const std::string& payload,
                         bool fin = true) {
  return to_string_bytes(ws_framing::write_frame(opcode, payload, true, fin));
}

void test_header_and_payload_split() {
  std::string stream = masked_frame(ws_framing::Opcode::TEXT, "{\"type\":\"vad_stop\"}");
  auto messages = decode_stream(stream, {1, 1, 1, 2, 3, 5});
  expect(messages.size() == 1, "split frame did not produce exactly one message");
  expect(messages[0].opcode == ws_framing::Opcode::TEXT, "split frame opcode mismatch");
  expect(payload_text(messages[0]) == "{\"type\":\"vad_stop\"}", "split frame payload mismatch");
}

void test_coalesced_frames() {
  std::string stream;
  stream += masked_frame(ws_framing::Opcode::TEXT, "{\"type\":\"vad_stop\"}");
  stream += masked_frame(ws_framing::Opcode::TEXT, "{\"type\":\"reset\",\"finalize\":true}");
  auto messages = decode_stream(stream, {stream.size()});
  expect(messages.size() == 2, "coalesced frames did not produce two messages");
  expect(payload_text(messages[0]) == "{\"type\":\"vad_stop\"}", "first coalesced payload mismatch");
  expect(payload_text(messages[1]) == "{\"type\":\"reset\",\"finalize\":true}", "second coalesced payload mismatch");
}

void test_one_byte_chunks_over_two_frames() {
  std::string stream;
  stream += masked_frame(ws_framing::Opcode::BINARY, std::string("\x01\x00\x02\x00", 4));
  stream += masked_frame(ws_framing::Opcode::TEXT, "{\"type\":\"reset\",\"finalize\":true}");
  auto messages = decode_stream(stream, {1});
  expect(messages.size() == 2, "one-byte chunks did not produce two messages");
  expect(messages[0].opcode == ws_framing::Opcode::BINARY, "one-byte binary opcode mismatch");
  expect(messages[0].payload.size() == 4, "one-byte binary payload size mismatch");
  expect(messages[1].opcode == ws_framing::Opcode::TEXT, "one-byte text opcode mismatch");
}

void test_fragmented_text_with_interleaved_ping() {
  std::string stream;
  stream += masked_frame(ws_framing::Opcode::TEXT, "{\"type\":\"", false);
  stream += masked_frame(ws_framing::Opcode::PING, "p");
  stream += masked_frame(ws_framing::Opcode::CONT, "reset\",\"finalize\":true}");
  auto messages = decode_stream(stream, {2, 1, 3});
  expect(messages.size() == 2, "fragmented stream did not produce ping plus text");
  expect(messages[0].opcode == ws_framing::Opcode::PING, "interleaved ping opcode mismatch");
  expect(payload_text(messages[0]) == "p", "interleaved ping payload mismatch");
  expect(messages[1].opcode == ws_framing::Opcode::TEXT, "fragmented text opcode mismatch");
  expect(payload_text(messages[1]) == "{\"type\":\"reset\",\"finalize\":true}",
         "fragmented text payload mismatch");
}

void test_oversize_fragmented_message() {
  std::string stream;
  stream += masked_frame(ws_framing::Opcode::TEXT, "12345", false);
  stream += masked_frame(ws_framing::Opcode::CONT, "67890");

  std::string buffer;
  ws_framing::MessageAssembler assembler(8);
  bool saw_too_large = false;
  buffer = stream;
  for (;;) {
    ws_framing::Frame frame;
    size_t consumed = 0;
    ws_framing::ReadResult read = ws_framing::read_frame(buffer, frame, consumed, 8);
    if (read == ws_framing::ReadResult::NEED_MORE) break;
    expect(read == ws_framing::ReadResult::OK, "oversize test frame decode failed early");
    buffer.erase(0, consumed);
    ws_framing::Message message;
    ws_framing::ReadResult assembled = assembler.push_frame(frame, message);
    if (assembled == ws_framing::ReadResult::FRAME_TOO_LARGE) {
      saw_too_large = true;
      break;
    }
  }
  expect(saw_too_large, "oversize fragmented message was not rejected");
}

}  // namespace

int main() {
  test_header_and_payload_split();
  test_coalesced_frames();
  test_one_byte_chunks_over_two_frames();
  test_fragmented_text_with_interleaved_ping();
  test_oversize_fragmented_message();
  std::cout << "ws_framing_selftest PASS\n";
  return 0;
}
