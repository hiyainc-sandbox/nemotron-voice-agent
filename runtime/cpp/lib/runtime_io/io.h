#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace runtime_io {

bool file_exists(const std::string& path);
bool directory_exists(const std::string& path);

std::string sha256_file(const std::string& path);
std::string read_text_file(const std::string& path);

size_t skip_ws(const std::string& s, size_t pos);
size_t find_matching_json_delim(const std::string& s, size_t open_pos);
std::string json_value_for_key(const std::string& object,
                               const std::string& key,
                               bool required = true);
std::string json_string_field(const std::string& object,
                              const std::string& key,
                              bool required = true);
int64_t json_int_field(const std::string& object, const std::string& key);

}  // namespace runtime_io
