#pragma once

#include "lib/runtime_io/io.h"

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace runtime_io {
namespace steady_batch_dir_detail {
namespace fs = std::filesystem;

inline bool regular_file_exists(const fs::path& path) {
  std::error_code ec;
  return fs::is_regular_file(path, ec);
}

inline std::string package_name_for_bucket(int bucket) {
  return "enc_steady_aoti_b" + std::to_string(bucket) + ".pt2";
}

inline std::vector<int> load_manifest_bucket_ids(const std::string& manifest_path) {
  std::string text = read_text_file(manifest_path);
  size_t top_buckets = text.rfind("\"buckets\"");
  if (top_buckets == std::string::npos) {
    throw std::runtime_error("steady batch manifest missing top-level buckets");
  }
  std::string buckets_arr = json_value_for_key(text.substr(top_buckets), "buckets");
  if (buckets_arr.empty() || buckets_arr.front() != '[') {
    throw std::runtime_error("steady batch manifest buckets is not an array");
  }

  std::vector<int> buckets;
  std::set<int> seen;
  size_t pos = 1;
  while (pos + 1 < buckets_arr.size()) {
    pos = skip_ws(buckets_arr, pos);
    if (pos >= buckets_arr.size() || buckets_arr[pos] == ']') break;
    if (buckets_arr[pos] == ',') {
      ++pos;
      continue;
    }
    if (buckets_arr[pos] != '{') {
      throw std::runtime_error("steady batch manifest bucket entry is not an object");
    }
    size_t end = find_matching_json_delim(buckets_arr, pos);
    std::string obj = buckets_arr.substr(pos, end - pos + 1);
    int bucket = static_cast<int>(json_int_field(obj, "B"));
    if (bucket <= 0) {
      throw std::runtime_error("steady batch manifest bucket B must be positive");
    }
    if (!seen.emplace(bucket).second) {
      throw std::runtime_error("steady batch manifest duplicate B=" + std::to_string(bucket));
    }
    buckets.push_back(bucket);
    pos = end + 1;
  }
  if (buckets.empty()) {
    throw std::runtime_error("steady batch manifest declares no buckets");
  }
  return buckets;
}

}  // namespace steady_batch_dir_detail

struct SteadyBatchDirValidation {
  bool ok = false;
  std::string error;
  std::vector<int> buckets;
};

inline SteadyBatchDirValidation validate_steady_batch_dir(const std::string& dir) {
  namespace detail = steady_batch_dir_detail;
  SteadyBatchDirValidation result;
  const std::filesystem::path root(dir);
  const std::filesystem::path manifest = root / "MANIFEST.json";
  if (!detail::regular_file_exists(manifest)) {
    result.error = "steady batch MANIFEST.json missing: " + manifest.string();
    return result;
  }

  try {
    result.buckets = detail::load_manifest_bucket_ids(manifest.string());
  } catch (const std::exception& e) {
    result.error = "steady batch MANIFEST.json invalid: " + manifest.string() + ": " + e.what();
    return result;
  }

  for (int bucket : result.buckets) {
    const std::filesystem::path package = root / detail::package_name_for_bucket(bucket);
    if (!detail::regular_file_exists(package)) {
      result.error = "steady batch package missing for manifest-declared B=" +
                     std::to_string(bucket) + ": " + package.string();
      result.ok = false;
      return result;
    }
  }
  result.ok = true;
  return result;
}

inline bool steady_batch_dir_has_declared_packages(const std::string& dir,
                                                   std::string* error = nullptr,
                                                   std::vector<int>* buckets = nullptr) {
  SteadyBatchDirValidation validation = validate_steady_batch_dir(dir);
  if (error != nullptr) *error = validation.error;
  if (buckets != nullptr) *buckets = validation.buckets;
  return validation.ok;
}

}  // namespace runtime_io
