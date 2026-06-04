#include "lib/runtime_io/jit_load.h"

#include <mutex>

torch::jit::Module load_jit_serialized(const std::string& path) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  return torch::jit::load(path);
}
