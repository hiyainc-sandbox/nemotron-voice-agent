#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace runtime_io {

// Kicks background page-cache prewarm for read-only artifact files. start()
// skips missing/non-regular files and never reports worker failures to callers;
// the destructor joins all owned workers.
class Prewarmer {
 public:
  Prewarmer() = default;
  ~Prewarmer();

  Prewarmer(const Prewarmer&) = delete;
  Prewarmer& operator=(const Prewarmer&) = delete;

  bool start(const std::vector<std::string>& paths) noexcept;
  void join() noexcept;

  size_t queued_file_count() const noexcept;
  uint64_t queued_bytes() const noexcept;
  std::string queued_paths_csv() const;

 private:
  struct File {
    std::string path;
    uint64_t bytes = 0;
  };

  void worker_loop(size_t worker_id) noexcept;

  std::vector<File> files_;
  std::vector<std::thread> workers_;
  std::atomic<size_t> next_index_{0};
  std::atomic<bool> cancel_{false};
};

}  // namespace runtime_io
