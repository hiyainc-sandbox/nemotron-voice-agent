#include "lib/runtime_io/prewarm.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <sstream>

namespace runtime_io {
namespace {

constexpr size_t kReadBlockBytes = 16U * 1024U * 1024U;
constexpr size_t kMaxWorkerThreads = 2;

bool stat_regular_file(const std::string& path, uint64_t* bytes) noexcept {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) return false;
  if (!S_ISREG(st.st_mode)) return false;
  *bytes = st.st_size > 0 ? static_cast<uint64_t>(st.st_size) : 0;
  return true;
}

void log_errno_failure(const char* op, const std::string& path, int err) noexcept {
  std::fprintf(stderr, "PREWARM %s failed path=%s error=%s\n", op, path.c_str(), std::strerror(err));
  std::fflush(stderr);
}

void log_fadvise_failure(const std::string& path, int err) noexcept {
  std::fprintf(stderr,
               "PREWARM posix_fadvise failed path=%s error=%s\n",
               path.c_str(),
               std::strerror(err));
  std::fflush(stderr);
}

void prewarm_one_file(const std::string& path,
                      char* buffer,
                      size_t buffer_bytes,
                      const std::atomic<bool>& cancel) noexcept {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno != ENOENT && errno != ENOTDIR) log_errno_failure("open", path, errno);
    return;
  }

  int advise_err = ::posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
  if (advise_err != 0) log_fadvise_failure(path, advise_err);

  while (!cancel.load(std::memory_order_acquire)) {
    ssize_t got = ::read(fd, buffer, buffer_bytes);
    if (got > 0) continue;
    if (got == 0) break;
    if (errno == EINTR) continue;
    log_errno_failure("read", path, errno);
    break;
  }

  if (::close(fd) != 0) log_errno_failure("close", path, errno);
}

}  // namespace

Prewarmer::~Prewarmer() {
  join();
}

bool Prewarmer::start(const std::vector<std::string>& paths) noexcept {
  try {
    join();
    files_.clear();
    next_index_.store(0, std::memory_order_release);
    cancel_.store(false, std::memory_order_release);

    for (const auto& path : paths) {
      if (path.empty()) continue;
      uint64_t bytes = 0;
      if (!stat_regular_file(path, &bytes)) continue;
      files_.push_back({path, bytes});
    }

    if (files_.empty()) return true;

    const size_t worker_count = std::min(files_.size(), kMaxWorkerThreads);
    workers_.reserve(worker_count);
    for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
      workers_.emplace_back([this, worker_id]() { worker_loop(worker_id); });
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "PREWARM start failed error=%s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "PREWARM start failed error=unknown\n");
  }

  std::fflush(stderr);
  cancel_.store(true, std::memory_order_release);
  join();
  files_.clear();
  return false;
}

void Prewarmer::join() noexcept {
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      try {
        worker.join();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "PREWARM join failed error=%s\n", e.what());
        std::fflush(stderr);
      } catch (...) {
        std::fprintf(stderr, "PREWARM join failed error=unknown\n");
        std::fflush(stderr);
      }
    }
  }
  workers_.clear();
}

size_t Prewarmer::queued_file_count() const noexcept {
  return files_.size();
}

uint64_t Prewarmer::queued_bytes() const noexcept {
  uint64_t total = 0;
  for (const auto& file : files_) {
    if (file.bytes > std::numeric_limits<uint64_t>::max() - total) {
      return std::numeric_limits<uint64_t>::max();
    }
    total += file.bytes;
  }
  return total;
}

std::string Prewarmer::queued_paths_csv() const {
  std::ostringstream oss;
  for (size_t i = 0; i < files_.size(); ++i) {
    if (i != 0) oss << ",";
    oss << files_[i].path;
  }
  return oss.str();
}

void Prewarmer::worker_loop(size_t worker_id) noexcept {
  (void)worker_id;
  try {
    std::unique_ptr<char[]> buffer(new char[kReadBlockBytes]);
    for (;;) {
      if (cancel_.load(std::memory_order_acquire)) return;
      const size_t index = next_index_.fetch_add(1, std::memory_order_acq_rel);
      if (index >= files_.size()) return;
      prewarm_one_file(files_[index].path, buffer.get(), kReadBlockBytes, cancel_);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "PREWARM worker failed error=%s\n", e.what());
    std::fflush(stderr);
  } catch (...) {
    std::fprintf(stderr, "PREWARM worker failed error=unknown\n");
    std::fflush(stderr);
  }
}

}  // namespace runtime_io
