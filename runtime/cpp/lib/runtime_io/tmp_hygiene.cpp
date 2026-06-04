#include "lib/runtime_io/tmp_hygiene.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

namespace runtime_io {
namespace {

constexpr const char* kTmpDir = "/tmp";
constexpr int kMaxMarkerDepth = 4;
constexpr long long kDefaultMinAgeSeconds = 60;

struct Candidate {
  std::string name;
  std::string path;
  struct stat st {};
};

enum class ProcPidOwner {
  kGone,
  kOtherEuid,
  kSameEuid,
  kUnsafe,
};

bool is_dot_entry(const char* name) {
  return std::strcmp(name, ".") == 0 || std::strcmp(name, "..") == 0;
}

bool is_mkdtemp_basename(const char* name) {
  if (std::strlen(name) != 6) return false;
  for (const char* p = name; *p != '\0'; ++p) {
    char ch = *p;
    bool ascii_alnum = (ch >= 'A' && ch <= 'Z') ||
                       (ch >= 'a' && ch <= 'z') ||
                       (ch >= '0' && ch <= '9');
    if (!ascii_alnum) return false;
  }
  return true;
}

bool is_numeric_pid(const char* name) {
  if (name[0] == '\0') return false;
  for (const char* p = name; *p != '\0'; ++p) {
    if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
  }
  return true;
}

std::string tmp_path_for_name(const std::string& name) {
  return std::string(kTmpDir) + "/" + name;
}

bool same_inode(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

bool stat_is_real_dir(const struct stat& st) {
  return S_ISDIR(st.st_mode);
}

bool timespec_after(const struct timespec& a, const struct timespec& b) {
  if (a.tv_sec != b.tv_sec) return a.tv_sec > b.tv_sec;
  return a.tv_nsec > b.tv_nsec;
}

void update_newest_mtime(const struct stat& st, struct timespec* newest) {
  if (timespec_after(st.st_mtim, *newest)) *newest = st.st_mtim;
}

bool errno_is_pid_gone(int err) {
  return err == ENOENT || err == ESRCH;
}

ProcPidOwner classify_proc_pid(const std::string& pid, uid_t euid) {
  std::string proc_path = std::string("/proc/") + pid;
  struct stat st {};
  if (::stat(proc_path.c_str(), &st) != 0) {
    return errno_is_pid_gone(errno) ? ProcPidOwner::kGone : ProcPidOwner::kUnsafe;
  }
  return st.st_uid == euid ? ProcPidOwner::kSameEuid : ProcPidOwner::kOtherEuid;
}

int open_tmp_dir_fd() {
  return ::open(kTmpDir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
}

int open_dir_no_follow_at(int parent_fd, const char* name) {
  return ::openat(parent_fd, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
}

int open_candidate_dir(const Candidate& candidate) {
  int fd = ::open(candidate.path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return -1;
  struct stat opened {};
  if (::fstat(fd, &opened) != 0 || !same_inode(opened, candidate.st) || !stat_is_real_dir(opened)) {
    ::close(fd);
    return -1;
  }
  return fd;
}

bool add_stat_bytes(const struct stat& st, size_t* total) {
  unsigned long long bytes = 0;
  if (st.st_blocks > 0) {
    bytes = static_cast<unsigned long long>(st.st_blocks) * 512ULL;
  } else if (st.st_size > 0) {
    bytes = static_cast<unsigned long long>(st.st_size);
  }
  if (bytes > static_cast<unsigned long long>(std::numeric_limits<size_t>::max() - *total)) {
    *total = std::numeric_limits<size_t>::max();
    return false;
  }
  *total += static_cast<size_t>(bytes);
  return true;
}

bool fd_child_is_dir(int parent_fd, const char* name, struct stat* st_out = nullptr) {
  struct stat st {};
  if (::fstatat(parent_fd, name, &st, AT_SYMLINK_NOFOLLOW) != 0) return false;
  if (!stat_is_real_dir(st)) return false;
  if (st_out != nullptr) *st_out = st;
  return true;
}

bool contains_aoti_marker_fd(int dir_fd, int depth) {
  struct stat data_st {};
  if (fd_child_is_dir(dir_fd, "data", &data_st)) {
    int data_fd = open_dir_no_follow_at(dir_fd, "data");
    if (data_fd >= 0) {
      bool found = fd_child_is_dir(data_fd, "aotinductor");
      ::close(data_fd);
      if (found) return true;
    }
  }

  if (depth >= kMaxMarkerDepth) return false;

  int scan_fd = ::dup(dir_fd);
  if (scan_fd < 0) return false;
  DIR* dir = ::fdopendir(scan_fd);
  if (dir == nullptr) {
    ::close(scan_fd);
    return false;
  }

  bool found = false;
  while (!found) {
    errno = 0;
    dirent* ent = ::readdir(dir);
    if (ent == nullptr) break;
    if (is_dot_entry(ent->d_name)) continue;

    struct stat child_st {};
    if (::fstatat(dir_fd, ent->d_name, &child_st, AT_SYMLINK_NOFOLLOW) != 0) continue;
    if (!stat_is_real_dir(child_st)) continue;

    int child_fd = open_dir_no_follow_at(dir_fd, ent->d_name);
    if (child_fd < 0) continue;
    found = contains_aoti_marker_fd(child_fd, depth + 1);
    ::close(child_fd);
  }

  ::closedir(dir);
  return found;
}

bool contains_aoti_marker(const Candidate& candidate) {
  int fd = open_candidate_dir(candidate);
  if (fd < 0) return false;
  bool found = contains_aoti_marker_fd(fd, 0);
  ::close(fd);
  return found;
}

bool path_starts_with_candidate(const std::string& path, const std::string& candidate_path) {
  if (path.compare(0, candidate_path.size(), candidate_path) != 0) return false;
  if (path.size() == candidate_path.size()) return true;
  char next = path[candidate_path.size()];
  return next == '/' || std::isspace(static_cast<unsigned char>(next));
}

bool proc_maps_references_candidate(const std::string& pid,
                                    const std::string& candidate_path,
                                    bool pid_same_euid) {
  std::string maps_path = std::string("/proc/") + pid + "/maps";
  int maps_fd = ::open(maps_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (maps_fd < 0) {
    return pid_same_euid && !errno_is_pid_gone(errno);
  }

  FILE* maps = ::fdopen(maps_fd, "r");
  if (maps == nullptr) {
    ::close(maps_fd);
    return true;
  }

  bool found = false;
  bool unsafe = false;
  char* line = nullptr;
  size_t line_cap = 0;
  while (!found) {
    errno = 0;
    ssize_t n = ::getline(&line, &line_cap, maps);
    if (n < 0) {
      if (!::feof(maps) && pid_same_euid) unsafe = true;
      break;
    }
    std::string line_text(line, static_cast<size_t>(n));
    size_t path_pos = line_text.find('/');
    if (path_pos == std::string::npos) continue;
    if (path_starts_with_candidate(line_text.substr(path_pos), candidate_path)) found = true;
  }

  std::free(line);
  if (::fclose(maps) != 0 && !found) unsafe = true;
  return found || unsafe;
}

bool proc_fds_reference_candidate(const std::string& pid,
                                  const std::string& candidate_path,
                                  bool pid_same_euid) {
  std::string fd_dir_path = std::string("/proc/") + pid + "/fd";
  DIR* fd_dir = ::opendir(fd_dir_path.c_str());
  if (fd_dir == nullptr) return pid_same_euid && !errno_is_pid_gone(errno);

  bool found = false;
  bool unsafe = false;
  std::vector<char> target(8192);
  while (!found && !unsafe) {
    errno = 0;
    dirent* ent = ::readdir(fd_dir);
    if (ent == nullptr) {
      if (errno != 0 && pid_same_euid) unsafe = true;
      break;
    }
    if (is_dot_entry(ent->d_name)) continue;

    std::string link_path = fd_dir_path + "/" + ent->d_name;
    errno = 0;
    ssize_t n = ::readlink(link_path.c_str(), target.data(), target.size() - 1);
    if (n < 0) {
      if (pid_same_euid && !errno_is_pid_gone(errno)) unsafe = true;
      continue;
    }
    std::string resolved(target.data(), static_cast<size_t>(n));
    if (path_starts_with_candidate(resolved, candidate_path)) found = true;
  }

  if (::closedir(fd_dir) != 0 && !found && pid_same_euid) unsafe = true;
  return found || unsafe;
}

bool live_process_references(const std::string& candidate_path) {
  DIR* proc = ::opendir("/proc");
  if (proc == nullptr) {
    // If /proc cannot be scanned, deletion is not provably safe.
    return true;
  }

  bool found = false;
  while (!found) {
    errno = 0;
    dirent* ent = ::readdir(proc);
    if (ent == nullptr) break;
    if (!is_numeric_pid(ent->d_name)) continue;

    std::string pid(ent->d_name);
    ProcPidOwner owner = classify_proc_pid(pid, ::geteuid());
    if (owner == ProcPidOwner::kGone) continue;
    if (owner == ProcPidOwner::kUnsafe) {
      found = true;
      break;
    }

    bool pid_same_euid = owner == ProcPidOwner::kSameEuid;
    if (proc_maps_references_candidate(pid, candidate_path, pid_same_euid) ||
        proc_fds_reference_candidate(pid, candidate_path, pid_same_euid)) {
      found = true;
    }
  }

  ::closedir(proc);
  return found;
}

bool sum_tree_bytes_fd(int dir_fd,
                       const struct stat& dir_st,
                       dev_t root_dev,
                       size_t* total,
                       struct timespec* newest_mtime,
                       bool* mount_boundary) {
  if (dir_st.st_dev != root_dev) {
    *mount_boundary = true;
    return false;
  }
  update_newest_mtime(dir_st, newest_mtime);
  if (!add_stat_bytes(dir_st, total)) return false;

  int scan_fd = ::dup(dir_fd);
  if (scan_fd < 0) return false;
  DIR* dir = ::fdopendir(scan_fd);
  if (dir == nullptr) {
    ::close(scan_fd);
    return false;
  }

  bool ok = true;
  while (ok) {
    errno = 0;
    dirent* ent = ::readdir(dir);
    if (ent == nullptr) break;
    if (is_dot_entry(ent->d_name)) continue;

    struct stat child_st {};
    if (::fstatat(dir_fd, ent->d_name, &child_st, AT_SYMLINK_NOFOLLOW) != 0) {
      ok = false;
      break;
    }
    update_newest_mtime(child_st, newest_mtime);
    if (child_st.st_dev != root_dev) {
      *mount_boundary = true;
      ok = false;
      break;
    }

    if (stat_is_real_dir(child_st)) {
      int child_fd = open_dir_no_follow_at(dir_fd, ent->d_name);
      if (child_fd < 0) {
        ok = false;
        break;
      }
      ok = sum_tree_bytes_fd(child_fd, child_st, root_dev, total, newest_mtime, mount_boundary);
      ::close(child_fd);
    } else {
      ok = add_stat_bytes(child_st, total);
    }
  }

  ::closedir(dir);
  return ok;
}

bool compute_tree_metadata(const Candidate& candidate,
                           size_t* bytes,
                           struct timespec* newest_mtime,
                           bool* mount_boundary) {
  *bytes = 0;
  *newest_mtime = candidate.st.st_mtim;
  *mount_boundary = false;

  int tmp_fd = open_tmp_dir_fd();
  if (tmp_fd < 0) return false;

  struct stat current {};
  bool ok = ::fstatat(tmp_fd, candidate.name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
            same_inode(current, candidate.st) &&
            stat_is_real_dir(current) &&
            current.st_uid == ::geteuid();
  int dir_fd = -1;
  if (ok) {
    dir_fd = open_dir_no_follow_at(tmp_fd, candidate.name.c_str());
    ok = dir_fd >= 0;
  }
  if (ok) {
    struct stat opened {};
    ok = ::fstat(dir_fd, &opened) == 0 &&
         same_inode(opened, candidate.st) &&
         stat_is_real_dir(opened);
  }
  if (ok) {
    ok = sum_tree_bytes_fd(dir_fd, current, current.st_dev, bytes, newest_mtime, mount_boundary);
  }

  if (dir_fd >= 0) ::close(dir_fd);
  ::close(tmp_fd);
  return ok;
}

bool remove_tree_contents_fd(int dir_fd, dev_t root_dev, bool* mount_boundary) {
  int scan_fd = ::dup(dir_fd);
  if (scan_fd < 0) return false;
  DIR* dir = ::fdopendir(scan_fd);
  if (dir == nullptr) {
    ::close(scan_fd);
    return false;
  }

  bool ok = true;
  while (ok) {
    errno = 0;
    dirent* ent = ::readdir(dir);
    if (ent == nullptr) break;
    if (is_dot_entry(ent->d_name)) continue;

    struct stat child_st {};
    if (::fstatat(dir_fd, ent->d_name, &child_st, AT_SYMLINK_NOFOLLOW) != 0) {
      ok = false;
      break;
    }
    if (child_st.st_dev != root_dev) {
      *mount_boundary = true;
      ok = false;
      break;
    }

    if (stat_is_real_dir(child_st)) {
      int child_fd = open_dir_no_follow_at(dir_fd, ent->d_name);
      if (child_fd < 0) {
        ok = false;
        break;
      }
      ok = remove_tree_contents_fd(child_fd, root_dev, mount_boundary);
      ::close(child_fd);
      if (ok && ::unlinkat(dir_fd, ent->d_name, AT_REMOVEDIR) != 0) ok = false;
    } else {
      if (::unlinkat(dir_fd, ent->d_name, 0) != 0) ok = false;
    }
  }

  ::closedir(dir);
  return ok;
}

bool remove_candidate_tree(const Candidate& candidate, bool* mount_boundary) {
  *mount_boundary = false;

  int tmp_fd = open_tmp_dir_fd();
  if (tmp_fd < 0) return false;

  struct stat current {};
  bool ok = ::fstatat(tmp_fd, candidate.name.c_str(), &current, AT_SYMLINK_NOFOLLOW) == 0 &&
            same_inode(current, candidate.st) &&
            stat_is_real_dir(current) &&
            current.st_uid == ::geteuid();
  int dir_fd = -1;
  if (ok) {
    dir_fd = open_dir_no_follow_at(tmp_fd, candidate.name.c_str());
    ok = dir_fd >= 0;
  }
  if (ok) {
    struct stat opened {};
    ok = ::fstat(dir_fd, &opened) == 0 &&
         same_inode(opened, candidate.st) &&
         stat_is_real_dir(opened);
  }
  if (ok) {
    ok = remove_tree_contents_fd(dir_fd, current.st_dev, mount_boundary);
  }
  if (dir_fd >= 0) ::close(dir_fd);
  if (ok && ::unlinkat(tmp_fd, candidate.name.c_str(), AT_REMOVEDIR) != 0) ok = false;

  ::close(tmp_fd);
  return ok;
}

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool tmp_hygiene_enabled() {
  const char* raw = std::getenv("NEMOTRON_WS_TMP_HYGIENE");
  if (raw == nullptr || raw[0] == '\0') return true;
  std::string value = lower_ascii(raw);
  return value != "0" && value != "false" && value != "off";
}

long long tmp_hygiene_min_age_seconds() {
  const char* raw = std::getenv("NEMOTRON_WS_TMP_HYGIENE_MIN_AGE_S");
  if (raw == nullptr || raw[0] == '\0') return kDefaultMinAgeSeconds;

  errno = 0;
  char* end = nullptr;
  long long parsed = std::strtoll(raw, &end, 10);
  if (end == raw) return kDefaultMinAgeSeconds;
  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (*end != '\0') return kDefaultMinAgeSeconds;
  if (errno == ERANGE && parsed > 0) return std::numeric_limits<long long>::max();
  if (parsed < 0) return 0;
  return parsed;
}

bool mtime_within_min_age(const struct timespec& newest_mtime, long long min_age_s) {
  if (min_age_s <= 0) return false;
  std::time_t now = std::time(nullptr);
  if (now == static_cast<std::time_t>(-1)) return true;

  struct timespec cutoff {};
  if (min_age_s >= static_cast<long long>(now)) {
    cutoff.tv_sec = 0;
  } else {
    cutoff.tv_sec = now - static_cast<std::time_t>(min_age_s);
  }
  cutoff.tv_nsec = 0;
  return timespec_after(newest_mtime, cutoff);
}

void log_summary(const TmpHygieneResult& result, bool disabled = false) {
  std::printf("TMP_HYGIENE_SUMMARY reclaimed_dirs=%zu reclaimed_mib=%.3f skipped_live=%zu%s\n",
              result.reclaimed_dirs,
              static_cast<double>(result.reclaimed_bytes) / (1024.0 * 1024.0),
              result.skipped_live,
              disabled ? " disabled=1" : "");
  std::fflush(stdout);
}

}  // namespace

TmpHygieneResult reclaim_stale_aoti_tmp_dirs(bool dry_run) {
  TmpHygieneResult result;
  if (!tmp_hygiene_enabled()) {
    log_summary(result, true);
    return result;
  }

  long long min_age_s = tmp_hygiene_min_age_seconds();
  DIR* tmp = ::opendir(kTmpDir);
  if (tmp == nullptr) {
    log_summary(result);
    return result;
  }

  std::vector<Candidate> candidates;
  uid_t euid = ::geteuid();
  while (true) {
    errno = 0;
    dirent* ent = ::readdir(tmp);
    if (ent == nullptr) break;
    if (is_dot_entry(ent->d_name) || !is_mkdtemp_basename(ent->d_name)) continue;

    Candidate candidate;
    candidate.name = ent->d_name;
    candidate.path = tmp_path_for_name(candidate.name);
    if (::lstat(candidate.path.c_str(), &candidate.st) != 0) continue;
    if (!stat_is_real_dir(candidate.st)) continue;
    if (candidate.st.st_uid != euid) continue;
    if (!contains_aoti_marker(candidate)) continue;
    candidates.push_back(candidate);
  }
  ::closedir(tmp);

  for (const Candidate& candidate : candidates) {
    if (live_process_references(candidate.path)) {
      ++result.skipped_live;
      std::printf("TMP_HYGIENE skipped_live dir=%s\n", candidate.path.c_str());
      std::fflush(stdout);
      continue;
    }

    size_t bytes = 0;
    struct timespec newest_mtime {};
    bool mount_boundary = false;
    if (!compute_tree_metadata(candidate, &bytes, &newest_mtime, &mount_boundary)) {
      std::printf("TMP_HYGIENE skipped dir=%s reason=%s\n",
                  candidate.path.c_str(),
                  mount_boundary ? "mount_boundary" : "sum_failed");
      std::fflush(stdout);
      continue;
    }

    if (mtime_within_min_age(newest_mtime, min_age_s)) {
      std::printf("TMP_HYGIENE skipped reason=too_recent dir=%s min_age_s=%lld latest_mtime_s=%lld\n",
                  candidate.path.c_str(),
                  min_age_s,
                  static_cast<long long>(newest_mtime.tv_sec));
      std::fflush(stdout);
      continue;
    }

    if (live_process_references(candidate.path)) {
      ++result.skipped_live;
      std::printf("TMP_HYGIENE skipped_live dir=%s\n", candidate.path.c_str());
      std::fflush(stdout);
      continue;
    }

    mount_boundary = false;
    if (!dry_run && !remove_candidate_tree(candidate, &mount_boundary)) {
      std::printf("TMP_HYGIENE skipped dir=%s reason=%s\n",
                  candidate.path.c_str(),
                  mount_boundary ? "mount_boundary" : "remove_failed");
      std::fflush(stdout);
      continue;
    }

    ++result.reclaimed_dirs;
    result.reclaimed_bytes += bytes;
    std::printf("TMP_HYGIENE reclaimed dir=%s bytes=%zu%s\n",
                candidate.path.c_str(),
                bytes,
                dry_run ? " dry_run=1" : "");
    std::fflush(stdout);
  }

  log_summary(result);
  return result;
}

}  // namespace runtime_io
