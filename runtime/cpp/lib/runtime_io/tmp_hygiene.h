#pragma once

#include <cstddef>

namespace runtime_io {

struct TmpHygieneResult {
  size_t reclaimed_dirs = 0;
  size_t reclaimed_bytes = 0;
  size_t skipped_live = 0;
};

// Reclaim stale, owned, dead-process AOTI extraction trees left by non-graceful
// exits. Safe by construction: only /tmp/<6 alnum> directories owned by this
// euid, containing data/aotinductor, and unreferenced by live /proc maps/fds
// are removed.
TmpHygieneResult reclaim_stale_aoti_tmp_dirs(bool dry_run = false);

}  // namespace runtime_io
