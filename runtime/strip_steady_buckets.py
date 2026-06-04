#!/usr/bin/env python3
"""Strip steady scheduler bucket weights in place and refresh MANIFEST.json.

The steady B buckets share the same encoder weights as finalize.  The runtime
binds those tensors from artifacts/finalize_shared_weights.ts via
load_constants(..., user_managed=True), so the package-local data/weights/*
payloads can be empty.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from strip_bucket_weights import fmt_bytes, is_weight_entry, sha256_file, strip_package


DEFAULT_BUCKETS = (1, 2, 4)


def package_name(bucket: int) -> str:
    return f"enc_steady_aoti_b{bucket}.pt2"


def ep_name(bucket: int) -> str:
    return f"enc_steady_t2a_b{bucket}.pt2"


def weight_payload_bytes(path: Path) -> tuple[int, int]:
    entries = 0
    payload_bytes = 0
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir() and is_weight_entry(info.filename):
                entries += 1
                payload_bytes += int(info.file_size)
    return entries, payload_bytes


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict) or not isinstance(value.get("buckets"), list):
        raise RuntimeError(f"{path} is not a steady bucket manifest")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def parse_buckets(text: str) -> list[int]:
    buckets = [int(part) for part in text.split(",") if part.strip()]
    if not buckets:
        raise RuntimeError("--buckets produced no bucket ids")
    unknown = sorted(set(buckets) - set(DEFAULT_BUCKETS))
    if unknown:
        raise RuntimeError(f"unsupported steady buckets: {unknown}")
    return buckets


def ensure_backup(src: Path, backup: Path) -> Path | None:
    entries, payload_bytes = weight_payload_bytes(src)
    if entries == 0:
        raise RuntimeError(
            f"{src} has no data/weights entries; recompile from the ExportedProgram with "
            "aot_inductor.package_constants_in_so=False and package_constants_on_disk=True first"
        )

    if payload_bytes == 0:
        return None

    if backup.exists():
        print(f"using existing full-package backup: {backup}")
        return src

    shutil.move(str(src), str(backup))
    print(f"moved full package backup: {src} -> {backup}")
    return backup


def update_manifest(manifest_path: Path, artifact_dir: Path, shared_weights: Path, buckets: list[int]) -> None:
    manifest = load_manifest(manifest_path)
    by_bucket = {int(bucket["B"]): bucket for bucket in manifest["buckets"]}
    shared_sha = sha256_file(str(shared_weights))
    shared_size = shared_weights.stat().st_size

    for bucket in buckets:
        if bucket not in by_bucket:
            raise RuntimeError(f"manifest missing B={bucket}")
        entry = by_bucket[bucket]
        pkg = artifact_dir / package_name(bucket)
        ep = artifact_dir / ep_name(bucket)
        if not pkg.exists():
            raise FileNotFoundError(pkg)
        if not ep.exists():
            raise FileNotFoundError(ep)

        entry["package"] = package_name(bucket)
        entry["package_sha256"] = sha256_file(str(pkg))
        entry["ep"] = ep_name(bucket)
        entry["ep_sha256"] = sha256_file(str(ep))
        entry["shared_weight"] = str(shared_weights)
        entry["shared_weight_sha256"] = shared_sha
        byte_sizes = entry.setdefault("byte_sizes", {})
        byte_sizes["package"] = pkg.stat().st_size
        byte_sizes["ep"] = ep.stat().st_size
        byte_sizes["shared_weight"] = shared_size

    parity = manifest.get("a1_b1_parity")
    if isinstance(parity, dict) and 1 in buckets:
        b1 = artifact_dir / package_name(1)
        parity["new_b1_package"] = str(artifact_dir / package_name(1))
        parity["new_b1_package_sha256"] = sha256_file(str(b1))
        production_sha = parity.get("production_package_sha256")
        if isinstance(production_sha, str):
            parity["package_sha256_equal"] = production_sha == parity["new_b1_package_sha256"]

    atomic_write_json(manifest_path, manifest)
    print(f"updated {manifest_path}")


def run(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.dir)
    manifest_path = artifact_dir / "MANIFEST.json"
    shared_weights = Path(args.shared_weights)
    buckets = parse_buckets(args.buckets)

    if not artifact_dir.is_dir():
        raise FileNotFoundError(artifact_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not shared_weights.exists():
        raise FileNotFoundError(shared_weights)

    for bucket in buckets:
        dst = artifact_dir / package_name(bucket)
        backup = artifact_dir / f"{package_name(bucket)}{args.backup_suffix}"
        if args.dry_run:
            entries, payload_bytes = weight_payload_bytes(dst)
            if entries == 0:
                action = "needs constants-on-disk recompile before stripping"
            elif payload_bytes == 0:
                action = "already stripped"
            elif backup.exists():
                action = f"strip in place using existing backup {backup}"
            else:
                action = f"move to {backup} and strip back in place"
            print(f"dry-run: B={bucket} entries={entries} payload={fmt_bytes(payload_bytes)} action={action}")
            continue
        source = ensure_backup(dst, backup)
        if source is None:
            print(f"steady B={bucket} already stripped: {dst}")
            continue
        src_size = source.stat().st_size
        stats = strip_package(str(source), str(dst))
        stats.src_size = src_size
        entries, payload_bytes = weight_payload_bytes(dst)
        if payload_bytes != 0:
            raise RuntimeError(f"{dst} still has {payload_bytes} bytes in data/weights payloads")
        print(
            f"stripped steady B={bucket}: entries={entries} "
            f"{fmt_bytes(stats.src_size)} -> {fmt_bytes(stats.dst_size)} "
            f"sha256={sha256_file(str(dst))}"
        )

    if args.dry_run:
        return 0

    update_manifest(manifest_path, artifact_dir, shared_weights, buckets)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="steady_b_artifacts")
    parser.add_argument("--shared-weights", default="artifacts/finalize_shared_weights.ts")
    parser.add_argument("--buckets", default="1,2,4")
    parser.add_argument("--backup-suffix", default=".full.pt2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
