#!/usr/bin/env python3
"""Build the checked manifest for stripped finalize AOTI buckets.

The manifest is the deploy-time contract consumed by cpp/finalize_main.cpp:
model/geometry constants, the shared-weights digest, and a complete inventory
of every stripped bucket package with its SHA-256.  Strip validation records are
kept per bucket when the package hash still matches.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python build_bucket_manifest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any


MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
BUCKET_RE = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path) or ".",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def existing_validations(manifest_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    manifest = load_json(manifest_path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket in manifest.get("buckets", []):
        if not isinstance(bucket, dict):
            continue
        validation = bucket.get("strip_validation")
        pkg = bucket.get("pkg")
        digest = bucket.get("pkg_sha256")
        if isinstance(validation, dict) and isinstance(pkg, str) and isinstance(digest, str):
            out[(pkg, digest)] = validation
    return out


def derive_contract(weights_path: str) -> dict[str, Any]:
    from finalize_ref import BLANK, MAX_SYMBOLS, RIGHT_CONTEXT, ContinuousFinalizeRef, load_model

    model = load_model()
    rt = ContinuousFinalizeRef(model)
    g = rt.geometry
    att_context = [int(x) for x in getattr(model.encoder, "att_context_size")]
    return {
        "model_id": MODEL_ID,
        "att_context": att_context,
        "right_context": int(RIGHT_CONTEXT),
        "shift": int(g.shift_frames),
        "pre_encode_cache": int(g.pre_encode_cache_size),
        "drop_extra": int(g.drop_extra),
        "final_padding_frames": int(g.final_padding_frames),
        "blank": int(BLANK),
        "max_symbols": int(MAX_SYMBOLS),
        "weights_sha256": sha256_file(weights_path),
    }


def discover_buckets(buckets_dir: str, preserved: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    if not os.path.isdir(buckets_dir):
        raise FileNotFoundError(f"bucket directory does not exist: {buckets_dir}")

    for name in sorted(os.listdir(buckets_dir)):
        match = BUCKET_RE.match(name)
        if not match:
            continue
        path = os.path.join(buckets_dir, name)
        if not os.path.isfile(path):
            continue
        digest = sha256_file(path)
        entry: dict[str, Any] = {
            "drop": int(match.group("drop")),
            "T": int(match.group("T")),
            "pkg": name,
            "pkg_sha256": digest,
        }
        validation = preserved.get((name, digest))
        if validation is not None:
            entry["strip_validation"] = validation
        buckets.append(entry)

    buckets.sort(key=lambda b: (int(b["drop"]), int(b["T"]), str(b["pkg"])))
    return buckets


def build_manifest(buckets_dir: str, weights_path: str, out_path: str) -> dict[str, Any]:
    preserved = existing_validations(out_path)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "CONTRACT": derive_contract(weights_path),
        "buckets": discover_buckets(buckets_dir, preserved),
    }
    atomic_write_json(out_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets-dir", default="artifacts/stripped_finalize_buckets")
    parser.add_argument("--weights", default="artifacts/finalize_shared_weights.pt")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.buckets_dir, "manifest.json")
    manifest = build_manifest(args.buckets_dir, args.weights, out_path)
    print(
        f"wrote {out_path}: buckets={len(manifest['buckets'])} "
        f"weights_sha256={manifest['CONTRACT']['weights_sha256']}"
    )
    if manifest["buckets"]:
        print("sample bucket:", json.dumps(manifest["buckets"][0], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
