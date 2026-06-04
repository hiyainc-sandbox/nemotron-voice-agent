#!/usr/bin/env python3
"""Compile and strip the first-chunk encoder AOTI package.

The package is compiled constants-on-disk, then every packaged weight payload is
zeroed.  At runtime, C++ wires the real tensors from finalize_shared_weights.ts
through load_constants(..., user_managed=True), sharing the encoder constants
with finalize buckets.

Run in the CUDA container from runtime/:
  TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0 \
    python3 aot_compile_enc_first.py --artifacts ./artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from datetime import datetime, timezone
from typing import Any

import torch


def _force_noexecstack_on_link():
    import torch._inductor.cpp_builder as cb

    orig = cb.CppBuilder.get_command_line
    seen = {"shared_link": False, "flagged": False}

    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd:
            seen["shared_link"] = True
            if "-Wl,-z,noexecstack" not in cmd:
                cmd += " -Wl,-z,noexecstack"
            seen["flagged"] = True
        return cmd

    cb.CppBuilder.get_command_line = patched
    return seen


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_weight_entry(name: str) -> bool:
    return "/data/weights/" in name


def clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    out = zipfile.ZipInfo(info.filename)
    out.date_time = info.date_time if info.date_time[0] >= 1980 else (1980, 1, 1, 0, 0, 0)
    out.compress_type = info.compress_type
    out.comment = info.comment
    out.extra = info.extra
    out.internal_attr = info.internal_attr
    out.external_attr = info.external_attr
    out.create_system = info.create_system
    return out


def strip_package(src: str, dst: str) -> dict[str, Any]:
    tmp = dst + ".tmp"
    zeroed_entries = 0
    zeroed_uncompressed = 0
    zeroed_compressed = 0
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(tmp, "w", allowZip64=True) as zout:
        for info in zin.infolist():
            cloned = clone_zipinfo(info)
            if info.is_dir():
                zout.writestr(cloned, b"")
                continue
            if is_weight_entry(info.filename):
                zeroed_entries += 1
                zeroed_uncompressed += int(info.file_size)
                zeroed_compressed += int(info.compress_size)
                zout.writestr(cloned, b"")
                continue
            with zin.open(info, "r") as src_f, zout.open(cloned, "w", force_zip64=True) as dst_f:
                shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
    os.replace(tmp, dst)
    return {
        "zeroed_entries": zeroed_entries,
        "zeroed_uncompressed_bytes": zeroed_uncompressed,
        "zeroed_compressed_bytes": zeroed_compressed,
        "src_size_bytes": os.path.getsize(src),
        "dst_size_bytes": os.path.getsize(dst),
    }


def resolve_shared_weight(weights: dict[str, Any], fqn: str) -> tuple[Any | None, str | None, bool]:
    if fqn in weights:
        return weights[fqn], fqn, False
    if fqn.startswith("encoder."):
        alt = "e." + fqn[len("encoder.") :]
        if alt in weights:
            return weights[alt], alt, True
    if fqn.startswith("e."):
        alt = "encoder." + fqn[len("e.") :]
        if alt in weights:
            return weights[alt], alt, True
    return None, None, False


class SharedWeights:
    def __init__(self, path: str):
        self.path = path
        self.cpu: dict[str, Any] | None = None
        self.cuda: dict[str, Any] = {}

    def load_cpu(self) -> dict[str, Any]:
        if self.cpu is None:
            self.cpu = torch.load(self.path, map_location="cpu", weights_only=False)
            if not isinstance(self.cpu, dict):
                raise TypeError(f"{self.path} did not contain a dict")
        return self.cpu

    def constants_for_runner(self, runner: Any) -> tuple[dict[str, Any], int, int, int]:
        fqns = list(runner.loader.get_constant_fqns())
        weights = self.load_cpu()
        cmap: dict[str, Any] = {}
        missing: list[str] = []
        direct = 0
        alias = 0
        for fqn in fqns:
            tensor, source_key, used_alias = resolve_shared_weight(weights, fqn)
            if tensor is None or source_key is None:
                missing.append(fqn)
                continue
            if source_key not in self.cuda:
                self.cuda[source_key] = tensor.cuda()
            cmap[fqn] = self.cuda[source_key]
            if used_alias:
                alias += 1
            else:
                direct += 1
        if missing:
            raise RuntimeError(f"missing {len(missing)} shared constants; first={missing[:5]}")
        return cmap, len(fqns), direct, alias


def normalize_outputs(out: Any) -> tuple[Any, ...]:
    if isinstance(out, (tuple, list)):
        return tuple(out)
    return (out,)


def self_check(pkg_path: str, io_path: str, shared: SharedWeights, atol: float) -> dict[str, Any]:
    runner = torch._inductor.aoti_load_package(pkg_path)
    cmap, fqn_count, direct, alias = shared.constants_for_runner(runner)
    runner.loader.load_constants(cmap, False, False, True)
    io = torch.load(io_path, weights_only=False)
    inputs = [
        io["chunk"].cuda().contiguous(),
        io["L"].cuda().contiguous(),
        io["clc"].cuda().contiguous(),
        io["clt"].cuda().contiguous(),
        io["clcl"].cuda().contiguous(),
    ]
    ref = [t.cuda() for t in io["out"]]
    with torch.inference_mode():
        got = normalize_outputs(runner(*inputs))
    if len(got) < len(ref):
        raise RuntimeError(f"output count mismatch: got={len(got)} ref={len(ref)}")
    outputs: list[dict[str, Any]] = []
    ok = True
    max_abs_all = 0.0
    for index, (actual, expected) in enumerate(zip(got, ref)):
        if not torch.is_tensor(actual) or not torch.is_tensor(expected):
            same = actual == expected
            ok = ok and bool(same)
            outputs.append({"index": index, "tensor": False, "equal": bool(same)})
            continue
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(
                f"tensor metadata mismatch output {index}: "
                f"{tuple(actual.shape)}/{actual.dtype} vs {tuple(expected.shape)}/{expected.dtype}"
            )
        finite = bool(torch.isfinite(actual).all().item())
        byte_exact = bool(torch.equal(actual, expected))
        max_abs = 0.0
        if actual.numel():
            max_abs = float((actual.float() - expected.float()).abs().max().item())
        max_abs_all = max(max_abs_all, max_abs)
        within = finite and (byte_exact or max_abs <= atol)
        ok = ok and within
        outputs.append(
            {
                "index": index,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "finite": finite,
                "byte_exact": byte_exact,
                "max_abs": max_abs,
            }
        )
    if not ok:
        raise RuntimeError(f"self-check failed: max_abs={max_abs_all:.6g} atol={atol}")
    return {
        "ok": True,
        "constant_fqns": fqn_count,
        "matched_constants": len(cmap),
        "direct_matches": direct,
        "alias_fallbacks": alias,
        "outputs": outputs,
        "max_abs": max_abs_all,
        "atol": atol,
    }


def compile_package(ep_path: str, tmp_pkg: str, args: argparse.Namespace) -> dict[str, Any]:
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
    os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "0"
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = not bool(args.no_cudnn_tf32)
    configured: dict[str, Any] = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    try:
        import torch._inductor.config as config

        if hasattr(config, "max_autotune"):
            config.max_autotune = False
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = False
        if args.force_same_precision and hasattr(config, "force_same_precision"):
            config.force_same_precision = True
            configured["force_same_precision"] = True
        if args.emulate_precision_casts and hasattr(config, "emulate_precision_casts"):
            config.emulate_precision_casts = True
            configured["emulate_precision_casts"] = True
        if args.max_autotune_gemm_backends and hasattr(config, "max_autotune_gemm_backends"):
            config.max_autotune_gemm_backends = args.max_autotune_gemm_backends
            configured["max_autotune_gemm_backends"] = args.max_autotune_gemm_backends
        if args.max_autotune_conv_backends and hasattr(config, "max_autotune_conv_backends"):
            config.max_autotune_conv_backends = args.max_autotune_conv_backends
            configured["max_autotune_conv_backends"] = args.max_autotune_conv_backends
    except Exception:
        pass

    seen = _force_noexecstack_on_link()
    ep = torch.export.load(ep_path)
    out = torch._inductor.aoti_compile_and_package(
        ep,
        package_path=tmp_pkg,
        inductor_configs={
            "aot_inductor.package_constants_in_so": False,
            "aot_inductor.package_constants_on_disk": True,
        },
    )
    if not seen["flagged"]:
        raise RuntimeError("noexecstack shim never fired on a shared-lib link")
    return {"package_path": out, "noexecstack": seen, "configured": configured}


def atomic_write_json(path: str, payload: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="./artifacts")
    parser.add_argument("--ep", default=None)
    parser.add_argument("--io", default=None)
    parser.add_argument("--pkg", default=None)
    parser.add_argument("--shared-weights", default=None)
    parser.add_argument("--self-check-atol", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "medium"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--no-cudnn-tf32", action="store_true")
    parser.add_argument("--force-same-precision", action="store_true")
    parser.add_argument("--emulate-precision-casts", action="store_true")
    parser.add_argument("--max-autotune-gemm-backends", default=None)
    parser.add_argument("--max-autotune-conv-backends", default=None)
    args = parser.parse_args()

    art = os.path.abspath(args.artifacts)
    ep_path = os.path.abspath(args.ep or os.path.join(art, "enc_first_t2a.pt2"))
    io_path = os.path.abspath(args.io or os.path.join(art, "enc_first_t2a_io.pt"))
    pkg_path = os.path.abspath(args.pkg or os.path.join(art, "enc_first_aoti.pt2"))
    shared_path = os.path.abspath(args.shared_weights or os.path.join(art, "finalize_shared_weights.pt"))
    manifest_path = os.path.join(os.path.dirname(pkg_path), "compile_enc_first_manifest.json")

    if not os.path.exists(ep_path):
        raise FileNotFoundError(ep_path)
    if not os.path.exists(io_path):
        raise FileNotFoundError(io_path)
    if not os.path.exists(shared_path):
        raise FileNotFoundError(shared_path)
    if os.path.exists(pkg_path) and not args.force:
        shared = SharedWeights(shared_path)
        check = self_check(pkg_path, io_path, shared, args.self_check_atol)
        print(f"existing {pkg_path} self-check OK max_abs={check['max_abs']:.3e}")
        return 0

    tmp_pkg = pkg_path + f".unstripped.{os.getpid()}.pt2"
    try:
        os.unlink(tmp_pkg)
    except FileNotFoundError:
        pass

    start = time.time()
    compile_info = compile_package(ep_path, tmp_pkg, args)
    shared = SharedWeights(shared_path)
    unstripped_check = self_check(tmp_pkg, io_path, shared, args.self_check_atol)
    strip_stats = strip_package(tmp_pkg, pkg_path)
    stripped_check = self_check(pkg_path, io_path, shared, args.self_check_atol)
    try:
        os.unlink(tmp_pkg)
    except FileNotFoundError:
        pass

    manifest = {
        "artifact": os.path.basename(pkg_path),
        "compiled_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": time.time() - start,
        "ep_path": ep_path,
        "ep_sha256": sha256_file(ep_path),
        "io_path": io_path,
        "io_sha256": sha256_file(io_path),
        "shared_weights_path": shared_path,
        "shared_weights_sha256": sha256_file(shared_path),
        "package_path": pkg_path,
        "package_size_bytes": os.path.getsize(pkg_path),
        "package_sha256": sha256_file(pkg_path),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": None if torch.backends.cudnn.version() is None else int(torch.backends.cudnn.version()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
        "env": {
            "TORCHINDUCTOR_MAX_AUTOTUNE": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE", ""),
            "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": os.environ.get(
                "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", ""
            ),
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        },
        "compile": compile_info,
        "strip": strip_stats,
        "unstripped_self_check": unstripped_check,
        "stripped_self_check": stripped_check,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"accepted {pkg_path} ({os.path.getsize(pkg_path) / 1e6:.1f} MB, "
        f"sha={manifest['package_sha256'][:12]}..., max_abs={stripped_check['max_abs']:.3e})"
    )
    print(f"manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
