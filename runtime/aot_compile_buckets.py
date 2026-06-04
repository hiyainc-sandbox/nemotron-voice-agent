#!/usr/bin/env python3
"""AOTI-compile finalize bucket ExportedPrograms with fail-closed self-checks.

Each bucket is compiled to a temporary package path in a child process, retried
on transient failures, loaded with ``aoti_load_package``, wired with
``load_constants(..., user_managed=True)``, and run on the ExportedProgram's
captured example inputs before the package is accepted.

Run in-container from runtime/:
  python3 aot_compile_buckets.py --dir ./artifacts/finalize_buckets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import torch


EP_RE = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)_ep\.pt2$")


def _force_noexecstack_on_link():
    import torch._inductor.cpp_builder as cb

    orig = cb.CppBuilder.get_command_line
    fired = {"ok": False}

    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd and "-Wl,-z,noexecstack" not in cmd:
            cmd += " -Wl,-z,noexecstack"
            fired["ok"] = True
        return cmd

    cb.CppBuilder.get_command_line = patched
    return fired


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, payload: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def parse_ep_name(name: str) -> tuple[int, int] | None:
    match = EP_RE.match(name)
    if not match:
        return None
    return int(match.group("drop")), int(match.group("T"))


def load_bucket_entries(bucket_dir: str) -> list[dict[str, Any]]:
    entries_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    manifest_path = os.path.join(bucket_dir, "buckets_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise TypeError(f"{manifest_path} is not a list")
        for item in raw:
            if not isinstance(item, dict) or "drop" not in item or "T" not in item or "ep" not in item:
                continue
            entries_by_key[(int(item["drop"]), int(item["T"]))] = dict(item)

    for name in sorted(os.listdir(bucket_dir)):
        key = parse_ep_name(name)
        if key is None:
            continue
        entries_by_key.setdefault(key, {"drop": key[0], "T": key[1], "ep": name})

    out = [entries_by_key[key] for key in sorted(entries_by_key)]
    if not out:
        raise RuntimeError(f"no exported finalize bucket programs found in {bucket_dir}")
    return out


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
                raise TypeError(f"{self.path} did not contain a Dict[str, Tensor]")
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
    if not isinstance(out, (tuple, list)):
        return (out,)
    return tuple(out)


def tensor_max_abs(lhs: Any, rhs: Any) -> float:
    if lhs.numel() == 0 and rhs.numel() == 0:
        return 0.0
    return float((lhs.float() - rhs.float()).abs().max().item())


def self_check_package(ep_path: str, pkg_path: str, shared: SharedWeights, atol: float) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    ep = torch.export.load(ep_path)
    args, kwargs = ep.example_inputs
    args = tuple(x.cuda().contiguous() if torch.is_tensor(x) else x for x in args)
    kwargs = {k: (v.cuda().contiguous() if torch.is_tensor(v) else v) for k, v in dict(kwargs).items()}

    with torch.inference_mode():
        expected = normalize_outputs(ep.module()(*args, **kwargs))
        runner = torch._inductor.aoti_load_package(pkg_path)
        cmap, fqn_count, direct, alias = shared.constants_for_runner(runner)
        runner.loader.load_constants(cmap, False, False, True)
        got = normalize_outputs(runner(*args))

    if len(got) != len(expected):
        raise RuntimeError(f"self-check output count mismatch: got={len(got)} expected={len(expected)}")

    outputs: list[dict[str, Any]] = []
    ok = True
    max_abs_all = 0.0
    for index, (actual, exp) in enumerate(zip(got, expected)):
        if not torch.is_tensor(actual) or not torch.is_tensor(exp):
            same = actual == exp
            outputs.append({"index": index, "tensor": False, "equal": bool(same)})
            ok = ok and bool(same)
            continue
        if actual.shape != exp.shape or actual.dtype != exp.dtype:
            raise RuntimeError(
                f"self-check tensor metadata mismatch output {index}: "
                f"{tuple(actual.shape)}/{actual.dtype} vs {tuple(exp.shape)}/{exp.dtype}"
            )
        finite = bool(torch.isfinite(actual).all().item())
        byte_exact = bool(torch.equal(actual, exp))
        max_abs = tensor_max_abs(actual, exp)
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


def compile_one_child(ep_path: str, pkg_path: str) -> int:
    fired = _force_noexecstack_on_link()
    ep = torch.export.load(ep_path)
    torch._inductor.aoti_compile_and_package(
        ep,
        package_path=pkg_path,
        inductor_configs={
            "aot_inductor.package_constants_in_so": False,
            "aot_inductor.package_constants_on_disk": True,
        },
    )
    if not fired["ok"]:
        raise RuntimeError("noexecstack shim never fired on a shared-lib link")
    print("COMPILE_ONE_JSON " + json.dumps({"ok": True, "noexecstack": True}), flush=True)
    return 0


def run_compile_child(ep_path: str, tmp_pkg: str, *, matmul_precision: str, allow_tf32: bool) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--compile-one",
        "--ep",
        ep_path,
        "--pkg",
        tmp_pkg,
        "--matmul-precision",
        matmul_precision,
    ]
    if allow_tf32:
        cmd.append("--allow-tf32")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    return proc.returncode == 0, proc.stdout


def remove_if_exists(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def compile_with_retries(
    ep_path: str,
    pkg_path: str,
    shared: SharedWeights,
    *,
    retries: int,
    atol: float,
    matmul_precision: str,
    allow_tf32: bool,
) -> tuple[dict[str, Any], int]:
    attempts = retries + 1
    last_error = ""
    for attempt in range(1, attempts + 1):
        tmp_pkg = f"{pkg_path}.tmp.{os.getpid()}.{attempt}.pt2"
        remove_if_exists(tmp_pkg)
        print(f"  compile attempt {attempt}/{attempts}: {os.path.basename(ep_path)} -> {os.path.basename(pkg_path)}")
        ok, output = run_compile_child(
            ep_path,
            tmp_pkg,
            matmul_precision=matmul_precision,
            allow_tf32=allow_tf32,
        )
        if not ok:
            last_error = output.strip().splitlines()[-1] if output.strip() else "compile child failed"
            remove_if_exists(tmp_pkg)
            print(f"    compile failed; deleting temp and retrying: {last_error}")
            continue
        try:
            check = self_check_package(ep_path, tmp_pkg, shared, atol)
        except BaseException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            remove_if_exists(tmp_pkg)
            print(f"    self-check failed; deleting temp and retrying: {last_error}")
            continue
        os.replace(tmp_pkg, pkg_path)
        return check, attempt
    raise RuntimeError(f"compile failed after {attempts} attempts: {last_error}")


def torch_version_record() -> dict[str, Any]:
    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": None if torch.backends.cudnn.version() is None else int(torch.backends.cudnn.version()),
    }


def configure_matmul_precision(matmul_precision: str, allow_tf32: bool) -> dict[str, Any]:
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    return {
        "torch.get_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch.backends.cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch.backends.cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./artifacts/finalize_buckets")
    parser.add_argument("--shared-weights", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    # AOTI is NOT byte-exact vs eager (F' finding: Triton fp reassociation ~1e-2; observed finalize enc-out drift ~1.1e-2).
    # The compile self-check is an encoder-level SANITY gate (loads+runs+not-garbage): accept the known fp drift but reject
    # gross corruption (a broken bucket diverges by orders of magnitude, e.g. the padding-bleed ~26). TOKEN-exactness is
    # validated downstream (strip --all token-validation + the corpus/drop0 shadow), not here.
    parser.add_argument("--self-check-atol", type=float, default=0.1)
    parser.add_argument("--compile-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ep", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pkg", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "medium"])
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args()

    precision = configure_matmul_precision(args.matmul_precision, args.allow_tf32)
    print("matmul precision config:", json.dumps(precision, sort_keys=True), flush=True)

    if args.compile_one:
        if not args.ep or not args.pkg:
            raise SystemExit("--compile-one requires --ep and --pkg")
        try:
            return compile_one_child(args.ep, args.pkg)
        except BaseException as exc:
            traceback.print_exc()
            print("COMPILE_ONE_JSON " + json.dumps({"ok": False, "error": str(exc)}), flush=True)
            return 1

    bucket_dir = args.dir
    shared_weights = args.shared_weights
    if shared_weights is None:
        shared_weights = os.path.join(os.path.dirname(os.path.abspath(bucket_dir)), "finalize_shared_weights.pt")
    if not os.path.exists(shared_weights):
        raise FileNotFoundError(f"shared weights not found: {shared_weights}")

    entries = load_bucket_entries(bucket_dir)
    shared = SharedWeights(shared_weights)
    versions = torch_version_record()
    out_manifest: list[dict[str, Any]] = []

    for entry in entries:
        drop = int(entry["drop"])
        T = int(entry["T"])
        ep_name = str(entry["ep"])
        ep_path = os.path.join(bucket_dir, ep_name)
        if not os.path.exists(ep_path):
            raise FileNotFoundError(f"missing ExportedProgram for drop={drop} T={T}: {ep_path}")
        pkg_name = f"enc_finalize_d{drop}_T{T}.pt2"
        pkg_path = os.path.join(bucket_dir, pkg_name)

        compiled = False
        attempts = 0
        if os.path.exists(pkg_path) and not args.force:
            print(f"  existing package found; self-checking {pkg_name}")
            try:
                check = self_check_package(ep_path, pkg_path, shared, args.self_check_atol)
            except BaseException as exc:
                print(f"    existing package self-check failed, recompiling: {type(exc).__name__}: {exc}")
                check, attempts = compile_with_retries(
                    ep_path,
                    pkg_path,
                    shared,
                    retries=args.retries,
                    atol=args.self_check_atol,
                    matmul_precision=args.matmul_precision,
                    allow_tf32=args.allow_tf32,
                )
                compiled = True
            else:
                attempts = 0
        else:
            check, attempts = compile_with_retries(
                ep_path,
                pkg_path,
                shared,
                retries=args.retries,
                atol=args.self_check_atol,
                matmul_precision=args.matmul_precision,
                allow_tf32=args.allow_tf32,
            )
            compiled = True

        pkg_size = os.path.getsize(pkg_path)
        b2 = dict(entry)
        b2.update(
            {
                "pkg": pkg_name,
                "pkg_mb": round(pkg_size / 1e6, 1),
                "pkg_sha256": sha256_file(pkg_path),
                "compiled": compiled,
                "compile_attempts": attempts,
                "self_check": check,
                "torch_versions": versions,
                "precision": precision,
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
        out_manifest.append(b2)
        print(
            f"  accepted drop={drop} T={T} -> {pkg_name} "
            f"({pkg_size / 1e6:.1f} MB, sha={b2['pkg_sha256'][:12]}..., "
            f"self_check max_abs={check['max_abs']:.3e})"
        )

    atomic_write_json(os.path.join(bucket_dir, "buckets_manifest.json"), out_manifest)
    print(f"=== accepted {len(out_manifest)} constants-on-disk finalize buckets ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
