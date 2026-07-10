#!/usr/bin/env python3
"""Strip packaged AOTI finalize bucket weights and validate the stripped packages.

Canonical stripped representation: keep every ``data/weights/*`` zip entry but
write an empty payload.  The runtime supplies the real tensors through
``loader.load_constants(..., user_managed=True)`` from
``artifacts/finalize_shared_weights.pt``.

Validation is token-exact:
* when the unstripped package exists, stripped output must decode identically to it;
* when the unstripped package was deleted, stripped output must decode identically
  to eager ``finalize_ref`` tokens for a matching ``(drop,T)`` example.

Run all buckets from runtime/:
  python3 strip_bucket_weights.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


from model_profile import get_profile

_PROFILE = get_profile()
BLANK = _PROFILE.blank
MAX_SYMBOLS = _PROFILE.max_symbols
BUCKET_RE = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")
DEFAULT_BUCKET = f"artifacts/finalize_buckets/enc_finalize_d{_PROFILE.drop}_T{_PROFILE.drop2_T[1]}.pt2"


@dataclass
class StripStats:
    src: str
    dst: str
    src_size: int
    dst_size: int
    zeroed_entries: int
    zeroed_uncompressed: int
    zeroed_compressed: int


@dataclass
class ValidationExample:
    drop: int
    T: int
    chunk_mel: Any
    cache_last_channel: Any
    cache_last_time: Any
    cache_last_channel_len: Any
    pre_tokens: list[int]
    g: Any
    h: Any
    c: Any
    gold_tokens: list[int]
    basis: str
    source: dict[str, Any]


@dataclass
class PackageRun:
    tokens: list[int]
    enc_len: int
    constant_fqns: int
    matched_constants: int
    direct_matches: int
    alias_fallbacks: int


def parse_bucket_name(path: str) -> tuple[int, int]:
    match = BUCKET_RE.match(os.path.basename(path))
    if not match:
        raise ValueError(f"cannot parse drop/T from bucket filename: {path}")
    return int(match.group("drop")), int(match.group("T"))


def discover_bucket_files(path: str) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    if not os.path.isdir(path):
        return out
    for name in sorted(os.listdir(path)):
        if not BUCKET_RE.match(name):
            continue
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        key = parse_bucket_name(name)
        if key in out:
            raise RuntimeError(f"duplicate bucket for {key}: {out[key]} and {full}")
        out[key] = full
    return out


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_bytes(size: int) -> str:
    return f"{size / 1_000_000_000:.3f} GB ({size / 1024 / 1024:.1f} MiB)"


def is_weight_entry(name: str) -> bool:
    return "/data/weights/" in name


def clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    out = zipfile.ZipInfo(info.filename)
    year, month, day, hour, minute, second = info.date_time
    if year < 1980 or not (1 <= month <= 12) or not (1 <= day <= 31):
        out.date_time = (1980, 1, 1, 0, 0, 0)
    else:
        out.date_time = (year, month, day, hour, minute, second)
    out.compress_type = info.compress_type
    out.comment = info.comment
    out.extra = info.extra
    out.internal_attr = info.internal_attr
    out.external_attr = info.external_attr
    out.create_system = info.create_system
    return out


def strip_package(src: str, dst: str) -> StripStats:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    zeroed_entries = 0
    zeroed_uncompressed = 0
    zeroed_compressed = 0
    try:
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
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return StripStats(
        src=src,
        dst=dst,
        src_size=os.path.getsize(src),
        dst_size=os.path.getsize(dst),
        zeroed_entries=zeroed_entries,
        zeroed_uncompressed=zeroed_uncompressed,
        zeroed_compressed=zeroed_compressed,
    )


def default_stripped_path(bucket: str, out_dir: str) -> str:
    return os.path.join(out_dir, os.path.basename(bucket))


def attr_tensor(module: Any, name: str):
    return getattr(module, name)


def row_tensor(bundle: Any, row: int, name: str):
    return attr_tensor(bundle, f"row{row}_{name}")


def scalar_i64(tensor: Any) -> int:
    return int(tensor.detach().cpu().reshape(-1)[0].item())


def tensor_to_vec(tensor: Any) -> list[int]:
    import torch

    return [int(x) for x in tensor.detach().cpu().to(dtype=torch.long).reshape(-1).tolist()]


def cpu_tensor(tensor: Any):
    return tensor.detach().cpu().contiguous()


def state_hc(state: Any) -> tuple[Any, Any]:
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        raise TypeError(f"expected decoder state (h,c), got {type(state)}")
    return state[0], state[1]


def decode_range(torch: Any, joint: Any, predict: Any, enc_out: Any, enc_len: int, g: Any, h: Any, c: Any, hyp: list[int]) -> list[int]:
    if enc_len < 0 or enc_len > int(enc_out.shape[2]):
        raise RuntimeError(f"enc_len={enc_len} out of range for enc_out shape={tuple(enc_out.shape)}")
    f = enc_out.transpose(1, 2).contiguous()
    device = f.device
    for t in range(enc_len):
        f_t = f[:, t : t + 1, :]
        for _ in range(MAX_SYMBOLS):
            logits = joint(f_t, g)
            k = int(logits.reshape(-1).argmax().item())
            if k == BLANK:
                break
            hyp.append(k)
            y = torch.full((1, 1), k, dtype=torch.long, device=device)
            out = predict(y, h, c)
            g, h, c = out[0], out[1], out[2]
    return hyp


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


class PackageValidator:
    def __init__(self, args: argparse.Namespace):
        import faulthandler

        import torch

        faulthandler.enable()
        torch.set_grad_enabled(False)
        self.torch = torch
        self.device = torch.device("cuda")
        self.shared_weights_path = args.shared_weights
        self.bundle_path = args.bundle
        self.joint_path = args.joint
        self.predict_path = args.predict
        self.scan = int(args.scan)
        self.shared_cpu: dict[str, Any] | None = None
        self.shared_cuda: dict[str, Any] = {}
        self.examples: dict[tuple[int, int], ValidationExample] = {}
        self.joint = torch.jit.load(self.joint_path).to(self.device).eval()
        self.predict = torch.jit.load(self.predict_path).to(self.device).eval()
        # Prompted (multilingual) profile: gold tokens were produced from
        # prompt-conditioned encoder outputs; condition bucket outputs the same
        # way (default target language) before decoding.
        self.prompt_apply = None
        self.prompt_onehot = None
        if _PROFILE.prompted:
            prompt_path = os.path.join(os.path.dirname(self.joint_path), "prompt_apply.ts")
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"prompted profile requires {prompt_path}")
            self.prompt_apply = torch.jit.load(prompt_path).to(self.device).eval()
            onehot = torch.zeros(1, _PROFILE.num_prompts, device=self.device)
            onehot[0, _PROFILE.default_prompt_index] = 1.0
            self.prompt_onehot = onehot
        self._load_fixture_examples()

    def _condition(self, enc_out: Any) -> Any:
        if self.prompt_apply is None:
            return enc_out
        return self.prompt_apply(enc_out, self.prompt_onehot)

    def _load_fixture_examples(self) -> None:
        torch = self.torch
        bundle = torch.jit.load(self.bundle_path)
        rows = scalar_i64(attr_tensor(bundle, "num_rows"))
        for row in range(rows):
            drop = scalar_i64(row_tensor(bundle, row, "drop_extra"))
            T = int(row_tensor(bundle, row, "chunk_mel").shape[-1])
            key = (drop, T)
            self.examples[key] = ValidationExample(
                drop=drop,
                T=T,
                chunk_mel=cpu_tensor(row_tensor(bundle, row, "chunk_mel")),
                cache_last_channel=cpu_tensor(row_tensor(bundle, row, "cache_last_channel")),
                cache_last_time=cpu_tensor(row_tensor(bundle, row, "cache_last_time")),
                cache_last_channel_len=cpu_tensor(row_tensor(bundle, row, "cache_last_channel_len")),
                pre_tokens=tensor_to_vec(row_tensor(bundle, row, "pre_final_tokens")),
                g=cpu_tensor(row_tensor(bundle, row, "pre_final_pred_out")),
                h=cpu_tensor(row_tensor(bundle, row, "pre_final_h")),
                c=cpu_tensor(row_tensor(bundle, row, "pre_final_c")),
                gold_tokens=tensor_to_vec(row_tensor(bundle, row, "finalize_ref_final_tokens")),
                basis="finalize_bundle",
                source={"kind": "finalize_bundle", "row": row},
            )

    def _load_shared_cpu(self) -> dict[str, Any]:
        if self.shared_cpu is None:
            self.shared_cpu = self.torch.load(self.shared_weights_path, map_location="cpu", weights_only=False)
            if not isinstance(self.shared_cpu, dict):
                raise TypeError(f"{self.shared_weights_path} did not contain a dict")
        return self.shared_cpu

    def constants_for_runner(self, runner: Any) -> tuple[dict[str, Any], int, int, int]:
        fqns = list(runner.loader.get_constant_fqns())
        weights = self._load_shared_cpu()
        cmap: dict[str, Any] = {}
        missing: list[str] = []
        direct = 0
        alias = 0
        for fqn in fqns:
            tensor, source_key, used_alias = resolve_shared_weight(weights, fqn)
            if tensor is None or source_key is None:
                missing.append(fqn)
                continue
            if source_key not in self.shared_cuda:
                self.shared_cuda[source_key] = tensor.to(self.device)
            cmap[fqn] = self.shared_cuda[source_key]
            if used_alias:
                alias += 1
            else:
                direct += 1
        if missing:
            raise RuntimeError(f"missing {len(missing)} shared constants; first={missing[:5]}")
        return cmap, len(fqns), direct, alias

    def run_package(self, pkg: str, example: ValidationExample) -> PackageRun:
        torch = self.torch
        print(f"    aoti_load_package {pkg}", flush=True)
        runner = torch._inductor.aoti_load_package(pkg)
        cmap, fqn_count, direct, alias = self.constants_for_runner(runner)
        runner.loader.load_constants(cmap, False, False, True)
        inputs = (
            example.chunk_mel.to(self.device).contiguous(),
            example.cache_last_channel.to(self.device).contiguous(),
            example.cache_last_time.to(self.device).contiguous(),
            example.cache_last_channel_len.to(self.device).contiguous(),
        )
        out = runner(*inputs)
        if not isinstance(out, (tuple, list)):
            out = (out,)
        if len(out) < 2:
            raise RuntimeError(f"bucket returned {len(out)} outputs, expected at least 2")
        enc_len = scalar_i64(out[1])
        tokens = decode_range(
            torch,
            self.joint,
            self.predict,
            self._condition(out[0]),
            enc_len,
            example.g.to(self.device).contiguous(),
            example.h.to(self.device).contiguous(),
            example.c.to(self.device).contiguous(),
            list(example.pre_tokens),
        )
        return PackageRun(
            tokens=tokens,
            enc_len=enc_len,
            constant_fqns=fqn_count,
            matched_constants=len(cmap),
            direct_matches=direct,
            alias_fallbacks=alias,
        )

    def ensure_examples(self, keys: set[tuple[int, int]]) -> None:
        missing = {key for key in keys if key not in self.examples}
        if not missing:
            return

        from finalize_ref import ContinuousFinalizeRef, clone_tree, load_benchmark_dataset, load_model, load_wav, tensor_clone
        from ref_decode import ref_greedy_range

        torch = self.torch
        print(f"  generating eager finalize_ref examples for {sorted(missing)}", flush=True)
        model = load_model()
        rt = ContinuousFinalizeRef(model)

        def capture(wav: Any, source: dict[str, Any]) -> None:
            nonlocal missing
            if not missing:
                return
            session = rt.new_session(str(source))
            rt.append_audio(session, wav)
            rt.vad_stop(session)
            fork = rt.build_continuous_finalize_fork(session)
            fi = rt.prepare_finalize_inputs(fork)
            if fi is None:
                return
            key = (int(fi.drop_extra), int(fi.chunk_mel.shape[-1]))
            if key not in missing:
                return

            pre_tokens = list(fork.hyp_tokens)
            pre_state = clone_tree(fork.decoder_state)
            pre_pred_out = tensor_clone(fork.pred_out_stream)
            h, c = state_hc(pre_state)
            with torch.inference_mode():
                enc_out, enc_len, _clc, _clt, _clcl = rt.encoder.cache_aware_stream_step(
                    processed_signal=fi.chunk_mel,
                    processed_signal_length=fi.chunk_len,
                    cache_last_channel=fi.cache_last_channel,
                    cache_last_time=fi.cache_last_time,
                    cache_last_channel_len=fi.cache_last_channel_len,
                    keep_all_outputs=True,
                    drop_extra_pre_encoded=fi.drop_extra,
                )
                new_tokens, _state, _pred_out = ref_greedy_range(
                    rt.decoder,
                    rt.joint,
                    enc_out.transpose(1, 2).contiguous(),
                    0,
                    int(enc_len.detach().reshape(-1)[0].item()),
                    clone_tree(pre_state),
                    tensor_clone(pre_pred_out),
                )

            self.examples[key] = ValidationExample(
                drop=key[0],
                T=key[1],
                chunk_mel=cpu_tensor(fi.chunk_mel),
                cache_last_channel=cpu_tensor(fi.cache_last_channel),
                cache_last_time=cpu_tensor(fi.cache_last_time),
                cache_last_channel_len=cpu_tensor(fi.cache_last_channel_len),
                pre_tokens=pre_tokens,
                g=cpu_tensor(pre_pred_out),
                h=cpu_tensor(h),
                c=cpu_tensor(c),
                gold_tokens=list(pre_tokens) + list(new_tokens),
                basis="eager_finalize_ref",
                source=source,
            )
            missing.remove(key)
            print(f"    captured eager example drop={key[0]} T={key[1]} source={source}", flush=True)

        drop0_keys = {key for key in missing if key[0] == 0}
        for _drop, T in sorted(drop0_keys):
            # First-finalize synthetic silence: emitted_frames stays 0 for lengths below one steady shift.
            samples = (T - rt.geometry.final_padding_frames - 1) * rt.geometry.hop_samples + 1
            if samples <= 0:
                continue
            wav = torch.zeros(samples, dtype=torch.float32).cpu().numpy()
            capture(wav, {"kind": "synthetic_drop0_silence", "samples": int(samples)})

        if missing:
            ds = load_benchmark_dataset()
            for index in range(min(self.scan, len(ds))):
                if not missing:
                    break
                capture(load_wav(ds[index]), {"kind": "stt_benchmark", "index": index})

        if missing:
            raise RuntimeError(f"could not find eager finalize_ref examples for {sorted(missing)} in scan={self.scan}")


def print_strip_stats(stats: StripStats) -> None:
    ratio = stats.dst_size / stats.src_size if stats.src_size else 0.0
    print(
        f"stripped {stats.zeroed_entries} weight entries from {stats.src}\n"
        f"  original: {fmt_bytes(stats.src_size)}\n"
        f"  stripped: {fmt_bytes(stats.dst_size)} ({ratio:.4%} of original)\n"
        f"  zeroed uncompressed: {fmt_bytes(stats.zeroed_uncompressed)}\n"
        f"  zeroed compressed:   {fmt_bytes(stats.zeroed_compressed)}\n"
        f"  output: {stats.dst}"
    )


def validate_bucket(
    validator: PackageValidator,
    stripped_pkg: str,
    original_pkg: str | None,
    key: tuple[int, int],
) -> dict[str, Any]:
    example = validator.examples[key]
    print(f"  validating drop={key[0]} T={key[1]} basis={example.basis}", flush=True)
    stripped = validator.run_package(stripped_pkg, example)
    basis = "unstripped_package" if original_pkg else example.basis
    original: PackageRun | None = None
    if original_pkg:
        original = validator.run_package(original_pkg, example)
        ok = stripped.tokens == original.tokens
        gold_ok = stripped.tokens == example.gold_tokens
        compare_tokens = original.tokens
    else:
        ok = stripped.tokens == example.gold_tokens
        gold_ok = ok
        compare_tokens = example.gold_tokens

    result: dict[str, Any] = {
        "ok": bool(ok),
        "basis": basis,
        "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "example": example.source,
        "tokens": len(stripped.tokens),
        "stripped_tokens_sha256": sha256_json(stripped.tokens),
        "compare_tokens_sha256": sha256_json(compare_tokens),
        "token_exact": bool(ok),
        "token_exact_vs_eager_finalize_ref": bool(gold_ok),
        "stripped": {
            "pkg": os.path.basename(stripped_pkg),
            "pkg_sha256": sha256_file(stripped_pkg),
            "enc_len": stripped.enc_len,
            "constant_fqns": stripped.constant_fqns,
            "matched_constants": stripped.matched_constants,
            "direct_matches": stripped.direct_matches,
            "alias_fallbacks": stripped.alias_fallbacks,
        },
    }
    if original_pkg and original is not None:
        result["unstripped"] = {
            "pkg": os.path.basename(original_pkg),
            "pkg_sha256": sha256_file(original_pkg),
            "enc_len": original.enc_len,
            "tokens_sha256": sha256_json(original.tokens),
            "direct_matches": original.direct_matches,
            "alias_fallbacks": original.alias_fallbacks,
        }
    print(
        f"    result token_exact={'PASS' if ok else 'FAIL'} "
        f"tokens={len(stripped.tokens)} enc_len={stripped.enc_len} "
        f"direct={stripped.direct_matches} alias={stripped.alias_fallbacks}",
        flush=True,
    )
    return result


def write_manifest_with_validations(out_dir: str, weights_path: str, results: dict[str, dict[str, Any]]) -> None:
    from build_bucket_manifest import atomic_write_json, build_manifest

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = build_manifest(out_dir, weights_path, manifest_path)
    for bucket in manifest["buckets"]:
        validation = results.get(bucket["pkg"])
        if validation is not None:
            bucket["strip_validation"] = validation
    atomic_write_json(manifest_path, manifest)
    print(f"updated {manifest_path} with {len(results)} strip validation records")


def validate_only_main(args: argparse.Namespace) -> int:
    try:
        key = parse_bucket_name(args.bucket)
        validator = PackageValidator(args)
        validator.ensure_examples({key})
        result = validate_bucket(validator, args.bucket, None, key)
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["ok"] else 1
    except BaseException as exc:
        payload = {
            "ok": False,
            "package": args.bucket,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print("VALIDATION_ERROR", type(exc).__name__, str(exc), flush=True)
        traceback.print_exc()
        print("RESULT_JSON " + json.dumps(payload, sort_keys=True), flush=True)
        return 1


def run_all(args: argparse.Namespace) -> int:
    source_buckets = discover_bucket_files(args.source_dir)
    stripped_buckets = discover_bucket_files(args.out_dir)
    originals: dict[tuple[int, int], str] = {}
    stats: list[StripStats] = []

    for key, src in sorted(source_buckets.items()):
        dst = default_stripped_path(src, args.out_dir)
        originals[key] = src
        if os.path.abspath(src) == os.path.abspath(dst):
            stripped_buckets[key] = dst
            continue
        if args.force or not os.path.exists(dst):
            stat = strip_package(src, dst)
            print_strip_stats(stat)
            stats.append(stat)
        stripped_buckets[key] = dst

    if not stripped_buckets:
        raise RuntimeError(f"no finalize buckets found in {args.source_dir} or {args.out_dir}")

    keys = set(stripped_buckets)
    validator = PackageValidator(args)
    validator.ensure_examples(keys)

    validation_results: dict[str, dict[str, Any]] = {}
    ok_all = True
    for key in sorted(keys):
        stripped = stripped_buckets[key]
        original = originals.get(key)
        try:
            result = validate_bucket(validator, stripped, original, key)
        except BaseException as exc:
            traceback.print_exc()
            result = {
                "ok": False,
                "basis": "error",
                "validated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        validation_results[os.path.basename(stripped)] = result
        ok_all = ok_all and bool(result.get("ok"))

    # Drop cached CUDA tensors before build_manifest reloads the model to derive geometry.
    del validator
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    write_manifest_with_validations(args.out_dir, args.shared_weights, validation_results)

    print("\n=== strip --all summary ===")
    print(f"source buckets stripped this run: {len(stats)}")
    for key in sorted(keys):
        result = validation_results[os.path.basename(stripped_buckets[key])]
        print(
            f"  drop={key[0]} T={key[1]}: {'PASS' if result.get('ok') else 'FAIL'} "
            f"basis={result.get('basis')}"
        )
    return 0 if ok_all else 1


def run_single(args: argparse.Namespace) -> int:
    stripped = args.out or default_stripped_path(args.bucket, args.out_dir)
    stats = strip_package(args.bucket, stripped)
    print_strip_stats(stats)

    if args.strip_only:
        return 0

    key = parse_bucket_name(stripped)
    validator = PackageValidator(args)
    validator.ensure_examples({key})
    result = validate_bucket(validator, stripped, args.bucket, key)
    write_manifest_with_validations(args.out_dir, args.shared_weights, {os.path.basename(stripped): result})
    return 0 if result["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--source-dir", default="artifacts/finalize_buckets")
    parser.add_argument("--out-dir", default="artifacts/stripped_finalize_buckets")
    parser.add_argument("--out", default=None)
    parser.add_argument("--shared-weights", default="artifacts/finalize_shared_weights.pt")
    parser.add_argument("--bundle", default="artifacts/finalize_bundle.ts")
    parser.add_argument("--joint", default="artifacts/joint_step.ts")
    parser.add_argument("--predict", default="artifacts/predict_step.ts")
    parser.add_argument("--scan", type=int, default=1200)
    parser.add_argument("--all", action="store_true", help="strip every source bucket and validate every stripped bucket")
    parser.add_argument("--force", action="store_true", help="re-strip outputs even when they already exist")
    parser.add_argument("--strip-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.validate_only:
        return validate_only_main(args)
    if args.all:
        if args.strip_only:
            raise SystemExit("--all always validates; omit --strip-only")
        return run_all(args)
    return run_single(args)


if __name__ == "__main__":
    raise SystemExit(main())
