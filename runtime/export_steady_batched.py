#!/usr/bin/env python3
"""Export and AOTI-compile fixed-B steady encoder packages for STEADY-BATCH-0.

The steady encoder input geometry is fixed at B x 128 x (PRE + SHIFT), with
PRE=9 and SHIFT=16.  Each exported program has a static batch size B and packs
B independent streaming caches:

  cache_last_channel:     24 x B x 70 x 1024
  cache_last_time:        24 x B x 1024 x 8
  cache_last_channel_len: B

Run from runtime/ on a NeMo-capable host for export+compile:
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_steady_batched.py --out ./artifacts

On an L40S torch-only sm_89 box, copy the enc_steady_t2a_b{1,2,4}.pt2 files
first, then compile only:
  TORCH_CUDA_ARCH_LIST=8.9 python export_steady_batched.py --out ./artifacts_sm89 --compile-only
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import torch


MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
SHIFT = 16
PRE = 9
DROP = 2
MELS = 128
EP_NAME = "enc_steady_t2a_b{b}.pt2"
PKG_NAME = "enc_steady_aoti_b{b}.pt2"
NAMES = ["enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_batches(text: str) -> list[int]:
    batches = [int(x) for x in text.split(",") if x.strip()]
    if not batches:
        raise ValueError("--batches cannot be empty")
    for b in batches:
        if b <= 0:
            raise ValueError(f"batch sizes must be positive, got {b}")
    return batches


def remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def force_noexecstack_on_link() -> dict[str, bool]:
    import torch._inductor.cpp_builder as cb

    orig = cb.CppBuilder.get_command_line
    seen = {"flagged": False}

    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd:
            if "-Wl,-z,noexecstack" not in cmd:
                cmd += " -Wl,-z,noexecstack"
            seen["flagged"] = True
            print("[noexecstack] injected into shared-lib link:", cmd[-160:], flush=True)
        return cmd

    cb.CppBuilder.get_command_line = patched
    return seen


def set_inductor_autotune_off() -> dict[str, object]:
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM", "0")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE", "0")
    os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "0")

    import torch._inductor.config as cfg

    requested = {
        "max_autotune": False,
        "max_autotune_gemm": False,
        "max_autotune_pointwise": False,
        "coordinate_descent_tuning": False,
    }
    applied: dict[str, object] = {}
    for key, value in requested.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            applied[key] = value
    return applied


def configure_matmul_precision(matmul_precision: str, allow_tf32: bool) -> dict[str, object]:
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    return {
        "torch.get_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch.backends.cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "torch.backends.cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def compile_configs() -> dict[str, object]:
    return {
        "aot_inductor.package": True,
        "aot_inductor.package_constants_in_so": False,
        "aot_inductor.package_constants_on_disk": True,
        "max_autotune": False,
        "max_autotune_gemm": False,
        "max_autotune_pointwise": False,
        "coordinate_descent_tuning": False,
    }


def emit_manifest(
    out_dir: Path,
    batches: list[int],
    shared_weights: Path,
    production_b1: Path,
    precision: dict[str, object],
) -> None:
    if not shared_weights.exists():
        raise FileNotFoundError(f"missing shared weights for manifest: {shared_weights}")
    shared_sha = sha256_file(shared_weights)
    buckets = []
    for batch in batches:
        pkg = out_dir / PKG_NAME.format(b=batch)
        ep = out_dir / EP_NAME.format(b=batch)
        if not pkg.exists():
            raise FileNotFoundError(f"missing AOTI package for manifest B={batch}: {pkg}")
        if not ep.exists() and batch == 1 and (out_dir / "enc_steady_t2a.pt2").exists():
            ep = out_dir / "enc_steady_t2a.pt2"
        bucket = {
            "B": batch,
            "package": pkg.name,
            "package_sha256": sha256_file(pkg),
            "ep": ep.name if ep.exists() else "",
            "ep_sha256": sha256_file(ep) if ep.exists() else "",
            "shared_weight": str(shared_weights),
            "shared_weight_sha256": shared_sha,
            "torch_version": torch.__version__,
            "cuda_arch": ".".join(map(str, torch.cuda.get_device_capability()))
            if torch.cuda.is_available()
            else "NA",
            "precision": precision,
            "inductor_configs": compile_configs(),
            "byte_sizes": {
                "package": pkg.stat().st_size,
                "ep": ep.stat().st_size if ep.exists() else 0,
                "shared_weight": shared_weights.stat().st_size,
            },
        }
        buckets.append(bucket)

    prod_sha = sha256_file(production_b1) if production_b1.exists() else ""
    new_b1 = out_dir / PKG_NAME.format(b=1)
    manifest = {
        "CONTRACT": {
            "schema": 1,
            "model_id": MODEL_ID,
            "shift": SHIFT,
            "pre_encode_cache": PRE,
            "drop_extra": DROP,
            "bucket_set": batches,
        },
        "buckets": buckets,
        "a1_b1_parity": {
            "production_package": str(production_b1),
            "production_package_sha256": prod_sha,
            "new_b1_package": str(new_b1),
            "new_b1_package_sha256": sha256_file(new_b1) if new_b1.exists() else "",
            "package_sha256_equal": bool(prod_sha and new_b1.exists() and prod_sha == sha256_file(new_b1)),
            "tensor_parity": "not_run_by_exporter",
        },
    }
    tmp = out_dir / "MANIFEST.json.tmp"
    final = out_dir / "MANIFEST.json"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(final)
    print(f"wrote steady batch manifest: {final}", flush=True)


def make_mel(batch: int, frames: int, device: torch.device, *, offset: float) -> torch.Tensor:
    base = torch.linspace(-0.75 + offset, 0.75 + offset, MELS * frames, device=device, dtype=torch.float32)
    base = base.reshape(1, MELS, frames)
    rows = [base + float(i) * 0.01 for i in range(batch)]
    return torch.cat(rows, dim=0).contiguous()


class SteadyStep(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, drop_extra: int):
        super().__init__()
        self.encoder = encoder
        self.drop_extra = int(drop_extra)

    def forward(
        self,
        chunk: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ):
        return self.encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )


def load_encoder():
    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        raise RuntimeError("NeMo is required unless --compile-only is used") from exc

    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID, map_location="cpu").cuda().eval()
    try:
        model.preprocessor.featurizer.dither = 0.0
    except Exception:
        pass
    model.encoder.set_default_att_context_size([70, 1])
    return model.encoder


def build_example(encoder, batch: int):
    device = next(encoder.parameters()).device
    cache = encoder.get_initial_cache_state(batch_size=batch)
    c0 = make_mel(batch, SHIFT, device, offset=0.0)
    l0 = torch.full((batch,), SHIFT, device=device, dtype=torch.long)
    with torch.inference_mode():
        first = encoder.cache_aware_stream_step(
            processed_signal=c0,
            processed_signal_length=l0,
            cache_last_channel=cache[0].clone(),
            cache_last_time=cache[1].clone(),
            cache_last_channel_len=cache[2].clone(),
            keep_all_outputs=False,
            drop_extra_pre_encoded=0,
        )

    c1 = torch.cat((c0[:, :, -PRE:], make_mel(batch, SHIFT, device, offset=0.25)), dim=-1).contiguous()
    l1 = torch.full((batch,), c1.shape[-1], device=device, dtype=torch.long)
    return (
        c1,
        l1,
        first[2].clone().contiguous(),
        first[3].clone().contiguous(),
        first[4].clone().contiguous(),
    )


def compare_outputs(tag: str, got: Iterable[torch.Tensor], ref: Iterable[torch.Tensor], atol: float, rtol: float) -> bool:
    ok = True
    max_all = 0.0
    for name, a, b in zip(NAMES, got, ref):
        shape_ok = tuple(a.shape) == tuple(b.shape)
        if not shape_ok:
            print(f"  {tag}:{name}: shape mismatch got={tuple(a.shape)} ref={tuple(b.shape)}", flush=True)
            ok = False
            continue
        if a.is_floating_point():
            diff = (a.float() - b.float()).abs()
            max_abs = diff.max().item() if diff.numel() else 0.0
            close = torch.allclose(a, b, atol=atol, rtol=rtol)
            max_all = max(max_all, max_abs)
            print(f"  {tag}:{name}: close={close} max_abs={max_abs:.3e}", flush=True)
            ok = ok and bool(close)
        else:
            equal = torch.equal(a, b)
            print(f"  {tag}:{name}: equal={equal}", flush=True)
            ok = ok and bool(equal)
    print(f"[{tag}] close={ok} max_abs={max_all:.3e} atol={atol:.1e} rtol={rtol:.1e}", flush=True)
    return ok


def export_batch(encoder, batch: int, out_dir: Path, atol: float, rtol: float) -> tuple[Path, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    step = SteadyStep(encoder, DROP).cuda().eval()
    example = build_example(encoder, batch)
    with torch.inference_mode():
        eager = tuple(step(*example))
        ep = torch.export.export(step, example)
        exported = tuple(ep.module()(*example))
    if not compare_outputs(f"export-b{batch}-vs-eager", exported, eager, atol, rtol):
        raise RuntimeError(f"torch.export output is not close to eager for B={batch}")
    ep_path = out_dir / EP_NAME.format(b=batch)
    remove_if_exists(ep_path)
    torch.export.save(ep, ep_path)
    print(f"saved ExportedProgram B={batch}: {ep_path}", flush=True)
    return ep_path, example, eager


def compile_batch(
    ep_path: Path,
    pkg_path: Path,
    *,
    example: tuple[torch.Tensor, ...] | None,
    eager: tuple[torch.Tensor, ...] | None,
    atol: float,
    rtol: float,
    self_check: bool,
) -> None:
    remove_if_exists(pkg_path)
    ep = torch.export.load(ep_path)
    print(f"compiling {ep_path.name} -> {pkg_path.name} (autotune OFF)", flush=True)
    out_path = torch._inductor.aoti_compile_and_package(
        ep,
        package_path=str(pkg_path),
        inductor_configs=compile_configs(),
    )
    print(f"AOTI package: {out_path}", flush=True)

    if self_check and example is not None and eager is not None:
        runner = torch._inductor.aoti_load_package(out_path)
        with torch.inference_mode():
            got = runner(*example)
        got_tuple = tuple(got) if isinstance(got, (tuple, list)) else (got,)
        if not compare_outputs(f"aoti-{pkg_path.stem}-vs-eager", got_tuple, eager, atol, rtol):
            raise RuntimeError(f"AOTI self-check failed for {pkg_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./artifacts")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--compile-only", action="store_true", help="compile saved enc_steady_t2a_bB.pt2 files; skip NeMo export")
    parser.add_argument("--export-only", action="store_true", help="save ExportedPrograms but do not AOTI-compile")
    parser.add_argument("--manifest-only", action="store_true", help="only write MANIFEST.json for existing AOTI packages")
    parser.add_argument("--no-self-check", action="store_true", help="skip post-compile AOTI vs eager check")
    parser.add_argument("--shared-weights", default="./artifacts/finalize_shared_weights.ts")
    parser.add_argument("--production-b1", default="./artifacts/enc_steady_aoti.pt2")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--matmul-precision", default="highest", choices=["highest", "high", "medium"])
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args()

    if args.compile_only and args.export_only:
        raise SystemExit("--compile-only and --export-only cannot both be set")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = parse_batches(args.batches)

    if args.manifest_only:
        precision = configure_matmul_precision(args.matmul_precision, args.allow_tf32)
        emit_manifest(out_dir, batches, Path(args.shared_weights), Path(args.production_b1), precision)
        return

    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.cuda.is_available(),
        "cc",
        torch.cuda.get_device_capability() if torch.cuda.is_available() else "NA",
        flush=True,
    )
    precision = configure_matmul_precision(args.matmul_precision, args.allow_tf32)
    print("matmul precision config:", precision, flush=True)
    print("inductor autotune config:", set_inductor_autotune_off(), flush=True)
    noexec_seen = force_noexecstack_on_link()

    encoder = None if args.compile_only else load_encoder()
    exported_examples: dict[int, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = {}

    for batch in batches:
        ep_path = out_dir / EP_NAME.format(b=batch)
        if not args.compile_only:
            ep_path, example, eager = export_batch(encoder, batch, out_dir, args.atol, args.rtol)
            exported_examples[batch] = (example, eager)
        elif not ep_path.exists() and batch == 1 and (out_dir / "enc_steady_t2a.pt2").exists():
            ep_path = out_dir / "enc_steady_t2a.pt2"
            print(f"using legacy B=1 ExportedProgram path: {ep_path}", flush=True)

        if not args.export_only:
            if not ep_path.exists():
                raise FileNotFoundError(f"missing ExportedProgram for B={batch}: {ep_path}")
            example_eager = exported_examples.get(batch)
            compile_batch(
                ep_path,
                out_dir / PKG_NAME.format(b=batch),
                example=example_eager[0] if example_eager else None,
                eager=example_eager[1] if example_eager else None,
                atol=args.atol,
                rtol=args.rtol,
                self_check=not args.no_self_check and not args.compile_only,
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.export_only and not noexec_seen["flagged"]:
        raise RuntimeError("noexecstack shim never fired on a shared-lib link")
    if not args.export_only:
        emit_manifest(out_dir, batches, Path(args.shared_weights), Path(args.production_b1), precision)
    print("=== steady batched export/compile complete ===", flush=True)


if __name__ == "__main__":
    main()
