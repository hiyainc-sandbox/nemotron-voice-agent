#!/usr/bin/env python3
"""Compute Step-4 session WER from C++ hypotheses and bundle references.

Run from runtime/ after the C++ session gate writes artifacts/session_hyps.jsonl:
  HF_HUB_OFFLINE=1 ./.venv/bin/python session_wer.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from importlib import metadata
from typing import Any

import jiwer
import torch
from whisper_normalizer.english import EnglishTextNormalizer


ART = os.path.join(os.path.dirname(__file__), "artifacts")


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return "unknown"


def _tensor_attr(module: Any, name: str) -> torch.Tensor:
    try:
        value = getattr(module, name)
    except AttributeError:
        value = module._c.getattr(name)  # type: ignore[attr-defined]
    if not torch.is_tensor(value):
        raise TypeError(f"bundle attribute {name!r} is not a tensor")
    return value.detach().cpu()


def _tensor_to_i64(tensor: torch.Tensor) -> list[int]:
    flat = tensor.to(torch.long).contiguous().view(-1)
    return [int(v) for v in flat.tolist()]


def _tensor_to_u8(tensor: torch.Tensor) -> bytes:
    flat = tensor.to(torch.uint8).contiguous().view(-1)
    return bytes(int(v) for v in flat.tolist())


def _unpack_utf8(flat_tensor: torch.Tensor, offsets_tensor: torch.Tensor, label: str) -> list[str]:
    flat = _tensor_to_u8(flat_tensor)
    offsets = _tensor_to_i64(offsets_tensor)
    if not offsets or offsets[0] != 0 or offsets[-1] != len(flat):
        raise ValueError(f"{label} offsets do not cover the UTF-8 payload")
    out: list[str] = []
    for start, end in zip(offsets, offsets[1:]):
        if start < 0 or end < start or end > len(flat):
            raise ValueError(f"{label} has invalid UTF-8 offsets")
        out.append(flat[start:end].decode("utf-8"))
    return out


def _one_text(module: Any, prefix: str, name: str) -> str:
    values = _unpack_utf8(
        _tensor_attr(module, f"{prefix}_{name}_bytes"),
        _tensor_attr(module, f"{prefix}_{name}_offsets"),
        f"{prefix}_{name}",
    )
    if len(values) != 1:
        raise ValueError(f"{prefix}_{name} expected one string, got {len(values)}")
    return values[0]


def _optional_one_text(module: Any, prefix: str, name: str) -> str | None:
    try:
        return _one_text(module, prefix, name)
    except Exception:
        return None


def _ids_to_text(ids: list[int], pieces: list[str]) -> str:
    if not ids:
        return ""
    marker = "\u2581"
    unk_surface = " \u2047 "
    text = []
    strip_dummy_prefix = pieces[ids[0]].startswith(marker)
    for token_id in ids:
        if token_id < 0 or token_id >= len(pieces):
            raise ValueError(f"token id out of tokenizer piece range: {token_id}")
        piece = pieces[token_id]
        text.append(unk_surface if piece == "<unk>" else piece)
    joined = "".join(text).replace(marker, " ")
    if strip_dummy_prefix and joined.startswith(" "):
        joined = joined[1:]
    return joined


def _load_bundle_rows(bundle_path: str) -> list[dict[str, Any]]:
    bundle = torch.jit.load(bundle_path, map_location="cpu")
    rows = int(_tensor_attr(bundle, "num_utts").reshape(-1)[0].item())
    pieces = _unpack_utf8(
        _tensor_attr(bundle, "token_piece_bytes"),
        _tensor_attr(bundle, "token_piece_offsets"),
        "token_piece",
    )

    out: list[dict[str, Any]] = []
    for utt in range(rows):
        prefix = f"utt{utt}"
        sample_index = int(_tensor_attr(bundle, f"{prefix}_sample_index").reshape(-1)[0].item())
        sample_id = _optional_one_text(bundle, prefix, "sample_id") or str(sample_index)
        reference = _optional_one_text(bundle, prefix, "reference_text")
        if reference is None:
            raise RuntimeError(
                "bundle is missing Step-4 reference_text buffers; re-export with the updated "
                "runtime/export_session_bundle.py"
            )
        oracle_text = _optional_one_text(bundle, prefix, "finalize_ref_text")
        if oracle_text is None:
            oracle_text = _ids_to_text(_tensor_to_i64(_tensor_attr(bundle, f"{prefix}_gold_tokens")), pieces)
        out.append(
            {
                "utt": utt,
                "sample_index": sample_index,
                "sample_id": str(sample_id),
                "reference": reference,
                "oracle_text": oracle_text,
            }
        )
    return out


def _load_hyps(path: str) -> dict[str, dict[str, Any]]:
    hyps: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in hyps:
                raise ValueError(f"duplicate sample_id in {path}: {sample_id!r}")
            if "final_text" not in row:
                raise ValueError(f"line {line_no} missing final_text")
            row["sample_id"] = sample_id
            hyps[sample_id] = row
    return hyps


def _wer_one(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else math.inf
    return float(jiwer.wer(reference, hypothesis))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _short(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=os.path.join(ART, "session_bundle.ts"))
    parser.add_argument("--hyps", default=os.path.join(ART, "session_hyps.jsonl"))
    parser.add_argument("--out", default=os.path.join(ART, "session_wer_summary.json"))
    parser.add_argument("--expect-n", type=int, default=None)
    parser.add_argument(
        "--max-delta-pp",
        type=float,
        default=0.0042,
        help="pooled session WER may exceed finalize_ref WER by at most this many percentage points",
    )
    parser.add_argument(
        "--max-utt-delta-pp",
        type=float,
        default=0.0,
        help="per-utterance session WER may exceed finalize_ref WER by at most this many percentage points",
    )
    parser.add_argument("--worst", type=int, default=10)
    args = parser.parse_args()

    normalizer = EnglishTextNormalizer()
    bundle_rows = _load_bundle_rows(args.bundle)
    hyps = _load_hyps(args.hyps)
    if args.expect_n is not None and len(bundle_rows) != args.expect_n:
        raise SystemExit(f"bundle row count {len(bundle_rows)} != --expect-n {args.expect_n}")

    missing = [row["sample_id"] for row in bundle_rows if row["sample_id"] not in hyps]
    extra = sorted(set(hyps) - {row["sample_id"] for row in bundle_rows})
    if missing or extra:
        raise SystemExit(
            f"hypothesis/sample mismatch: missing={missing[:10]} extra={extra[:10]} "
            f"(counts missing={len(missing)} extra={len(extra)})"
        )

    records: list[dict[str, Any]] = []
    for row in bundle_rows:
        sample_id = row["sample_id"]
        session_text = str(hyps[sample_id]["final_text"])
        ref_norm = normalizer(row["reference"]).strip()
        session_norm = normalizer(session_text).strip()
        oracle_norm = normalizer(row["oracle_text"]).strip()
        records.append(
            {
                **row,
                "session_text": session_text,
                "ref_norm": ref_norm,
                "session_norm": session_norm,
                "oracle_norm": oracle_norm,
                "text_exact_oracle": session_text == row["oracle_text"],
            }
        )

    keep = [record for record in records if record["ref_norm"].strip()]
    refs = [record["ref_norm"] for record in keep]
    session_h = [record["session_norm"] for record in keep]
    oracle_h = [record["oracle_norm"] for record in keep]
    pooled_session = float(jiwer.wer(refs, session_h)) if keep else math.nan
    pooled_oracle = float(jiwer.wer(refs, oracle_h)) if keep else math.nan
    delta_pp = (pooled_session - pooled_oracle) * 100.0

    per_utt: list[dict[str, Any]] = []
    for record in keep:
        session_wer = _wer_one(record["ref_norm"], record["session_norm"])
        oracle_wer = _wer_one(record["ref_norm"], record["oracle_norm"])
        per_utt.append(
            {
                "utt": record["utt"],
                "sample_index": record["sample_index"],
                "sample_id": record["sample_id"],
                "ref_words": len(record["ref_norm"].split()),
                "session_wer": session_wer,
                "oracle_wer": oracle_wer,
                "delta_pp": (session_wer - oracle_wer) * 100.0,
                "reference_norm": record["ref_norm"],
                "session_norm": record["session_norm"],
                "oracle_norm": record["oracle_norm"],
                "text_exact_oracle": record["text_exact_oracle"],
            }
        )

    wers = [row["session_wer"] for row in per_utt]
    deltas = [row["delta_pp"] for row in per_utt]
    max_utt_delta_pp = max(deltas) if deltas else 0.0
    pooled_pass = delta_pp <= args.max_delta_pp + 1.0e-12
    per_utt_pass = max_utt_delta_pp <= args.max_utt_delta_pp + 1.0e-12
    text_divergences = sum(1 for record in records if not record["text_exact_oracle"])
    worst = sorted(
        per_utt,
        key=lambda row: (row["session_wer"], row["delta_pp"], row["ref_words"]),
        reverse=True,
    )[: args.worst]
    worst_delta = sorted(per_utt, key=lambda row: row["delta_pp"], reverse=True)[: args.worst]

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "jiwer": _package_version("jiwer"),
        "whisper_normalizer": _package_version("whisper-normalizer", "whisper_normalizer"),
    }
    distribution = {
        "count": len(wers),
        "mean": sum(wers) / len(wers) if wers else math.nan,
        "p50": _quantile(wers, 0.50),
        "p90": _quantile(wers, 0.90),
        "p95": _quantile(wers, 0.95),
        "p99": _quantile(wers, 0.99),
        "max": max(wers) if wers else math.nan,
        "max_delta_pp": max_utt_delta_pp,
    }
    summary = {
        "bundle": args.bundle,
        "hyps": args.hyps,
        "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
        "versions": versions,
        "n": len(records),
        "non_empty_refs": len(keep),
        "empty_refs_skipped_for_pooled": len(records) - len(keep),
        "sample_ids": [record["sample_id"] for record in records],
        "text_divergences_vs_oracle": text_divergences,
        "pooled": {
            "session_wer": pooled_session,
            "oracle_wer": pooled_oracle,
            "delta_pp": delta_pp,
            "max_delta_pp": args.max_delta_pp,
            "noninferior": pooled_pass,
        },
        "per_utterance": {
            "distribution": distribution,
            "max_utt_delta_pp": args.max_utt_delta_pp,
            "bounded": per_utt_pass,
            "worst": worst,
            "worst_delta": worst_delta,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"=== SESSION WER ({len(records)} utterances, {len(keep)} non-empty normalized refs) ===")
    print("normalizer: whisper_normalizer.english.EnglishTextNormalizer")
    print(
        "versions: "
        f"jiwer={versions['jiwer']} torch={versions['torch']} "
        f"whisper_normalizer={versions['whisper_normalizer']}"
    )
    print(f"sample_ids: first={summary['sample_ids'][:5]} last={summary['sample_ids'][-5:]}")
    print(f"text divergences vs finalize_ref oracle: {text_divergences}/{len(records)}")
    print(f"pooled session WER     : {pooled_session * 100.0:.4f}%")
    print(f"pooled finalize_ref WER: {pooled_oracle * 100.0:.4f}%")
    print(f"delta session-oracle   : {delta_pp:+.6f} pp (limit +{args.max_delta_pp:.6f} pp)")
    print(
        "per-utt WER distribution: "
        f"mean={distribution['mean'] * 100.0:.3f}% "
        f"p50={distribution['p50'] * 100.0:.3f}% "
        f"p90={distribution['p90'] * 100.0:.3f}% "
        f"p95={distribution['p95'] * 100.0:.3f}% "
        f"p99={distribution['p99'] * 100.0:.3f}% "
        f"max={distribution['max'] * 100.0:.3f}%"
    )
    print(
        f"per-utt max positive delta: {max_utt_delta_pp:+.6f} pp "
        f"(limit +{args.max_utt_delta_pp:.6f} pp)"
    )
    print("worst utterances by session WER:")
    for row in worst:
        print(
            f"  sample_id={row['sample_id']} idx={row['sample_index']} "
            f"session={row['session_wer'] * 100.0:.2f}% "
            f"oracle={row['oracle_wer'] * 100.0:.2f}% "
            f"delta={row['delta_pp']:+.3f}pp ref={_short(row['reference_norm'])!r} "
            f"hyp={_short(row['session_norm'])!r}"
        )
    if worst_delta and worst_delta[0]["delta_pp"] > 0:
        print("worst utterances by positive session-oracle delta:")
        for row in worst_delta:
            if row["delta_pp"] <= 0:
                continue
            print(
                f"  sample_id={row['sample_id']} idx={row['sample_index']} "
                f"delta={row['delta_pp']:+.3f}pp session={row['session_wer'] * 100.0:.2f}% "
                f"oracle={row['oracle_wer'] * 100.0:.2f}%"
            )
    print(f"summary: {args.out}")
    print(
        "ASSERTIONS: "
        f"pooled_noninferior={'PASS' if pooled_pass else 'FAIL'} "
        f"per_utt_bound={'PASS' if per_utt_pass else 'FAIL'}"
    )
    return 0 if pooled_pass and per_utt_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
