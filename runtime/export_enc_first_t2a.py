#!/usr/bin/env python3
"""Export the first-chunk streaming encoder geometry as torch.export.

This is the drop_extra=0, no-prior-ring geometry used for chunk 0.  The
resulting ExportedProgram is architecture-agnostic and is the source for the
sm_120/sm_89 AOTI packages.

Run from runtime/ on the host Nemo venv:
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_enc_first_t2a.py --out ./artifacts
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr

from model_profile import get_profile, load_profile_model


class FirstChunkStep(torch.nn.Module):
    def __init__(self, encoder, valid_out: int, extra_out: int):
        super().__init__()
        self.encoder = encoder
        self.valid_out = valid_out
        self.extra_out = extra_out

    def forward(self, chunk, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        enc_out, enc_len, cache_ch, cache_t, cache_ch_len = self.encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=0,
        )
        # For the first shift-frame chunk, keep_all_outputs=True returns
        # extra encoder frames at the end; the prefix and recurrent caches are
        # byte-exact to keep_all_outputs=False in eager/TS.  Returning the
        # prefix keeps the runtime contract unchanged while giving AOTI the
        # numerically safer graph used by finalize-style exports.  valid_out /
        # extra_out are measured eagerly against keep_all_outputs=False before
        # export (2 / 1 for the en profile at shift 16).
        return (
            enc_out[:, :, : self.valid_out].contiguous(),
            enc_len - self.extra_out,
            cache_ch,
            cache_t,
            cache_ch_len,
        )


def _stream_cfg_int(value) -> int:
    return int(value[1]) if isinstance(value, (list, tuple)) else int(value)


def _load_fixture_audio():
    import datasets

    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column(
        "audio", datasets.Audio(decode=False)
    )
    wav, sr = sf.read(io.BytesIO(ds[1]["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        n = int(len(wav) * 16000 / sr)
        wav = np.interp(
            np.linspace(0, len(wav), n, endpoint=False),
            np.arange(len(wav)),
            wav,
        ).astype(np.float32)
    return wav


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./artifacts")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    profile = get_profile()
    model = load_profile_model(profile)
    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create(
            {
                "strategy": "greedy_batch",
                "greedy": {
                    "max_symbols": profile.max_symbols,
                    "loop_labels": True,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )
    encoder = model.encoder
    cfg = encoder.streaming_cfg
    shift = _stream_cfg_int(cfg.shift_size)
    pre = _stream_cfg_int(cfg.pre_encode_cache_size)
    drop = int(cfg.drop_extra_pre_encoded)
    device = next(model.parameters()).device

    wav = _load_fixture_audio()
    audio = torch.tensor(wav, device=device).unsqueeze(0)
    audio_len = torch.tensor([wav.shape[0]], device=device)
    with torch.inference_mode():
        mel, _ = model.preprocessor(input_signal=audio, length=audio_len)

    cache = encoder.get_initial_cache_state(batch_size=1)
    chunk = mel[:, :, 0:shift].contiguous()
    length = torch.full((1,), shift, dtype=torch.long, device=device)
    clc = cache[0].clone()
    clt = cache[1].clone()
    clcl = cache[2].clone()

    # Measure the valid first-chunk output length (keep_all_outputs=False) and
    # how many extra frames keep_all_outputs=True appends; those become static
    # slice constants in the exported graph.
    with torch.inference_mode():
        ref_valid = encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=clc.clone(),
            cache_last_time=clt.clone(),
            cache_last_channel_len=clcl.clone(),
            keep_all_outputs=False,
            drop_extra_pre_encoded=0,
        )
        ref_all = encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=clc.clone(),
            cache_last_time=clt.clone(),
            cache_last_channel_len=clcl.clone(),
            keep_all_outputs=True,
            drop_extra_pre_encoded=0,
        )
    valid_out = int(ref_valid[1][0])
    extra_out = int(ref_all[1][0]) - valid_out
    if extra_out < 0:
        raise RuntimeError(f"keep_all_outputs shrank the output: valid={valid_out} all={int(ref_all[1][0])}")
    print(f"first-chunk geometry: valid_out={valid_out} extra_out={extra_out}")
    step = FirstChunkStep(encoder, valid_out, extra_out).cuda().eval()

    with torch.inference_mode():
        eager = step(chunk, length, clc, clt, clcl)
        ep = torch.export.export(step, (chunk, length, clc, clt, clcl))
        exported = ep.module()(chunk, length, clc, clt, clcl)

    byte_exact = True
    max_abs = 0.0
    for lhs, rhs in zip(eager, exported):
        if not torch.is_tensor(lhs):
            continue
        same = torch.equal(lhs, rhs)
        byte_exact = byte_exact and same
        if lhs.numel():
            max_abs = max(max_abs, float((lhs.float() - rhs.float()).abs().max().item()))

    ts_path = os.path.join(args.out, "enc_first.ts")
    ts_exact = None
    if os.path.exists(ts_path):
        ts_mod = torch.jit.load(ts_path).to(device).eval()
        with torch.inference_mode():
            ts_out = ts_mod(chunk, length, clc, clt, clcl)
        ts_exact = all(
            torch.equal(lhs, rhs)
            for lhs, rhs in zip(eager, ts_out)
            if torch.is_tensor(lhs) and torch.is_tensor(rhs)
        )

    ep_path = os.path.join(args.out, "enc_first_t2a.pt2")
    io_path = os.path.join(args.out, "enc_first_t2a_io.pt")
    torch.export.save(ep, ep_path)
    torch.save(
        {
            "chunk": chunk.cpu(),
            "L": length.cpu(),
            "clc": clc.cpu(),
            "clt": clt.cpu(),
            "clcl": clcl.cpu(),
            "out": [t.cpu() for t in eager],
            "meta": {
                "shift": shift,
                "pre_encode_cache": pre,
                "drop_extra_steady": drop,
                "drop_extra_first": 0,
                "valid_out": valid_out,
                "extra_out": extra_out,
                "model_id": profile.model_id,
                "att_context_size": list(profile.att_context),
            },
        },
        io_path,
    )
    print(
        "first-chunk torch.export: "
        f"byte_exact_vs_eager={byte_exact} max_abs={max_abs:.3e} "
        f"ts_byte_exact_vs_eager={ts_exact}"
    )
    print(f"saved {ep_path} + {io_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
