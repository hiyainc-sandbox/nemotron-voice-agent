#!/usr/bin/env python3
"""0.2 / T2a — is the exported steady-encoder TorchScript byte-exact vs the eager NeMo cache_aware_stream_step?
Same inputs -> compare all 5 outputs bit-for-bit (torch.equal) + max-abs-diff. Run-to-run determinism (T0) too.

Run: ./.venv/bin/python validate_encoder.py
"""
from __future__ import annotations
import os, torch
import nemo.collections.asr as nemo_asr

def main():
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    enc = model.encoder
    scfg = enc.streaming_cfg
    def _int(v): return int(v[1]) if isinstance(v,(list,tuple)) else int(v)
    shift, pre = _int(scfg.shift_size), _int(scfg.pre_encode_cache_size)
    drop_extra = int(scfg.drop_extra_pre_encoded); T = pre + shift
    feat = int(model.cfg.preprocessor.features)
    cache = enc.get_initial_cache_state(batch_size=1); dev = cache[0].device

    torch.manual_seed(0)
    processed = torch.randn(1, feat, T, device=dev, dtype=cache[0].dtype)
    length = torch.full((1,), T, device=dev, dtype=torch.long)
    clc, clt, clcl = cache[0].clone(), cache[1].clone(), cache[2].clone()

    def eager():
        with torch.inference_mode():
            return enc.cache_aware_stream_step(
                processed_signal=processed, processed_signal_length=length,
                cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                keep_all_outputs=False, drop_extra_pre_encoded=drop_extra)
    o1 = eager(); o2 = eager()
    # T0: eager run-to-run determinism
    t0 = all(torch.equal(a, b) for a, b in zip(o1, o2) if torch.is_tensor(a))
    print(f"[T0] eager run-to-run byte-identical: {t0}")

    ts_path = os.path.join(os.path.dirname(__file__), "..", "spikes", "0.1-overlap-ablation", "microbench",
                           "artifacts", "encoder_steady_b1.ts")
    if not os.path.exists(ts_path):
        print(f"  (no exported .ts at {ts_path}; regenerate via microbench/export_encoder.py) — skipping T2a")
        return
    mod = torch.jit.load(ts_path); mod.to(dev).eval()
    with torch.inference_mode():
        ots = mod(processed, length, clc, clt, clcl)
    names = ["encoder_out","cache_ch","cache_t","cache_ch_len","enc_len"]
    print("[T2a] exported .ts vs eager (same inputs):")
    allok = True
    for n, a, b in zip(names, o1, ots):
        if not (torch.is_tensor(a) and torch.is_tensor(b)): continue
        eq = torch.equal(a, b); md = (a.float()-b.float()).abs().max().item() if a.numel() else 0.0
        allok &= eq
        print(f"   {n}: byte-equal={eq} max_abs_diff={md:.3e} shape={tuple(a.shape)}")
    print("=== T2a STEADY ENCODER: BYTE-EXACT" if allok else "=== T2a: NOT byte-exact (investigate)")

if __name__ == "__main__":
    main()
