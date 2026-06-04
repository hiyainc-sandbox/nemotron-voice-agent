#!/usr/bin/env python3
"""0.8 — native preprocessor byte-exactness. The server uses a CONSTANT-PLAN fixed-size mel preprocessor (dither=0) to
keep cuFFT deterministic (memory cufft-stft-plan-size-nondeterminism). Validate: (T0) fixed-size preprocessor is
run-to-run byte-identical, and (T2a) it exports to TorchScript byte-exact. If both hold, the native runtime can
reproduce the server's preprocessing byte-exact by using the same fixed-size constant-plan approach.

Run: ./.venv/bin/python validate_preproc.py
"""
from __future__ import annotations
import torch, nemo.collections.asr as nemo_asr

def main():
    m = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    pp = m.preprocessor
    try: pp.featurizer.dither = 0.0
    except Exception: pass
    try: print("dither:", pp.featurizer.dither, "n_mels:", m.cfg.preprocessor.features)
    except Exception: pass
    dev = next(m.parameters()).device

    # Fixed-size audio chunk (constant plan). Use a representative chunk size; values fixed (seed) for determinism test.
    torch.manual_seed(0)
    for K in (2720, 4096, 16000):   # preprocess_new_audio_samples, a power-of-2, ~1s
        audio = torch.randn(1, K, device=dev)
        length = torch.full((1,), K, device=dev, dtype=torch.long)
        with torch.inference_mode():
            a1, l1 = pp(input_signal=audio, length=length)
            a2, l2 = pp(input_signal=audio, length=length)
        t0 = torch.equal(a1, a2)
        print(f"[T0] K={K}: mel {tuple(a1.shape)} run-to-run byte-identical={t0} (max_diff={(a1-a2).abs().max().item():.3e})")

    # T2a: export the fixed-size preprocessor and compare byte-exact
    class PP(torch.nn.Module):
        def __init__(s): super().__init__(); s.pp = pp
        def forward(s, sig, length): return s.pp(input_signal=sig, length=length)
    K = 4096
    audio = torch.randn(1, K, device=dev); length = torch.full((1,), K, device=dev, dtype=torch.long)
    with torch.inference_mode():
        eager_mel, _ = pp(input_signal=audio, length=length)
        try:
            ts = torch.jit.trace(PP().cuda().eval(), (audio, length), check_trace=False)
            ts_mel, _ = ts(audio, length)
            eq = (eager_mel.shape == ts_mel.shape) and torch.equal(eager_mel, ts_mel)
            md = (eager_mel - ts_mel).abs().max().item()
            print(f"[T2a] preprocessor .ts vs eager: byte-equal={eq} max_abs_diff={md:.3e}")
        except Exception as ex:
            print(f"[T2a] preprocessor export FAILED: {type(ex).__name__}: {str(ex)[:200]}")

if __name__ == "__main__":
    main()
