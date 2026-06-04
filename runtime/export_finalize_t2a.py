#!/usr/bin/env python3
"""1.3b substrate (host side): torch.export the FINALIZE-geometry encoder (keep_all_outputs=True) with a DYNAMIC T
dimension, so the C++ finalize port has one AOTI-compilable program that handles the variable remainder length (the
fixture shows T from 44 to 261). TorchScript trace is not byte-exact for dynamic-T finalize (export_finalize_encoder.py),
so — exactly like the steady encoder (export_t2a.py + aot_compile.py) — we torch.export here on the host, then
AOTI-compile in the container (aot_compile_finalize.py). drop_extra is baked per-variant (0=first, 2=continuation).

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python export_finalize_t2a.py --out ./artifacts
"""
from __future__ import annotations
import argparse, os, torch
from finalize_ref import ContinuousFinalizeRef, load_model, load_benchmark_dataset, load_wav


class FinalizeStep(torch.nn.Module):
    """keep_all_outputs=True finalize encoder step. Length is derived from the chunk (finalize always uses the full
    chunk length), avoiding a data-dependent length input that would complicate dynamic-shape export."""
    def __init__(self, encoder, drop_extra: int):
        super().__init__()
        self.encoder = encoder
        self.drop_extra = int(drop_extra)

    def forward(self, chunk, cache_last_channel, cache_last_time, cache_last_channel_len):
        length = torch.full((chunk.shape[0],), chunk.shape[-1], dtype=torch.long, device=chunk.device)
        return self.encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=self.drop_extra,
        )


def _finalize_example(rt: ContinuousFinalizeRef, wav, drop_extra: int):
    """Drive a clip to a finalize point and return (chunk_mel, clc, clt, clcl) for the requested drop_extra variant.
    drop=2 => a continuation finalize (emitted_frames>0); drop=0 => first-chunk finalize (clip kept short)."""
    sess = rt.new_session("exp")
    rt.append_audio(sess, wav)
    if drop_extra == 0 and sess.emitted_frames != 0:
        raise RuntimeError("clip too long for a drop=0 first-finalize example; pass a short clip")
    rt.vad_stop(sess)
    fork = rt.build_continuous_finalize_fork(sess)
    fi = rt.prepare_finalize_inputs(fork)
    if fi is None:
        raise RuntimeError("no finalize remainder for this clip")
    if int(fi.drop_extra) != drop_extra:
        raise RuntimeError(f"got drop={fi.drop_extra}, wanted {drop_extra}")
    return fi.chunk_mel, fi.cache_last_channel, fi.cache_last_time, fi.cache_last_channel_len


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    ds = load_benchmark_dataset()

    # drop=2 continuation example (the common finalize geometry). idx 2 -> a steady-then-finalize clip.
    chunk, clc, clt, clcl = _finalize_example(rt, load_wav(ds[2]), drop_extra=2)
    step = FinalizeStep(model.encoder, drop_extra=2).cuda().eval()
    T = torch.export.Dim("T", min=20, max=320)
    dyn = {"chunk": {2: T}, "cache_last_channel": None, "cache_last_time": None, "cache_last_channel_len": None}
    with torch.inference_mode():
        ep = torch.export.export(step, (chunk, clc, clt, clcl), dynamic_shapes=dyn)
    print("torch.export(finalize drop=2, dynamic T): OK")

    # validate the exported program byte-exact vs eager across several T values from the corpus
    mod = ep.module()
    ok_all = True
    samples = {}
    for idx in (2, 3, 4, 9):
        c, a0, a1, a2 = _finalize_example(rt, load_wav(ds[idx]), drop_extra=2)
        with torch.inference_mode():
            eager = step(c, a0, a1, a2)
            exp = mod(c, a0, a1, a2)
        maxd = max((e.float() - x.float()).abs().max().item() for e, x in zip(eager, exp) if torch.is_tensor(e) and e.numel())
        be = all(torch.equal(e, x) for e, x in zip(eager, exp) if torch.is_tensor(e))
        ok_all = ok_all and be
        print(f"  idx={idx} T={c.shape[-1]} export-vs-eager byte_exact={be} max_abs_diff={maxd:.3e}")
        if idx == 2:
            samples = {"chunk": c.cpu(), "clc": a0.cpu(), "clt": a1.cpu(), "clcl": a2.cpu(),
                       "out": [t.cpu() for t in eager]}

    torch.export.save(ep, os.path.join(a.out, "enc_finalize_drop2_t2a.pt2"))
    torch.save(samples, os.path.join(a.out, "finalize_t2a_io.pt"))
    print(f"=== saved enc_finalize_drop2_t2a.pt2 + finalize_t2a_io.pt (export byte-exact across T: {ok_all}) ===")


if __name__ == "__main__":
    main()
