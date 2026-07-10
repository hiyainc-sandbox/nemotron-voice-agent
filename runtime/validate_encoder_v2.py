#!/usr/bin/env python3
"""0.2 / T2a (proper) — the steady encoder is a DATA-DEPENDENT geometry: output frame-count depends on cache_len, so a
trace made with an EMPTY initial cache (first-chunk) does NOT match a steady chunk (populated cache). Here we run a
realistic 2-chunk eager streaming sequence, then export the STEADY chunk's geometry and check byte-exactness vs eager.

Run: ./.venv/bin/python validate_encoder_v2.py
"""
from __future__ import annotations
import torch, nemo.collections.asr as nemo_asr
from model_profile import get_profile, load_profile_model

def main():
    m = load_profile_model(get_profile())
    e = m.encoder; sc = e.streaming_cfg
    _int = lambda v: int(v[1]) if isinstance(v,(list,tuple)) else int(v)
    T = _int(sc.pre_encode_cache_size) + _int(sc.shift_size); F = int(m.cfg.preprocessor.features)
    steady_drop = int(sc.drop_extra_pre_encoded)
    cache = e.get_initial_cache_state(batch_size=1); dev = cache[0].device
    torch.manual_seed(0)

    def step(proc, clc, clt, clcl, drop):
        with torch.inference_mode():
            return e.cache_aware_stream_step(processed_signal=proc, processed_signal_length=torch.full((1,),proc.shape[-1],device=dev,dtype=torch.long),
                cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                keep_all_outputs=False, drop_extra_pre_encoded=drop)

    # chunk1 = first chunk (drop_extra=0, empty cache) -> populates cache
    p1 = torch.randn(1, F, T, device=dev, dtype=cache[0].dtype)
    o1 = step(p1, cache[0].clone(), cache[1].clone(), cache[2].clone(), 0)
    c_ch, c_t, c_len = o1[2], o1[3], o1[4]
    print(f"chunk1(first,empty cache,drop=0): enc_out {tuple(o1[0].shape)} cache_len_out {c_len.tolist()}")

    # chunk2 = STEADY chunk (drop_extra=2, populated cache)
    p2 = torch.randn(1, F, T, device=dev, dtype=cache[0].dtype)
    o2a = step(p2, c_ch.clone(), c_t.clone(), c_len.clone(), steady_drop)
    o2b = step(p2, c_ch.clone(), c_t.clone(), c_len.clone(), steady_drop)
    t0 = all(torch.equal(a,b) for a,b in zip(o2a,o2b) if torch.is_tensor(a))
    print(f"chunk2(STEADY,populated cache,drop={steady_drop}): enc_out {tuple(o2a[0].shape)}  [T0 eager determinism: {t0}]")

    # export the STEADY geometry (populated cache as the example) + compare byte-exact
    class Steady(torch.nn.Module):
        def __init__(s): super().__init__(); s.e=e
        def forward(s, proc, length, clc, clt, clcl):
            return s.e.cache_aware_stream_step(processed_signal=proc, processed_signal_length=length,
                cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                keep_all_outputs=False, drop_extra_pre_encoded=steady_drop)
    length = torch.full((1,), T, device=dev, dtype=torch.long)
    with torch.inference_mode():
        ts = torch.jit.trace(Steady().cuda().eval(), (p2, length, c_ch.clone(), c_t.clone(), c_len.clone()), check_trace=False)
        ots = ts(p2, length, c_ch.clone(), c_t.clone(), c_len.clone())
    names = ["enc_out","enc_len","cache_ch","cache_t","cache_ch_len"]
    allok = True
    print("[T2a] re-exported STEADY .ts vs eager (same populated-cache inputs):")
    for n,a,b in zip(names,o2a,ots):
        if not (torch.is_tensor(a) and torch.is_tensor(b)): continue
        eq = (a.shape==b.shape) and torch.equal(a,b)
        md = (a.float()-b.float()).abs().max().item() if a.shape==b.shape and a.numel() else float('nan')
        allok &= eq; print(f"   {n}: byte-equal={eq} max_abs_diff={md:.3e} shapes {tuple(a.shape)}/{tuple(b.shape)}")
    print("=== T2a STEADY (populated cache): BYTE-EXACT" if allok else "=== T2a STEADY: NOT byte-exact")

if __name__ == "__main__":
    main()
