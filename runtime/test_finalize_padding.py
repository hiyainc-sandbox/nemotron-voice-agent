#!/usr/bin/env python3
"""1.3b-enc de-risk: is PADDING the finalize chunk up to a fixed bucket-T transparent? i.e. does
cache_aware_stream_step(chunk_padded_to_BUCKET, length=true_T, keep_all_outputs=True) produce BYTE-IDENTICAL valid
output to the unpadded chunk_at_true_T? If yes, a fixed-T AOTI bucket + pad/slice is a sound substrate (the production
"padded finalize"). If the depthwise-conv right-context bleeds across the pad boundary, it won't be byte-exact and we
need a different bucketing. Eager-only, no export/AOTI — the crux check before building the substrate.

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python test_finalize_padding.py
"""
from __future__ import annotations
import torch
from finalize_ref import ContinuousFinalizeRef, load_model, load_benchmark_dataset, load_wav
from ref_decode import ref_greedy_range

BUCKET_T = 64

def finalize_input(rt, wav, drop_extra):
    """Return (fork, fi) — fork carries the CONTINUED decoder state/pred_out + pre-final hyp_tokens."""
    s = rt.new_session("p"); rt.append_audio(s, wav); rt.vad_stop(s)
    fork = rt.build_continuous_finalize_fork(s); fi = rt.prepare_finalize_inputs(fork)
    return fork, fi

@torch.inference_mode()
def main():
    rt = ContinuousFinalizeRef(load_model())
    ds = load_benchmark_dataset()
    enc, dec, joint = rt.encoder, rt.decoder, rt.joint
    import copy
    tok_allok = True
    for idx in range(40):  # broaden coverage beyond the 4 canaries
        fork, fi = finalize_input(rt, load_wav(ds[idx]), drop_extra=2)
        if fi is None or int(fi.drop_extra) != 2:
            continue
        chunk = fi.chunk_mel; T = chunk.shape[-1]
        if T > BUCKET_T:
            print(f"  idx={idx}: T={T} > bucket {BUCKET_T} (drop0/long), needs larger bucket or eager"); continue
        clc, clt, clcl = fi.cache_last_channel, fi.cache_last_time, fi.cache_last_channel_len
        Lt = torch.full((1,), T, device=chunk.device, dtype=torch.long)
        u = enc.cache_aware_stream_step(processed_signal=chunk, processed_signal_length=Lt,
            cache_last_channel=clc.clone(), cache_last_time=clt.clone(), cache_last_channel_len=clcl.clone(),
            keep_all_outputs=True, drop_extra_pre_encoded=2)
        pad = torch.zeros(chunk.shape[0], chunk.shape[1], BUCKET_T - T, device=chunk.device, dtype=chunk.dtype)
        p = enc.cache_aware_stream_step(processed_signal=torch.cat((chunk, pad), dim=-1), processed_signal_length=Lt,
            cache_last_channel=clc.clone(), cache_last_time=clt.clone(), cache_last_channel_len=clcl.clone(),
            keep_all_outputs=True, drop_extra_pre_encoded=2)
        Tu, Tp = int(u[1][0]), int(p[1][0]); v = min(Tu, Tp)
        eod = (u[0][:, :, :v].float() - p[0][:, :, :v].float()).abs().max().item() if v > 0 else 0.0
        # THE real check: FINAL tokens (pre-final hyp + finalize decode CONTINUED from fork decoder state)
        base = list(fork.hyp_tokens)
        su, gu = copy.deepcopy(fork.decoder_state), fork.pred_out_stream.clone()
        tu, _, _ = ref_greedy_range(dec, joint, u[0].transpose(1,2).contiguous(), 0, Tu, su, gu)
        sp, gp = copy.deepcopy(fork.decoder_state), fork.pred_out_stream.clone()
        tp, _, _ = ref_greedy_range(dec, joint, p[0].transpose(1,2).contiguous(), 0, Tp, sp, gp)
        full_u, full_p = base + tu, base + tp
        tok_eq = (full_u == full_p)
        tok_allok = tok_allok and tok_eq
        print(f"  idx={idx} T={T}->{BUCKET_T} enc_out_diff={eod:.3e} final_tok {len(full_u)}/{len(full_p)} TOKEN_EXACT={tok_eq}")
    print(f"=== PADDED-BUCKET (pad to {BUCKET_T}) FINAL-TOKEN-EXACT vs unpadded (continuation decode): {tok_allok} ===")

if __name__ == "__main__":
    main()
