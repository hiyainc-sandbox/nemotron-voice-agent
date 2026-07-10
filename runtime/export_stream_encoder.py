#!/usr/bin/env python3
"""1.2b — export per-geometry streaming encoders (first drop=0 @16-frame; steady drop=2 @25-frame) and VALIDATE they
stay byte-exact vs eager cache_aware_stream_step ACROSS the whole streaming sequence (cache_channel_len grows 0..70).
This is the key risk: a frozen trace might not handle varying cache_len. If it holds, the C++ streaming port can load
these .ts; if not, switch to torch.export/dynamic. Also bundles the clip's mel + gold streaming tokens for the C++ test.

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python export_stream_encoder.py --out ./artifacts
"""
from __future__ import annotations
import argparse, io, os, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from model_profile import apply_prompt, get_profile, load_profile_model
from ref_decode import ref_greedy_range

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    profile = get_profile()
    m = load_profile_model(profile)
    m.change_decoding_strategy(decoding_cfg=OmegaConf.create({"strategy":"greedy_batch","greedy":{"max_symbols":profile.max_symbols,"loop_labels":True,"use_cuda_graph_decoder":False}}))
    e, dec, joint = m.encoder, m.decoder, m.joint
    sc = e.streaming_cfg; _int=lambda v:int(v[1]) if isinstance(v,(list,tuple)) else int(v)
    shift, pre, drop = _int(sc.shift_size), _int(sc.pre_encode_cache_size), int(sc.drop_extra_pre_encoded)
    dev = next(m.parameters()).device

    import datasets
    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column("audio", datasets.Audio(decode=False))
    wav, sr = sf.read(io.BytesIO(ds[1]["audio"]["bytes"]), dtype="float32")
    if wav.ndim>1: wav=wav.mean(1)
    if sr!=16000: n=int(len(wav)*16000/sr); wav=np.interp(np.linspace(0,len(wav),n,endpoint=False),np.arange(len(wav)),wav).astype(np.float32)
    audio=torch.tensor(wav,device=dev).unsqueeze(0); alen=torch.tensor([wav.shape[0]],device=dev)
    with torch.inference_mode(): mel,_=m.preprocessor(input_signal=audio,length=alen)
    Tm=mel.shape[-1]; assert Tm % shift == 0, f"clip mel {Tm} not multiple of {shift} (would need partial-chunk handling)"

    def estep(chunk, clc, clt, clcl, d):
        L=torch.full((1,),chunk.shape[-1],device=dev,dtype=torch.long)
        return e.cache_aware_stream_step(processed_signal=chunk,processed_signal_length=L,cache_last_channel=clc,
            cache_last_time=clt,cache_last_channel_len=clcl,keep_all_outputs=False,drop_extra_pre_encoded=d)

    class Step(torch.nn.Module):
        def __init__(s,d): super().__init__(); s.e=e; s.d=d
        def forward(s,chunk,L,clc,clt,clcl):
            return s.e.cache_aware_stream_step(processed_signal=chunk,processed_signal_length=L,cache_last_channel=clc,
                cache_last_time=clt,cache_last_channel_len=clcl,keep_all_outputs=False,drop_extra_pre_encoded=s.d)

    # trace first (@16, empty cache) + steady (@25, populated cache from chunk0)
    cache=e.get_initial_cache_state(batch_size=1)
    c0=mel[:,:,0:shift]; L0=torch.full((1,),shift,device=dev,dtype=torch.long)
    with torch.inference_mode():
        o0=estep(c0,cache[0].clone(),cache[1].clone(),cache[2].clone(),0)
        ring=mel[:,:,0:shift][:,:,-pre:]; c1=torch.cat((ring,mel[:,:,shift:2*shift]),dim=-1); L1=torch.full((1,),c1.shape[-1],device=dev,dtype=torch.long)
        first_ts=torch.jit.trace(Step(0).cuda().eval(),(c0,L0,cache[0].clone(),cache[1].clone(),cache[2].clone()),check_trace=False)
        steady_ts=torch.jit.trace(Step(drop).cuda().eval(),(c1,L1,o0[2].clone(),o0[3].clone(),o0[4].clone()),check_trace=False)
    first_ts.save(os.path.join(a.out,"enc_first.ts")); steady_ts.save(os.path.join(a.out,"enc_steady.ts"))

    # VALIDATE: run the full streaming loop with EAGER vs TRACED encoders; compare enc_out byte-exact per chunk + tokens
    def run(use_traced):
        clc,clt,clcl=cache[0].clone(),cache[1].clone(),cache[2].clone(); ring=None
        st=dec.initialize_state(torch.zeros(1,1,dtype=torch.float32,device=dev)); g,st=dec.predict(None,st,add_sos=False,batch_size=1)
        toks=[]; encs=[]; emitted=0; pos=0
        with torch.inference_mode():
            while pos<Tm:
                nm=mel[:,:,pos:pos+shift]
                if emitted==0: chunk=nm; d=0; mod=first_ts
                else: chunk=torch.cat((ring,nm),dim=-1); d=drop; mod=steady_ts
                if use_traced:
                    L=torch.full((1,),chunk.shape[-1],device=dev,dtype=torch.long); out=mod(chunk,L,clc,clt,clcl)
                else: out=estep(chunk,clc,clt,clcl,d)
                eo,elen,clc,clt,clcl=out; encs.append(eo.clone())
                f=apply_prompt(m,eo).transpose(1,2).contiguous(); t,st,g=ref_greedy_range(dec,joint,f,0,int(elen[0]),st,g); toks+=t
                ring=(torch.cat((ring,nm),dim=-1) if ring is not None else nm)[:,:,-pre:]; emitted+=nm.shape[-1]; pos+=shift
        return toks, encs
    eager_tok, eager_enc = run(False)
    traced_tok, traced_enc = run(True)
    enc_ok = all(torch.equal(a_,b_) for a_,b_ in zip(eager_enc,traced_enc))
    tok_ok = (eager_tok==traced_tok)
    print(f"chunks={Tm//shift} | traced-encoder enc_out byte-exact across ALL chunks: {enc_ok} | tokens match: {tok_ok}")
    print(f"  eager tokens={len(eager_tok)} text={m.tokenizer.ids_to_text(eager_tok)!r}")
    if not enc_ok:
        for i,(x,y) in enumerate(zip(eager_enc,traced_enc)):
            if not torch.equal(x,y): print(f"   chunk {i}: eager{tuple(x.shape)} traced{tuple(y.shape)} maxdiff={(x.float()-y.float()).abs().max().item():.3e}"); break
    # FINDING: traced per-geometry encoder is NOT byte-exact across varying cache_len (~1e-5 drift, trace freezes
    # cache_len-dependent ops) BUT is TOKEN-exact (greedy decode robust to the drift). T1 (transcript) PASS;
    # T2a (byte-exact encoder) across the stream needs torch.export/dynamic (refinement). Bundle on the T1 gate.
    if tok_ok:
        class B(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.register_buffer("mel",mel.cpu()); s.register_buffer("gold",torch.tensor(eager_tok,dtype=torch.int64))
                s.register_buffer("clc0",cache[0].cpu()); s.register_buffer("clt0",cache[1].cpu()); s.register_buffer("clcl0",cache[2].cpu())
                # metadata the C++ runtime asserts vs its compiled constants (Codex#4: don't hard-code + silently mismatch)
                s.register_buffer("meta", torch.tensor([shift, pre, drop, profile.blank, profile.max_symbols], dtype=torch.int64))  # shift,pre,drop,blank,max_symbols
            def forward(s): return s.mel
        torch.jit.script(B()).save(os.path.join(a.out,"stream_bundle.ts"))
        print(f"exported enc_first.ts + enc_steady.ts + stream_bundle.ts (mel {Tm}, {len(eager_tok)} gold tok)")
        print(f"  T1 (token-exact streaming): PASS | T2a (encoder byte-exact across cache_len): {'PASS' if enc_ok else 'NO (~1e-5 drift; needs torch.export/dynamic)'}")
        print("  -> C++ streaming port ready at the T1 bar")
    else:
        print("TOKENS DIVERGE -> investigate before C++ port")

if __name__ == "__main__":
    main()
