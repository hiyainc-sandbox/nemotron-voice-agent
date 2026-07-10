#!/usr/bin/env python3
"""T2a — attempt a BYTE-EXACT streaming encoder via torch.export (vs the trace's ~1e-5 cache_len drift). torch.export
captures tensor-value ops symbolically (unlike trace which bakes them); if cache_aware_stream_step's cache_len usage is
pure tensor ops it'll be byte-exact across cache_len; if it has data-dependent Python control flow, export errors (a
finding -> reimplement the step, or stay T1). Validate byte-exact across the 20-chunk streaming sequence.

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python export_t2a.py --out ./artifacts
"""
from __future__ import annotations
import argparse, io, os, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from model_profile import get_profile, load_profile_model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="./artifacts"); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    m=load_profile_model(get_profile())
    e=m.encoder; sc=e.streaming_cfg; _int=lambda v:int(v[1]) if isinstance(v,(list,tuple)) else int(v)
    shift,pre,drop=_int(sc.shift_size),_int(sc.pre_encode_cache_size),int(sc.drop_extra_pre_encoded); dev=next(m.parameters()).device

    import datasets
    ds=datasets.load_dataset("pipecat-ai/stt-benchmark-data",split="train").cast_column("audio",datasets.Audio(decode=False))
    wav,sr=sf.read(io.BytesIO(ds[1]["audio"]["bytes"]),dtype="float32")
    if wav.ndim>1: wav=wav.mean(1)
    if sr!=16000: n=int(len(wav)*16000/sr); wav=np.interp(np.linspace(0,len(wav),n,endpoint=False),np.arange(len(wav)),wav).astype(np.float32)
    audio=torch.tensor(wav,device=dev).unsqueeze(0); alen=torch.tensor([wav.shape[0]],device=dev)
    with torch.inference_mode(): mel,_=m.preprocessor(input_signal=audio,length=alen)
    Tm=mel.shape[-1]

    class Step(torch.nn.Module):
        def __init__(s,d): super().__init__(); s.e=e; s.d=d
        def forward(s,chunk,L,clc,clt,clcl):
            return s.e.cache_aware_stream_step(processed_signal=chunk,processed_signal_length=L,cache_last_channel=clc,
                cache_last_time=clt,cache_last_channel_len=clcl,keep_all_outputs=False,drop_extra_pre_encoded=s.d)

    cache=e.get_initial_cache_state(batch_size=1)
    c0=mel[:,:,0:shift]; L0=torch.full((1,),shift,device=dev,dtype=torch.long)
    with torch.inference_mode(): o0=e.cache_aware_stream_step(processed_signal=c0,processed_signal_length=L0,cache_last_channel=cache[0].clone(),cache_last_time=cache[1].clone(),cache_last_channel_len=cache[2].clone(),keep_all_outputs=False,drop_extra_pre_encoded=0)
    c1=torch.cat((c0[:,:,-pre:],mel[:,:,shift:2*shift]),dim=-1); L1=torch.full((1,),c1.shape[-1],device=dev,dtype=torch.long)

    # attempt torch.export of the steady step
    try:
        with torch.inference_mode():
            ep=torch.export.export(Step(drop).cuda().eval(),(c1,L1,o0[2].clone(),o0[3].clone(),o0[4].clone()))
        steady_mod=ep.module()
        print("torch.export(steady): OK")
    except Exception as ex:
        print(f"torch.export(steady) FAILED: {type(ex).__name__}: {str(ex)[:300]}")
        print("=> T2a via torch.export not directly available; options: reimplement the streaming step natively, or stay T1.")
        return

    # validate byte-exact across the streaming sequence vs eager
    def estep(chunk,clc,clt,clcl,d):
        L=torch.full((1,),chunk.shape[-1],device=dev,dtype=torch.long)
        return e.cache_aware_stream_step(processed_signal=chunk,processed_signal_length=L,cache_last_channel=clc,cache_last_time=clt,cache_last_channel_len=clcl,keep_all_outputs=False,drop_extra_pre_encoded=d)
    clc,clt,clcl=cache[0].clone(),cache[1].clone(),cache[2].clone(); ring=None; emitted=0; pos=0; allok=True; maxd=0.0; nchk=0
    with torch.inference_mode():
        while pos<Tm:
            nm=mel[:,:,pos:pos+shift]
            if emitted==0:
                eo,el,clc,clt,clcl=estep(nm,clc,clt,clcl,0)  # first chunk eager (export only steady here)
            else:
                chunk=torch.cat((ring,nm),dim=-1); L=torch.full((1,),chunk.shape[-1],device=dev,dtype=torch.long)
                eg=estep(chunk,clc,clt,clcl,drop)
                ex_out=steady_mod(chunk,L,clc,clt,clcl)
                for x,y in zip(eg,ex_out):
                    if torch.is_tensor(x) and torch.is_tensor(y):
                        if not torch.equal(x,y): allok=False; maxd=max(maxd,(x.float()-y.float()).abs().max().item())
                eo,el,clc,clt,clcl=eg; nchk+=1
            ring=(torch.cat((ring,nm),dim=-1) if ring is not None else nm)[:,:,-pre:]; emitted+=nm.shape[-1]; pos+=shift
    print(f"[T2a] torch.export steady vs eager across {nchk} steady chunks: byte-exact={allok} max_abs_diff={maxd:.3e}")
    if allok:
        torch.export.save(ep, os.path.join(a.out,"enc_steady_t2a.pt2"))
        # I/O fixture for the container AOTI byte-exact check (container has torch, no nemo): real steady-chunk inputs +
        # eager 5-output reference. Use chunk1's geometry (the export's example) for a representative steady step.
        with torch.inference_mode():
            ref = estep(c1, o0[2].clone(), o0[3].clone(), o0[4].clone(), drop)
        torch.save({"chunk":c1.cpu(),"L":L1.cpu(),"clc":o0[2].cpu(),"clt":o0[3].cpu(),"clcl":o0[4].cpu(),
                    "out":[t.cpu() for t in ref]}, os.path.join(a.out,"t2a_io.pt"))
        print("=== T2a: BYTE-EXACT streaming encoder via torch.export -> saved enc_steady_t2a.pt2 + t2a_io.pt")
    else:
        print("=== T2a: torch.export NOT byte-exact across cache_len (same class as trace) -> reimplement the step")

if __name__=="__main__":
    main()
