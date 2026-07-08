#!/usr/bin/env python3
"""0.6a — capture NeMo greedy_batch label-looping decode I/O as GOLDEN FIXTURES + characterize the contract.

The native C++ decode (B1b) must reproduce NeMo's decode BYTE/STATE-exact given identical encoder outputs. This dumps
(encoder_output -> Hypothesis + decoder state) fixtures for the deployed config (greedy_batch, loop_labels=True,
use_cuda_graph_decoder=False, max_symbols=10) and prints the state structure the native impl must match. Synthetic audio
is fine — the decode ALGORITHM is exercised identically; realistic/streaming-continuation fixtures are a follow-up.

Run: ./.venv/bin/python capture_decode_fixtures.py --out ./fixtures
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch
from omegaconf import OmegaConf

from model_profile import apply_prompt, get_profile, load_profile_model

PROFILE = get_profile()

def describe(x, name):
    if torch.is_tensor(x): return f"{name}: Tensor {tuple(x.shape)} {x.dtype} {x.device}"
    if isinstance(x,(list,tuple)): return f"{name}: {type(x).__name__}[{len(x)}]" + (" -> "+describe(x[0],name+"[0]") if x else "")
    return f"{name}: {type(x).__name__}={x!r}" if not hasattr(x,'__dict__') else f"{name}: {type(x).__name__} fields={list(vars(x).keys())}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=f"./{PROFILE.fixtures_dirname}")
    ap.add_argument("--wav", nargs="*", default=[], help="16 kHz wav files captured as decode_real_N fixtures")
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    model = load_profile_model(PROFILE)
    # Deployed decode config.
    cfg = OmegaConf.create({"strategy":"greedy_batch",
                            "greedy":{"max_symbols":PROFILE.max_symbols,"loop_labels":True,"use_cuda_graph_decoder":False}})
    model.change_decoding_strategy(decoding_cfg=cfg)
    print("decoding:", type(model.decoding).__name__,
          "infer:", type(getattr(model.decoding,'decoding',None)).__name__)

    rng = np.random.default_rng(0)
    cases = {"noise_3s": rng.standard_normal(48000).astype(np.float32)*0.05,
             "noise_loud_3s": rng.standard_normal(48000).astype(np.float32)*0.3,
             "quiet_1s": rng.standard_normal(16000).astype(np.float32)*0.005}
    if a.wav:
        import soundfile as sf
        for i, wav_path in enumerate(a.wav):
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1: audio = audio[:, 0]
            assert sr == 16000, f"{wav_path}: expected 16 kHz, got {sr}"
            cases[f"real_{i}"] = np.ascontiguousarray(audio)
    manifest={}
    for name, audio in cases.items():
        sig = torch.tensor(audio, device="cuda").unsqueeze(0)
        length = torch.tensor([audio.shape[0]], device="cuda")
        with torch.inference_mode():
            proc, proc_len = model.preprocessor(input_signal=sig, length=length)
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)   # [B, D, T]
            enc = apply_prompt(model, enc)
            hyps = model.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=True)
        h = hyps[0]
        if name=="noise_3s":
            print("=== Hypothesis structure (native decode must reproduce) ===")
            print(describe(h,"hyp"))
            for f in ("y_sequence","text","score","timestamp","dec_state","alignments","length"):
                if hasattr(h,f): print("  ", describe(getattr(h,f),f))
        ys = h.y_sequence
        ys = ys.tolist() if torch.is_tensor(ys) else list(ys)
        rec = {"name":name, "enc_shape":list(enc.shape), "enc_len":int(enc_len[0]),
               "text":getattr(h,"text",None), "y_sequence":ys, "n_tokens":len(ys),
               "score":float(getattr(h,"score",0.0) or 0.0)}
        manifest[name]=rec
        torch.save({"enc":enc.cpu(),"enc_len":enc_len.cpu(),"y_sequence":h.y_sequence,
                    "text":getattr(h,"text",None)}, os.path.join(a.out,f"decode_{name}.pt"))
        print(f"[{name}] enc{list(enc.shape)} len={int(enc_len[0])} -> {len(ys)} tokens, text={getattr(h,'text','')!r}")
    with open(os.path.join(a.out,"manifest.json"),"w") as f: json.dump(manifest,f,indent=2)
    print(f"saved fixtures + manifest to {a.out}")

if __name__=="__main__":
    main()
