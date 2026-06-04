#!/usr/bin/env python3
"""1.2b-pre — export the FULL pipeline modules (preprocessor + full encoder) and bundle a real clip's audio + gold
tokens, so a C++ program can run audio -> preproc -> encoder -> decode end-to-end and self-check BYTE-EXACT. (Full
non-streaming encoder, matching how the fixtures were made — proves the C++ chain composes byte-exact on real speech.
The streaming cache-aware chunk loop is the separate 1.2b proper.)

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python export_pipeline.py --out ./artifacts
"""
from __future__ import annotations
import argparse, io, os, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    try: m.preprocessor.featurizer.dither = 0.0
    except Exception: pass
    m.change_decoding_strategy(decoding_cfg=OmegaConf.create(
        {"strategy":"greedy_batch","greedy":{"max_symbols":10,"loop_labels":True,"use_cuda_graph_decoder":False}}))
    dev = next(m.parameters()).device

    # real clip
    import datasets
    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column("audio", datasets.Audio(decode=False))
    ex = ds[1]   # real_1: "How much juice is in one lime?"
    wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000:
        n = int(len(wav)*16000/sr); wav = np.interp(np.linspace(0,len(wav),n,endpoint=False), np.arange(len(wav)), wav).astype(np.float32)
    audio = torch.tensor(wav, device=dev).unsqueeze(0); length = torch.tensor([wav.shape[0]], device=dev)

    # gold (full pipeline) for the self-check
    with torch.inference_mode():
        proc, proc_len = m.preprocessor(input_signal=audio, length=length)
        enc, enc_len = m.encoder(audio_signal=proc, length=proc_len)
        hyps = m.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=True)
    gold = hyps[0].y_sequence; gold = gold if torch.is_tensor(gold) else torch.tensor(gold)
    print(f"gold: {len(gold)} tokens, text={hyps[0].text!r}, enc{list(enc.shape)}")

    # export preprocessor + full encoder
    class PP(torch.nn.Module):
        def __init__(s): super().__init__(); s.pp=m.preprocessor
        def forward(s, sig, length): return s.pp(input_signal=sig, length=length)
    class ENC(torch.nn.Module):
        def __init__(s): super().__init__(); s.e=m.encoder
        def forward(s, proc, proc_len): return s.e(audio_signal=proc, length=proc_len)
    with torch.inference_mode():
        pp_ts = torch.jit.trace(PP().cuda().eval(), (audio, length), check_trace=False)
        enc_ts = torch.jit.trace(ENC().cuda().eval(), (proc, proc_len), check_trace=False)
    pp_ts.save(os.path.join(a.out,"preproc.ts")); enc_ts.save(os.path.join(a.out,"encoder_full.ts"))

    class Bundle(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.register_buffer("audio", audio.cpu()); s.register_buffer("alen", length.cpu().to(torch.int64))
            s.register_buffer("gold", gold.cpu().to(torch.int64))
        def forward(s): return s.audio
    torch.jit.script(Bundle()).save(os.path.join(a.out,"pipeline_bundle.ts"))
    print(f"exported preproc.ts + encoder_full.ts + pipeline_bundle.ts ({len(wav)/16000:.1f}s audio, {len(gold)} gold tokens)")

if __name__ == "__main__":
    main()
