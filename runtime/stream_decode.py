#!/usr/bin/env python3
"""1.2b-pre (Python) — streaming pipeline: cache-aware encoder chunk loop (server.py logic: first chunk drop=0; steady
chunk = cat(mel_ring[9], new_mel[16]) drop=2; ring=last 9) + decode with state-carry. Validates MY decode composed in
the streaming loop is BYTE-EXACT vs NeMo's streaming decode (partial_hypotheses continuation) on the SAME encoder
outputs. Proves the streaming COMPOSITION (cache threading + decode state-carry) before the C++ port.

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python stream_decode.py
"""
from __future__ import annotations
import io, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from model_profile import apply_prompt, get_profile, load_profile_model
from ref_decode import ref_greedy_range  # reuse the verified resumable decode

def main():
    profile = get_profile()
    m = load_profile_model(profile)
    m.change_decoding_strategy(decoding_cfg=OmegaConf.create(
        {"strategy":"greedy_batch","greedy":{"max_symbols":profile.max_symbols,"loop_labels":True,"use_cuda_graph_decoder":False}}))
    e, dec, joint = m.encoder, m.decoder, m.joint
    sc = e.streaming_cfg
    _int = lambda v: int(v[1]) if isinstance(v,(list,tuple)) else int(v)
    shift, pre, drop = _int(sc.shift_size), _int(sc.pre_encode_cache_size), int(sc.drop_extra_pre_encoded)
    dev = next(m.parameters()).device

    import datasets
    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column("audio", datasets.Audio(decode=False))
    ex = ds[1]; wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000:
        n=int(len(wav)*16000/sr); wav=np.interp(np.linspace(0,len(wav),n,endpoint=False),np.arange(len(wav)),wav).astype(np.float32)
    audio = torch.tensor(wav, device=dev).unsqueeze(0); alen = torch.tensor([wav.shape[0]], device=dev)

    with torch.inference_mode():
        mel, _ = m.preprocessor(input_signal=audio, length=alen)   # [1,128,Tm]
    Tm = mel.shape[-1]

    cache = e.get_initial_cache_state(batch_size=1)
    clc, clt, clcl = cache[0].clone(), cache[1].clone(), cache[2].clone()
    ring = None
    # my decode state
    my_state = dec.initialize_state(torch.zeros(1,1,dtype=torch.float32,device=dev))
    g, my_state = dec.predict(None, my_state, add_sos=False, batch_size=1)
    my_tokens = []
    # nemo streaming decode state
    nemo_prev = None
    emitted = 0; pos = 0
    with torch.inference_mode():
        while pos < Tm:
            new_mel = mel[:, :, pos:pos+shift]
            if emitted == 0:
                chunk = new_mel; d = 0
            else:
                chunk = torch.cat((ring, new_mel), dim=-1); d = drop
            length = torch.full((1,), chunk.shape[-1], device=dev, dtype=torch.long)
            enc_out, enc_len_out, clc, clt, clcl = e.cache_aware_stream_step(
                processed_signal=chunk, processed_signal_length=length,
                cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                keep_all_outputs=False, drop_extra_pre_encoded=d)
            To = int(enc_len_out[0])
            enc_out = apply_prompt(m, enc_out)
            # MY decode (state-carry) over this chunk's encoder frames
            f = enc_out.transpose(1, 2).contiguous()
            toks, my_state, g = ref_greedy_range(dec, joint, f, 0, To, my_state, g)
            my_tokens += toks
            # NeMo streaming decode (partial_hypotheses continuation) — the oracle
            hyps = m.decoding.rnnt_decoder_predictions_tensor(
                enc_out, enc_len_out, return_hypotheses=True, partial_hypotheses=nemo_prev)
            nemo_prev = hyps
            ring = (torch.cat((ring, new_mel), dim=-1) if ring is not None else new_mel)[:, :, -pre:]
            emitted += new_mel.shape[-1]; pos += shift

    nemo_tokens = nemo_prev[0].y_sequence
    nemo_tokens = nemo_tokens.tolist() if torch.is_tensor(nemo_tokens) else list(nemo_tokens)
    ok = (my_tokens == nemo_tokens)
    print(f"streaming chunks={ (Tm + shift - 1)//shift } mel_frames={Tm}")
    print(f"MY   streaming: {len(my_tokens)} tok -> {m.tokenizer.ids_to_text(my_tokens)!r}")
    print(f"NeMo streaming: {len(nemo_tokens)} tok -> {nemo_prev[0].text!r}")
    print(f"=== streaming-composition byte-exact (my decode == NeMo streaming decode): {ok} ===")
    if not ok:
        print("  my  :", my_tokens[:30]); print("  nemo:", nemo_tokens[:30])

if __name__ == "__main__":
    main()
