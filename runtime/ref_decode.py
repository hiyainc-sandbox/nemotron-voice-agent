#!/usr/bin/env python3
"""1.1a — VERIFIED Python reference of the RNNT greedy decode (the executable spec for the C++ port).

Reimplements classic RNNT greedy (frame-loop + inner symbol-loop, max_symbols, blank advance) using ONLY
model.decoder (prediction net) + model.joint (joint net) — NOT NeMo's decode. Validates it produces BYTE-IDENTICAL
y_sequence to NeMo's deployed greedy_batch label-looping decode on every golden fixture. If it matches, we've proven
the algorithm exactly → the spec for B1b's C++ decode. (Greedy label-looping and frame-looping yield the same greedy
argmax sequence; this is the simplest correct reference.)

Run: ./.venv/bin/python ref_decode.py
"""
from __future__ import annotations
import glob, os, torch
import nemo.collections.asr as nemo_asr

BLANK = 1024
MAX_SYMBOLS = 10

@torch.inference_mode()
def ref_greedy_range(decoder, joint, f, t0, t1, state, g):
    """Decode encoder frames [t0,t1) of f:[1,T,1024], carrying (LSTM state, pred output g). RESUMABLE — this is the
    streaming-continuation contract the C++ decode must match. Returns (tokens, state, g)."""
    hyp = []
    for t in range(t0, t1):
        f_t = f[:, t:t+1, :]                        # [1,1,1024]
        n_sym = 0
        while n_sym < MAX_SYMBOLS:
            logits = joint.joint(f_t, g)            # [1,1,1,1025]
            k = int(logits.reshape(-1).argmax().item())
            if k == BLANK:
                break
            hyp.append(k)
            y = torch.full((1, 1), k, dtype=torch.long, device=f.device)
            g, state = decoder.predict(y, state, add_sos=False, batch_size=1)
            n_sym += 1
    return hyp, state, g

@torch.inference_mode()
def ref_greedy(decoder, joint, enc, enc_len):
    """Full-utterance decode (fresh state). enc: [1,1024,T]."""
    f = enc.transpose(1, 2).contiguous()
    state = decoder.initialize_state(torch.zeros(1, 1, dtype=torch.float32, device=enc.device))
    g, state = decoder.predict(None, state, add_sos=False, batch_size=1)
    hyp, _, _ = ref_greedy_range(decoder, joint, f, 0, int(enc_len[0]), state, g)
    return hyp

@torch.inference_mode()
def ref_greedy_streamed(decoder, joint, enc, enc_len, n_chunks=2):
    """Decode in n_chunks carrying state across them (the streaming partial-hyp continuation). Returns concatenated tokens."""
    f = enc.transpose(1, 2).contiguous(); T = int(enc_len[0])
    state = decoder.initialize_state(torch.zeros(1, 1, dtype=torch.float32, device=enc.device))
    g, state = decoder.predict(None, state, add_sos=False, batch_size=1)
    bounds = [round(T * i / n_chunks) for i in range(n_chunks + 1)]
    out = []
    for i in range(n_chunks):
        toks, state, g = ref_greedy_range(decoder, joint, f, bounds[i], bounds[i+1], state, g)
        out += toks
    return out

def main():
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    decoder, joint = model.decoder, model.joint
    fixtures = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "decode_*.pt")))
    n_pass = n_total = 0
    for fp in fixtures:
        d = torch.load(fp, weights_only=False)
        enc = d["enc"].cuda(); enc_len = d["enc_len"].cuda()
        gold = d["y_sequence"]; gold = gold.tolist() if torch.is_tensor(gold) else list(gold)
        got = ref_greedy(decoder, joint, enc, enc_len)
        ok = (got == gold)
        n_total += 1; n_pass += int(ok)
        name = os.path.basename(fp)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: ref {len(got)} tok vs gold {len(gold)} tok"
              + ("" if ok else f"\n   gold[:20]={gold[:20]}\n   ref [:20]={got[:20]}"))
    print(f"\n=== {n_pass}/{n_total} fixtures byte-exact (full-utterance vs NeMo) ===")
    # 1.1b: streaming-continuation self-consistency (state carried across chunks == full decode -> transitively == NeMo)
    print("--- streaming-continuation (state-carry across chunks) ---")
    n_cpass = n_ctot = 0
    for fp in fixtures:
        d = torch.load(fp, weights_only=False); enc = d["enc"].cuda(); enc_len = d["enc_len"].cuda()
        full = ref_greedy(decoder, joint, enc, enc_len)
        for nc in (2, 3, 5):
            streamed = ref_greedy_streamed(decoder, joint, enc, enc_len, n_chunks=nc)
            ok = (streamed == full); n_ctot += 1; n_cpass += int(ok)
            if not ok:
                print(f"   [FAIL] {os.path.basename(fp)} nchunks={nc}: streamed != full")
    print(f"=== {n_cpass}/{n_ctot} continuation checks pass (2/3/5-chunk carry == full) ===")
    ok_all = (n_pass == n_total and n_cpass == n_ctot)
    print("VERIFIED REFERENCE (full + streaming-continuation)" if ok_all else "DIVERGENCE — investigate before C++ port")

if __name__ == "__main__":
    main()
