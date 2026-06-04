#!/usr/bin/env python3
"""1.2a-pre — export the decode modules (pred + joint) to TorchScript and validate the exported-module greedy decode
is BYTE-EXACT vs the fixtures. This proves the decode path is exportable -> the C++ port (1.2a) loads the SAME .ts and
implements the (verified, simple) loop. The SOS pred-output is precomputed once (avoids tracing the y=None branch).

Run: ./.venv/bin/python export_decode.py --out ./artifacts
"""
from __future__ import annotations
import argparse, glob, os, torch
import nemo.collections.asr as nemo_asr

BLANK, MAX_SYMBOLS = 1024, 10

class JointStep(torch.nn.Module):
    def __init__(self, joint): super().__init__(); self.joint = joint
    def forward(self, f, g):                      # f:[1,1,1024], g:[1,1,640] -> [1,1,1,1025]
        return self.joint.joint(f, g)

class PredictStep(torch.nn.Module):
    def __init__(self, decoder): super().__init__(); self.decoder = decoder
    def forward(self, y, h, c):                   # y:[1,1] long, (h,c):[2,1,640] -> (g[1,1,640], h,c)
        g, (nh, nc) = self.decoder.predict(y, (h, c), add_sos=False, batch_size=1)
        return g, nh, nc

@torch.inference_mode()
def decode_with_modules(jt, pt, sos_g, init_h, init_c, enc, enc_len):
    f = enc.transpose(1, 2).contiguous()
    g, h, c = sos_g, init_h, init_c
    hyp = []
    for t in range(int(enc_len[0])):
        f_t = f[:, t:t+1, :]
        for _ in range(MAX_SYMBOLS):
            k = int(jt(f_t, g).reshape(-1).argmax().item())
            if k == BLANK: break
            hyp.append(k)
            y = torch.full((1, 1), k, dtype=torch.long, device=enc.device)
            g, h, c = pt(y, h, c)
    return hyp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    dec, joint = model.decoder, model.joint

    # initial (SOS) pred output + state — precomputed constants the C++ runtime ships.
    init_state = dec.initialize_state(torch.zeros(1, 1, dtype=torch.float32, device="cuda"))
    sos_g, (init_h, init_c) = dec.predict(None, init_state, add_sos=False, batch_size=1)

    jstep = JointStep(joint).cuda().eval(); pstep = PredictStep(dec).cuda().eval()
    ex_f = torch.zeros(1, 1, 1024, device="cuda"); ex_g = sos_g.clone()
    ex_y = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        jt = torch.jit.trace(jstep, (ex_f, ex_g), check_trace=False)
        pt = torch.jit.trace(pstep, (ex_y, init_h, init_c), check_trace=False)
    jt.save(os.path.join(a.out, "joint_step.ts")); pt.save(os.path.join(a.out, "predict_step.ts"))
    torch.save({"sos_g": sos_g.cpu(), "init_h": init_h.cpu(), "init_c": init_c.cpu()},
               os.path.join(a.out, "decode_init.pt"))
    print("exported joint_step.ts + predict_step.ts + decode_init.pt")

    # validate exported-module decode == fixtures (byte-exact)
    jt = torch.jit.load(os.path.join(a.out, "joint_step.ts")); pt = torch.jit.load(os.path.join(a.out, "predict_step.ts"))
    fixtures = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "decode_*.pt")))
    npass = ntot = 0
    for fp in fixtures:
        d = torch.load(fp, weights_only=False); enc = d["enc"].cuda(); enc_len = d["enc_len"].cuda()
        gold = d["y_sequence"]; gold = gold.tolist() if torch.is_tensor(gold) else list(gold)
        got = decode_with_modules(jt, pt, sos_g, init_h, init_c, enc, enc_len)
        ok = (got == gold); ntot += 1; npass += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {os.path.basename(fp)}: exported {len(got)} vs gold {len(gold)}")
    print(f"=== {npass}/{ntot} exported-module decode byte-exact ===")
    print("EXPORT VERIFIED — decode path is C++-portable" if npass == ntot else "EXPORT DIVERGENCE")

    # self-contained C++ test bundle: init constants + the real_1 fixture (enc/gold) as named buffers (C++ reads via
    # bundle.attr(name)). Lets decode_main.cpp run + self-check byte-exact without python-pickle marshalling.
    rf = os.path.join(os.path.dirname(__file__), "fixtures", "decode_real_1.pt")
    fx = torch.load(rf, weights_only=False)
    gold = fx["y_sequence"]; gold = gold if torch.is_tensor(gold) else torch.tensor(gold)
    class Bundle(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.register_buffer("sos_g", sos_g.cpu()); s.register_buffer("init_h", init_h.cpu())
            s.register_buffer("init_c", init_c.cpu()); s.register_buffer("enc", fx["enc"].cpu())
            s.register_buffer("enc_len", fx["enc_len"].cpu().to(torch.int64))
            s.register_buffer("gold", gold.cpu().to(torch.int64))
        def forward(s): return s.sos_g
    torch.jit.script(Bundle()).save(os.path.join(a.out, "cpp_bundle.ts"))
    print(f"wrote cpp_bundle.ts (real_1: {int(fx['enc_len'][0])} frames, {len(gold)} gold tokens) for the C++ decode")

if __name__ == "__main__":
    main()
