#!/usr/bin/env python3
"""1.2a-pre — export the decode modules (pred + joint) to TorchScript and validate the exported-module greedy decode
is BYTE-EXACT vs the fixtures. This proves the decode path is exportable -> the C++ port (1.2a) loads the SAME .ts and
implements the (verified, simple) loop. The SOS pred-output is precomputed once (avoids tracing the y=None branch).

Run: ./.venv/bin/python export_decode.py --out ./artifacts
"""
from __future__ import annotations
import argparse, glob, os, torch
import nemo.collections.asr as nemo_asr
from model_profile import get_profile, load_profile_model

PROFILE = get_profile()
BLANK, MAX_SYMBOLS = PROFILE.blank, PROFILE.max_symbols

class JointStep(torch.nn.Module):
    def __init__(self, joint): super().__init__(); self.joint = joint
    def forward(self, f, g):                      # f:[1,1,1024], g:[1,1,640] -> [1,1,1,1025]
        return self.joint.joint(f, g)

class PromptApply(torch.nn.Module):
    """Post-encoder language conditioning for prompted (multilingual) models.

    Mirrors PromptStreamingMixin._apply_prompt_to_encoded: enc [B,D,T] is
    transposed, concatenated per-frame with a [B,num_prompts] one-hot, passed
    through the model's prompt_kernel MLP, and transposed back. Scripted (not
    traced) so B and T stay dynamic."""
    def __init__(self, kernel, num_prompts: int):
        super().__init__()
        self.kernel = kernel
        self.num_prompts = num_prompts
    def forward(self, enc, prompt):               # enc:[B,D,T], prompt:[B,P] -> [B,D,T]
        x = enc.transpose(1, 2)                   # [B,T,D]
        p = prompt.unsqueeze(1).expand(x.shape[0], x.shape[1], self.num_prompts)
        y = self.kernel(torch.cat([x, p], dim=-1))
        return y.transpose(1, 2).contiguous()

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
    model = load_profile_model(PROFILE)
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

    if PROFILE.prompted:
        pa_ts = torch.jit.script(PromptApply(model.prompt_kernel, PROFILE.num_prompts).cuda().eval())
        # validate byte-exact vs NeMo's _apply_prompt_to_encoded across shapes/batch/prompt ids
        d_model = int(model.cfg.encoder.d_model)
        pa_ok = True
        with torch.inference_mode():
            for B, T in ((1, 2), (1, 4), (2, 4), (4, 33), (1, 170)):
                enc = torch.randn(B, d_model, T, device="cuda")
                for pid in (0, 1, PROFILE.num_prompts - 1):
                    model._inference_prompt_index = pid
                    ref = model._apply_prompt_to_encoded(enc.clone())
                    onehot = torch.zeros(B, PROFILE.num_prompts, device="cuda")
                    onehot[:, pid] = 1.0
                    got = pa_ts(enc.clone(), onehot)
                    if not torch.equal(ref, got):
                        pa_ok = False
                        print(f"[FAIL] prompt_apply B={B} T={T} pid={pid} "
                              f"max_abs={(ref - got).abs().max().item():.3e}")
        if not pa_ok:
            raise SystemExit("prompt_apply.ts is not byte-exact vs NeMo _apply_prompt_to_encoded")
        model.set_inference_prompt(PROFILE.default_target_lang)  # restore
        pa_ts.save(os.path.join(a.out, "prompt_apply.ts"))
        print(f"exported prompt_apply.ts (num_prompts={PROFILE.num_prompts}, byte-exact across shapes/prompts)")

    # validate exported-module decode == fixtures (byte-exact)
    jt = torch.jit.load(os.path.join(a.out, "joint_step.ts")); pt = torch.jit.load(os.path.join(a.out, "predict_step.ts"))
    fixtures = sorted(glob.glob(os.path.join(os.path.dirname(__file__), PROFILE.fixtures_dirname, "decode_*.pt")))
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
    real_fixtures = sorted(glob.glob(os.path.join(os.path.dirname(__file__), PROFILE.fixtures_dirname, "decode_real_*.pt")))
    if not real_fixtures:
        raise SystemExit(f"no decode_real_* fixture in {PROFILE.fixtures_dirname}/ (run capture_decode_fixtures.py --wav ...)")
    preferred = os.path.join(os.path.dirname(__file__), PROFILE.fixtures_dirname, "decode_real_1.pt")
    rf = preferred if preferred in real_fixtures else real_fixtures[0]
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
