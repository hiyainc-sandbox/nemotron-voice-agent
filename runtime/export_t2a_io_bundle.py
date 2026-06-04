#!/usr/bin/env python3
"""Pack the t2a_io fixture (5 steady-chunk inputs + 5 eager-reference outputs) into a scripted module the C++ AOTI
harness (cpp/aoti_encoder_main.cpp) loads via torch::jit::load. Must be a real file (torch.jit.script needs source).
Run in-container: python3 export_t2a_io_bundle.py
"""
import os, torch

ART = os.path.join(os.path.dirname(__file__), "artifacts")

class IO(torch.nn.Module):
    def __init__(self, d, aoti_out):
        super().__init__()
        self.register_buffer("chunk", d["chunk"]); self.register_buffer("L", d["L"])
        self.register_buffer("clc", d["clc"]); self.register_buffer("clt", d["clt"]); self.register_buffer("clcl", d["clcl"])
        for i, o in enumerate(d["out"]):
            self.register_buffer(f"out{i}", o)               # eager reference (T2a target)
        for i, o in enumerate(aoti_out):
            self.register_buffer(f"aoti{i}", o)              # Python aoti_load_package output (the C++==Python target)

    def forward(self):
        return self.chunk

def main():
    io = torch.load(os.path.join(ART, "t2a_io.pt"), weights_only=False)
    # capture Python's AOTI output so C++ can byte-for-byte verify it reproduces the Python runtime exactly
    runner = torch._inductor.aoti_load_package(os.path.join(ART, "enc_steady_aoti.pt2"))
    ins = [io["chunk"].cuda(), io["L"].cuda(), io["clc"].cuda(), io["clt"].cuda(), io["clcl"].cuda()]
    with torch.inference_mode():
        out = runner(*ins)
    aoti_out = [t.detach().cpu() for t in (out if isinstance(out, (list, tuple)) else [out])]
    m = torch.jit.script(IO(io, aoti_out))
    outp = os.path.join(ART, "t2a_io_bundle.ts")
    m.save(outp)
    print("wrote", outp, "(eager-ref out0..4 + python-aoti aoti0..4)")

if __name__ == "__main__":
    main()
