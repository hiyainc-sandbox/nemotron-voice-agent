#!/usr/bin/env python3
"""Action D (in the CUDA container) — AOTInductor-compile the T2a steady-encoder ExportedProgram (.pt2) and validate
the compiled artifact is byte-exact vs the eager reference (t2a_io.pt). Container has torch + nvcc + GPU, no nemo (the
.pt2 is self-contained). Run: ./container/enter.sh python3 aot_compile.py  (or docker run ... python3 aot_compile.py)
"""
from __future__ import annotations
import os, torch

ART = os.environ.get(
    "NEMOTRON_ART_DIR", os.path.join(os.path.dirname(__file__), "artifacts")
)

def _force_noexecstack_on_link():
    """Fix the BUILD (not the binary): the AOTI wrapper.so links an input (embedded constants/cubin blob) without a
    .note.GNU-stack, so ld marks the output stack RWX and a hardened host kernel refuses to dlopen it ("cannot enable
    executable stack"). (Root cause asserted from build behaviour; readelf appendix TODO.) Inductor exposes no ldflags
    config knob. Append -Wl,-z,noexecstack to the SHARED-LIB LINK command only (gated on '-shared', so Inductor's
    compile/ISA probes are untouched); -z noexecstack forces the output PT_GNU_STACK non-exec regardless of any input
    object's missing note. FAIL-CLOSED: assert the flag actually reached a shared-lib link, else raise (so a silent
    Inductor refactor can't ship an exec-stack .so). (constants-on-disk was tried+rejected: didn't clear the flag AND
    segfaulted on a fresh aoti_load_package.)"""
    import torch._inductor.cpp_builder as cb
    orig = cb.CppBuilder.get_command_line
    seen = {"shared_link": False, "flagged": False}
    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd:
            seen["shared_link"] = True
            if "-Wl,-z,noexecstack" not in cmd:
                cmd += " -Wl,-z,noexecstack"
            seen["flagged"] = True
            print("[noexecstack] injected into shared-lib link:", cmd[-120:])
        return cmd
    cb.CppBuilder.get_command_line = patched
    return seen  # caller asserts seen["flagged"] after the compile

def main():
    seen = _force_noexecstack_on_link()
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability())
    ep = torch.export.load(os.path.join(ART, "enc_steady_t2a.pt2"))
    print("loaded ExportedProgram")

    # AOTInductor compile -> package (.pt2 with the .so for CUDA sm_120); constants embedded in the .so (default; the
    # link flag above makes the stack non-exec).
    pkg_path = os.path.join(ART, "enc_steady_aoti.pt2")
    out_path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg_path)
    print("AOTI package:", out_path)
    # fail-closed: the noexecstack flag MUST have reached a real shared-lib link
    assert seen["flagged"], "noexecstack shim never fired on a shared-lib link — Inductor build path changed; refusing to ship a possibly exec-stack .so"

    runner = torch._inductor.aoti_load_package(out_path)
    io = torch.load(os.path.join(ART, "t2a_io.pt"), weights_only=False)
    ins = [io["chunk"].cuda(), io["L"].cuda(), io["clc"].cuda(), io["clt"].cuda(), io["clcl"].cuda()]
    with torch.inference_mode():
        out = runner(*ins)
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    ref = [t.cuda() for t in io["out"]]
    names = ["enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"]
    allok = True; maxd = 0.0
    for n, a, b in zip(names, ref, outs):
        if torch.is_tensor(a) and torch.is_tensor(b):
            eq = (a.shape == b.shape) and torch.equal(a, b)
            d = (a.float() - b.float()).abs().max().item() if a.shape == b.shape and a.numel() else float("nan")
            allok &= eq; maxd = max(maxd, d if d == d else 0.0)
            print(f"  {n}: byte-equal={eq} max_abs_diff={d:.3e} shapes {tuple(a.shape)}/{tuple(b.shape)}")
    print(f"=== AOTI vs eager: {'BYTE-EXACT' if allok else f'NOT byte-exact (maxdiff {maxd:.3e})'} ===")

if __name__ == "__main__":
    main()
