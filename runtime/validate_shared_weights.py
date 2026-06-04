#!/usr/bin/env python3
"""1.3b-enc shared-weights UNLOCK (proven): AOTI constants-on-disk packages do NOT auto-wire weights
(plain aoti_load_package + run segfaults); loader.load_constants(cmap, user_managed=True) wires a SHARED weight set in.
This lets N per-exact-T finalize buckets (tiny wrapper .so each) share ONE 2.5GB weight set. Validates the loaded
constants-on-disk runner matches the constants-in-so result (1.66e-2 vs eager). Run in-container.
Needs: artifacts/enc_steady_codisk.pt2 (constants-on-disk build) + artifacts/finalize_shared_weights.pt (export_shared_weights.py).
"""
import os, torch, faulthandler; faulthandler.enable()
ART="artifacts"; PKG=os.path.join(ART,"enc_steady_codisk.pt2")
io=torch.load(os.path.join(ART,"t2a_io.pt"), weights_only=False)
ins=[io["chunk"].cuda(),io["L"].cuda(),io["clc"].cuda(),io["clt"].cuda(),io["clcl"].cuda()]
ref=[t.cuda() for t in io["out"]]
W=torch.load(os.path.join(ART,"finalize_shared_weights.pt"), weights_only=False)
m=torch._inductor.aoti_load_package(PKG)
fqns=m.loader.get_constant_fqns()
cmap={f:W[f].cuda() for f in fqns if f in W}
missing=[f for f in fqns if f not in W]
print(f"fqns={len(fqns)} matched={len(cmap)} missing={len(missing)} sample_missing={missing[:3]}")
if missing: raise SystemExit("missing fqns -> naming mismatch")
m.loader.load_constants(cmap, False, False, True)   # user_managed=True (SHARED, not copied)
print("load_constants OK; running...")
out=m(*ins)
maxd=max((a.float()-b.float()).abs().max().item() for a,b in zip(ref,out) if torch.is_tensor(a) and a.numel())
be=all(torch.equal(a,b) for a,b in zip(ref,out) if torch.is_tensor(a))
print(f"=== constants-on-disk + load_constants: RAN OK, vs eager-ref max_abs_diff={maxd:.3e} (expect ~1.66e-2, same as constants-in-so) byte_exact_vs_eager={be} ===")
