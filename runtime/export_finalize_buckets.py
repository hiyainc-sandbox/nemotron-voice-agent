#!/usr/bin/env python3
"""1.3b-enc-build (host side): torch.export the finalize encoder at FIXED exact-T per (drop_extra, T) bucket needed by the
fixture (drop=2 continuation T={44,45,58}; drop=0 first-chunk T=49). Fixed-shape export is byte-exact (like the steady T2a)
and EXACT-T = no padding bleed. Saves one ExportedProgram per bucket; aot_compile_buckets.py AOTI-compiles them
constants-on-disk in the container so they share ONE weight set (validate_shared_weights.py).

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python export_finalize_buckets.py --out ./artifacts/finalize_buckets
"""
from __future__ import annotations
import argparse, json, os, torch
from finalize_ref import load_model
from export_finalize_t2a import FinalizeStep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts/finalize_buckets")
    ap.add_argument("--fixture", default="./artifacts/finalize_fixture.pt"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load_model()
    rows = torch.load(a.fixture, weights_only=False)["rows"]

    # distinct (drop, T) buckets from the fixture
    seen = {}
    for r in rows:
        cm = r["chunk_mel"]; T = int(cm.shape[-1]); drop = int(r["drop_extra"])
        seen.setdefault((drop, T), r)

    manifest = []
    for (drop, T), r in sorted(seen.items()):
        chunk = r["chunk_mel"].cuda()
        clc = r["cache_last_channel"].cuda(); clt = r["cache_last_time"].cuda(); clcl = r["cache_last_channel_len"].cuda()
        step = FinalizeStep(model.encoder, drop_extra=drop).cuda().eval()
        with torch.inference_mode():
            ep = torch.export.export(step, (chunk, clc, clt, clcl))  # FIXED shape -> no dynamic-T constraints
            # sanity: exported == eager byte-exact at this fixed shape
            eager = step(chunk, clc, clt, clcl); exp = ep.module()(chunk, clc, clt, clcl)
            be = all(torch.equal(e, x) for e, x in zip(eager, exp) if torch.is_tensor(e))
        name = f"enc_finalize_d{drop}_T{T}_ep.pt2"
        torch.export.save(ep, os.path.join(a.out, name))
        manifest.append({"drop": drop, "T": T, "ep": name})
        print(f"  bucket drop={drop} T={T}: export byte_exact={be} -> {name}")
    json.dump(manifest, open(os.path.join(a.out, "buckets_manifest.json"), "w"), indent=2)
    print(f"=== exported {len(manifest)} finalize buckets -> {a.out}/buckets_manifest.json ===")


if __name__ == "__main__":
    main()
