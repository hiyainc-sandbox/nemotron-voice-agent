#!/usr/bin/env python3
"""1.3b-enc shared-weights source: save the encoder's params+buffers keyed by the AOTI constant FQN
('encoder.<name>', matching export_finalize_t2a.FinalizeStep.encoder and built bucket get_constant_fqns()) so per-T
finalize buckets (constants-on-disk) share ONE weight set via loader.load_constants(user_managed=True).

Saves BOTH a .pt (Python consumers) and a TorchScript .ts holding a Dict[str,Tensor] attribute "weights" keyed by FQN
(C++ consumers: FQNs have dots so they can't be buffer names). Older artifacts used "e.<name>"; consumers keep an
e.* <-> encoder.* compatibility fallback, but the primary exported contract is encoder.*.

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python export_shared_weights.py
"""
from __future__ import annotations

import argparse
import os, itertools
from typing import Dict
import torch

ART = "artifacts"
PT = os.path.join(ART, "finalize_shared_weights.pt")
TS = os.path.join(ART, "finalize_shared_weights.ts")


class SharedWeights(torch.nn.Module):
    def __init__(self, cmap: Dict[str, torch.Tensor]):
        super().__init__()
        self.weights: Dict[str, torch.Tensor] = cmap

    def forward(self) -> Dict[str, torch.Tensor]:
        return self.weights


def normalize_existing_cmap(path: str) -> Dict[str, torch.Tensor]:
    """Compatibility escape hatch for old e.* files; not the default production path."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} did not contain a Dict[str, Tensor]")
    cmap: Dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError(f"{path} has a non string/tensor item: {type(key)} {type(value)}")
        if key.startswith("encoder."):
            out_key = key
        elif key.startswith("e."):
            out_key = "encoder." + key[len("e.") :]
        else:
            out_key = key
        cmap[out_key] = value.detach().cpu()
    return cmap


def build_cmap(*, reuse_existing: bool = False) -> Dict[str, torch.Tensor]:
    if reuse_existing:
        if not os.path.exists(PT):
            raise FileNotFoundError(f"--reuse-existing requested but {PT} does not exist")
        return normalize_existing_cmap(PT)

    from finalize_ref import load_model
    enc = load_model().encoder
    cmap = {}
    for name, t in itertools.chain(enc.named_parameters(), enc.named_buffers()):
        cmap["encoder." + name] = t.detach().cpu()
    return cmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="compatibility-only: rewrite an existing .pt to encoder.* keys without reloading the model",
    )
    args = parser.parse_args()

    os.makedirs(ART, exist_ok=True)
    cmap = build_cmap(reuse_existing=args.reuse_existing)
    torch.save(cmap, PT)
    sm = torch.jit.script(SharedWeights(dict(cmap)))
    sm.save(TS)
    encoder_keys = sum(1 for key in cmap if key.startswith("encoder."))
    e_keys = sum(1 for key in cmap if key.startswith("e."))
    print(
        f"saved {len(cmap)} weights -> finalize_shared_weights.pt + .ts "
        f"(encoder.*={encoder_keys}, e.*={e_keys})"
    )


if __name__ == "__main__":
    main()
