#!/usr/bin/env python3
"""Pack finalize_fixture.pt rows into a TorchScript bundle for the C++ finalize harness.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python export_finalize_bundle.py
"""
import os

import torch


ART = os.path.join(os.path.dirname(__file__), "artifacts")


def _as_cpu_tensor(value):
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value).__name__}")
    return value.detach().cpu().clone()


class FinalizeBundle(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("num_rows", torch.tensor([len(rows)], dtype=torch.int64))
        self.register_buffer(
            "meta",
            torch.tensor([len(rows), 1024, 10, 16, 9, 2], dtype=torch.int64),
        )

        for i, row in enumerate(rows):
            prefix = f"row{i}"
            self.register_buffer(f"{prefix}_chunk_mel", _as_cpu_tensor(row["chunk_mel"]))
            self.register_buffer(f"{prefix}_chunk_len", _as_cpu_tensor(row["chunk_len"]))
            self.register_buffer(
                f"{prefix}_cache_last_channel",
                _as_cpu_tensor(row["cache_last_channel"]),
            )
            self.register_buffer(
                f"{prefix}_cache_last_time",
                _as_cpu_tensor(row["cache_last_time"]),
            )
            self.register_buffer(
                f"{prefix}_cache_last_channel_len",
                _as_cpu_tensor(row["cache_last_channel_len"]),
            )

            enc_out, enc_len = row["eager_outputs"][0], row["eager_outputs"][1]
            self.register_buffer(f"{prefix}_eager_enc_out", _as_cpu_tensor(enc_out))
            self.register_buffer(f"{prefix}_eager_enc_len", _as_cpu_tensor(enc_len))

            h, c = row["pre_final_decoder_state"]
            self.register_buffer(f"{prefix}_pre_final_h", _as_cpu_tensor(h))
            self.register_buffer(f"{prefix}_pre_final_c", _as_cpu_tensor(c))
            self.register_buffer(
                f"{prefix}_pre_final_pred_out",
                _as_cpu_tensor(row["pre_final_pred_out"]),
            )
            self.register_buffer(
                f"{prefix}_pre_final_tokens",
                _as_cpu_tensor(row["pre_final_tokens"]).to(torch.int64),
            )
            self.register_buffer(
                f"{prefix}_finalize_ref_final_tokens",
                _as_cpu_tensor(row["finalize_ref_final_tokens"]).to(torch.int64),
            )
            self.register_buffer(
                f"{prefix}_nemo_stream_finalize_tokens",
                _as_cpu_tensor(row["nemo_stream_finalize_tokens"]).to(torch.int64),
            )
            self.register_buffer(
                f"{prefix}_drop_extra",
                torch.tensor([int(row["drop_extra"])], dtype=torch.int64),
            )
            self.register_buffer(
                f"{prefix}_emitted_gt0",
                torch.tensor([int(row["drop_extra"]) != 0], dtype=torch.bool),
            )

    def forward(self):
        return self.num_rows


def main() -> int:
    fixture_path = os.path.join(ART, "finalize_fixture.pt")
    fixture = torch.load(fixture_path, weights_only=False)
    rows = fixture["rows"]
    bundle = torch.jit.script(FinalizeBundle(rows))
    out_path = os.path.join(ART, "finalize_bundle.ts")
    bundle.save(out_path)
    print(f"wrote {out_path} ({len(rows)} finalize rows)")
    for i, row in enumerate(rows):
        enc_len = int(row["eager_outputs"][1][0].item())
        print(
            f"  row{i}: drop_extra={int(row['drop_extra'])} "
            f"emitted_gt0={int(row['drop_extra']) != 0} "
            f"chunk_T={int(row['chunk_mel'].shape[-1])} enc_len={enc_len} "
            f"pre/final={len(row['pre_final_tokens'])}/"
            f"{len(row['finalize_ref_final_tokens'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
