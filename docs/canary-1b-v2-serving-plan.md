# Plan: Serving nvidia/canary-1b-v2 with this server

Status: proposal — not implemented.

## Context

The repo serves NeMo **cache-aware streaming FastConformer-RNNT** models (en: `nemotron-speech-streaming-en-0.6b`, ml: `nemotron-3.5-asr-streaming-0.6b`) two ways: the Python reference server (`src/nemotron_speech/server.py`) and the native C++ `ws_server` (`runtime/cpp/`, libtorch TorchScript + AOTI artifacts, one model per binary via `-DNEMOTRON_PROFILE=en|ml`). This document investigates whether **nvidia/canary-1b-v2** can also be served, what code changes are needed, and how to export the model.

## Feasibility verdict

**Yes — but it's a new architecture, not a new profile like the en→ml jump was.** Canary-1b-v2 is an offline attention-encoder-decoder (AED) model, `EncDecMultiTaskModel` (978M params): full-attention FastConformer encoder (32 layers) + 8-layer Transformer decoder with cross-attention, driven by a 10-token "canary2" prompt, unified SentencePiece vocab 16,384 + ~1,162 special tokens, 25 European languages, ASR + English↔X translation, word/segment timestamps (via auxiliary CTC alignment — out of scope here).

| | current (nemotron RNN-T) | canary-1b-v2 |
|---|---|---|
| Decode | LSTM predict + joint, blank-break greedy (`session.cpp:1959`) | autoregressive Transformer decoder w/ KV cache, EOS stop |
| Streaming | native (enc_first/enc_steady/finalize buckets/scheduler) | none — full-utterance decode |
| Lang control | post-encoder one-hot→MLP (`prompt_apply.ts`) | decoder prompt tokens: `[" ", <\|startofcontext\|>, <\|startoftranscript\|>, <\|emo:undefined\|>, <\|src\|>, <\|tgt\|>, <\|pnc\|>, <\|noitn\|>, <\|notimestamp\|>, <\|nodiarize\|>]` |

**Reusable:** WS transport (`runtime/cpp/lib/ws/`), transport-agnostic `SessionRuntime`/`WireEvent` (`runtime/cpp/lib/session/runtime.h` — `SessionConfig.language` already exists), mel preproc wrapper, jit_load/prewarm/admission/telemetry, profile + export scaffolding (`runtime/model_profile.py`, `export_session_bundle.py`), oracle/compat harness (`tests/server_compat/run_compat.py`), dual-model router.

**Not reusable:** the entire RNN-T decode path, streaming machinery, batched steady scheduler, prompt MLP; `server.py`'s `_batch_model_rnnt_pure_status` (~:1135) explicitly rejects non-RNNT models.

**Export gotcha:** NeMo's official `model.export()` is **broken** for `EncDecMultiTaskModel` ([NeMo issue #11004](https://github.com/NVIDIA-NeMo/NeMo/issues/11004)); no official ONNX/TensorRT/Riva recipe exists for v2. Community blueprint: [`onnx-asr`](https://github.com/istupakov/onnx-asr) exports two graphs — encoder (`audio, len → enc_embeddings, enc_mask`) and decoder step (`input_ids, enc_embeddings, enc_mask, decoder_mems → logits`) with a growing self-attn KV cache; step 0 feeds the full prompt, then one token/step, greedy, stop at `<|endoftext|>`. This plan mirrors that split in **TorchScript/torch.export** (the repo runtime is libtorch, not ONNX Runtime).

**Effort:** ~4–6 engineer-weeks total. Phase A ~1 week and independently shippable; B ~1–2 weeks; C ~2–3 weeks. B/C are optional if Python throughput suffices (canary is offline, RTFx ~749 — the per-utterance latency story is far less punishing than streaming).

## Phase A — Python serving + oracle (ship first)

**A0. Environment spike (cheap de-risk):** verify `runtime/.venv` (or the production en venv) can `ASRModel.from_pretrained("nvidia/canary-1b-v2")` and `.transcribe()` a clip (needs NeMo ≥2.4 for canary2 prompts). If incompatible, it becomes a third venv/process — the dual-router deployment already normalizes that.

**A1. New `src/nemotron_speech/canary_server.py` (~600–900 lines)** speaking the existing wire protocol (binary PCM16@16k in; `ready`/`transcript`(`is_final`)/`error` out; `vad_start`/`vad_stop` controls — frame shapes pinned by `server.py` and `run_compat.py`). Do NOT graft into the 10.8k-line `server.py`. Logic:

- Buffer PCM from `vad_start`; cap ~120 s (forced final past cap).
- On `vad_stop`/finalize: decode buffered audio, emit one final event, clear.
- Two decode paths: (a) `model.transcribe()` greedy (sanity gold); (b) **manual step decode** — preproc → encoder → build canary2 prompt ids from (source_lang, target_lang, flags) → greedy over decoder with explicit `decoder_mems`, stop at `<|endoftext|>`. Path (b) is the byte-exact oracle for Phases B/C; assert (a)==(b) on a startup self-test clip.
- Query params: `?model=` validated against MODEL_ID; `?language=` → source lang; new `?target_language=` (= source ⇒ ASR; ≠ ⇒ translation), validated fail-closed (pattern of `ws_server.cpp:1369-1387`).
- Detokenizer skips all special/control tokens.

**A2. Routing:** extend `deploy/dual-model-router/router.py` `decide()` to route `?model=` → canary backend before the `?language=` en/ml split; add the backend to `run_dual_model.sh`. This alone gives production serving with zero export work.

## Phase B — Export pipeline

**B1. Profile plumbing:** add `arch: str = "rnnt"` to `ModelProfile` in `runtime/model_profile.py` + a `canary` profile (`arch="aed"`, `mels=128`, vocab/EOS/prompt constants; RNN-T fields sentinel). `validate_model_against_profile` branches on arch (AED checks: decoder layers, d_model, vocab, EOS id, prompt format `canary2`). Streaming export scripts `SystemExit` on `arch=="aed"`.

**B2. New `runtime/export_canary.py`** (each artifact byte-exact-validated against the Phase A oracle, per the `export_decode.py` "EXPORT VERIFIED"-or-fail convention):

1. `preproc.ts` — re-export from the canary checkpoint via the `export_pipeline.py` wrapper (don't reuse the en artifact; normalization config may differ).
2. `encoder.ts` — full-utterance encoder `(mel[1,128,T], len) → (enc, enc_len)`; `torch.jit.trace` then **byte-exact T-sweep validation** (0.5–60 s, odd lengths); fall back to script/torch.export with dynamic T if trace shape-specializes.
3. `aed_decode_step.ts` — embeddings + TransformerDecoder + classifier head: `(input_ids[1,S], enc[1,T,D], enc_mask, mems) → (logits[1,V], new_mems)`; validated at S=10/P=0 (prompt step) and S=1/P>0 (incremental); split into two graphs only if trace can't unify.
4. Prompt-template token table per (src, tgt) + EOS + max-gen-len (mirrors `decode_init.pt`).
5. Optional v1.1: `aot_compile_canary.py` — AOTI encoder at bucketed T (mirroring `aot_compile_buckets.py`), sm_120 + sm_89; decoder step stays TorchScript (per-step cost tiny, dynamic shapes AOTI-hostile).

**B3. New `runtime/export_canary_bundle.py`** (modeled on `export_session_bundle.py`) → `runtime/artifacts_canary/session_bundle.ts`: tokenizer piece table + **special-token id skip-set** + detok self-tests (must include control-token cases), prompt table (generalizing `_register_prompt_table`), audio geometry, gold fixtures from the Phase A oracle (`runtime/fixtures_canary/` via a canary mode of `capture_decode_fixtures.py`), `meta.model_id` fail-closed cross-check.

## Phase C — C++ ws_server profile

1. **`runtime/cpp/lib/session/model_constants.h`:** new `#ifdef NEMOTRON_PROFILE_CANARY` block (MODEL_ID, EOS_ID, VOCAB_SIZE, MAX_DECODE_TOKENS, MAX_AUDIO_SECONDS, `MODEL_ARCH_AED=true`) that does **not** define streaming constants — any streaming TU accidentally compiled into the canary binary fails at compile time.
2. **New `runtime/cpp/lib/session/aed_session.{h,cpp}`** implementing the `SessionRuntime` contract: `append_pcm_and_drain` buffers; `handle_vad_stop`/`finalize_now` runs preproc → encoder → AED greedy (prompt ids from bundle table keyed by session language/target_language; token-at-a-time with mems; EOS/cap stop) → special-token-skipping detok → one final `WireEvent`. **Don't `#ifdef` inside the 6.2k-line `session.cpp`** — the canary CMake profile simply doesn't compile `session.cpp`/`first_encoder.cpp`/scheduler sources. (Known gotcha: `PromptTable` is declared in both `session.h:63` and `session.cpp:862` — the parallel-file approach sidesteps that class of drift.)
3. **`runtime/cpp/ws_server.cpp`:** `?language=` validated against the bundle prompt table (same pattern as ml); new `?target_language=` under the canary ifdef; scheduler disabled.
4. **`runtime/cpp/CMakeLists.txt`:** `-DNEMOTRON_PROFILE=canary` → define + canary source set; keep one-model-per-binary.

## Interim transcripts

**v1: final-only on `vad_stop`** (offline model has no incremental hypothesis; the protocol tolerates finals-only). Optional fast-follow behind a flag: periodic re-decode of accumulated audio every ~2 s emitted as `is_final:false` (at RTFx ~749 a 20 s re-decode is tens of ms; disable past ~60 s). AlignAtt/wait-k streaming: explicitly out of scope (research-grade in NeMo, changes the encoder invocation pattern).

## Verification

1. Phase A startup self-test: manual step-decode == `.transcribe()` greedy on a test clip; 20-clip offline check.
2. Fixture capture → `runtime/fixtures_canary/` (mel, enc, prompt ids, per-step gold tokens, final text).
3. Export gates: byte-exact vs eager shape sweeps inside each export script; full-chain audio→tokens on fixtures.
4. Phase C: parameterize `tests/server_compat/run_compat.py` by profile (Python side launches `canary_server.py`); byte-exact final-transcript JSON comparison (WER-tolerance fallback only if the full-attention encoder shows cross-kernel nondeterminism at large T).
5. WER sanity via a canary variant of `runtime/session_wer.py` per language vs NeMo-published numbers + one translation smoke.
6. Router/deploy smoke.

## Risks (ranked)

1. **TorchScriptability of NeMo's TransformerDecoder with `decoder_mems`** — the cache flows through Python-level code; the wrapper may need to reimplement the per-layer cache append. Spike in week 1 of Phase B (onnx-asr proves the decomposition exports in principle).
2. Encoder trace shape-specialization (rel-pos/masking) — mitigated by the T-sweep gate + script/export fallback.
3. NeMo/venv compatibility — possibly a third venv (the repo already runs two incompatible ones); contained by the router. Verified in A0.
4. Memory: ~4 GB fp32 weights + O(T'²) full-attention activations; cap buffered audio; check L40S VRAM budget for co-residency.
5. Finalize latency scales with turn length (fine for <30 s voice-agent turns; document the cap).
6. Greedy-vs-beam parity: pin the decoding config explicitly in both oracles.
7. Special-token leakage into transcripts — covered by detok self-tests with control-token cases.

## References

- Model card: https://huggingface.co/nvidia/canary-1b-v2
- Paper: https://arxiv.org/abs/2509.14128
- NeMo export limitation: https://github.com/NVIDIA-NeMo/NeMo/issues/11004
- onnx-asr (two-graph export blueprint, supports v2): https://github.com/istupakov/onnx-asr — pre-exported weights: https://huggingface.co/istupakov/canary-1b-v2-onnx
- NeMo Canary chunked & streaming decoding: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/streaming_decoding/canary_chunked_and_streaming_decoding.html
