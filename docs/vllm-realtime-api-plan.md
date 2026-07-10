# Plan: expose vLLM's `/v1/realtime` streaming audio API from the C++ ws_server

Status: proposal (plan only, not implemented). Date: 2026-07-09.

## Context

**Question:** does `runtime/cpp/ws_server.cpp` expose the same API as vLLM? **Answer: no — but it's close, and adding it is very feasible.**

vLLM's streaming audio surface is a WebSocket endpoint at **`/v1/realtime`** (OpenAI Realtime-inspired, used with streaming ASR models like Voxtral-Mini-Realtime): the client sends JSON events carrying base64 PCM16@16kHz mono audio; the server streams back partial and final transcriptions. `ws_server` already implements the same *mechanics* — WebSocket, 16 kHz s16le mono PCM in, interim/final transcripts out — but with a proprietary wire protocol (binary PCM frames + `vad_start`/`vad_stop`/`reset`/`end` control messages, `ready`/`transcript`/`error` responses) at `GET /`. So the work is a **protocol adapter on the existing WS path**, not new server machinery. No POST bodies, no multipart parsing, no audio file decoding needed.

For reference, vLLM's other audio endpoints: `POST /v1/audio/transcriptions` (batch, multipart file upload) and `GET /v1/models` / `GET /health`. `ws_server` already matches `/health`; `/v1/models` is trivial to add; the batch endpoint is scoped to v2 (see end).

### Protocol mapping (vLLM realtime ↔ ws_server today)

| vLLM `/v1/realtime` | ws_server equivalent | Gap |
|---|---|---|
| WS upgrade at `/v1/realtime` | WS upgrade at `/` | new route, trivial |
| `session.created` (server, on connect) | `{"type":"ready"}` | rename + add session id |
| `session.update` `{model}` (client) | `?model=` query validation at connect | defer model validation to first event |
| `input_audio_buffer.append` `{audio:<base64 PCM16>}` | binary WS frames of raw PCM16 | base64 decode → same `process_binary_pcm` path |
| `input_audio_buffer.commit` | `vad_stop` (finalize) | direct map |
| `input_audio_buffer.commit` `{final:true}` | `{"type":"end"}` | direct map |
| `transcription.delta` `{delta}` | `transcript` `is_final:false` (full hypothesis) | **delta diffing** (see step 4) |
| `transcription.done` `{text, usage}` | `transcript` `is_final:true` | rename + usage stub |
| `error` `{error}` | `error` `{message}` | rename field |

`response.create` / `response.text.delta` / `response.audio.delta` are for LLM/TTS generation models — not applicable to an RNN-T transcription server; reject `response.create` with an `error` event.

## Implementation

All server internals stay untouched: per-connection `ws_worker()` thread (`ws_server.cpp:1625`), `SessionRuntime` (`runtime/cpp/lib/session/runtime.h:109-143` — already transport-agnostic and synchronous), admission control, keepalive, drain. Only the message translation layer changes per protocol.

### 1. Route: `/v1/realtime` (~0.5 day)

- Add `RouteKind::REALTIME` (or a `WsProtocol::OPENAI_REALTIME` tag on the existing `WEBSOCKET` kind) in `runtime/cpp/lib/ws/routes.h:10-17`; match `path == "/v1/realtime"` + upgrade in `dispatch()` (`routes.cpp:62-87`). Same RFC6455 handshake (`handshake.cpp:234`).
- In `handle_accepted()` (`ws_server.cpp:~2152`) spawn the same `ws_worker` with a protocol flag.
- Connect-time differences: send `{"type":"session.created","session":{"id":...}}` instead of `ready` (id from an atomic counter or stream_id); **defer** model validation from `validate_ws_query()` (`ws_server.cpp:1356`) to the first `session.update` — extract its model/language checks into a shared `validate_model_and_language()` helper. Also honor `?model=`/`?language=` query params if present (harmless extension).

### 2. Inbound event adapter (~1 day)

Extend `process_text_control()` (`ws_server.cpp:1516`) — or a parallel `process_realtime_event()` selected by the protocol flag:

- `session.update`: validate `model` against `accepted_model_aliases()` (`ws_server.cpp:1331`); accept optional `language` (ml profile: validate against `prompt_runtime_table()`, supports `auto`) as an extension since vLLM's protocol doesn't carry one. Invalid → `error` event + close.
- `input_audio_buffer.append`: base64-decode `audio` (new ~40-line decoder in `lib/runtime_io/base64.{h,cpp}` — nothing vendored exists), enforce even byte count, feed the existing PCM path (`process_binary_pcm`, `ws_server.cpp:1580`). Note ~33% size overhead vs binary frames — the 10 MiB `kMaxMessageSize` still comfortably fits 4–8 KB chunks.
- `input_audio_buffer.commit`: → the `vad_stop` finalize path; each commit produces one `transcription.done`. With `"final":true` → the `end` path (finalize + close 1000).
- Unknown/`response.create` → `error` event (vLLM protocol has no LLM generation here).
- Binary frames on `/v1/realtime`: accept as raw PCM too (harmless superset) or close 1003 — pick one, document it.

### 3. Outbound event adapter (~0.5 day)

Branch in the event serializer (`wire_event_json()`, `ws_server.cpp:1218`) on the protocol flag:

- interim `WireEvent` (`is_final:false`) → `{"type":"transcription.delta","delta":...}`
- final (`is_final:true`) → `{"type":"transcription.done","text":<full final>,"usage":{"seconds":<audio_s>}}` (+ `language` as an extension field on the ml profile)
- `error` → `{"type":"error","error":<message>}`

### 4. Delta semantics — the one real design issue (~0.5 day)

ws_server interims carry the **full current hypothesis**; `transcription.delta` is **incremental append-only text**. Adapter: track the last emitted string per utterance; when the new hypothesis extends it, emit only the suffix as `delta`. RNN-T greedy streaming decode is append-mostly, but the finalize pass (finalize buckets re-decode) can *revise* earlier text — and the protocol has no retraction event. Policy: deltas are best-effort; `transcription.done.text` is authoritative and carries the complete finalized text (this matches how vLLM clients consume it — they replace accumulated deltas with `done.text`). When a new hypothesis is *not* an extension, skip the delta and let `done` correct it.

### 5. Config + `/v1/models` (~0.5 day)

- Route on by default; kill switch `NEMOTRON_REALTIME_API=0` (pattern: `read_env_enabled`, `ws_server.cpp:~676`). Surface in `format_config()`.
- `GET /v1/models` (trivial, admin pool, next to `health_json()` `ws_server.cpp:1071`): `{"object":"list","data":[{"id":MODEL_ID,...}]}` — vLLM clients commonly probe it. `/health` already matches vLLM.

### 6. Verification (~1 day)

1. Unit selftest for the base64 decoder + delta-diffing logic, modeled on `ws_framing_selftest.cpp` (torch-free standalone target in `runtime/cpp/CMakeLists.txt`).
2. Extend `run_selftest()` (`ws_server.cpp:2537`) — it already drives real WS sessions against fixture audio: run the same utterance through `/` and `/v1/realtime`, assert `transcription.done.text` equals the legacy path's final transcript; assert event ordering (`session.created` → deltas → `done`); bad model in `session.update` → `error`; `commit final:true` → close 1000.
3. Real client: vLLM ships a reference client — [examples/online_serving/openai_realtime_client.py](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_realtime_client.py) — point it at `ws://127.0.0.1:8081/v1/realtime` and confirm it transcribes unmodified. That's the compatibility acceptance test.
4. Accuracy: adapt `runtime/session_wer.py` to speak the realtime protocol and compare WER against the legacy WS path.

## Effort & scope

**v1 total ≈ 3.5–4 days**: route + adapters + base64 + delta diffing + `/v1/models` + selftests + vLLM-client acceptance.

**v2 / out of scope for now:**

- Batch `POST /v1/audio/transcriptions` (multipart + WAV parsing + one-shot `SessionRuntime` session, ~5-6 days — requires adding POST/body support to the HTTP layer: `read_http_request_from_socket()` at `ws_server.cpp:933` currently discards bytes past the header terminator, so a `residual` fix plus a body reader would be needed; independent of the realtime work).
- OpenAI-official Realtime event names (`conversation.item.input_audio_transcription.delta`, `transcription_session.update`, server-VAD `speech_started/stopped` events) — vLLM uses the simplified names above; match vLLM, not OpenAI, unless a client needs it. ws_server's silence-based auto-finalize could later back OpenAI-style server VAD events.
- `usage` token accounting parity with vLLM (v1 ships audio-seconds only).

**Risks:** delta-vs-revision mismatch (mitigated in step 4; `done.text` authoritative); base64 CPU cost per frame (negligible at 4–8 KB/20–250 ms chunks); protocol flag leaking into session logic (keep the adapter purely at the serialize/parse boundary — `SessionRuntime` and `WireEvent` stay protocol-agnostic).

## Sources

- [vLLM Realtime example (protocol events, audio format)](https://docs.vllm.ai/en/latest/examples/speech_to_text/realtime/)
- [vLLM blog: Streaming Requests & Realtime API](https://vllm.ai/blog/2026-01-31-streaming-realtime)
- [vLLM realtime api_router](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/realtime/api_router/)
