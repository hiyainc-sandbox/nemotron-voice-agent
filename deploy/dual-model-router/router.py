#!/usr/bin/env python3
"""router.py — language-routing WebSocket front-end for dual-model ASR (single public endpoint).

Each *connection* is routed by its ?language= query (a handshake contract — language cannot be
changed mid-session in this deployment, which runs NEMOTRON_CONTINUOUS=1 where post-connect
set_language is ignored) to one of two single-model batched backends:
  - en / en-* / (absent)  -> EN backend (English specialist, omni venv, validated, batched)
  - any other language     -> ML backend (multilingual prompted, EA venv, batched)
then frames are piped transparently both directions (binary audio up, text control+transcripts
back). The EN backend REJECTS a language query, so we strip it for en routes; the ML backend
NEEDS it (per-language prompt), so we preserve it. Path + other query params are preserved.

Run in any environment with the `websockets` package (>=13, asyncio API). No torch/NeMo needed.

  python router.py --host 0.0.0.0 --port 8080 --en ws://127.0.0.1:8081 --ml ws://127.0.0.1:8082 -v
"""
import argparse, asyncio, json
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import websockets


def decide(path, en_url, ml_url):
    """Return (backend_ws_url, route_name, normalized_lang) for a client request path."""
    parts = urlsplit(path)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    lang = ""
    for k, v in pairs:
        if k == "language":
            lang = (v or "").strip().lower().replace("_", "-")
    # Mirror the server's dual-model predicate: only "en" or "en-*" count as English.
    is_en = (not lang) or lang == "en" or lang.startswith("en-")
    if is_en:
        pairs = [(k, v) for (k, v) in pairs if k != "language"]  # EN backend rejects a language query
        base = en_url
    else:
        base = ml_url
    b = urlsplit(base)
    # Forward the client's path onto the backend host; backends serve "/".
    backend_path = parts.path or b.path or "/"
    query = urlencode(pairs, doseq=True)
    backend_url = urlunsplit((b.scheme, b.netloc, backend_path, query, ""))
    return backend_url, ("en" if is_en else "ml"), lang


async def _pump(src, dst):
    """Forward complete messages src->dst; on src end, close dst preserving code/reason."""
    try:
        async for msg in src:
            await dst.send(msg)
    except websockets.ConnectionClosed:
        pass
    finally:
        code = getattr(src, "close_code", None) or 1000
        reason = getattr(src, "close_reason", "") or ""
        try:
            await dst.close(code, reason)
        except Exception:
            pass


def make_handler(en_url, ml_url, verbose):
    async def handler(client):
        path = getattr(getattr(client, "request", None), "path", "/")
        backend_url, route, lang = decide(path, en_url, ml_url)
        if verbose:
            print(f"[router] conn path={path!r} lang={lang!r} -> {route} {backend_url}", flush=True)
        try:
            backend = await websockets.connect(backend_url, max_size=None, compression=None)
        except Exception as e:
            # Surface backend-connect failures to the client instead of a silent close.
            print(f"[router] backend connect FAILED ({route} {backend_url}): {type(e).__name__}: {e}", flush=True)
            try:
                await client.send(json.dumps({"type": "error", "error": f"backend unavailable: {e}"}))
            except Exception:
                pass
            await client.close(1011, "backend unavailable")
            return
        try:
            # First pump to finish cancels the other so neither leaks.
            up = asyncio.create_task(_pump(client, backend))
            down = asyncio.create_task(_pump(backend, client))
            done, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for ws in (backend, client):
                try:
                    await ws.close()
                except Exception:
                    pass
    return handler


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--en", default="ws://127.0.0.1:8081")
    ap.add_argument("--ml", default="ws://127.0.0.1:8082")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    handler = make_handler(a.en, a.ml, a.verbose)
    print(f"[router] listening ws://{a.host}:{a.port}  en->{a.en}  ml->{a.ml}", flush=True)
    async with websockets.serve(handler, a.host, a.port, max_size=None, compression=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
