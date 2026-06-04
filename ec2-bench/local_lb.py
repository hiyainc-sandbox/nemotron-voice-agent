#!/usr/bin/env python3
"""Minimal leastconn TCP load balancer — a haproxy stand-in for LOCAL benches (no sudo / no deps).

haproxy in the cloud harness runs `mode tcp` (it byte-pipes the websocket, it does not parse it). This does the
same at the TCP level: accept a client connection, route it to the least-loaded backend (respecting an optional
per-backend maxconn), and pipe bytes both ways until either side closes. Because it is pure TCP passthrough it
carries the websocket (HTTP upgrade + frames) transparently — identical routing semantics to
`balance leastconn` + `server ... maxconn N` for the keep-up sweep.

Usage: local_lb.py --front 8080 --backends 8081,8082,8083 [--maxconn 0]
  --maxconn 0 (default) = unlimited (let the server's own keep-up be the limit, which is what the density sweep
  measures). Set it to mirror the cloud `maxconn` only for overload/shed tests (note: this stand-in *closes* a
  client when all backends are full, vs haproxy which queues — fine for density, approximate for overload).
"""
import argparse
import asyncio


class LB:
    def __init__(self, backend_ports: list[int], maxconn: int):
        self.backends = [{"port": p, "conns": 0} for p in backend_ports]
        self.maxconn = maxconn

    def pick(self):
        elig = [b for b in self.backends if self.maxconn <= 0 or b["conns"] < self.maxconn]
        return min(elig, key=lambda b: b["conns"]) if elig else None

    async def _pipe(self, reader, writer):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def handle(self, cr, cw):
        b = self.pick()
        if b is None:  # all backends at maxconn -> shed (haproxy would queue; fine for density sweeps)
            cw.close()
            return
        b["conns"] += 1
        try:
            br, bw = await asyncio.open_connection("127.0.0.1", b["port"])
        except Exception:
            b["conns"] -= 1
            cw.close()
            return
        try:
            await asyncio.gather(self._pipe(cr, bw), self._pipe(br, cw))
        finally:
            b["conns"] -= 1
            for x in (cw, bw):
                try:
                    x.close()
                except Exception:
                    pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=int, required=True)
    ap.add_argument("--backends", required=True, help="comma-separated backend ports")
    ap.add_argument("--maxconn", type=int, default=0)
    a = ap.parse_args()
    lb = LB([int(p) for p in a.backends.split(",") if p.strip()], a.maxconn)
    srv = await asyncio.start_server(lb.handle, "127.0.0.1", a.front)
    print(f"local_lb up: front={a.front} backends={a.backends} maxconn={a.maxconn} (leastconn)", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
