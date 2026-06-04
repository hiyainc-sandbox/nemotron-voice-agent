# Deploy Directory

Start here: [RUNBOOK.md](RUNBOOK.md) is the Phase 1 procedure-of-record, the HOW.

Why these decisions: [DEPLOYMENT.md](DEPLOYMENT.md) explains the topology, measurements, trade-offs, scope, and fallback rules.

## Files

- [README.md](README.md) - this navigation index.
- [RUNBOOK.md](RUNBOOK.md) - Phase 1 hand-deploy runbook and operating procedure.
- [DEPLOYMENT.md](DEPLOYMENT.md) - Phase 1 rationale, sizing caveats, non-goals, and accepted risks.
- [launch_single.sh](launch_single.sh) - Phase-1 single-proc launcher, no MPS.
- [nemotron-asr.service](nemotron-asr.service) - systemd unit; systemd is the sole backend supervisor.
- [asr.env.example](asr.env.example) - environment override template for `/etc/nemotron/asr.env`.
- [gen_haproxy.py](gen_haproxy.py) - HAProxy config generator using `leastconn` and `/health` `http-check rstring`.
- [drain.sh](drain.sh) - operator drain helper for the HAProxy Runtime API.
- [_drain_fixtures/](_drain_fixtures/) - sample `show stat` CSVs for drain smoke tests.
- [smoke_local.sh](smoke_local.sh), [_smoke_backend.py](_smoke_backend.py), and [_smoke_haproxy_check.py](_smoke_haproxy_check.py) - laptop-safe deploy artifact smoke test.
- [launch_multiproc.sh](launch_multiproc.sh) - K=3+MPS sustained-load fallback; see [DEPLOYMENT.md](DEPLOYMENT.md#k3mps-fallback-decision-rule) for trip-wires.
- [haproxy.cfg.example](haproxy.cfg.example) - original HAProxy design artifact; predates [gen_haproxy.py](gen_haproxy.py).

## What To Open Next

- Deploying or operating by hand: [RUNBOOK.md](RUNBOOK.md).
- Reviewing why Phase 1 is single-proc/no-MPS: [DEPLOYMENT.md](DEPLOYMENT.md#tldr--phase-1-architecture).
- Checking scope and non-goals: [DEPLOYMENT.md](DEPLOYMENT.md#phase-1-scope--non-goals).
- Running a laptop-safe local artifact smoke: [smoke_local.sh](smoke_local.sh).
- Switching after sustained-load trip-wires: [RUNBOOK.md](RUNBOOK.md#7-switching-to-the-k3mps-fallback).

## Phase 1 Scope

Phase 1 ships runnable deploy artifacts and operator docs only: no IaC, no live infra creation beyond manual EC2 commands, no autoscaling, no managed LB/ASG, no app auth, no TLS automation, no monitoring stack, and no log shipping.

Sustained-load fallback: K=3+MPS via [launch_multiproc.sh](launch_multiproc.sh), with the compatibility notes in [RUNBOOK.md](RUNBOOK.md#7-switching-to-the-k3mps-fallback).
