# Nemotron streaming ASR - Phase 1 deployment rationale

For the HOW, see [RUNBOOK.md](RUNBOOK.md). This file is the WHY:
topology, measurements, trade-offs, and the decision rules behind the Phase 1
runbook.

## TL;DR + Phase-1 Architecture

Phase 1's primary topology is **single process, no MPS**: one
`python -m nemotron_speech.server` per L40S backend box, listening on `:8080`,
fronted by one HAProxy host using `leastconn`. The default generated HAProxy
cap is `maxconn 20` per backend box, but that value is the
**one-utterance-per-connection** default; use
`deploy/gen_haproxy.py --maxconn-conservative` for `maxconn 12` until sustained
multi-turn load validation lands.

```text
Approved clients
    |
    | wss://<lb>:8443  (LB SG: approved client CIDR only)
    v
+----------------------+       private VPC IPs only       +----------------------+
| HAProxy EC2 host     |  leastconn + http-check /health  | L40S backend box 1   |
| frontend asr_ws      |---------------------------------->| nemotron-asr.service |
| backend asr_pool     | maxconn 20 default, 12 safer     | launch_single.sh     |
| admin.sock for drain |                                  | server.py :8080      |
+----------------------+                                  +----------------------+
          |                                                 ...
          | leastconn
          v
                                                        +----------------------+
                                                        | L40S backend box N   |
                                                        | one server.py :8080  |
                                                        +----------------------+
```

**Primary sizing lead.** Same g6e/L40S, single-utterance burst: a single
full-GPU process is SLO-robust to about **20 streams/box** with **ttfs p50
about 42-54 ms** and **p95 about 158 ms at 20 concurrent**. The 10-repeat
follow-up shows 20 concurrent at p50 53.3 ms, p95 174.4 ms, p99 212.0 ms.
The retained historical note below records the original caveat; the Phase 1
decision is stated immediately after it.


> **Re-measure correction (2026-05-27, same g6e/L40S, single-utterance burst):** a **single** server process on the *full* GPU is SLO-robust to **~20 streams** at **ttfs p50 ~42-54ms** (p95 158ms @20; server-side vad_stop->final). The multi-proc **K=3+MPS** config reaches the *same* ~16-20/box but at **ttfs p50 ~245ms** at matched load - so MPS+multi-proc here buys **about no density over one process** yet pays a **~190ms median-latency tax** (it was sized on the since-refuted "48/box" keep-up belief). The "~5-7/proc" above is the *MPS-degraded* per-proc, **not** a single proc's ceiling. **This is a LEAD, not yet a flip:** the loadgen is one-utterance-per-connection; the genuine multi-proc rationale is per-proc asyncio **intake** parallelism under *sustained multi-turn* load, which this test does not exercise. **Action before changing prod:** run a single-proc *sustained multi-turn* load test - if a single proc holds ~16-20 under sustained load, prefer **fewer procs / no MPS** for the latency win.

Phase 1 does make that lead the default topology because this phase is a manual,
observable rollout with an explicit fallback rule. K=3+MPS remains documented as
the fallback, not the primary path.

## Phase-1 Scope + Non-goals

Phase 1 ships runnable deploy artifacts and operator docs. It does not stand up
a live cluster or replace the future production substrate work. The scoped
implementation is EC2 boxes provisioned by hand, a separate HAProxy EC2 host,
one backend process per L40S box, network-level access control, manual checks,
and a K=3+MPS fallback if sustained traffic invalidates the single-proc
assumption.

- No IaC: no Terraform, CloudFormation, ASG, or ALB definitions in Phase 1;
  Phase 2 moves to EC2 + ASG + ALB or equivalent.
- No live cluster stood up by this phase; the deliverable is scripts and docs.
- No autoscaling automation; Phase 2 owns scaling policy and managed target
  groups.
- No sustained-multi-turn load validation; the `~20` default came from
  one-utterance-per-connection runs only.
- No app-level WebSocket auth; the network allowlist is the Phase 1 access
  control.
- No TLS certificate automation or renewal; Phase 1 assumes an operator-provided
  HAProxy PEM and manual reload.
- No LB HA; one HAProxy host is a single point of failure. Phase 2 adds
  keepalived/dual-LB or ALB.
- No monitoring beyond manual checks; no Prometheus, log shipping, or alerting.
- No GPU memory monitoring beyond operator `nvidia-smi`.
- Torch is unpinned in the reused bootstrap; Phase 2 pins or mirrors the runtime
  dependency set.

## Per-GPU Config Matrix

SLO-robust means the latency-safe operating point, not the old keep-up knee.
The old L40S "48/box" and L4 "~24/box" keep-up estimates overstate deployable
density; the cloud SLO-robust validation refuted them.

| GPU | instance | Phase-1 primary | lanes/proc | SLO-robust /box | Caveat | notes |
|---|---|---:|---:|---:|---|---|
| L4 | g6.4xlarge | not Phase-1 primary | 2 | **~6** in historical K=2+MPS validation | no single-proc L4 baseline yet | BW-bound; remeasure before using L4 for Phase 1 |
| L40S | g6e.4xlarge | **1 process, no MPS** | 2 | **~20** one-utterance default; **12 safer** pre-sustained test | `maxconn 20` is not a sustained-multi-turn claim | preferred Phase-1 box |
| L40S | g6e.8xlarge | **1 process, no MPS** | 2 | **~20** expected | extra vCPU did not buy K=3+MPS density | use only if capacity/availability requires |

The L4 row is preserved as a sizing warning, not a recommendation to deploy L4
without a fresh single-proc measurement. Historical L4 validation was K=2+MPS,
about 3 streams/proc and about 6/box at the p95 edge; profiling-off did not help
because L4 is memory-bandwidth-bound.

The L40S primary row is the 2026-05-27 single-proc lead. The older K=3+MPS L40S
validation remains useful as the fallback baseline: K=3 clean was solid around
16/box and marginal around 20/box, with p50 around 246-251 ms.

## `maxconn` Caveat

Every `maxconn 20` in this Phase 1 design means: **one backend box, one server
process, one-utterance-per-connection measurement**. It is the default because
the generator needs an operator-friendly starting point and the single-proc
measurement held about 20 streams. It is not proof that long-lived, multi-turn
WebSocket sessions keep the same tail behavior.

The safer setting is `deploy/gen_haproxy.py --maxconn-conservative`, which emits
`maxconn 12`. Use that for production-leaning smoke or any fleet where the first
sustained multi-turn test has not been run. HAProxy queues past `maxconn` for up
to `timeout queue 5s`; server-side overload shedding remains
`NEMOTRON_ADMISSION_MAX_BACKLOG` and WebSocket close 1013.

## `SRV_ENV` Provenance

`launch_single.sh` deliberately keeps the production optimization stack from the
old multiproc launcher, but expresses it with canonical `NEMOTRON_*` variables:
continuous mode, silence0, warmup, scheduler/batching, lanes=2, barrier drain,
batch finalize, encoder CUDA graphs, padded finalize graph, sync compression,
and finalize priority.

The important caveat is provenance. The optimization stack was validated in the
K=3+MPS topology, not in the new single-proc Phase 1 topology. These are
per-process optimizations and should carry over, but that is an untested
assumption until the single-proc smoke and sustained test run.

If single-proc smoke shows unexpected p95, first A/B-test
`NEMOTRON_SYNC_COMPRESS=0` and `NEMOTRON_FINALIZE_PRIORITY=0`. Those are the
two tail levers most likely to interact with scheduling policy; padded finalize
should stay on unless GPU memory or correctness evidence points elsewhere.


## Reproducibility Risk

`ec2-bench/bootstrap.sh` pins NeMo by commit
`056d937544064df164b1751e9c8a1c3b597389fd`, but it leaves `torch` unpinned and
installs `uv` through live `curl | sh`. The script then smoke-prints torch and
NeMo versions, but the preserved EC2 validation logs retained only
`bootstrap DONE`, not the full bootstrap output.

Runtime version note: a local production ASR venv check reported
`torch 2.11.0+cu130` and `nemo 2.8.0rc0`. Treat that as an observed reference,
not a Phase 1 bootstrap pin. Reusing the validated `ec2-bench/bootstrap.sh`
verbatim keeps the deployment close to the measured EC2 runs, but accepts
future drift until a later phase pins torch, pins or vendors `uv`, and mirrors
the NeMo source artifact.

## Network Topology Contract

`ec2-bench/ec2_up.py` provisions into public subnets and associates public IPv4
addresses. Private subnets plus NAT/bastion are Phase 2 hardening, not what the
reused provisioner supports in Phase 1.

The Phase 1 contract is:

- HAProxy uses backend **private IPs**, so LB-to-backend traffic stays inside
  the VPC.
- Backend security groups block public `:8080`; only the LB security group may
  reach backend `:8080`.
- The LB security group allows `:8443` only from an approved client CIDR.
- SSH remains `:22` from `MY_IP` through the existing `nemotron-bench-sg`.
- App-level WebSocket auth is deferred; this is acceptable only because the
  network allowlist is the access control.

The procedure for reconciling those security groups belongs in
[RUNBOOK.md section 1.5](RUNBOOK.md#15-network-and-security-groups).
This section states the contract and why it matters; the runbook is the
copy-pasteable AWS CLI.

## MPS Hardening - Informational Only for Phase 1

Phase 1 single-proc does **not** run CUDA MPS. This section is preserved for the
K=3+MPS fallback path only. Use `deploy/launch_multiproc.sh` only if a sustained
multi-turn Phase 2 load test proves that one process's asyncio intake path fails
below the needed load.

MPS shares one CUDA context, so a CUDA fault in one process can corrupt the
context and take down the other processes on that GPU. That is a larger blast
radius than the Phase 1 single-proc model. Mitigations for the fallback are:

- Per-process supervision in the fallback launcher; if multiple procs die
  together, restart MPS and all procs.
- LB health-check and operator drain so a restarting process does not receive
  new streams until `/health` passes.
- Optional per-client SM caps with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for QoS.
- MIG is not available on L40S/Ada, so it is not an isolation option here.
- If the MPS blast radius is unacceptable, run without MPS at lower density
  rather than treating MPS as transparent isolation.



## K=3+MPS Fallback Decision Rule

Abandon single-proc only on externally observable evidence. Do not require
Prometheus or trace instrumentation that Phase 1 does not ship.

1. Sustained p95 stays above the SLO at a known concurrent-stream load below
   about 16 on a single L40S box.
   Observe by running the cloud smoke directly against the box, or drain all but
   one backend and run through the LB; use `deploy/drain.sh status` or HAProxy
   `show stat` to confirm the box's active stream count.

2. A single box repeatedly returns 1013 admission rejections under target load.
   Observe with `curl http://<box>:8080/health` and compare
   `admission.attempted`, `admission.admitted`, and `admission.rejected` before
   and after the run.

3. Any box's median TTFS climbs past about 150 ms under steady multi-turn
   traffic.
   Observe with the cloud smoke, preferably one backend at a time, and correlate
   with HAProxy `show stat` so the load per box is known.

The fallback is operationally heavier: `deploy/launch_multiproc.sh` currently
uses flat `server.py` layout, starts MPS, launches K processes, and has its own
supervisor loop. The Phase 1 runbook documents the compatibility gap before an
operator flips to it.

## Substrate Decision

For a long-lived WebSocket service, Phase 2 should move production scaling to
managed infrastructure. Phase 1 is intentionally simpler: EC2 manual provision
via `ec2-bench/ec2_up.py`, HAProxy on a separate EC2 host, and one
systemd-managed backend process per L40S box.

| Substrate | Fit | Notes |
|---|---|---|
| **Phase 1: EC2 + manual provision + HAProxy** | **shipping now** | Full control, minimal moving parts, no IaC, single LB host. |
| **Phase 2: EC2 + ASG + ALB** | **recommended production start** | ALB supports WebSockets, least-outstanding-requests, and target drain; ASG owns box lifecycle. |
| ECS on EC2 + ALB | good if already standardized | Task entrypoint can be the launcher; MPS fallback needs host/task care. |
| EKS | only if k8s-native | Heaviest control plane for this phase. |
| SageMaker real-time endpoint | **not a fit** | Request/response endpoint shape does not match raw long-lived WebSockets. |

The original HAProxy design already named the ALB equivalent for the routing
shape; Phase 1 keeps self-hosted HAProxy because it is faster to inspect and
ship by hand.

## Day-1 Observability

Operators can see these signals today:

- `curl http://<box>:8080/health` shows `status` and, when admission is enabled,
  admission counters.
- `curl http://<box>:8080/stats` returns a JSON rolling-latency snapshot:
  per-signal p50/p90/p95/p99/max + count over the last ~2048 finalizes for
  `vad_stop_to_sent_ms` (the server-side TTFS), `fork_flush_wall_ms`,
  `vad_stop_recv_to_process_ms` (intake-backlog signal), `lock_wait_ms`,
  `vad_stop_to_finalize_start_ms`, plus the concurrent-session distribution
  at finalize time and the admission counters. `?last=N` narrows to the most
  recent N samples. Always on by default (cost: one deque append per
  finalize, no GPU sync); toggle with `NEMOTRON_STATS_ENABLED=0` or resize
  with `NEMOTRON_STATS_WINDOW`. Use this for ongoing latency observation;
  the heavier `NEMOTRON_FINALIZE_PROFILE=1` log path is reserved for
  diagnostic deep-dives (it adds ~2× intake tax on L40S).
- `echo 'show stat' | socat /run/haproxy/admin.sock stdio` shows HAProxy backend
  status and current sessions; `deploy/drain.sh status` parses the same data.
- `journalctl -u nemotron-asr -f` streams backend server logs from systemd.
- `nvidia-smi` shows GPU memory and process presence on a backend box.

What is not instrumented in Phase 1: no Prometheus scrape format (the
`/stats` JSON is operator-pollable but not Prometheus-native), no log
shipping, no alerting, no metric retention beyond the in-memory sliding
window, no GPU memory time series, and no automatic p95 alert. A small
external poller (cron + `curl /stats` + append to a JSONL or push to
CloudWatch/Datadog) bridges to a proper monitoring system without changing
the server.

## Autoscaling Roadmap

Phase 2 should scale boxes on aggregate utilization:
`active_streams / (boxes * per_box_budget)`, targeting about 75% so a single
box failure or drain does not immediately force overload. Use an ALB target
group with deregistration delay equal to the chosen drain window, and keep warm
headroom because model load plus warmup is measured in minutes, not seconds.


This roadmap is not Phase 1 automation. Phase 1's only scaling action is:
provision another EC2 box, bootstrap it, add its private IP to the HAProxy
config, validate, then reload HAProxy.

## Native Runtime Roadmap

The Python single-proc Phase 1 topology is the pragmatic shipping path. The
native C++/libtorch runtime remains the density play if fewer boxes become worth
the engineering cost. The native L40S gate found an SLO-robust knee around
36 streams/box, about 1.8-2.25x over the accepted Python 16-20/box baseline, and
the native path avoids MPS and multiple Python processes.

This is not a Phase 1 dependency. Invest when the reduction in L40S fleet size
beats the cost of a second serving stack, native correctness gates, and ongoing
version drift management.

## $/stream Summary

Do not price the fleet from the old keep-up knees. Use SLO-robust streams per
box: L40S Phase 1 `~20/box` at `maxconn 20` one-utterance default, or `12/box`
with `--maxconn-conservative`; historical L4 validation is about `~6/box` and
needs a single-proc remeasure before Phase 1 use.

The sizing formula is:

```text
cost_per_stream_hour = instance_on_demand_hourly_price / SLO_robust_streams_per_box
```

For current procurement, recompute with live AWS pricing and the selected
region. The old "$0.031/stream L4" note was tied to an obsolete keep-up density
and should not drive Phase 1 capacity planning. With the validated numbers, L40S
is the Phase 1 density path; L4 is a horizontal-scale/cost option only after a
fresh single-proc measurement or an explicit K=2+MPS fallback decision. [memory
pointer: pre-rewrite `deploy/DEPLOYMENT.md:111-117`; validation sources above]

### Sizing caveat: discrete-burst vs continuous-arrival

The `~20/box` SLO-robust number describes **discrete-burst** load: N concurrent
WS sessions arrive in batches, each does one utterance, the batch finishes,
then the next batch arrives with a brief drain in between. This pattern is
typical of voice-agent traffic where conversation pauses naturally space
session arrivals.

A 2026-05-28 2-hour load test against a 4-backend cluster surfaced a distinct
load pattern: **continuous-arrival** at the same wall concurrency. The
`run_full1000_conc12.py --concurrency 50 --limit 0` bench keeps 50 sessions in
flight by replenishing each finished session immediately from a 1000-sample
queue. Same instantaneous concurrency (50), but the server's intake never
drains. Result on 4 backends at `maxconn 20`:

| Pattern              | Phase   | Sessions | 1013 sheds | OK rate |
|----------------------|---------|---------:|-----------:|--------:|
| Discrete-burst 50    | B + D   |    16,300|        ~480|   97.0% |
| Continuous-arrival 50| E       |     1,000|         482|   51.8% |

The cluster shed correctly in both patterns — admitted sessions held p95
~32–36ms server-side throughout the 2h run. But the continuous-arrival
pattern needs ~2× the box count of the discrete-burst pattern for the same
client-visible OK rate. Operators sizing for high-throughput
ASR-as-a-batch-job workloads (transcription pipelines, audio-import flows)
should plan against arrival rate, not instantaneous concurrency:

```text
boxes_continuous = ceil(target_arrival_rate_per_sec × utterance_seconds /
                        (maxconn × admission_cap_ratio × headroom))
```

where `admission_cap_ratio ≈ 0.6` (server admits 12 of HAProxy's 20). For
voice-agent loads stick with the simpler discrete-burst formula in section
"Per-GPU Config Matrix" above.

## Artifacts

- [launch_single.sh](launch_single.sh) - Phase-1 launcher: one process, no MPS,
  no in-script supervisor.
- [nemotron-asr.service](nemotron-asr.service) - systemd unit; systemd is the
  only backend supervisor.
- [asr.env.example](asr.env.example) - environment override template and
  provenance notes.
- [gen_haproxy.py](gen_haproxy.py) - HAProxy config generator: `leastconn`,
  Runtime API socket, `/health` `http-check expect rstring`.
- [drain.sh](drain.sh) - operator drain wrapper around the HAProxy Runtime API.
- [_drain_fixtures/](_drain_fixtures/) - sample `show stat` CSVs for drain
  smoke tests.
- [smoke_local.sh](smoke_local.sh) plus helpers - laptop-safe smoke test for the
  deploy artifacts.
- [RUNBOOK.md](RUNBOOK.md) - procedure of record; step 7 lands it.
- [README.md](README.md) - deploy navigation index; step 7 lands it.
- [launch_multiproc.sh](launch_multiproc.sh) - K=3+MPS fallback only; untouched
  by Phase 1.
- [haproxy.cfg.example](haproxy.cfg.example) - original routing design artifact;
  predates the generator and still documents the leastconn rationale.
- [../ec2-bench/](../ec2-bench/) - provisioning and benchmark toolkit referenced
  by the runbook.
