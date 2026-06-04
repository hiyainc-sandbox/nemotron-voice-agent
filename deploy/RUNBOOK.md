# Nemotron ASR Phase 1 Runbook

This is the Phase 1 procedure-of-record: the hand-run sequence an operator
follows to stand up, smoke, operate, and tear down the cluster.

For rationale and trade-offs, see [DEPLOYMENT.md](DEPLOYMENT.md). For the
short deploy directory index, see [README.md](README.md). The production
artifacts used by this runbook are [launch_single.sh](launch_single.sh),
[nemotron-asr.service](nemotron-asr.service),
[asr.env.example](asr.env.example), [gen_haproxy.py](gen_haproxy.py),
[drain.sh](drain.sh), [smoke_local.sh](smoke_local.sh),
[_smoke_backend.py](_smoke_backend.py),
[_smoke_haproxy_check.py](_smoke_haproxy_check.py),
[_drain_fixtures/](_drain_fixtures/), [launch_multiproc.sh](launch_multiproc.sh),
and [haproxy.cfg.example](haproxy.cfg.example).

Assumptions:

- Run workstation commands from the repository root.
- EC2 SSH user is `ubuntu`, matching [ec2-bench/ec2_up.py](../ec2-bench/ec2_up.py).
- State files live under `ec2-bench/` because `ec2_up.py` resolves
  `NEMOTRON_EC2_STATE` relative to its own directory.
- Backend serving topology is one full L40S process per box on `:8080`, no MPS.
- HAProxy uses backend private IPs and drains through its Runtime API socket.

## (0.1) C++ `ws_server` B16 Artifact Requirement

The Step-10 C++ production default is the adaptive B16 policy: `B_max=16`,
min-fill enabled, `NEMOTRON_DENSITY_BATCH_MAX_QUEUE_DELAY_MS=2`, and
`NEMOTRON_DENSITY_BATCH_WINDOW_MS=10`, with queue capacity 32. Any L40S/sm_89 deployment of
`ws_server` MUST stage and launch with:

```text
--steady-batch-dir ~/density/steady_b_artifacts_b16
```

That directory must contain the FP32 sm_89 B16 ladder with manifest bucket set
`[1,2,4,8,16]`. The bucket-16 package SHA256 is:

```text
a426d313de5196b0e24ae934fdc65c8ef996e70e73b6adefcad11f60d1fb7e68
```

Do not point the default B16 server at the legacy `steady_b_artifacts/`
`{1,2,4}` package. Startup now fails closed if the selected steady batch dir is
valid but lacks bucket 16.

Capacity planning for this C++ B16 path uses the off-box canonical harness:
separate c7i driver, private VPC URL, driver idle, gate = 0 errors plus
`lag_p95<500ms` plus `TTFS_p95<400ms`. The honest L40S cap=16 result is
**88 SLO-robust streams/box** (96 marginal), not the older co-located
104/box artifact. The off-box dispatch re-diagnosis
found cap=16 was the admission limiter starving the dispatch pipeline; cap
16->32 moves the robust knee to roughly **112-120 streams/box** (~+30%) with
byte-identical outputs because this is admission/in-flight buffering only.
Cap 32->64 gave no further gain; the next limiter is the single dispatch
stream. Never co-locate the load generator for capacity-of-record.

## (0) Prereqs

Export AWS identity and region before any command in this runbook so
[../ec2-bench/ec2_up.py](../ec2-bench/ec2_up.py), AWS CLI snippets, and the
security-group reconciliation target the same account and region:

```bash
export AWS_PROFILE=<your-sso-or-named-profile>          # or leave unset for the default credential chain
export NEMOTRON_AWS_PROFILE="$AWS_PROFILE"              # ec2_up.py / ec2_down.py read this
export AWS_REGION=us-west-2                              # set to your region
export AWS_DEFAULT_REGION="$AWS_REGION"                  # belt-and-braces
export MY_IP=$(curl -s https://checkip.amazonaws.com)    # SG ingress for SSH
```

Expected identity check:

```bash
aws sso login
aws sts get-caller-identity --output json
```

Expected output shape:

```text
{
  "UserId": "...",
  "Account": "<AWS_ACCOUNT_ID>",
  "Arn": "arn:aws:sts::<AWS_ACCOUNT_ID>:assumed-role/..."
}
```

Workstation prerequisites are Python with `boto3` for provisioning,
`websockets` for the cloud smoke, `jq`, `rsync`, and `ssh`:

```bash
python3 -m venv .venv-deploy
source .venv-deploy/bin/activate
python -m pip install -U pip boto3 websockets
for cmd in aws jq rsync ssh curl python3; do
  command -v "$cmd"
done
python - <<'PY'
import boto3
import websockets
print("boto3", boto3.__version__)
print("websockets", websockets.__version__)
PY
```

Expected output shape:

```text
/usr/local/bin/aws
/usr/bin/jq
/usr/bin/rsync
/usr/bin/ssh
/usr/bin/curl
/usr/bin/python3
boto3 1.x.y
websockets x.y
```

The LB host prerequisites are installed in section 4, but the final LB host must
have `haproxy`, `socat`, `python3`, `jq`, and AWS CLI v2. Ubuntu's `awscli`
package can be v1; that is enough for simple EC2/SG queries, but use AWS CLI v2
if SSO operations are needed on the LB host.

Operator safety checks before provisioning:

```bash
test -d deploy
test -f deploy/launch_single.sh
test -f deploy/nemotron-asr.service
test -f ec2-bench/ec2_up.py
test -f ec2-bench/bootstrap.sh
```

Expected output: no output and exit code `0`.

## (0.5) Sizing

Use:

```text
boxes = ceil(target_streams / (maxconn * headroom))
```

Phase 1 defaults:

- `maxconn=20` for the one-utterance-per-connection L40S single-process lead.
- `headroom=0.7` so a normal drain or one failed box does not immediately blow
  the SLO budget.
- Use `maxconn=12` via [gen_haproxy.py](gen_haproxy.py)
  `--maxconn-conservative` when sustained multi-turn behavior has not been
  validated for the workload.

Worked example:

```bash
python3 - <<'PY'
import math
target_streams = 100
maxconn = 20
headroom = 0.7
boxes = math.ceil(target_streams / (maxconn * headroom))
print(f"boxes={boxes}")
PY
```

Expected output:

```text
boxes=8
```

Interpretation: 100 target streams with 20 max connections per box and 70%
operating headroom requires `ceil(100 / 14) = 8` L40S backend boxes. At roughly
`$2/hr` per `g6e.4xlarge`, 8 backends plus the LB host is meaningful spend; run
section 8 teardown when finished.

## (0.6) Network Topology

Goal state:

```text
Approved clients
    |
    | wss://<lb-dns>:8443
    | SG: nemotron-asr-lb-sg allows FRONT_PORT only from CLIENT_CIDR
    v
+----------------------+        private VPC IPs         +----------------------+
| HAProxy EC2 host     |  leastconn + /health checks     | L40S backend box 1   |
| frontend asr_ws      |--------------------------------->| nemotron-asr.service |
| backend asr_pool     | maxconn 20 default              | launch_single.sh     |
| admin.sock for drain | drain.sh Runtime API            | server.py :8080      |
+----------------------+                                  +----------------------+
          |                                                 ...
          v
                                                        +----------------------+
                                                        | L40S backend box N   |
                                                        | one server.py :8080  |
                                                        +----------------------+
```

Concrete security-group rules:

| Security group | Attached to | Ingress | Source |
|---|---|---|---|
| `nemotron-bench-sg` | LB and backend instances | TCP `:22` | `$MY_IP/32` |
| `nemotron-asr-lb-sg` | LB instance | TCP `$FRONT_PORT`, normally `:8443` | Approved `$CLIENT_CIDR` |
| `nemotron-asr-backend-sg` | backend instances | TCP `:8080` | `nemotron-asr-lb-sg` source group |

Important constraint: [../ec2-bench/ec2_up.py](../ec2-bench/ec2_up.py)
only provisions the shared `nemotron-bench-sg` and only authorizes `:22` from
`MY_IP`. It does not create the LB or backend access-control security groups.
Section 1.5 reconciles and attaches those additional SGs after provisioning.

Public IPs exist because `ec2_up.py` provisions public-subnet instances. Phase 1
still routes LB-to-backend traffic by private IP, and backend `:8080` must not
be reachable from the public internet.

## (1) Provision N g6e.4xlarge L40S Boxes

Cost reminder before running this section: each `g6e.4xlarge` L40S box is
roughly `$2/hr`. Provision only the count you intend to smoke, and run section 8
when finished.

Set the target box count from section 0.5:

```bash
export BOX_COUNT=8
```

Provision each backend with both a unique `NEMOTRON_EC2_NAME` and a unique
`NEMOTRON_EC2_STATE`. The name prevents `ec2_up.py` from reusing another box,
and the state file keeps later commands unambiguous.

**On-demand (the recommended default at any cluster size):**

```bash
source .venv-deploy/bin/activate
for n in $(seq 1 "$BOX_COUNT"); do
  NEMOTRON_EC2_ITYPE=g6e.4xlarge \
  NEMOTRON_EC2_NAME="nemotron-asr-box$n" \
  NEMOTRON_EC2_STATE=".instance_box$n.json" \
  python ec2-bench/ec2_up.py
done
```

**Spot (optional, ~35% cheaper but flaky):** set `NEMOTRON_EC2_SPOT=1`. The
script falls back across AZs the same way as on-demand, but real-world spot
availability for g6e.4xlarge is very capacity-constrained — multiple recent
attempts (2026-05-29) hit `InsufficientInstanceCapacity` in every AZ even with
default spot pricing. Spot quota for G & VT vCPUs also defaults to a small
number (e.g. 64 vCPUs = 4 g6e.4xlarge boxes max), so check the quota first:

```bash
aws service-quotas get-service-quota --service-code ec2 \
  --quota-code L-3819A6DF \
  --query 'Quota.Value' --output text
# Compare to BOX_COUNT × 16 vCPU.

# Then try spot:
for n in $(seq 1 "$BOX_COUNT"); do
  NEMOTRON_EC2_ITYPE=g6e.4xlarge \
  NEMOTRON_EC2_NAME="nemotron-asr-box$n" \
  NEMOTRON_EC2_STATE=".instance_box$n.json" \
  NEMOTRON_EC2_SPOT=1 \
  python ec2-bench/ec2_up.py
done
```

Recommended fallback pattern: try spot first; if any box returns "no
g6e.4xlarge capacity in any AZ", re-run that box's command without
`NEMOTRON_EC2_SPOT=1` to get on-demand. The on-demand G quota default is
much larger (e.g. 768 vCPUs = 48 boxes).

Expected output shape for each box:

```text
[key] created /.../ec2-bench/nemotron-bench-key.pem
[sg] reuse sg-...
[ami] ami-... Deep Learning Base ...
[launch] g6e.4xlarge us-west-2 az=... vpc=... subnet=... sg=... ...
[launch] i-0123456789abcdef0 starting
[wait] instance running ...
[ip] 198.51.100.10
[wait] ssh:22 reachable ...
[ssh] open

INSTANCE_ID=i-0123456789abcdef0
PUBLIC_IP=198.51.100.10
SSH: ssh -i /.../ec2-bench/nemotron-bench-key.pem ubuntu@198.51.100.10
```

Provision the LB host in the same VPC with a small instance:

```bash
NEMOTRON_EC2_ITYPE=t3.medium \
NEMOTRON_EC2_NAME=nemotron-asr-lb \
NEMOTRON_EC2_STATE=.instance_lb.json \
python ec2-bench/ec2_up.py
```

Expected output shape:

```text
[launch] t3.medium us-west-2 az=... vpc=... subnet=... sg=... ...
[wait] instance running ...
[ip] 198.51.100.20
[ssh] open
INSTANCE_ID=i-0abcdef0123456789
PUBLIC_IP=198.51.100.20
```

Fetch private backend IPs for HAProxy. The state files hold public IPs; HAProxy
must use private IPs.

```bash
: > /tmp/asr-backends.ips
: > /tmp/asr-fleet.txt
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  iid=$(jq -r .instance_id "$s")
  name=$(basename "$s" .json | sed 's/^\.instance_//')
  private_ip=$(aws ec2 describe-instances \
    --instance-ids "$iid" \
    --query 'Reservations[].Instances[].PrivateIpAddress' \
    --output text)
  printf '%s\n' "$private_ip" >> /tmp/asr-backends.ips
  printf '%s=%s\n' "$name" "$private_ip" >> /tmp/asr-fleet.txt
done
cat /tmp/asr-backends.ips
cat /tmp/asr-fleet.txt
```

Expected output shape:

```text
10.0.12.31
10.0.44.87
10.0.73.104
box1=10.0.12.31
box2=10.0.44.87
box3=10.0.73.104
```

This runbook uses `gen_haproxy.py --boxes "$(paste -sd, /tmp/asr-backends.ips)"`
so backend names are the generator defaults, for example `box_10-0-12-31`. The
`/tmp/asr-fleet.txt` file is only an audit record unless you intentionally want
custom backend names.

Rollback for a bad provision: terminate only the instance whose state file you
are discarding.

```bash
BAD_STATE=ec2-bench/.instance_box1.json
aws ec2 terminate-instances --instance-ids "$(jq -r .instance_id "$BAD_STATE")"
aws ec2 wait instance-terminated --instance-ids "$(jq -r .instance_id "$BAD_STATE")"
```

Expected output shape:

```text
{
  "TerminatingInstances": [
    {
      "InstanceId": "i-0123456789abcdef0",
      "CurrentState": { "Name": "shutting-down", ... }
    }
  ]
}
```

## (1.5) Network and Security Groups
<a id="15-network-and-security-groups"></a>

`ec2_up.py` attaches the single fixed `nemotron-bench-sg` to every instance.
That SG gives SSH access only. The block below is idempotent and re-runnable on
a partially configured fleet, including after a failed-box replacement in
section 6b.

Set `CLIENT_CIDR` to the approved client range and `FRONT_PORT` to the LB
listener port. Use `8443` for TLS and `8080` for a plain internal smoke.

```bash
CLIENT_CIDR="203.0.113.0/24"   # set to your approved range
FRONT_PORT=8443                # 8443 for TLS, 8080 plain — must match gen_haproxy.py
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
      --query 'Vpcs[0].VpcId' --output text)
# describe-or-create both SGs (idempotent)
ensure_sg(){ local name="$1" desc="$2"
  local id=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$name \
        Name=vpc-id,Values=$VPC --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [ -z "$id" ] || [ "$id" = None ]; then
    id=$(aws ec2 create-security-group --group-name "$name" --description "$desc" \
          --vpc-id $VPC --query GroupId --output text)
  fi
  echo "$id"; }
LB_SG=$(ensure_sg nemotron-asr-lb-sg "Nemotron ASR LB ingress")
BE_SG=$(ensure_sg nemotron-asr-backend-sg "Nemotron ASR backend ingress")
# ingress rules — duplicates yield InvalidPermission.Duplicate (harmless, tolerate)
aws ec2 authorize-security-group-ingress --group-id $LB_SG \
      --protocol tcp --port $FRONT_PORT --cidr "$CLIENT_CIDR" 2>/dev/null || true
aws ec2 authorize-security-group-ingress --group-id $BE_SG \
      --protocol tcp --port 8080 --source-group $LB_SG 2>/dev/null || true
```

Then attach the new SGs to instances alongside `nemotron-bench-sg` so SSH still
works. `modify-instance-attribute --groups` replaces the full SG list, so the
function includes existing groups and de-duplicates before modifying.

```bash
attach_sg(){ local id="$1" new_sg="$2"
  local existing=$(aws ec2 describe-instances --instance-ids $id \
        --query 'Reservations[].Instances[].SecurityGroups[].GroupId' --output text)
  # de-dupe: only add new_sg if not already attached
  local final=$(echo "$existing $new_sg" | tr ' \t' '\n' | sort -u | tr '\n' ' ')
  aws ec2 modify-instance-attribute --instance-id $id --groups $final; }
for s in ec2-bench/.instance_box*.json; do
  attach_sg "$(jq -r .instance_id "$s")" "$BE_SG"
done
attach_sg "$(jq -r .instance_id ec2-bench/.instance_lb.json)" "$LB_SG"
# verify
aws ec2 describe-instances --instance-ids \
      $(jq -r .instance_id ec2-bench/.instance_lb.json) \
      --query 'Reservations[].Instances[].SecurityGroups'
```

Expected verify output includes both the original bench SG and the LB SG:

```text
[
  [
    {
      "GroupName": "nemotron-bench-sg",
      "GroupId": "sg-..."
    },
    {
      "GroupName": "nemotron-asr-lb-sg",
      "GroupId": "sg-..."
    }
  ]
]
```

Verify backend SG membership:

```bash
aws ec2 describe-instances \
  --instance-ids $(jq -r .instance_id ec2-bench/.instance_box*.json) \
  --query 'Reservations[].Instances[].{InstanceId:InstanceId,Groups:SecurityGroups[].GroupName}' \
  --output table
```

Expected output shape:

```text
------------------------------------------------------------
|                   DescribeInstances                      |
+----------------------+-----------------------------------+
|  InstanceId          |  i-0123456789abcdef0              |
|  Groups              |  nemotron-bench-sg                |
|                      |  nemotron-asr-backend-sg          |
+----------------------+-----------------------------------+
```

Fix if the LB cannot reach backend `:8080`: re-run both code blocks above, then
confirm `nemotron-asr-backend-sg` has source-group ingress from
`nemotron-asr-lb-sg`, not a CIDR typo.

## (2) Bootstrap Each Box

Rsync the repository to each backend at `$HOME/nemotron` using a **whitelist**
of the files the backend actually needs. The whitelist idiom is safer than
trying to maintain a complete blacklist — blacklist failure caught in the
2026-05-28 load-test validation: with a `--exclude=`-only list, the operator's
local `eou-collect/` (53 GB), `.cache/huggingface/` (multi-GB), and several
other unlisted top-level dirs would have transferred and made the rsync run
for an hour. The whitelist below sends ~1 MB, ~40 files.

The included paths cover everything the backend needs to run
`python -m nemotron_speech.server` (the package + pyproject.toml for the
editable install + bootstrap.sh for the venv setup). `--exclude='*'` at the
end is required by rsync's whitelist idiom — without it everything else
slips through.

```bash
KEY=ec2-bench/nemotron-bench-key.pem
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  rsync -az \
    --include='/pyproject.toml' \
    --include='/README.md' \
    --include='/src/' \
    --include='/src/nemotron_speech/' \
    --include='/src/nemotron_speech/***' \
    --include='/deploy/' \
    --include='/deploy/***' \
    --include='/ec2-bench/' \
    --include='/ec2-bench/bootstrap.sh' \
    --include='/ec2-bench/local_lb.py' \
    --include='/ec2-bench/ec2_up.py' \
    --include='/ec2-bench/ec2_down.py' \
    --exclude='*' \
    -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    ./ "ubuntu@${pub_ip}:~/nemotron/"
done
```

The whitelist deliberately excludes `ec2-bench/*.pem` and `ec2-bench/*.key`
by default (they don't match any `--include`), so the cluster SSH key
`ec2-bench/nemotron-bench-key.pem` cannot leak.

Expected output: `rsync` is quiet on success. To confirm that secrets did not
copy:

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" \
    'find ~/nemotron/ec2-bench -maxdepth 1 \( -name "*.pem" -o -name "*.key" \) -print'
done
```

Expected output: no output.

Run the DLAMI bootstrap on each backend. This creates `~/nemo-venv`, installs
torch and NeMo from the pinned NeMo commit, and pre-downloads the model when the
network allows it.

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" \
    'bash ~/nemotron/ec2-bench/bootstrap.sh'
done
```

Expected output tail:

```text
[bootstrap 12:34:56] smoke: import torch+nemo + GPU
torch 2.x.y cuda_avail True NVIDIA L40S
nemo 2.x.y /home/ubuntu/nemo-venv/lib/python3.11/site-packages/nemo/__init__.py
[bootstrap 12:35:10] DONE
```

Install the local package into the bootstrap venv:

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" \
    'export PATH="$HOME/.local/bin:$PATH" && uv pip install --no-deps --python "$HOME/nemo-venv" -e "$HOME/nemotron"'
done
```

`--no-deps` is required. The bootstrap intentionally installs the validated
runtime subset, while `pyproject.toml` declares unrelated dependencies such as
`pipecat-ai`, `riva-client`, and `gdown`. Installing without `--no-deps` can
pull a broader graph and overwrite validated torch/NeMo wheels with newer
versions.

Expected output shape:

```text
Resolved 1 package in ...
Installed 1 package in ...
 + nemotron-speech==...
```

Verify imports:

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" \
    '$HOME/nemo-venv/bin/python -c "import nemotron_speech.server; print(\"server import ok\")"'
done
```

Expected output:

```text
server import ok
server import ok
```

Fix if bootstrap fails during NeMo install: retry once for network flake. If the
pinned commit no longer fetches, see troubleshooting item 14 before changing
[../ec2-bench/bootstrap.sh](../ec2-bench/bootstrap.sh).

## (3) Install Systemd Unit and Env

Install [launch_single.sh](launch_single.sh), [nemotron-asr.service](nemotron-asr.service),
and the environment template on every backend:

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" <<'REMOTE'
set -euo pipefail
chmod +x ~/nemotron/deploy/launch_single.sh
sudo cp ~/nemotron/deploy/nemotron-asr.service /etc/systemd/system/nemotron-asr.service
sudo install -d -m 0755 /etc/nemotron
sudo install -m 0644 ~/nemotron/deploy/asr.env.example /etc/nemotron/asr.env
sudo grep -n '^NEMOTRON_ADMISSION_MAX_BACKLOG=12' /etc/nemotron/asr.env
sudo systemctl daemon-reload
sudo systemctl enable --now nemotron-asr
sudo systemctl --no-pager --full status nemotron-asr | sed -n '1,12p'
REMOTE
done
```

Expected output shape:

```text
31:NEMOTRON_ADMISSION_MAX_BACKLOG=12
Created symlink /etc/systemd/system/multi-user.target.wants/nemotron-asr.service -> /etc/systemd/system/nemotron-asr.service.
* nemotron-asr.service - Nemotron streaming ASR single-process server
     Loaded: loaded (/etc/systemd/system/nemotron-asr.service; enabled; preset: enabled)
     Active: active (running) since ...
```

The `RECOMMENDED-EXPLICIT` block in [asr.env.example](asr.env.example) contains
values that match launcher defaults but are worth making visible in ops. The
minimum recommended explicit value is:

```text
NEMOTRON_ADMISSION_MAX_BACKLOG=12
```

It is not functionally required because [launch_single.sh](launch_single.sh) and
the unit already default to `12`. It is set explicitly so an operator can see
the admission cap in `/etc/nemotron/asr.env`. Treat the `OPTIONAL` block as
tuning-only.

Follow model load on one backend at a time:

```bash
BACKEND_PUB_IP=$(jq -r .ip ec2-bench/.instance_box1.json)
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" \
  'sudo journalctl -u nemotron-asr -f'
```

Expected log line after several minutes on first boot:

```text
ASR server listening on ws://0.0.0.0:8080
Health check available at http://0.0.0.0:8080/health
```

Press `Ctrl-C` after the listening line appears, then verify local health on
each backend:

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" \
    'curl -fsS http://127.0.0.1:8080/health | jq .'
done
```

Expected output for each backend:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "admission": {
    "enabled": true
  }
}
```

If `status` stays `loading`, the HTTP server may not have reached bind yet.
Use the journal, `nvidia-smi`, and troubleshooting item 1.

Once healthy, the server also exposes `GET /stats` (always-on rolling-latency
endpoint added post-Phase-1):

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  pub_ip=$(jq -r .ip "$s")
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${pub_ip}" \
    'curl -fsS http://127.0.0.1:8080/stats?last=100 | jq .metrics.vad_stop_to_sent_ms'
done
```

Returns the latest 100 finalizes' server-side TTFS percentiles
(`vad_stop_to_sent_ms` = vad_stop → final_sent, the server-side equivalent
of the client TTFB). The full `/stats` response also includes
`fork_flush_wall_ms`, `vad_stop_recv_to_process_ms` (intake-backlog signal),
`lock_wait_ms`, `vad_stop_to_finalize_start_ms`, the concurrent-session
distribution, and admission counters. Drop `?last=N` for the entire sliding
window (default 2048 samples; set via `NEMOTRON_STATS_WINDOW`).

## (4) Configure LB Host

First copy deploy artifacts to the LB host using the same **whitelist**
pattern as section (2) — the LB needs `deploy/` (gen_haproxy.py, drain.sh)
and `ec2-bench/local_lb.py`, nothing else. The whitelist also keeps
`ec2-bench/*.pem` out: the cluster SSH key must not land on the LB.

```bash
LB_PUB_IP=$(jq -r .ip ec2-bench/.instance_lb.json)
KEY=ec2-bench/nemotron-bench-key.pem
rsync -az \
  --include='/pyproject.toml' \
  --include='/README.md' \
  --include='/src/' \
  --include='/src/nemotron_speech/' \
  --include='/src/nemotron_speech/***' \
  --include='/deploy/' \
  --include='/deploy/***' \
  --include='/ec2-bench/' \
  --include='/ec2-bench/bootstrap.sh' \
  --include='/ec2-bench/local_lb.py' \
  --include='/ec2-bench/ec2_up.py' \
  --include='/ec2-bench/ec2_down.py' \
  --exclude='*' \
  -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  ./ "ubuntu@${LB_PUB_IP}:~/nemotron/"
scp -i "$KEY" -o StrictHostKeyChecking=accept-new /tmp/asr-backends.ips "ubuntu@${LB_PUB_IP}:~/asr-backends.ips"
```

Expected output: `rsync` and `scp` complete without errors.

Install LB prerequisites:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" <<'REMOTE'
# NB: apt-get update is intentionally NOT under `set -e`. On the DLAMI, the
# bundled CUDA repos sometimes register duplicate Translations/Packages targets,
# which makes apt-get update emit warnings and exit nonzero. With set -e
# active, the whole heredoc aborts before haproxy/socat install — observed
# twice during 2026-05-28/29 LB provisioning. Run update separately and
# tolerate its rc, then proceed under set -e for the install steps.
sudo apt-get update -qq || true
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y haproxy socat python3 jq unzip curl
if ! command -v aws >/dev/null 2>&1 || ! aws --version 2>&1 | grep -q 'aws-cli/2'; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  rm -rf /tmp/aws
  unzip -q -o /tmp/awscliv2.zip -d /tmp
  if command -v aws >/dev/null 2>&1; then
    sudo /tmp/aws/install --update
  else
    sudo /tmp/aws/install
  fi
fi
haproxy -v | head -1
socat -V | head -1
python3 --version
jq --version
aws --version
REMOTE
```

Expected output shape:

```text
HAProxy version 2.x.y
socat by Gerhard Rieger ...
Python 3.x.y
jq-1.x
aws-cli/2.x.y Python/...
```

The following steps must be run in this exact order: install PEM with correct
permissions, generate into `/tmp`, install config with `sudo`, validate as root,
then start HAProxy.

1. Install the TLS PEM at `/etc/haproxy/asr.pem`. The PEM must be concatenated
   `cert + chain + privkey` in HAProxy format.

```bash
TLS_PEM_LOCAL=${TLS_PEM_LOCAL:-./asr.pem}
scp -i "$KEY" -o StrictHostKeyChecking=accept-new "$TLS_PEM_LOCAL" "ubuntu@${LB_PUB_IP}:~/asr.pem"
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" <<'REMOTE'
set -euo pipefail
sudo install -d -m 0755 /etc/haproxy
sudo cp "$HOME/asr.pem" /etc/haproxy/asr.pem
sudo chown haproxy:haproxy /etc/haproxy/asr.pem
sudo chmod 0600 /etc/haproxy/asr.pem
sudo ls -l /etc/haproxy/asr.pem
REMOTE
```

Expected output:

```text
-rw------- 1 haproxy haproxy ... /etc/haproxy/asr.pem
```

2. Generate to `/tmp/haproxy.cfg`, then install to `/etc/haproxy/haproxy.cfg`.

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" <<'REMOTE'
set -euo pipefail
BOXES=$(paste -sd, "$HOME/asr-backends.ips")
python3 ~/nemotron/deploy/gen_haproxy.py \
  --boxes "$BOXES" \
  --maxconn 20 \
  --tls-port 8443 \
  --tls-pem /etc/haproxy/asr.pem \
  -o /tmp/haproxy.cfg
sudo install -m 0644 -o root -g root /tmp/haproxy.cfg /etc/haproxy/haproxy.cfg
REMOTE
```

Expected output: no output and exit code `0`.

Plain internal smoke variant, only if section 1.5 used `FRONT_PORT=8080`:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" <<'REMOTE'
set -euo pipefail
BOXES=$(paste -sd, "$HOME/asr-backends.ips")
python3 ~/nemotron/deploy/gen_haproxy.py \
  --boxes "$BOXES" \
  --maxconn 20 \
  --front-port 8080 \
  -o /tmp/haproxy.cfg
sudo install -m 0644 -o root -g root /tmp/haproxy.cfg /etc/haproxy/haproxy.cfg
REMOTE
```

3. Validate as root so HAProxy can read the PEM:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  'sudo haproxy -c -f /etc/haproxy/haproxy.cfg'
```

Expected output:

```text
Configuration file is valid
```

4. Stop any package default instance, then enable and start HAProxy:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  'sudo systemctl stop haproxy 2>/dev/null || true; sudo systemctl enable --now haproxy'
```

Expected output shape:

```text
Created symlink /etc/systemd/system/multi-user.target.wants/haproxy.service -> /lib/systemd/system/haproxy.service.
```

5. Verify the HAProxy Runtime API sees the ASR pool:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  "echo 'show stat' | sudo socat /run/haproxy/admin.sock stdio | grep asr_pool"
```

Expected output shape after roughly `rise * inter` seconds:

```text
asr_pool,box_10-0-12-31,0,0,0,20,...,UP,...
asr_pool,box_10-0-44-87,0,0,0,20,...,UP,...
asr_pool,BACKEND,0,0,0,200,...
```

Certificate rotation: replace `/etc/haproxy/asr.pem` with the new concatenated
PEM, keep owner `haproxy:haproxy` and mode `0600`, then run:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  'sudo systemctl reload haproxy'
```

Expected output: no output and existing WebSocket streams remain connected. Use
`reload`, not `restart`.

## (4b) Stable LB Endpoint via Elastic IP

By default the LB instance's public IP is auto-assigned and is **released
when the instance is stopped or terminated**. For any client that hardcodes
the LB URL — including a DNS record at daily.co — this is the wrong default.
Allocate an Elastic IP and associate it with the LB before publishing the
endpoint. Once attached, the EIP follows the instance through stop/start
(needed for the resize in section 4c), and survives a terminate-and-replace
of the LB instance (re-associate with the new instance).

EIPs are **free while associated** with a running instance and accrue
roughly `$3.60/month` when sitting unassociated.

```bash
LB_ID=$(jq -r .instance_id ec2-bench/.instance_lb.json)

# Allocate-or-reuse, by Name tag (idempotent across re-runs)
ALLOC=$(aws ec2 describe-addresses \
  --filters Name=tag:Name,Values=nemotron-asr-lb-eip \
  --query 'Addresses[0].AllocationId' --output text 2>/dev/null)
if [ -z "$ALLOC" ] || [ "$ALLOC" = None ]; then
  ALLOC=$(aws ec2 allocate-address --domain vpc \
    --tag-specifications 'ResourceType=elastic-ip,Tags=[{Key=Name,Value=nemotron-asr-lb-eip}]' \
    --query AllocationId --output text)
  echo "[eip] allocated $ALLOC"
else
  echo "[eip] reused $ALLOC"
fi

# Associate with the LB instance (the existing auto-assigned IP is released)
aws ec2 associate-address --allocation-id "$ALLOC" --instance-id "$LB_ID" \
  --query AssociationId --output text

# Verify + persist the new IP into the local state file so subsequent
# commands and clients use it.
NEW_IP=$(aws ec2 describe-instances --instance-ids "$LB_ID" \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text)
echo "[eip] LB public IP is now $NEW_IP"
TMP=$(mktemp); jq --arg ip "$NEW_IP" '.ip = $ip' ec2-bench/.instance_lb.json > "$TMP" \
  && mv "$TMP" ec2-bench/.instance_lb.json
```

Expected behaviour: a brief WebSocket disconnect window (single-digit
seconds) while AWS releases the old IP and routes the new one. Existing
streams drop and must reconnect; the LB SG ingress rules and backend SG are
unaffected.

After this section, smoke (section 5) uses `ws://$NEW_IP:8080`. If you also
plan to publish a DNS name, add an `A` record at daily.co pointing to the
EIP — see DEPLOYMENT.md for the broader DNS+TLS path.

Rollback: `aws ec2 disassociate-address --association-id <assoc-id>` and the
LB falls back to a fresh auto-assigned public IP. Release the EIP with
`aws ec2 release-address --allocation-id "$ALLOC"` if you want to stop the
idle-allocation charge.

## (4c) Resizing the LB without losing the EIP

At cluster sizes around 24+ backends, t3.medium's 2 vCPU and 4 GiB may feel
tight under sustained load. With section 4b's EIP attached, you can bump
the LB instance type without changing the URL — the EIP and private IP both
follow the instance through stop/start.

```bash
LB_ID=$(jq -r .instance_id ec2-bench/.instance_lb.json)
NEW_TYPE=t3.xlarge   # or t3.large; pick based on observed CPU under target conc

aws ec2 stop-instances --instance-ids "$LB_ID" --query 'StoppingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-stopped --instance-ids "$LB_ID"

aws ec2 modify-instance-attribute --instance-id "$LB_ID" --instance-type "Value=$NEW_TYPE"

aws ec2 start-instances --instance-ids "$LB_ID" --query 'StartingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-running --instance-ids "$LB_ID"

# Verify type + IPs + that HAProxy auto-recovered
aws ec2 describe-instances --instance-ids "$LB_ID" \
  --query 'Reservations[].Instances[].[InstanceType,PublicIpAddress,PrivateIpAddress,State.Name]' \
  --output table

# Update local state's itype field
TMP=$(mktemp); jq --arg t "$NEW_TYPE" '.itype = $t' ec2-bench/.instance_lb.json > "$TMP" \
  && mv "$TMP" ec2-bench/.instance_lb.json
```

Expected behaviour: ~60 seconds of LB downtime (stop ~30s + start ~20s +
boot/HAProxy auto-start ~10s). HAProxy starts on boot because
`systemctl enable --now haproxy` was set in section 4. Verify it came up
and all backends rejoined:

```bash
LB_PUB_IP=$(jq -r .ip ec2-bench/.instance_lb.json)
KEY=ec2-bench/nemotron-bench-key.pem
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  "echo 'show stat' | sudo socat /run/haproxy/admin.sock stdio | \
   awk -F, '\$1==\"asr_pool\" && \$2!=\"BACKEND\" {printf \"  %-22s %s\n\", \$2, \$18}'"
```

Expected output: all N backends with status `UP` within ~5 seconds of
HAProxy starting.

Instance-type guidance from 2026-05-29 measurements at the 50-conc / 4-box
scale:

- `t3.medium` (2 vCPU, 4 GiB): fine up to ~50 conc through ~4-5 boxes.
- `t3.large` (2 vCPU, 8 GiB): same vCPU but ~50% more baseline CPU credit
  budget — better for sustained load, not more cores.
- `t3.xlarge` (4 vCPU, 16 GiB): doubled cores — recommended floor for any
  cluster going above 100-conc or 12+ backends. HAProxy at this scale
  rarely saturates a single core, but ssh/socat-based ops tooling and
  syslog have headroom.

## (5) End-to-End Smoke Through the LB

Run this section from the operator workstation in a venv that has `websockets`
installed. The smoke client is
[../tests/test_websocket_client.py](../tests/test_websocket_client.py), which
streams a fixture WAV over the WebSocket in realtime and reports the transcript
plus the finalization latency.

Pick the URL:

```bash
source .venv-deploy/bin/activate
LB_IP=$(jq -r .ip ec2-bench/.instance_lb.json)
```

With TLS, the certificate is issued for a DNS name, not the EC2 public IP. Point
corporate DNS, Route53, or a local `/etc/hosts` entry at `$LB_IP`, then set:

```bash
LB_HOST=lb.example.com
LB_URL="wss://${LB_HOST}:8443"
```

Without TLS, only for an internal Phase 1 smoke where the LB SG allows your
CIDR on `:8080`:

```bash
LB_URL="ws://${LB_IP}:8080"
```

Run the cloud smoke (one stream through the LB; append `/?model=en` to select
the model):

```bash
python tests/test_websocket_client.py tests/fixtures/harvard_16k.wav "$LB_URL/?model=en"
```

Expected output shape:

```text
  Sent <N> chunks (<bytes> bytes) in <ms>ms

Waiting for final transcript...

============================================================
FINAL TRANSCRIPT:
============================================================
<a sensible transcript of the fixture WAV>
============================================================

Statistics:
  Interim results: <N>
  Total time: <ms>ms
  Real-time factor: <~1.0>x

Finalization latency:
  Last audio chunk -> final transcript: <ms>ms
  End signal -> final transcript: <ms>ms
```

A clean smoke shows a sensible transcript and no WebSocket errors; the
finalization latency should be small (tens of ms on an idle box). To drive
concurrency, run several clients in parallel (`for i in $(seq 5); do python
tests/test_websocket_client.py ... & done`). If a client closes with WebSocket
`1013`, server-side admission fired; lower the concurrency, verify per-box
active streams with [drain.sh](drain.sh) `status`, and consider
`NEMOTRON_ADMISSION_MAX_BACKLOG` only after understanding the load.

Verify zero-drop reload during the smoke. Start the smoke in one terminal; while
it is running, reload HAProxy from a second terminal:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  'sudo systemctl reload haproxy'
```

Expected result: the smoke still finishes with `ok=10 errors=0`. A reload
should preserve active streams; a restart will drop them.

## (6) Rolling Redeploy

Run drains against the LB host, and run backend restarts from the workstation.
The SSH key is deliberately not present on the LB host because section 2 and
section 4 exclude `*.pem` and `*.key`.

For one backend, derive the HAProxy server name from its private IP:

```bash
STATE=ec2-bench/.instance_box1.json
BACKEND_PUB_IP=$(jq -r .ip "$STATE")
BACKEND_ID=$(jq -r .instance_id "$STATE")
BACKEND_PRIV_IP=$(aws ec2 describe-instances \
  --instance-ids "$BACKEND_ID" \
  --query 'Reservations[].Instances[].PrivateIpAddress' \
  --output text)
BOX="box_${BACKEND_PRIV_IP//./-}"
printf '%s %s %s\n' "$BOX" "$BACKEND_PUB_IP" "$BACKEND_PRIV_IP"
```

Expected output:

```text
box_10-0-12-31 198.51.100.10 10.0.12.31
```

Drain, wait empty, restart, wait local health, then mark ready:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  "cd ~/nemotron && sudo deploy/drain.sh drain '$BOX'"
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  "cd ~/nemotron && sudo deploy/drain.sh wait-empty '$BOX' 300"
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" \
  'sudo systemctl restart nemotron-asr'
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" \
  'until curl -fsS http://127.0.0.1:8080/health | jq -e ".status==\"healthy\"" >/dev/null; do sleep 2; done; echo healthy'
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  "cd ~/nemotron && sudo deploy/drain.sh ready '$BOX'"
```

Expected output:

```text
[drain] asr_pool/box_10-0-12-31 state drain requested
[drain] asr_pool/box_10-0-12-31 is empty
healthy
[drain] asr_pool/box_10-0-12-31 state ready requested
```

Loop version for the whole fleet:

```bash
for s in $(ls ec2-bench/.instance_box*.json | sort -V); do
  BACKEND_PUB_IP=$(jq -r .ip "$s")
  BACKEND_ID=$(jq -r .instance_id "$s")
  BACKEND_PRIV_IP=$(aws ec2 describe-instances \
    --instance-ids "$BACKEND_ID" \
    --query 'Reservations[].Instances[].PrivateIpAddress' \
    --output text)
  BOX="box_${BACKEND_PRIV_IP//./-}"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
    "cd ~/nemotron && sudo deploy/drain.sh drain '$BOX'"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
    "cd ~/nemotron && sudo deploy/drain.sh wait-empty '$BOX' 300"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" \
    'sudo systemctl restart nemotron-asr'
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" \
    'until curl -fsS http://127.0.0.1:8080/health | jq -e ".status==\"healthy\"" >/dev/null; do sleep 2; done; echo healthy'
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
    "cd ~/nemotron && sudo deploy/drain.sh ready '$BOX'"
done
```

Recovery if `wait-empty` times out: active streams can legally last up to one
hour because generated HAProxy config uses `timeout server 1h`. The operator
must choose either to extend the wait window or force-kill/restart the backend,
which drops active clients and relies on client reconnect through the LB.

## (6b) Replacing a Failed Box

This is not a rolling redeploy. Use it when a box died or was terminated.

Provision the replacement with the same `NEMOTRON_EC2_NAME` and state file. If
the old instance is no longer pending/running, `ec2_up.py` creates a fresh one.

```bash
N=1
NEMOTRON_EC2_ITYPE=g6e.4xlarge \
NEMOTRON_EC2_NAME="nemotron-asr-box$N" \
NEMOTRON_EC2_STATE=".instance_box$N.json" \
python ec2-bench/ec2_up.py
```

Expected output shape:

```text
[launch] i-0replacement starting
[wait] instance running ...
[ip] 198.51.100.111
[ssh] open
```

Rehydrate the backend SG from AWS by name because the original shell variables
are usually gone:

```bash
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
BE_SG=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=nemotron-asr-backend-sg Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)
echo "$BE_SG"
```

Expected output:

```text
sg-0123456789abcdef0
```

Re-run the attach function from section 1.5 for the replacement:

```bash
attach_sg(){ local id="$1" new_sg="$2"
  local existing=$(aws ec2 describe-instances --instance-ids $id \
        --query 'Reservations[].Instances[].SecurityGroups[].GroupId' --output text)
  # de-dupe: only add new_sg if not already attached
  local final=$(echo "$existing $new_sg" | tr ' \t' '\n' | sort -u | tr '\n' ' ')
  aws ec2 modify-instance-attribute --instance-id $id --groups $final; }
attach_sg "$(jq -r .instance_id "ec2-bench/.instance_box${N}.json")" "$BE_SG"
```

Expected output: no output and exit code `0`.

Fetch the replacement private IP:

```bash
NEW_STATE="ec2-bench/.instance_box${N}.json"
NEW_ID=$(jq -r .instance_id "$NEW_STATE")
NEW_PUB_IP=$(jq -r .ip "$NEW_STATE")
NEW_PRIV_IP=$(aws ec2 describe-instances \
  --instance-ids "$NEW_ID" \
  --query 'Reservations[].Instances[].PrivateIpAddress' \
  --output text)
printf 'public=%s private=%s\n' "$NEW_PUB_IP" "$NEW_PRIV_IP"
```

Expected output:

```text
public=198.51.100.111 private=10.0.88.42
```

Bootstrap the replacement and install systemd by re-running sections 2 and 3 for
`$NEW_STATE`. Confirm local health before touching HAProxy:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${NEW_PUB_IP}" \
  'curl -fsS http://127.0.0.1:8080/health | jq .'
```

Expected output:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Update the backend IP file, regenerate HAProxy config, validate, and reload:

```bash
cp /tmp/asr-backends.ips /tmp/asr-backends.ips.bak
OLD_PRIV_IP=10.0.12.31
grep -v -x "$OLD_PRIV_IP" /tmp/asr-backends.ips.bak > /tmp/asr-backends.ips
printf '%s\n' "$NEW_PRIV_IP" >> /tmp/asr-backends.ips
scp -i "$KEY" -o StrictHostKeyChecking=accept-new /tmp/asr-backends.ips "ubuntu@${LB_PUB_IP}:~/asr-backends.ips"
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" <<'REMOTE'
set -euo pipefail
BOXES=$(paste -sd, "$HOME/asr-backends.ips")
python3 ~/nemotron/deploy/gen_haproxy.py \
  --boxes "$BOXES" \
  --maxconn 20 \
  --tls-port 8443 \
  --tls-pem /etc/haproxy/asr.pem \
  -o /tmp/haproxy.cfg
sudo install -m 0644 -o root -g root /tmp/haproxy.cfg /etc/haproxy/haproxy.cfg
sudo haproxy -c -f /etc/haproxy/haproxy.cfg
sudo systemctl reload haproxy
echo 'show stat' | sudo socat /run/haproxy/admin.sock stdio | grep asr_pool
REMOTE
```

Expected output includes:

```text
Configuration file is valid
asr_pool,box_10-0-88-42,...,UP,...
```

Order note: HAProxy health checks make regen-first safe because an unready
backend is marked DOWN after `fall * inter`; regen-after-healthy is cleaner
because it avoids a wasted reload and brief DOWN-flap noise.

## (7) Switching to the K=3+MPS Fallback

Use the fallback only when a trip-wire in [DEPLOYMENT.md](DEPLOYMENT.md) fires
under sustained multi-turn load. The fallback is [launch_multiproc.sh](launch_multiproc.sh):
it starts CUDA MPS, launches K processes, and runs its own supervisor loop.

Known compatibility gap: [launch_multiproc.sh](launch_multiproc.sh) runs
`python server.py` from `$NEMOTRON_APP_DIR`, but Phase 1 installs the package
layout with `src/nemotron_speech/server.py`, not a flat `$HOME/nemotron/server.py`.
Choose one of these operator bridges before swapping systemd.

Option A, symlink the flat legacy entrypoints into the app directory:

```bash
BACKEND_PUB_IP=$(jq -r .ip ec2-bench/.instance_box1.json)
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" <<'REMOTE'
set -euo pipefail
cd "$HOME/nemotron"
ln -sfn "$HOME/nemotron/src/nemotron_speech/server.py" "$HOME/nemotron/server.py"
ln -sfn "$HOME/nemotron/src/nemotron_speech/batch_primitives.py" "$HOME/nemotron/batch_primitives.py"
ln -sfn "$HOME/nemotron/src/nemotron_speech/cudagraph_encoder.py" "$HOME/nemotron/cudagraph_encoder.py"
"$HOME/nemo-venv/bin/python" -c 'import pathlib; assert pathlib.Path("server.py").exists(); print("flat layout ready")'
REMOTE
```

Expected output:

```text
flat layout ready
```

Option B, edit a local copy of the launcher to use the package module:

```bash
BACKEND_PUB_IP=$(jq -r .ip ec2-bench/.instance_box1.json)
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" <<'REMOTE'
set -euo pipefail
cp ~/nemotron/deploy/launch_multiproc.sh ~/nemotron/deploy/launch_multiproc_phase1_module.sh
python3 - <<'PY'
from pathlib import Path
p = Path.home() / "nemotron/deploy/launch_multiproc_phase1_module.sh"
text = p.read_text()
old = '"$VENV/bin/python" server.py --model "$MODEL" \\'
new = '"$VENV/bin/python" -m nemotron_speech.server --model "$MODEL" \\'
if old not in text:
    raise SystemExit("expected legacy server.py invocation not found")
p.write_text(text.replace(old, new))
PY
chmod +x ~/nemotron/deploy/launch_multiproc_phase1_module.sh
grep -n -- '-m nemotron_speech.server' ~/nemotron/deploy/launch_multiproc_phase1_module.sh
REMOTE
```

Expected output:

```text
77:  env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" -m nemotron_speech.server --model "$MODEL" \
```

Swap the systemd `ExecStart` with a drop-in. Use `launch_multiproc.sh` for
Option A or `launch_multiproc_phase1_module.sh` for Option B:

```bash
BACKEND_PUB_IP=$(jq -r .ip ec2-bench/.instance_box1.json)
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${BACKEND_PUB_IP}" <<'REMOTE'
set -euo pipefail
sudo install -d -m 0755 /etc/systemd/system/nemotron-asr.service.d
sudo tee /etc/systemd/system/nemotron-asr.service.d/10-multiproc.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/home/ubuntu/nemotron/deploy/launch_multiproc.sh
EOF
sudo systemctl daemon-reload
sudo systemctl restart nemotron-asr
sudo journalctl -u nemotron-asr -n 40 --no-pager
REMOTE
```

Expected output includes:

```text
[mps] daemon up
[launch] proc 0 -> pid ... port 8080
[launch] proc 1 -> pid ... port 8081
[launch] proc 2 -> pid ... port 8082
```

Regenerate HAProxy for three ports per box. Known gap:
[gen_haproxy.py](gen_haproxy.py) does not yet have a `--ports-per-box` flag, so
this runbook uses the existing generator for the base config and then performs a
manual K=3 server-line expansion before install. Keep this as an operator patch
until the generator grows the flag.

```bash
scp -i "$KEY" -o StrictHostKeyChecking=accept-new /tmp/asr-backends.ips "ubuntu@${LB_PUB_IP}:~/asr-backends.ips"
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" <<'REMOTE'
set -euo pipefail
BOXES=$(paste -sd, "$HOME/asr-backends.ips")
python3 ~/nemotron/deploy/gen_haproxy.py \
  --boxes "$BOXES" \
  --maxconn 7 \
  --tls-port 8443 \
  --tls-pem /etc/haproxy/asr.pem \
  -o /tmp/haproxy.cfg
python3 - <<'PY'
from pathlib import Path
cfg = Path("/tmp/haproxy.cfg")
lines = cfg.read_text().splitlines()
out = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("server box_") and ":8080 " in stripped:
        parts = stripped.split()
        base_name = parts[1]
        host = parts[2].rsplit(":", 1)[0]
        tail = " ".join(parts[3:])
        indent = line[:len(line) - len(line.lstrip())]
        for port in (8080, 8081, 8082):
            out.append(f"{indent}server {base_name}_p{port - 8080} {host}:{port} {tail}")
    else:
        out.append(line)
cfg.write_text("\n".join(out) + "\n")
PY
sudo install -m 0644 -o root -g root /tmp/haproxy.cfg /etc/haproxy/haproxy.cfg
sudo haproxy -c -f /etc/haproxy/haproxy.cfg
sudo systemctl reload haproxy
grep -E 'server box_.*:808[012]' /etc/haproxy/haproxy.cfg | head
REMOTE
```

Expected output:

```text
Configuration file is valid
    server box_10-0-12-31_p0 10.0.12.31:8080 check inter 2s fall 3 rise 2 maxconn 7
    server box_10-0-12-31_p1 10.0.12.31:8081 check inter 2s fall 3 rise 2 maxconn 7
    server box_10-0-12-31_p2 10.0.12.31:8082 check inter 2s fall 3 rise 2 maxconn 7
```

Systemd interaction note: with [launch_multiproc.sh](launch_multiproc.sh), the
unit PID is the shell supervisor. `Restart=on-failure` restarts that supervisor,
which restarts MPS and all K processes. Accept the wider blast radius as the
documented MPS trade-off.

## (8) Teardown

Drain all backends first so no new streams are assigned:

```bash
for ip in $(cat /tmp/asr-backends.ips); do
  BOX="box_${ip//./-}"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
    "cd ~/nemotron && sudo deploy/drain.sh drain '$BOX'"
done
```

Expected output shape:

```text
[drain] asr_pool/box_10-0-12-31 state drain requested
[drain] asr_pool/box_10-0-44-87 state drain requested
```

Optionally wait for every backend to empty:

```bash
for ip in $(cat /tmp/asr-backends.ips); do
  BOX="box_${ip//./-}"
  ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
    "cd ~/nemotron && sudo deploy/drain.sh wait-empty '$BOX' 300"
done
```

Stop HAProxy:

```bash
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@${LB_PUB_IP}" \
  'sudo systemctl stop haproxy'
```

Terminate the backend and LB instances from their state files:

```bash
INSTANCE_IDS=$(jq -r .instance_id ec2-bench/.instance_box*.json ec2-bench/.instance_lb.json)
aws ec2 terminate-instances --instance-ids $INSTANCE_IDS
aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS
```

Expected output shape:

```text
{
  "TerminatingInstances": [
    { "InstanceId": "i-...", "CurrentState": { "Name": "shutting-down" } }
  ]
}
```

Cost reminder: the state files are not the source of billing; EC2 instances are.
Confirm there are no running Phase 1 instances:

```bash
aws ec2 describe-instances \
  --filters Name=tag:Name,Values='nemotron-asr-*' Name=instance-state-name,Values=pending,running \
  --query 'Reservations[].Instances[].{Name:Tags[?Key==`Name`]|[0].Value,InstanceId:InstanceId,State:State.Name}' \
  --output table
```

Expected output after teardown:

```text
---------------------
|DescribeInstances  |
+-------------------+
```

## (9) Troubleshooting Matrix

1. Symptom: model never loads or `/health` stays `loading`.
   Cause: NeMo import error, model download stall, no GPU, or full HF cache disk.
   Fix:
   ```bash
   sudo journalctl -u nemotron-asr -n 200 --no-pager
   nvidia-smi
   df -h "$HOME/hf"
   ```
   Resolve the logged import/download/GPU/disk error, then
   `sudo systemctl restart nemotron-asr`.

2. Symptom: `/health` flaps UP/DOWN at the LB.
   Cause: the server process is crashing and systemd is restarting it.
   Fix:
   ```bash
   sudo systemctl status nemotron-asr --no-pager
   sudo journalctl -u nemotron-asr -n 200 --no-pager
   ```
   Fix the trace, restart the unit, and wait for local `/health` green before
   marking ready.

3. Symptom: LB shows backend DOWN but the server says healthy locally.
   Cause: backend SG does not allow `:8080` from `nemotron-asr-lb-sg`.
   Fix: re-run section 1.5 and verify the backend ingress source is the LB SG
   ID, not a CIDR or the wrong SG name.

4. Symptom: `drain.sh wait-empty` always times out.
   Cause: active streams are still open; generated HAProxy permits streams up to
   `timeout server 1h`.
   Fix: extend the timeout or force-kill after accepting client impact:
   ```bash
   cd ~/nemotron
   sudo deploy/drain.sh wait-empty box_10-0-12-31 1800
   ```

5. Symptom: `systemctl restart` loop.
   Cause: the launcher or Python exits repeatedly; `RestartSec=5` only paces
   retries and can flood journald.
   Fix:
   ```bash
   sudo journalctl -u nemotron-asr -n 300 --no-pager
   sudo systemctl reset-failed nemotron-asr
   ```
   Fix the first Python/launcher error in the journal.

6. Symptom: GPU not visible to a systemd-launched process.
   Cause: DLAMI driver issue or unit user mismatch.
   Fix:
   ```bash
   nvidia-smi
   sudo -u ubuntu nvidia-smi
   systemctl cat nemotron-asr | grep '^User='
   ```
   The unit should run as `ubuntu`; fix the user/driver issue and restart.

7. Symptom: port `8080` conflict on a backend.
   Cause: another service is bound to the backend port.
   Fix:
   ```bash
   sudo ss -tnlp | grep ':8080'
   CONFLICTING_SERVICE=example.service
   sudo systemctl stop "$CONFLICTING_SERVICE"
   sudo systemctl restart nemotron-asr
   ```

8. Symptom: server OOM kill.
   Cause: GPU or host memory pressure.
   Fix:
   ```bash
   sudo dmesg -T | grep -i 'out of memory\|killed process'
   nvidia-smi
   ```
   Reduce `NEMOTRON_BATCH_MAX_SIZE` or re-verify
   `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1` in `/etc/nemotron/asr.env`,
   then restart after the box is drained.

9. Symptom: `bootstrap.sh` NeMo install fails.
   Cause: pinned commit fetch flake, GitHub outage, or dependency build failure.
   Fix:
   ```bash
   bash ~/nemotron/ec2-bench/bootstrap.sh
   ```
   Retry once. If the commit is gone, use item 14.

10. Symptom: TLS certificate expired.
    Cause: Phase 1 has manual TLS material and no renewal automation.
    Fix:
    ```bash
    sudo cp ~/asr.pem /etc/haproxy/asr.pem
    sudo chown haproxy:haproxy /etc/haproxy/asr.pem
    sudo chmod 0600 /etc/haproxy/asr.pem
    sudo systemctl reload haproxy
    ```

11. Symptom: `1013` close-storm from one box.
    Cause: that box's server-side admission cap is firing.
    Fix:
    ```bash
    curl -fsS http://127.0.0.1:8080/health | jq .admission
    curl -fsS http://127.0.0.1:8080/stats?last=200 \
      | jq '{ttfs: .metrics.vad_stop_to_sent_ms, intake: .metrics.vad_stop_recv_to_process_ms, active_sessions: .active_sessions_at_emit}'
    ```
    Check attempted/admitted/rejected counters and the recent latency
    distribution. The intake metric climbing into seconds while
    `active_sessions_at_emit.p95` is at/over `maxconn` is the canonical
    overload pattern. Lower HAProxy `maxconn` for that box, lower client
    concurrency, or tune `NEMOTRON_ADMISSION_MAX_BACKLOG` after validating
    tail latency.

12. Symptom: HAProxy reload drops connections.
    Cause: operator used `restart` or killed HAProxy instead of reload.
    Fix:
    ```bash
    sudo systemctl reload haproxy
    ```
    Do not use `restart` during active traffic.

13. Symptom: all backends DOWN at the LB.
    Cause: every server crashed, SG/network path broke, or HAProxy health check
    cannot match healthy JSON.
    Fix:
    ```bash
    echo 'show stat' | sudo socat /run/haproxy/admin.sock stdio | grep asr_pool
    ```
    Then recover each backend locally with `journalctl`, `systemctl restart`, and
    `/health` before marking ready.

14. Symptom: pinned NeMo commit no longer fetchable from GitHub.
    Cause: upstream force-pushed, removed, or temporarily cannot serve the
    pinned commit.
    Fix: document the next-known-good commit in [DEPLOYMENT.md](DEPLOYMENT.md),
    update `NEMO_COMMIT` in [../ec2-bench/bootstrap.sh](../ec2-bench/bootstrap.sh),
    then re-run bootstrap. Phase 2 should mirror the source artifact in S3.

15. Symptom: timestamps disagree across boxes.
    Cause: NTP/chrony drift.
    Fix:
    ```bash
    timedatectl status
    systemctl status chrony --no-pager || systemctl status systemd-timesyncd --no-pager
    ```
    DLAMI should self-heal within minutes once time sync is running.

16. Symptom: journald disk full.
    Cause: restart loops or verbose logs filled journal storage.
    Fix:
    ```bash
    sudo journalctl --vacuum-size=500M
    printf '%s\n' 'SystemMaxUse=500M' | sudo tee -a /etc/systemd/journald.conf
    sudo systemctl restart systemd-journald
    ```

17. Symptom: `haproxy -c` rejects the generated config.
    Cause: PEM unreadable, unsupported HAProxy version, malformed manual edit, or
    bad backend list.
    Fix:
    ```bash
    sudo ls -l /etc/haproxy/asr.pem
    haproxy -v
    sudo haproxy -c -f /etc/haproxy/haproxy.cfg
    ```
    Fix the first parser error, regenerate with [gen_haproxy.py](gen_haproxy.py),
    and re-run `sudo haproxy -c -f /etc/haproxy/haproxy.cfg` before reload.

## (10) Open Risks Accepted by Phase 1

1. Sustained multi-turn load is unvalidated. The `maxconn 20` default comes from
   one-utterance-per-connection tests, so sustained streams may need lower caps
   or the K=3+MPS fallback.

2. No IaC/ASG/ALB. Manual EC2 and HAProxy are faster for Phase 1 but leave
   repeatability and managed lifecycle work for Phase 2.

3. No autoscaling. Capacity changes are manual provision, bootstrap, HAProxy
   regenerate, and reload.

4. Single LB host. One HAProxy EC2 instance is a single point of failure until a
   dual-LB, keepalived, or ALB design replaces it.

5. No TLS automation or renewal. Operators provide and rotate
   `/etc/haproxy/asr.pem` by hand.

6. No app-level WebSocket auth. Phase 1 relies on `CLIENT_CIDR` and SG isolation
   as the access-control boundary.

7. No monitoring or alerting. Operators use `/health`, `show stat`, `journalctl`,
   and smoke output manually.

8. No log shipping. Journald is local to each EC2 host, so terminated boxes lose
   their logs unless an operator collects them first.

9. No GPU memory time series. Operators use point-in-time `nvidia-smi`; there is
   no retained GPU memory or utilization history.

10. Runtime dependency drift is accepted. `bootstrap.sh` pins NeMo by commit but
    leaves torch unpinned and installs `uv` live; Phase 2 should pin or mirror
    the full runtime set.
