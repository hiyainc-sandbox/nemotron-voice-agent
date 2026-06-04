# ec2-bench — minimal EC2 GPU provisioning helpers

Small, dependency-light helpers to bring up (and tear down) an EC2 GPU box for running the
Nemotron streaming-ASR server. These are the scripts the cluster deploy runbook
([`../deploy/RUNBOOK.md`](../deploy/RUNBOOK.md)) uses to provision L40S (`g6e`) instances. They
create nothing destructive and are idempotent.

This is **not** an IaC framework — it is a thin boto3 wrapper for "launch one Deep-Learning-AMI GPU
box, scope SSH to my IP, write its id/ip to a local state file." Bring your own provisioning if you
prefer Terraform/CloudFormation; the rest of the deploy runbook only needs N reachable Ubuntu boxes.

## What's here

| file | role |
|---|---|
| `ec2_up.py`   | launch (or reuse) a `nemotron-bench-<itype>` instance on a Deep Learning Base GPU AMI — key pair, SSH-only security group scoped to your IP, public IP, 200 GB gp3. Writes `.instance.json`. |
| `ec2_down.py` | terminate the instance (stops billing). Security group + key pair are left for reuse. |
| `bootstrap.sh`| on-box one-time setup: `uv` venv (py3.11), `torch` + deps + `nemo_toolkit[asr]` at a pinned commit, pre-download the public checkpoint, smoke-test torch+NeMo+GPU. |
| `local_lb.py` | a pure-TCP leastconn load balancer (a haproxy stand-in for local testing — no sudo, no deps). |

## Prerequisites

- **AWS credentials.** Set `NEMOTRON_AWS_PROFILE` to your SSO/named profile, or leave it unset to use
  the default boto3 credential chain (env vars, default profile, or an instance role).
- **boto3** in your Python environment.
- **EC2 G-instance quota** in your region for the instance type you pick.
- **Your public IP** for the SSH ingress rule: `export MY_IP=$(curl -s https://checkip.amazonaws.com)`.

## Usage

```bash
# 1. launch (default g6.4xlarge; override the type for an L40S box)
export MY_IP=$(curl -s https://checkip.amazonaws.com)
NEMOTRON_EC2_ITYPE=g6e.4xlarge python ec2-bench/ec2_up.py   # writes ec2-bench/.instance.json (id+ip+key)

# 2. bootstrap the runtime ON the box (~3-5 min: pip + NeMo-from-git build + checkpoint download)
IP=$(python -c "import json;print(json.load(open('ec2-bench/.instance.json'))['ip'])")
ssh -i ec2-bench/$NEMOTRON_EC2_KEY.pem -o StrictHostKeyChecking=no ubuntu@$IP \
    'cd ~ && bash bootstrap.sh'

# 3. deploy + serve: follow deploy/RUNBOOK.md from here (systemd unit, HAProxy, smoke).

# 4. ALWAYS tear down when done (the box bills while running)
python ec2-bench/ec2_down.py
```

## Config knobs (env vars)

- `NEMOTRON_EC2_ITYPE` — instance type, e.g. `g6.2xlarge`, `g6e.4xlarge`, `g6e.8xlarge`.
- `AWS_REGION` — region (default `us-west-2`).
- `NEMOTRON_AWS_PROFILE` — AWS profile (default: the default credential chain).
- `NEMOTRON_EC2_KEY` / `NEMOTRON_EC2_SG` — key-pair / security-group names to create or reuse.
- `NEMOTRON_EC2_NAME` / `NEMOTRON_EC2_STATE` — instance Name tag / state file (override for parallel boxes).
- `MY_IP` — your public IP for the SSH ingress rule (**required** by `ec2_up.py`).
- `NEMOTRON_EC2_SPOT=1` — request a one-time spot instance (saves ~35% on `g6e`; acceptable when losing a box to interruption is fine).

## Cost & safety

- `g6` ≈ $1/hr, `g6e` ≈ $2–4/hr on-demand. Always run `ec2_down.py` when finished.
- The `.pem` private key and `.instance.json` state files are git-ignored — **never commit them**.
- The checkpoint `nvidia/nemotron-speech-streaming-en-0.6b` is public (no HF token needed).
