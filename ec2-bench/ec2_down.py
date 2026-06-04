#!/usr/bin/env python3
"""Terminate the benchmark EC2 instance (stops billing). SG + key pair are left for reuse.

  stt-benchmark/.venv/bin/python ec2-bench/ec2_down.py
"""
import json
import os
from pathlib import Path

import boto3

# Set NEMOTRON_AWS_PROFILE to your SSO/named profile, or leave unset for the default credential chain.
PROFILE = os.environ.get("NEMOTRON_AWS_PROFILE") or None

HERE = Path(__file__).resolve().parent
st = json.loads((HERE / os.environ.get("NEMOTRON_EC2_STATE", ".instance.json")).read_text())  # match ec2_up.py STATE
ec2 = boto3.Session(profile_name=PROFILE).client("ec2", region_name=st["region"])
r = ec2.terminate_instances(InstanceIds=[st["instance_id"]])
state = r["TerminatingInstances"][0]["CurrentState"]["Name"]
print(f"terminate {st['instance_id']} ({st.get('itype')}) {st['region']} -> {state}")
print("(security group 'nemotron-bench-sg' + key pair left in place for reuse)")
