#!/usr/bin/env python3
"""Spin up (or reuse) a g6.4xlarge in us-west-2 for the ASR baseline benchmark.

Idempotent: reuses a running 'nemotron-bench' instance / existing key+SG if present.
Creates NOTHING destructive. Writes ec2-bench/.instance.json with id+ip for the other scripts.

  stt-benchmark/.venv/bin/python ec2-bench/ec2_up.py
"""
import json
import os
import socket
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# AWS profile: set NEMOTRON_AWS_PROFILE to your SSO/named profile, or leave unset to
# use the default boto3 credential chain (env vars, default profile, instance role).
PROFILE = os.environ.get("NEMOTRON_AWS_PROFILE") or None
REGION = os.environ.get("AWS_REGION", "us-west-2")
ITYPE = os.environ.get("NEMOTRON_EC2_ITYPE", "g6.4xlarge")
KEY = os.environ.get("NEMOTRON_EC2_KEY", "nemotron-bench-key")
SG = os.environ.get("NEMOTRON_EC2_SG", "nemotron-bench-sg")
NAME = os.environ.get("NEMOTRON_EC2_NAME", "nemotron-bench-" + ITYPE)  # override so same-ITYPE parallel runs don't reuse each other's box
HERE = Path(__file__).resolve().parent
PEM = HERE / f"{KEY}.pem"
STATE = HERE / os.environ.get("NEMOTRON_EC2_STATE", ".instance.json")  # override for parallel runs (separate boxes)
# Your public IP, used to scope the SSH (:22) ingress rule. Required — set MY_IP=$(curl -s ifconfig.me).
MY_IP = os.environ.get("MY_IP")
if not MY_IP:
    raise SystemExit("set MY_IP to your public IP for the SSH ingress rule, e.g. MY_IP=$(curl -s ifconfig.me)")

ec2 = boto3.Session(profile_name=PROFILE).client("ec2", region_name=REGION)


def find_ami():
    for pat in ("Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*",
                "Deep Learning Base GPU AMI (Ubuntu 22.04)*",
                "Deep Learning OSS Nvidia Driver AMI GPU PyTorch*(Ubuntu 22.04)*"):
        r = ec2.describe_images(Owners=["amazon"], Filters=[
            {"Name": "name", "Values": [pat]}, {"Name": "state", "Values": ["available"]}])
        imgs = sorted(r["Images"], key=lambda i: i["CreationDate"], reverse=True)
        if imgs:
            return imgs[0]["ImageId"], imgs[0]["Name"]
    raise SystemExit("no Deep Learning Base GPU AMI found in " + REGION)


r = ec2.describe_instances(Filters=[
    {"Name": "tag:Name", "Values": [NAME]},
    {"Name": "instance-state-name", "Values": ["pending", "running"]}])
existing = [i for res in r["Reservations"] for i in res["Instances"]]

if existing:
    inst = existing[0]
    print(f"[reuse] {inst['InstanceId']} state={inst['State']['Name']}")
else:
    if not PEM.exists():
        try:
            ec2.delete_key_pair(KeyName=KEY)
        except ClientError:
            pass
        kp = ec2.create_key_pair(KeyName=KEY)
        PEM.write_text(kp["KeyMaterial"])
        PEM.chmod(0o600)
        print(f"[key] created {PEM}")
    else:
        print(f"[key] reuse {PEM}")

    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["Subnets"]
    # Try across AZs (g6e capacity is often AZ-constrained -> InsufficientInstanceCapacity in one AZ).
    cand_subnets = [s for s in subnets if s.get("MapPublicIpOnLaunch")] or subnets

    try:
        sg = ec2.describe_security_groups(Filters=[
            {"Name": "group-name", "Values": [SG]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"][0]["GroupId"]
        print(f"[sg] reuse {sg}")
    except (IndexError, ClientError):
        sg = ec2.create_security_group(GroupName=SG, Description="nemotron asr bench ssh", VpcId=vpc)["GroupId"]
        print(f"[sg] created {sg}")
    try:
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": f"{MY_IP}/32", "Description": "bench ssh"}]}])
        print(f"[sg] ingress 22 <- {MY_IP}/32")
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise

    ami, aminame = find_ami()
    print(f"[ami] {ami}  {aminame}")
    # NEMOTRON_EC2_SPOT=1 requests a one-time spot instance with no max price
    # (defaults to on-demand ceiling per AWS). For multi-box clusters where
    # losing 1 box to spot interruption is acceptable, this saves ~35% on g6e.
    USE_SPOT = os.environ.get("NEMOTRON_EC2_SPOT", "") == "1"
    market_opts = ({"MarketType": "spot",
                    "SpotOptions": {"SpotInstanceType": "one-time",
                                    "InstanceInterruptionBehavior": "terminate"}}
                   if USE_SPOT else None)
    if USE_SPOT:
        print(f"[launch] SPOT requested (NEMOTRON_EC2_SPOT=1)")

    inst = None
    last_err = None
    for s in cand_subnets:
        subnet = s["SubnetId"]; az = s.get("AvailabilityZone", "?")
        print(f"[launch] {ITYPE} {REGION} az={az} vpc={vpc} subnet={subnet} sg={sg} ...")
        try:
            kwargs = dict(
                ImageId=ami, InstanceType=ITYPE, KeyName=KEY, MinCount=1, MaxCount=1,
                NetworkInterfaces=[{"DeviceIndex": 0, "AssociatePublicIpAddress": True,
                                    "Groups": [sg], "SubnetId": subnet}],
                BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                      "Ebs": {"VolumeSize": 200, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}]}])
            if market_opts is not None:
                kwargs["InstanceMarketOptions"] = market_opts
            r = ec2.run_instances(**kwargs)
            inst = r["Instances"][0]
            break
        except ClientError as e:
            msg = str(e)
            if ("InsufficientInstanceCapacity" in msg or "Unsupported" in msg
                or "SpotMaxPriceTooLow" in msg or "MaxSpotInstanceCountExceeded" in msg):
                print(f"[launch] az={az} no capacity ({type(e).__name__}); trying next AZ ...")
                last_err = e
                continue
            raise
    if inst is None:
        raise SystemExit(f"no {ITYPE} capacity in any AZ of {REGION}: {last_err}")
    print(f"[launch] {inst['InstanceId']} starting")

iid = inst["InstanceId"]
print("[wait] instance running ...")
ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
ip = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0].get("PublicIpAddress")
print(f"[ip] {ip}")

print("[wait] ssh:22 reachable ...")
ok = False
for _ in range(72):
    try:
        with socket.create_connection((ip, 22), timeout=5):
            ok = True
            break
    except OSError:
        time.sleep(5)
print("[ssh] open" if ok else "[ssh] not yet reachable (cloud-init still booting)")

STATE.write_text(json.dumps({"instance_id": iid, "ip": ip, "region": REGION,
                             "key": str(PEM), "itype": ITYPE, "user": "ubuntu"}, indent=2))
print(f"\nINSTANCE_ID={iid}\nPUBLIC_IP={ip}\nSSH: ssh -i {PEM} ubuntu@{ip}")
