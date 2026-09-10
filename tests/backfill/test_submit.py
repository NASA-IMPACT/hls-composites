import json

import boto3
import pytest
from moto import mock_aws

from hls_composites.backfill.submit import ACTIVE_JOB_STATUSES, BackfillSubmitter
from hls_composites.models import YearMonth

QUEUE = "test-queue"
JOB_DEF = "test-jd"
JUNE_2016 = YearMonth(2016, 6)


class FakePaginator:
    """Returns a fixed page of job summaries for every status."""

    def __init__(self, per_status):
        self.per_status = per_status

    def paginate(self, *, jobQueue, jobStatus):
        yield {"jobSummaryList": [{"jobId": str(i)} for i in range(self.per_status)]}


class FakeBatchClient:
    """Minimal Batch stub. moto does not keep jobs in active states."""

    def __init__(self, per_status=0, fail_on=None):
        self.per_status = per_status
        self.fail_on = fail_on or set()
        self.submitted = []

    def get_paginator(self, name):
        assert name == "list_jobs"
        return FakePaginator(self.per_status)

    def submit_job(self, **kwargs):
        name = kwargs["jobName"]
        if any(marker in name for marker in self.fail_on):
            raise RuntimeError(f"ThrottlingException on {name}")
        self.submitted.append(kwargs)
        return {"jobId": f"job-{len(self.submitted)}"}


@pytest.fixture
def batch_env():
    with mock_aws(config={"batch": {"use_docker": False}}):
        ec2 = boto3.client("ec2", region_name="us-west-2")
        iam = boto3.client("iam", region_name="us-west-2")
        batch = boto3.client("batch", region_name="us-west-2")

        vpc = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
        subnet = ec2.create_subnet(VpcId=vpc, CidrBlock="10.0.0.0/24")["Subnet"][
            "SubnetId"
        ]
        sg = ec2.create_security_group(
            GroupName="sg-test", Description="test", VpcId=vpc
        )["GroupId"]
        role = iam.create_role(RoleName="batch-role", AssumeRolePolicyDocument="{}")[
            "Role"
        ]["Arn"]
        profile = iam.create_instance_profile(InstanceProfileName="batch-profile")[
            "InstanceProfile"
        ]["Arn"]

        compute_env = batch.create_compute_environment(
            computeEnvironmentName="test-ce",
            type="MANAGED",
            state="ENABLED",
            computeResources={
                "type": "EC2",
                "maxvCpus": 16,
                "minvCpus": 0,
                "instanceRole": profile,
                "instanceTypes": ["optimal"],
                "subnets": [subnet],
                "securityGroupIds": [sg],
            },
            serviceRole=role,
        )["computeEnvironmentArn"]
        batch.create_job_queue(
            jobQueueName=QUEUE,
            state="ENABLED",
            priority=1,
            computeEnvironmentOrder=[{"order": 1, "computeEnvironment": compute_env}],
        )
        batch.register_job_definition(
            jobDefinitionName=JOB_DEF,
            type="container",
            containerProperties={
                "image": "busybox",
                "vcpus": 1,
                "memory": 128,
                "command": ["true"],
            },
        )
        yield batch


def test_active_job_statuses_cover_every_pre_terminal_state():
    assert set(ACTIVE_JOB_STATUSES) == {
        "SUBMITTED",
        "PENDING",
        "RUNNABLE",
        "STARTING",
        "RUNNING",
    }


def test_below_threshold_when_queue_is_empty():
    submitter = BackfillSubmitter(FakeBatchClient(per_status=0), QUEUE, JOB_DEF)
    assert submitter.active_jobs_below_threshold(10)


def test_not_below_threshold_when_queue_is_full():
    submitter = BackfillSubmitter(FakeBatchClient(per_status=100), QUEUE, JOB_DEF)
    assert not submitter.active_jobs_below_threshold(10)


def test_threshold_counts_across_all_statuses():
    submitter = BackfillSubmitter(FakeBatchClient(per_status=3), QUEUE, JOB_DEF)
    assert not submitter.active_jobs_below_threshold(15)
    assert submitter.active_jobs_below_threshold(16)


def test_submit_unit_sets_the_monitoring_parameters(batch_env):
    submitter = BackfillSubmitter(batch_env, QUEUE, JOB_DEF)
    job_id = submitter.submit_unit("14TPN", JUNE_2016)

    parameters = batch_env.describe_jobs(jobs=[job_id])["jobs"][0]["parameters"]
    assert parameters["bejm_job_type"] == "monthly-composite"
    assert json.loads(parameters["bejm_input_entity_ids"]) == ["14TPN_2016-06"]
    assert parameters["bejm_output_entity_id"] == "HLS.M30.T14TPN.2016153.2016182.v2.0"
    assert parameters["bejm_attempt"] == "1"


def test_submit_unit_partition_fields_match_the_athena_key_path(batch_env):
    """The Athena tables project job_type then year_month; only year_month is a field."""
    submitter = BackfillSubmitter(batch_env, QUEUE, JOB_DEF)
    job_id = submitter.submit_unit("14TPN", JUNE_2016)

    parameters = batch_env.describe_jobs(jobs=[job_id])["jobs"][0]["parameters"]
    fields = json.loads(parameters["bejm_partition_fields"])
    assert list(fields) == ["year_month"]
    assert fields["year_month"] == "2016-06"


def test_submit_unit_passes_the_cli_arguments():
    client = FakeBatchClient()
    submitter = BackfillSubmitter(client, QUEUE, JOB_DEF)
    submitter.submit_unit("14TPN", JUNE_2016)

    command = client.submitted[0]["containerOverrides"]["command"]
    assert command == ["--tile-id", "14TPN", "--year-month", "2016-06"]
    assert client.submitted[0]["jobQueue"] == QUEUE
    assert client.submitted[0]["jobDefinition"] == JOB_DEF


def test_submit_units_returns_the_number_submitted():
    client = FakeBatchClient()
    submitter = BackfillSubmitter(client, QUEUE, JOB_DEF)
    units = [("14TPN", JUNE_2016), ("14TPM", JUNE_2016), ("14TPL", JUNE_2016)]
    assert submitter.submit_units(units) == 3
    assert len(client.submitted) == 3


def test_submit_units_stops_at_the_first_failure():
    client = FakeBatchClient(fail_on={"14TPM"})
    submitter = BackfillSubmitter(client, QUEUE, JOB_DEF)
    units = [("14TPN", JUNE_2016), ("14TPM", JUNE_2016), ("14TPL", JUNE_2016)]

    assert submitter.submit_units(units) == 1
    assert len(client.submitted) == 1


def test_submit_units_handles_an_empty_batch():
    submitter = BackfillSubmitter(FakeBatchClient(), QUEUE, JOB_DEF)
    assert submitter.submit_units([]) == 0
