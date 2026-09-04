import os
from typing import ClassVar

import boto3
import pytest
from moto import mock_aws
from moto.core import DEFAULT_ACCOUNT_ID

from hls_composites.aws import (
    CREDENTIAL_ENV_VARS,
    REQUESTER,
    REQUESTER_PAYS_ENV_VAR,
    assumed_role_env,
    requester_pays_env,
    upload_directory,
)

ROLE_ARN = f"arn:aws:iam::{DEFAULT_ACCOUNT_ID}:role/reader"


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch):
    """No ambient credentials, so the code under test owns every AWS_* var."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeSts:
    """A minimal STS stub for the environment save/restore tests.

    moto seeds and restores the same AWS_* variables this module manages, so
    tests that control the starting state cannot run inside it.
    """

    CREDENTIALS: ClassVar[dict[str, str]] = {
        "AccessKeyId": "ASIAFAKE",
        "SecretAccessKey": "secret",
        "SessionToken": "token",
    }

    def __init__(self):
        self.calls: list[dict] = []

    def assume_role(self, **kwargs):
        self.calls.append(kwargs)
        return {"Credentials": self.CREDENTIALS}


class FakeSession:
    """A session whose only job is to hand back the stubbed STS client."""

    def __init__(self, sts: "FakeSts", **kwargs):
        self._sts = sts

    def client(self, service: str, *args, **kwargs):
        return self._sts


@pytest.fixture
def fake_sts(monkeypatch):
    sts = FakeSts()
    monkeypatch.setattr(boto3, "Session", lambda **kwargs: FakeSession(sts, **kwargs))
    return sts


class TestCredentialScoping:
    """How the context manager treats os.environ. No AWS involved."""

    def test_no_role_leaves_the_environment_untouched(self):
        """The local path: fall through to the ambient credential chain."""
        before = dict(os.environ)

        with assumed_role_env(None):
            assert dict(os.environ) == before

        assert dict(os.environ) == before

    @pytest.mark.parametrize("role_arn", [None, ""])
    def test_absent_role_never_calls_sts(self, fake_sts, role_arn):
        """An empty string counts as absent -- the container always sets the var."""
        with assumed_role_env(role_arn):
            pass

        assert fake_sts.calls == []

    def test_role_exports_credentials_for_gdal_and_boto(self, fake_sts):
        """Set in os.environ, not a session: dask worker threads read them too."""
        with assumed_role_env("arn:aws:iam::123456789012:role/reader"):
            assert os.environ["AWS_ACCESS_KEY_ID"] == "ASIAFAKE"
            assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
            assert os.environ["AWS_SESSION_TOKEN"] == "token"

        assert fake_sts.calls[0]["RoleArn"].endswith("role/reader")

    def test_previous_credentials_are_restored(self, fake_sts, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "OUTER")

        with assumed_role_env("arn:aws:iam::123456789012:role/reader"):
            assert os.environ["AWS_ACCESS_KEY_ID"] == "ASIAFAKE"

        assert os.environ["AWS_ACCESS_KEY_ID"] == "OUTER"

    def test_credentials_are_cleared_when_there_were_none(self, fake_sts):
        with assumed_role_env("arn:aws:iam::123456789012:role/reader"):
            pass

        for name in CREDENTIAL_ENV_VARS:
            assert name not in os.environ

    def test_credentials_are_restored_when_the_body_raises(self, fake_sts):
        with (
            pytest.raises(ZeroDivisionError),
            assumed_role_env("arn:aws:iam::123456789012:role/reader"),
        ):
            raise ZeroDivisionError

        assert "AWS_ACCESS_KEY_ID" not in os.environ


class TestRequesterPaysEnv:
    """LP DAAC's buckets are requester-pays; S3 ignores this elsewhere."""

    def test_sets_the_variable_gdal_reads(self, monkeypatch):
        monkeypatch.delenv(REQUESTER_PAYS_ENV_VAR, raising=False)

        with requester_pays_env():
            assert os.environ[REQUESTER_PAYS_ENV_VAR] == REQUESTER

    def test_clears_it_afterwards_when_it_was_unset(self, monkeypatch):
        monkeypatch.delenv(REQUESTER_PAYS_ENV_VAR, raising=False)

        with requester_pays_env():
            pass

        assert REQUESTER_PAYS_ENV_VAR not in os.environ

    def test_restores_a_previous_value(self, monkeypatch):
        monkeypatch.setenv(REQUESTER_PAYS_ENV_VAR, "outer")

        with requester_pays_env():
            assert os.environ[REQUESTER_PAYS_ENV_VAR] == REQUESTER

        assert os.environ[REQUESTER_PAYS_ENV_VAR] == "outer"

    def test_restores_when_the_body_raises(self, monkeypatch):
        monkeypatch.delenv(REQUESTER_PAYS_ENV_VAR, raising=False)

        with pytest.raises(ZeroDivisionError), requester_pays_env():
            raise ZeroDivisionError

        assert REQUESTER_PAYS_ENV_VAR not in os.environ


class TestAssumeRole:
    """Real STS behaviour, via moto."""

    @mock_aws
    def test_exported_credentials_replace_the_ambient_ones(self):
        """Reads the real STS response shape, against real botocore plumbing.

        Temporary keys start with ASIA (long-lived ones with AKIA), so this
        also catches exporting the wrong field out of the response.
        """
        ambient = os.environ["AWS_ACCESS_KEY_ID"]

        with assumed_role_env(ROLE_ARN):
            assumed = os.environ["AWS_ACCESS_KEY_ID"]
            assert assumed.startswith("ASIA")
            assert assumed != ambient
            assert os.environ["AWS_SESSION_TOKEN"]

        assert os.environ["AWS_ACCESS_KEY_ID"] == ambient

    @mock_aws
    def test_failure_to_assume_names_the_role(self):
        """A malformed ARN fails botocore's own validation, not a stub's."""
        with pytest.raises(RuntimeError, match="short"), assumed_role_env("short"):
            pass


def make_bucket(name: str = "out-bucket"):
    """A real (moto) S3 client with `name` created."""
    s3 = boto3.client("s3")
    s3.create_bucket(
        Bucket=name,
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )
    return s3


def keys_in(s3, bucket: str = "out-bucket") -> list[str]:
    listing = s3.list_objects_v2(Bucket=bucket)
    return sorted(item["Key"] for item in listing.get("Contents", []))


@mock_aws
class TestUploadDirectory:
    def test_uploads_every_file_under_the_granule_prefix(self, tmp_path):
        s3 = make_bucket()
        dest = tmp_path / "HLS.M30.T14TPN.2020032.2020060.v2.0"
        dest.mkdir()
        for name in ("a.NDVI.tif", "a.EVI.tif"):
            (dest / name).write_bytes(b"x")

        keys = upload_directory(s3, dest, "out-bucket", dest.name)

        assert keys == [
            "HLS.M30.T14TPN.2020032.2020060.v2.0/a.EVI.tif",
            "HLS.M30.T14TPN.2020032.2020060.v2.0/a.NDVI.tif",
        ]
        assert keys_in(s3) == keys

    def test_nested_files_keep_their_relative_path(self, tmp_path):
        s3 = make_bucket()
        dest = tmp_path / "granule"
        (dest / "sub").mkdir(parents=True)
        (dest / "sub" / "b.tif").write_bytes(b"x")

        assert upload_directory(s3, dest, "out-bucket", "granule") == [
            "granule/sub/b.tif"
        ]
        assert keys_in(s3) == ["granule/sub/b.tif"]

    def test_empty_directory_uploads_nothing(self, tmp_path):
        s3 = make_bucket()
        dest = tmp_path / "granule"
        dest.mkdir()

        assert upload_directory(s3, dest, "out-bucket", "granule") == []
        assert keys_in(s3) == []

    def test_content_survives_the_round_trip(self, tmp_path):
        s3 = make_bucket()
        dest = tmp_path / "granule"
        dest.mkdir()
        (dest / "a.NDVI.tif").write_bytes(b"pixels")

        upload_directory(s3, dest, "out-bucket", "M30/data/granule")

        body = s3.get_object(Bucket="out-bucket", Key="M30/data/granule/a.NDVI.tif")
        assert body["Body"].read() == b"pixels"
