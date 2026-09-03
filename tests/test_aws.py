import os
from typing import ClassVar

import boto3
import pytest
from moto import mock_aws
from moto.core import DEFAULT_ACCOUNT_ID

from hls_composites.aws import CREDENTIAL_ENV_VARS, assumed_role_env, upload_directory

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
    tests that need to control the *starting* state cannot run inside it --
    clearing those variables breaks moto's own teardown. The AWS behaviour
    (that assume_role works, and that the exported credentials authenticate)
    is covered by the moto tests in TestAssumeRole below.
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


@pytest.fixture
def fake_sts(monkeypatch):
    sts = FakeSts()
    monkeypatch.setattr(boto3, "client", lambda service, *a, **k: sts)
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


class FakeS3:
    def __init__(self):
        self.uploads: list[tuple[str, str, str]] = []

    def upload_file(self, filename, bucket, key):
        self.uploads.append((filename, bucket, key))


class TestUploadDirectory:
    def test_uploads_every_file_under_the_granule_prefix(self, tmp_path):
        dest = tmp_path / "HLS.M30.T14TPN.2020032.2020060.v2.0"
        dest.mkdir()
        for name in ("a.NDVI.tif", "a.EVI.tif"):
            (dest / name).write_bytes(b"x")
        s3 = FakeS3()

        keys = upload_directory(s3, dest, "out-bucket", dest.name)

        assert keys == [
            "HLS.M30.T14TPN.2020032.2020060.v2.0/a.EVI.tif",
            "HLS.M30.T14TPN.2020032.2020060.v2.0/a.NDVI.tif",
        ]
        assert [bucket for _, bucket, _ in s3.uploads] == ["out-bucket"] * 2

    def test_nested_files_keep_their_relative_path(self, tmp_path):
        dest = tmp_path / "granule"
        (dest / "sub").mkdir(parents=True)
        (dest / "sub" / "b.tif").write_bytes(b"x")

        assert upload_directory(FakeS3(), dest, "out", "granule") == [
            "granule/sub/b.tif"
        ]

    def test_empty_directory_uploads_nothing(self, tmp_path):
        dest = tmp_path / "granule"
        dest.mkdir()
        s3 = FakeS3()

        assert upload_directory(s3, dest, "out", "granule") == []
        assert s3.uploads == []

    @mock_aws
    def test_objects_actually_land_in_the_bucket(self, tmp_path):
        """End to end against a real S3 client, not a recording stub."""
        s3 = boto3.client("s3")
        s3.create_bucket(
            Bucket="out-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        dest = tmp_path / "granule"
        dest.mkdir()
        (dest / "a.NDVI.tif").write_bytes(b"pixels")

        upload_directory(s3, dest, "out-bucket", "M30/data/granule")

        body = s3.get_object(Bucket="out-bucket", Key="M30/data/granule/a.NDVI.tif")
        assert body["Body"].read() == b"pixels"
