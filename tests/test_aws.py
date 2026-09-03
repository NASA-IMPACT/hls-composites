import os

import pytest

from hls_composites.aws import CREDENTIAL_ENV_VARS, assumed_role_env, upload_directory


class FakeSts:
    """Stands in for an STS client, recording the assume_role call."""

    def __init__(self, error: Exception | None = None):
        self.calls: list[dict] = []
        self.error = error

    def assume_role(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "Credentials": {
                "AccessKeyId": "AKIAFAKE",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch):
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestAssumedRoleEnv:
    def test_no_role_leaves_the_environment_untouched(self):
        """The local path: fall through to the ambient credential chain."""
        before = dict(os.environ)

        with assumed_role_env(None):
            assert dict(os.environ) == before

        assert dict(os.environ) == before

    @pytest.mark.parametrize("role_arn", [None, ""])
    def test_absent_role_never_calls_sts(self, monkeypatch, role_arn):
        """An empty string counts as absent -- the container always sets the var."""
        sts = FakeSts()
        monkeypatch.setattr("hls_composites.aws._sts_client", lambda: sts)

        with assumed_role_env(role_arn):
            pass

        assert sts.calls == []

    def test_role_exports_credentials_for_gdal_and_boto(self, monkeypatch):
        """Set in os.environ, not a session: dask worker threads read them too."""
        sts = FakeSts()
        monkeypatch.setattr("hls_composites.aws._sts_client", lambda: sts)

        with assumed_role_env("arn:aws:iam::123456789012:role/reader"):
            assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAFAKE"
            assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret"
            assert os.environ["AWS_SESSION_TOKEN"] == "token"

        assert sts.calls[0]["RoleArn"] == "arn:aws:iam::123456789012:role/reader"

    def test_previous_credentials_are_restored(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "OUTER")
        monkeypatch.setattr("hls_composites.aws._sts_client", lambda: FakeSts())

        with assumed_role_env("arn:aws:iam::123456789012:role/reader"):
            assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAFAKE"

        assert os.environ["AWS_ACCESS_KEY_ID"] == "OUTER"

    def test_credentials_are_cleared_when_there_were_none(self, monkeypatch):
        monkeypatch.setattr("hls_composites.aws._sts_client", lambda: FakeSts())

        with assumed_role_env("arn:aws:iam::123456789012:role/reader"):
            pass

        for name in CREDENTIAL_ENV_VARS:
            assert name not in os.environ

    def test_credentials_are_restored_when_the_body_raises(self, monkeypatch):
        monkeypatch.setattr("hls_composites.aws._sts_client", lambda: FakeSts())

        with (
            pytest.raises(ZeroDivisionError),
            assumed_role_env("arn:aws:iam::123456789012:role/reader"),
        ):
            raise ZeroDivisionError

        assert "AWS_ACCESS_KEY_ID" not in os.environ

    def test_failure_to_assume_names_the_role(self, monkeypatch):
        sts = FakeSts(error=RuntimeError("AccessDenied"))
        monkeypatch.setattr("hls_composites.aws._sts_client", lambda: sts)

        with (
            pytest.raises(RuntimeError, match="role/reader"),
            assumed_role_env("arn:aws:iam::123456789012:role/reader"),
        ):
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
        s3 = FakeS3()

        assert upload_directory(s3, dest, "out", "granule") == ["granule/sub/b.tif"]

    def test_empty_directory_uploads_nothing(self, tmp_path):
        dest = tmp_path / "granule"
        dest.mkdir()
        s3 = FakeS3()

        assert upload_directory(s3, dest, "out", "granule") == []
        assert s3.uploads == []
