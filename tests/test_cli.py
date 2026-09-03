from datetime import date

import pytest
from click.testing import CliRunner

from hls_composites import cli
from hls_composites.models import DateRange, Granule


def _patch_pipeline(monkeypatch, captured, granules):
    def fake_scan(s3, bucket, tile, date_range, **kwargs):
        captured["scan"] = {"bucket": bucket, "tile": tile, "date_range": date_range}
        return granules

    def fake_build(granules, **kwargs):
        captured["build"] = {"n": len(granules)}
        captured["build_kwargs"] = kwargs
        return "DATASET"

    def fake_write(ds, out_dir, tile, date_range, **kwargs):
        captured["write"] = {"ds": ds, "tile": tile, "date_range": date_range}
        return "/tmp/dest"

    monkeypatch.setattr(cli, "scan_bucket_for_granules", fake_scan)
    monkeypatch.setattr(cli, "build_composite", fake_build)
    monkeypatch.setattr(cli, "write_composite", fake_write)
    monkeypatch.setattr(cli.boto3, "client", lambda *a, **k: object())


def test_cli_maps_year_month_to_month_range_and_wires_args(monkeypatch, tmp_path):
    captured: dict = {}
    _patch_pipeline(
        monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "--tile-id",
            "14TPN",
            "--year-month",
            "2015-07",
            "--bucket",
            "my-bucket",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["scan"]["bucket"] == "my-bucket"
    assert captured["scan"]["tile"] == "14TPN"
    assert captured["scan"]["date_range"] == DateRange(
        date(2015, 7, 1), date(2015, 7, 31)
    )
    assert captured["build"] == {"n": 1}
    assert captured["write"]["tile"] == "14TPN"
    assert captured["write"]["ds"] == "DATASET"


def test_cli_bucket_falls_back_to_env(monkeypatch, tmp_path):
    captured: dict = {}
    _patch_pipeline(
        monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
    )
    monkeypatch.setenv("HLS_BUCKET", "env-bucket")

    result = CliRunner().invoke(
        cli.main,
        [
            "--tile-id",
            "14TPN",
            "--year-month",
            "2015-07",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["scan"]["bucket"] == "env-bucket"


def test_cli_rejects_malformed_year_month(monkeypatch, tmp_path):
    captured: dict = {}
    _patch_pipeline(monkeypatch, captured, [])

    result = CliRunner().invoke(
        cli.main,
        [
            "--tile-id",
            "14TPN",
            "--year-month",
            "2015/07",
            "--bucket",
            "my-bucket",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "YYYY-MM" in result.output


def test_cli_no_granules_skips_build_and_writes_marker(monkeypatch, tmp_path):
    captured: dict = {}
    _patch_pipeline(monkeypatch, captured, [])

    result = CliRunner().invoke(
        cli.main,
        [
            "--tile-id",
            "14TPN",
            "--year-month",
            "2015-07",
            "--bucket",
            "my-bucket",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "build" not in captured  # build_composite must not run
    marker = tmp_path / "HLS.M30.T14TPN.2015182.2015212.v2.0" / "No granules found"
    assert marker.exists()


def _invoke(tmp_path, *extra):
    return CliRunner().invoke(
        cli.main,
        [
            "--tile-id",
            "14TPN",
            "--year-month",
            "2015-07",
            "--bucket",
            "my-bucket",
            "--output-dir",
            str(tmp_path),
            *extra,
        ],
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [((), "indexes"), (("--indexes",), "indexes"), (("--bands",), "bands")],
)
def test_cli_flags_select_the_composite_output(monkeypatch, tmp_path, flags, expected):
    captured: dict = {}
    _patch_pipeline(
        monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
    )

    result = _invoke(tmp_path, *flags)

    assert result.exit_code == 0, result.output
    assert captured["build_kwargs"]["output"] == expected


def test_cli_rejects_bands_and_indexes_together(monkeypatch, tmp_path):
    captured: dict = {}
    _patch_pipeline(
        monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
    )

    result = _invoke(tmp_path, "--bands", "--indexes")

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert "build" not in captured


class TestOutputTarget:
    """--output-dir and --output-bucket are mutually exclusive, one required."""

    def _args(self, *extra):
        return [
            "--tile-id",
            "14TPN",
            "--year-month",
            "2015-07",
            "--bucket",
            "in-bucket",
            *extra,
        ]

    def test_both_targets_is_an_error(self, monkeypatch, tmp_path):
        _patch_pipeline(
            monkeypatch, {}, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )

        result = CliRunner().invoke(
            cli.main,
            self._args("--output-dir", str(tmp_path), "--output-bucket", "out"),
        )

        assert result.exit_code != 0
        assert "exactly one of" in result.output

    def test_neither_target_is_an_error(self, monkeypatch):
        _patch_pipeline(
            monkeypatch, {}, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )

        result = CliRunner().invoke(cli.main, self._args(), env={"OUTPUT_BUCKET": ""})

        assert result.exit_code != 0
        assert "exactly one of" in result.output

    def test_bucket_target_uploads_under_the_granule_id(self, monkeypatch, tmp_path):
        captured: dict = {}
        _patch_pipeline(
            monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )
        dest = tmp_path / "HLS.M30.T14TPN.2015182.2015212.v2.0"
        dest.mkdir()
        monkeypatch.setattr(cli, "write_composite", lambda *a, **k: dest)

        def fake_upload(client, local_dir, bucket, prefix):
            captured["upload"] = {"dir": local_dir, "bucket": bucket, "prefix": prefix}
            return ["key-a", "key-b"]

        monkeypatch.setattr(cli, "upload_directory", fake_upload)

        result = CliRunner().invoke(
            cli.main, self._args("--output-bucket", "out-bucket")
        )

        assert result.exit_code == 0, result.output
        assert captured["upload"]["bucket"] == "out-bucket"
        # The prefix is the granule id, so keys mirror the local layout.
        assert captured["upload"]["prefix"] == dest.name
        assert "Uploaded 2 files" in result.output

    def test_output_prefix_is_prepended_to_the_granule_id(self, monkeypatch, tmp_path):
        captured: dict = {}
        _patch_pipeline(
            monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )
        dest = tmp_path / "HLS.M30.T14TPN.2015182.2015212.v2.0"
        dest.mkdir()
        monkeypatch.setattr(cli, "write_composite", lambda *a, **k: dest)
        monkeypatch.setattr(
            cli,
            "upload_directory",
            lambda client, d, bucket, prefix: (
                captured.setdefault("prefix", prefix) or []
            ),
        )

        result = CliRunner().invoke(
            cli.main,
            self._args(
                "--output-bucket", "out-bucket", "--output-prefix", "HLS_M30/data"
            ),
        )

        assert result.exit_code == 0, result.output
        assert captured["prefix"] == f"HLS_M30/data/{dest.name}"

    def test_empty_output_prefix_writes_at_the_root(self, monkeypatch, tmp_path):
        captured: dict = {}
        _patch_pipeline(
            monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )
        dest = tmp_path / "HLS.M30.T14TPN.2015182.2015212.v2.0"
        dest.mkdir()
        monkeypatch.setattr(cli, "write_composite", lambda *a, **k: dest)
        monkeypatch.setattr(
            cli,
            "upload_directory",
            lambda client, d, bucket, prefix: (
                captured.setdefault("prefix", prefix) or []
            ),
        )

        result = CliRunner().invoke(
            cli.main,
            self._args("--output-bucket", "out-bucket"),
            env={"OUTPUT_PREFIX": ""},
        )

        assert result.exit_code == 0, result.output
        assert captured["prefix"] == dest.name

    def test_dir_target_does_not_upload(self, monkeypatch, tmp_path):
        _patch_pipeline(
            monkeypatch, {}, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )

        def fail_upload(*a, **k):
            raise AssertionError("should not upload when writing locally")

        monkeypatch.setattr(cli, "upload_directory", fail_upload)

        result = CliRunner().invoke(
            cli.main,
            self._args("--output-dir", str(tmp_path)),
            env={"OUTPUT_BUCKET": ""},
        )

        assert result.exit_code == 0, result.output
        assert "Wrote composite to" in result.output


class TestReaderRole:
    def test_role_arn_is_passed_through_and_reported(self, monkeypatch, tmp_path):
        captured: dict = {}
        _patch_pipeline(
            monkeypatch, captured, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )

        from contextlib import contextmanager

        @contextmanager
        def fake_env(role_arn, **kwargs):
            captured["role_arn"] = role_arn
            yield role_arn

        monkeypatch.setattr(cli, "assumed_role_env", fake_env)

        result = CliRunner().invoke(
            cli.main,
            [
                "--tile-id",
                "14TPN",
                "--year-month",
                "2015-07",
                "--bucket",
                "in-bucket",
                "--output-dir",
                str(tmp_path),
                "--role-arn",
                "arn:aws:iam::123456789012:role/reader",
            ],
            env={"OUTPUT_BUCKET": ""},
        )

        assert result.exit_code == 0, result.output
        assert captured["role_arn"] == "arn:aws:iam::123456789012:role/reader"
        assert "Reading via assumed role" in result.output

    def test_without_a_role_it_reports_ambient_credentials(self, monkeypatch, tmp_path):
        _patch_pipeline(
            monkeypatch, {}, [Granule("s3://b/g", "L30", date(2015, 7, 10))]
        )

        result = CliRunner().invoke(
            cli.main,
            [
                "--tile-id",
                "14TPN",
                "--year-month",
                "2015-07",
                "--bucket",
                "in-bucket",
                "--output-dir",
                str(tmp_path),
            ],
            env={"OUTPUT_BUCKET": "", "LPDAAC_READER_ROLE_ARN": ""},
        )

        assert result.exit_code == 0, result.output
        assert "Reading with ambient credentials" in result.output
