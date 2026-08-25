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
