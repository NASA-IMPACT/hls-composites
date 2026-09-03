"""The CLI's own job: parse arguments, validate them, call the pipeline."""

from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from hls_composites import cli
from hls_composites.models import DateRange
from hls_composites.pipeline import CompositeResult, LocalDestination, S3Destination


@pytest.fixture
def called(monkeypatch):
    """Capture the keyword arguments the CLI hands to the pipeline."""
    seen: dict = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return CompositeResult("GRAN", 1, [])

    monkeypatch.setattr(cli, "create_composite", fake_create)
    return seen


def invoke(*extra, env=None):
    base = ["--tile-id", "14TPN", "--year-month", "2015-07", "--bucket", "in-bucket"]
    return CliRunner().invoke(
        cli.main, [*base, *extra], env={"OUTPUT_BUCKET": "", **(env or {})}
    )


class TestArgumentWiring:
    def test_year_month_becomes_that_calendar_month(self, called, tmp_path):
        result = invoke("--output-dir", str(tmp_path))

        assert result.exit_code == 0, result.output
        assert called["date_range"] == DateRange(date(2015, 7, 1), date(2015, 7, 31))
        assert called["tile_id"] == "14TPN"
        assert called["input_bucket"] == "in-bucket"

    def test_output_dir_becomes_a_local_destination(self, called, tmp_path):
        invoke("--output-dir", str(tmp_path))

        assert called["destination"] == LocalDestination(Path(tmp_path))

    def test_output_bucket_and_prefix_become_an_s3_destination(self, called):
        invoke("--output-bucket", "out-bucket", "--output-prefix", "M30/data")

        assert called["destination"] == S3Destination("out-bucket", "M30/data")

    def test_role_arn_is_passed_through(self, called, tmp_path):
        invoke("--output-dir", str(tmp_path), "--role-arn", "arn:aws:iam::1:role/r")

        assert called["role_arn"] == "arn:aws:iam::1:role/r"

    def test_indexes_is_the_default_output(self, called, tmp_path):
        invoke("--output-dir", str(tmp_path))

        assert called["output"] == "indexes"

    def test_bands_selects_the_reflectance_output(self, called, tmp_path):
        invoke("--output-dir", str(tmp_path), "--bands")

        assert called["output"] == "bands"

    def test_env_vars_supply_defaults(self, called):
        """The container sets these; flags override them."""
        result = invoke(
            env={"OUTPUT_BUCKET": "env-bucket", "OUTPUT_PREFIX": "M30/data"}
        )

        assert result.exit_code == 0, result.output
        assert called["destination"] == S3Destination("env-bucket", "M30/data")


class TestValidation:
    def test_both_output_targets_is_an_error(self, called, tmp_path):
        result = invoke("--output-dir", str(tmp_path), "--output-bucket", "out")

        assert result.exit_code != 0
        assert "exactly one of" in result.output

    def test_neither_output_target_is_an_error(self, called):
        result = invoke()

        assert result.exit_code != 0
        assert "exactly one of" in result.output

    def test_indexes_and_bands_are_mutually_exclusive(self, called, tmp_path):
        result = invoke("--output-dir", str(tmp_path), "--indexes", "--bands")

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_a_malformed_month_is_reported_against_the_flag(self, called, tmp_path):
        result = CliRunner().invoke(
            cli.main,
            [
                "--tile-id",
                "14TPN",
                "--year-month",
                "July 2015",
                "--bucket",
                "in-bucket",
                "--output-dir",
                str(tmp_path),
            ],
            env={"OUTPUT_BUCKET": ""},
        )

        assert result.exit_code != 0
        assert "--year-month" in result.output
        assert "expected YYYY-MM" in result.output
