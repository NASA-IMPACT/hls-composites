from datetime import date
from pathlib import Path

import pytest

from hls_composites import pipeline
from hls_composites.models import DateRange, Granule
from hls_composites.pipeline import (
    LocalDestination,
    S3Destination,
    create_composite,
    object_prefix,
)

JULY = DateRange(date(2015, 7, 1), date(2015, 7, 31))
GRANULES = [Granule("s3://b/g", "L30", date(2015, 7, 10))]


@pytest.fixture
def stages(monkeypatch, tmp_path):
    """Stub out discovery, compositing, and writing; record what they saw."""
    captured: dict = {}

    def fake_scan(s3, bucket, tile, date_range, **kwargs):
        captured["scan"] = {"bucket": bucket, "tile": tile, "date_range": date_range}
        return captured.get("granules", GRANULES)

    def fake_build(granules, **kwargs):
        captured["build"] = {"n": len(granules), "kwargs": kwargs}
        return "DATASET"

    def fake_write(ds, out_dir, tile, date_range, **kwargs):
        dest = Path(out_dir) / "HLS.M30.T14TPN.2015182.2015212.v2.0"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "a.NDVI.tif").write_bytes(b"x")
        captured["write"] = {"out_dir": Path(out_dir), "dest": dest}
        return dest

    monkeypatch.setattr(pipeline, "scan_bucket_for_granules", fake_scan)
    monkeypatch.setattr(pipeline, "build_composite", fake_build)
    monkeypatch.setattr(pipeline, "write_composite", fake_write)
    # Metadata reads the written rasters, which these stubs do not produce.
    # Tests that care about it override this.
    monkeypatch.setattr(pipeline, "write_metadata", lambda *a, **k: [])
    monkeypatch.setattr(pipeline.boto3, "client", lambda *a, **k: object())
    return captured


def run(destination, **overrides):
    kwargs = {
        "tile_id": "14TPN",
        "date_range": JULY,
        "input_bucket": "in-bucket",
        "destination": destination,
    }
    kwargs.update(overrides)
    return create_composite(**kwargs)


class TestObjectPrefix:
    def test_prefix_precedes_the_granule_id(self):
        assert object_prefix("M30/data", "GRAN") == "M30/data/GRAN"

    @pytest.mark.parametrize("prefix", ["", "/", "  "])
    def test_blank_prefix_leaves_the_granule_at_the_root(self, prefix):
        assert object_prefix(prefix, "GRAN") == "GRAN"

    def test_surrounding_slashes_do_not_double_up(self):
        assert object_prefix("/M30/data/", "GRAN") == "M30/data/GRAN"


class TestLocalDestination:
    def test_writes_into_the_given_directory(self, stages, tmp_path):
        result = create_composite(
            tile_id="14TPN",
            date_range=JULY,
            input_bucket="in-bucket",
            destination=LocalDestination(tmp_path),
        )

        assert stages["write"]["out_dir"] == tmp_path
        assert result.granule_count == 1
        assert result.uploaded_keys == []
        assert result.found_granules

    def test_does_not_upload(self, stages, monkeypatch, tmp_path):
        def fail(*a, **k):
            raise AssertionError("must not upload for a local destination")

        monkeypatch.setattr(pipeline, "upload_directory", fail)

        run(LocalDestination(tmp_path))


class TestS3Destination:
    def test_uploads_under_prefix_and_granule_id(self, stages, monkeypatch):
        seen: dict = {}

        def fake_upload(client, local_dir, bucket, prefix):
            seen.update(dir=Path(local_dir), bucket=bucket, prefix=prefix)
            return ["key-a", "key-b"]

        monkeypatch.setattr(pipeline, "upload_directory", fake_upload)

        result = run(S3Destination("out-bucket", "M30/data"))

        assert seen["bucket"] == "out-bucket"
        assert seen["prefix"] == "M30/data/HLS.M30.T14TPN.2015182.2015212.v2.0"
        assert result.uploaded_keys == ["key-a", "key-b"]

    def test_builds_in_a_temporary_directory_that_is_cleaned_up(
        self, stages, monkeypatch
    ):
        """A failed run must not leave the product behind on the instance."""
        monkeypatch.setattr(pipeline, "upload_directory", lambda *a, **k: [])

        run(S3Destination("out-bucket"))

        work_dir = stages["write"]["out_dir"]
        assert not work_dir.exists()


class TestNoGranules:
    def test_writes_a_marker_and_reports_zero(self, stages, tmp_path):
        stages["granules"] = []

        result = create_composite(
            tile_id="14TPN",
            date_range=JULY,
            input_bucket="in-bucket",
            destination=LocalDestination(tmp_path),
        )

        assert result.granule_count == 0
        assert not result.found_granules
        marker = tmp_path / result.granule_id / pipeline.NO_GRANULES_MARKER
        assert marker.exists()

    def test_does_not_composite(self, stages, tmp_path):
        stages["granules"] = []

        run(LocalDestination(tmp_path))

        assert "build" not in stages


class TestReaderRole:
    def test_role_is_passed_to_the_credential_scope(
        self, stages, monkeypatch, tmp_path
    ):
        seen: dict = {}

        from contextlib import contextmanager

        @contextmanager
        def fake_env(role_arn, **kwargs):
            seen["role_arn"] = role_arn
            yield role_arn

        monkeypatch.setattr(pipeline, "assumed_role_env", fake_env)

        run(LocalDestination(tmp_path), role_arn="arn:aws:iam::1:role/r")

        assert seen["role_arn"] == "arn:aws:iam::1:role/r"

    def test_progress_names_the_identity_used(self, stages, tmp_path):
        messages: list[str] = []

        run(LocalDestination(tmp_path), on_progress=messages.append)

        assert any("ambient credentials" in m for m in messages)

    def test_progress_is_optional(self, stages, tmp_path):
        """The default callback discards messages, so no caller is required."""
        run(LocalDestination(tmp_path))


class TestMetadata:
    def test_metadata_is_written_for_the_granule_directory(
        self, stages, monkeypatch, tmp_path
    ):
        written: dict = {}

        def fake_write_metadata(tile_id, date_range, granule_dir, inputs=None):
            written.update(
                tile_id=tile_id,
                granule_dir=Path(granule_dir),
                inputs=list(inputs or []),
            )
            return []

        monkeypatch.setattr(pipeline, "write_metadata", fake_write_metadata)

        run(LocalDestination(tmp_path))

        assert written["tile_id"] == "14TPN"
        assert written["granule_dir"] == stages["write"]["dest"]
        # Provenance: the discovered granules reach the metadata.
        assert written["inputs"] == GRANULES

    def test_no_metadata_when_no_granules_were_found(
        self, stages, monkeypatch, tmp_path
    ):
        """There is no product to describe."""
        stages["granules"] = []

        def fail(*args, **kwargs):
            raise AssertionError("must not write metadata without a composite")

        monkeypatch.setattr(pipeline, "write_metadata", fail)

        run(LocalDestination(tmp_path))

    def test_a_metadata_failure_fails_the_run(self, stages, monkeypatch, tmp_path):
        """Better a retryable failure than a granule the DAAC cannot ingest."""

        def boom(*args, **kwargs):
            raise RuntimeError("no CRS")

        monkeypatch.setattr(pipeline, "write_metadata", boom)

        with pytest.raises(RuntimeError, match="no CRS"):
            run(LocalDestination(tmp_path))

    def test_metadata_is_written_before_upload(self, stages, monkeypatch, tmp_path):
        """Order matters: the documents must be in the directory to be uploaded."""
        order: list[str] = []

        monkeypatch.setattr(
            pipeline,
            "write_metadata",
            lambda *a, **k: order.append("metadata") or [],
        )
        monkeypatch.setattr(
            pipeline,
            "upload_directory",
            lambda *a, **k: order.append("upload") or [],
        )

        run(S3Destination("out-bucket"))

        assert order == ["metadata", "upload"]
