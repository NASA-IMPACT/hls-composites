"""The CNM submission message written beside a finished granule."""

import json

import pytest

from hls_composites.metadata.manifest import (
    COLLECTION,
    MANIFEST_SUFFIX,
    job_id,
    write_manifest,
)
from tests.metadata.conftest import GRANULE_ID

BUCKET_URI = f"s3://out-bucket/M30/data/{GRANULE_ID}"


@pytest.fixture
def written(granule_dir, browse_image):
    """A granule directory with every artefact, then its manifest."""
    (granule_dir / f"{GRANULE_ID}.cmr.xml").write_bytes(b"<Granule/>")
    (granule_dir / f"{GRANULE_ID}_stac.json").write_bytes(b"{}")
    path = write_manifest(granule_dir, BUCKET_URI, GRANULE_ID, identifier="job-1")
    return path, json.loads(path.read_text())


class TestJobId:
    def test_uses_the_batch_job_id_when_present(self, monkeypatch):
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "batch-abc")

        assert job_id() == "batch-abc"

    def test_falls_back_to_a_uuid_outside_batch(self, monkeypatch):
        monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)

        first, second = job_id(), job_id()

        assert first != second
        assert len(first) == 36

    def test_an_empty_batch_id_falls_back(self, monkeypatch):
        """An unset container variable arrives as empty, not absent."""
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "")

        assert len(job_id()) == 36


class TestManifest:
    def test_is_named_for_the_granule(self, written):
        path, _ = written

        assert path.name == f"{GRANULE_ID}{MANIFEST_SUFFIX}"

    def test_names_the_collection_and_granule(self, written):
        _, manifest = written

        assert manifest["collection"] == COLLECTION
        assert manifest["identifier"] == "job-1"
        assert manifest["duplicationid"] == GRANULE_ID
        assert manifest["product"]["id"] == GRANULE_ID

    def test_lists_every_artefact_with_its_type(self, written):
        _, manifest = written

        by_name = {f["name"]: f["type"] for f in manifest["product"]["files"]}
        assert by_name == {
            f"{GRANULE_ID}.NDVI.tif": "data",
            f"{GRANULE_ID}.ValidCount.tif": "data",
            f"{GRANULE_ID}.cmr.xml": "metadata",
            f"{GRANULE_ID}_stac.json": "metadata",
            f"{GRANULE_ID}.jpg": "browse",
        }

    def test_does_not_list_itself(self, written):
        """A file cannot carry its own checksum."""
        path, manifest = written

        assert path.name not in {f["name"] for f in manifest["product"]["files"]}

    def test_every_file_has_a_checksum_and_size(self, written):
        _, manifest = written

        for entry in manifest["product"]["files"]:
            assert entry["checksumType"] == "SHA512"
            assert len(entry["checksum"]) == 128
            assert entry["size"] > 0

    def test_uris_point_at_where_the_files_will_land(self, written):
        _, manifest = written

        for entry in manifest["product"]["files"]:
            assert entry["uri"] == f"{BUCKET_URI}/{entry['name']}"

    def test_an_empty_granule_directory_is_an_error(self, tmp_path):
        """A manifest listing nothing would ask the DAAC to ingest nothing."""
        empty = tmp_path / "HLS.M30.T14TPN.2020032.2020060.v2.0"
        empty.mkdir()

        with pytest.raises(FileNotFoundError, match="no product files"):
            write_manifest(empty, BUCKET_URI, empty.name)

    def test_defaults_its_identifier_to_the_job(
        self, granule_dir, browse_image, monkeypatch
    ):
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "batch-xyz")

        path = write_manifest(granule_dir, BUCKET_URI, GRANULE_ID)

        assert json.loads(path.read_text())["identifier"] == "batch-xyz"
