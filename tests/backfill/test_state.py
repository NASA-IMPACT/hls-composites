import boto3
import pytest
from moto import mock_aws

from hls_composites.backfill.plan import BackfillPlan, Segment, tile_list_digest
from hls_composites.backfill.state import (
    PlanConflictError,
    PlanNotFoundError,
    PlanStore,
)
from hls_composites.models import YearMonth

BUCKET = "hls-composites-processing-test"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-west-2")
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        yield client


@pytest.fixture
def store(s3):
    return PlanStore(bucket=BUCKET, client=s3)


def a_plan(submitted=0):
    return BackfillPlan(
        plan_version="sha256:abc",
        tile_list_source="https://example.invalid/tiles@deadbee",
        segments=[Segment(YearMonth(2013, 4), submitted, 10)],
    )


def test_get_raises_when_no_plan_exists(store):
    with pytest.raises(PlanNotFoundError):
        store.get()


def test_create_then_get_round_trips(store):
    created = store.create(a_plan())
    fetched = store.get()
    assert fetched.plan == created.plan
    assert fetched.etag == created.etag


def test_create_refuses_to_clobber_an_existing_plan(store):
    store.create(a_plan())
    with pytest.raises(PlanConflictError):
        store.create(a_plan())


def test_update_with_current_etag_succeeds_and_returns_new_etag(store):
    stored = store.create(a_plan())
    stored.plan.segments[0].submitted_count = 4
    updated = store.update(stored)
    assert updated.etag != stored.etag
    assert store.get().plan.segments[0].submitted_count == 4


def test_update_with_stale_etag_raises_rather_than_clobbering(store):
    stored = store.create(a_plan())

    # A concurrent feeder tick advances the cursor and writes it back.
    concurrent = store.get()
    concurrent.plan.segments[0].submitted_count = 4
    store.update(concurrent)

    # Our copy still carries the pre-tick ETag, so its write must not land.
    stored.plan.segments[0].submitted_count = 999
    with pytest.raises(PlanConflictError):
        store.update(stored)

    assert store.get().plan.segments[0].submitted_count == 4


def test_rewriting_identical_content_keeps_the_same_etag(store):
    """S3 ETags are content hashes, not version counters.

    An interleaved write of byte-identical content is therefore invisible to
    the CAS. That is harmless -- the two writers agreed -- but it means the
    guard catches diverging writes, not merely concurrent ones.
    """
    stored = store.create(a_plan())

    assert store.update(stored).etag == stored.etag


def test_read_tile_list_returns_tiles_and_digest(store, s3):
    body = b"01FBE\n01FBF\n60WWV\n"
    s3.put_object(Bucket=BUCKET, Key="tiles.txt", Body=body)
    tiles, digest = store.read_tile_list("tiles.txt")
    assert tiles == ["01FBE", "01FBF", "60WWV"]
    assert digest == tile_list_digest(body)


def test_read_tile_list_raises_when_absent(store):
    with pytest.raises(PlanNotFoundError):
        store.read_tile_list("backfill/missing.txt")
