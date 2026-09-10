import boto3
import pytest
from moto import mock_aws

from backfill_feeder.handler import TileListMismatchError, backfill_feeder
from hls_composites.backfill.plan import BackfillPlan, Segment, tile_list_digest
from hls_composites.backfill.state import PlanNotFoundError, PlanStore
from hls_composites.models import YearMonth

BUCKET = "hls-composites-processing-test"
TILES = b"01FBE\n01FBF\n01GBH\n01GDM\n01GEL\n"
TILE_KEY = "tiles.txt"
APRIL = YearMonth(2013, 4)
MAY = YearMonth(2013, 5)
JUNE_2016 = YearMonth(2016, 6)


class StubSubmitter:
    def __init__(self, below_threshold=True, accept=None):
        self.below_threshold = below_threshold
        self.accept = accept
        self.units = []
        self.threshold_checks = []

    def active_jobs_below_threshold(self, threshold):
        self.threshold_checks.append(threshold)
        return self.below_threshold

    def submit_units(self, units):
        units = list(units)
        limit = len(units) if self.accept is None else min(self.accept, len(units))
        self.units.extend(units[:limit])
        return limit


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-west-2")
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
        )
        client.put_object(Bucket=BUCKET, Key=TILE_KEY, Body=TILES)
        yield client


@pytest.fixture
def store(s3):
    return PlanStore(bucket=BUCKET, client=s3)


def seed(store, segments, plan_version=None):
    plan = BackfillPlan(
        plan_version=plan_version or tile_list_digest(TILES),
        tile_list_source="https://example.invalid/tiles@deadbee",
        segments=segments,
    )
    return store.create(plan)


def feed(store, submitter, submit_count=2, max_active_jobs=100):
    return backfill_feeder(
        store=store,
        submitter=submitter,
        tile_list_key=TILE_KEY,
        max_active_jobs=max_active_jobs,
        submit_count=submit_count,
    )


def test_submits_the_next_n_units_and_advances_the_cursor(store):
    seed(store, [Segment(APRIL, 0, 5)])
    submitter = StubSubmitter()

    result = feed(store, submitter, submit_count=2)

    assert result.status == "submitted"
    assert result.submitted == 2
    assert submitter.units == [("01FBE", APRIL), ("01FBF", APRIL)]
    assert store.get().plan.segments[0].submitted_count == 2


def test_resumes_from_the_stored_cursor(store):
    seed(store, [Segment(APRIL, 3, 5)])
    submitter = StubSubmitter()

    feed(store, submitter, submit_count=10)

    assert submitter.units == [("01GDM", APRIL), ("01GEL", APRIL)]
    assert store.get().plan.segments[0].submitted_count == 5


def test_moves_to_the_next_segment_once_one_completes(store):
    seed(store, [Segment(APRIL, 5, 5), Segment(MAY, 0, 5)])
    submitter = StubSubmitter()

    result = feed(store, submitter, submit_count=1)

    assert result.year_month == MAY
    assert submitter.units == [("01FBE", MAY)]


def test_skips_the_tick_when_the_queue_is_full(store):
    seed(store, [Segment(APRIL, 0, 5)])
    submitter = StubSubmitter(below_threshold=False)

    result = feed(store, submitter, submit_count=2, max_active_jobs=42)

    assert result.status == "throttled"
    assert result.submitted == 0
    assert submitter.threshold_checks == [42]
    assert submitter.units == []
    assert store.get().plan.segments[0].submitted_count == 0


def test_reports_completion_when_every_segment_is_done(store):
    seed(store, [Segment(APRIL, 5, 5)])
    submitter = StubSubmitter()

    result = feed(store, submitter)

    assert result.status == "complete"
    assert submitter.units == []


def test_refuses_to_run_when_the_tile_list_changed(store):
    seed(store, [Segment(APRIL, 0, 5)], plan_version="sha256:stale")
    submitter = StubSubmitter()

    with pytest.raises(TileListMismatchError, match="sha256:stale"):
        feed(store, submitter)

    assert submitter.units == []
    assert store.get().plan.segments[0].submitted_count == 0


def test_advances_only_by_the_number_actually_submitted(store):
    seed(store, [Segment(APRIL, 0, 5)])
    submitter = StubSubmitter(accept=1)

    result = feed(store, submitter, submit_count=3)

    assert result.submitted == 1
    assert store.get().plan.segments[0].submitted_count == 1


def test_does_not_write_the_plan_when_nothing_was_submitted(store):
    stored = seed(store, [Segment(APRIL, 0, 5)])
    submitter = StubSubmitter(accept=0)

    feed(store, submitter, submit_count=3)

    assert store.get().etag == stored.etag


def test_sparse_segment_draws_units_from_its_own_tile_list(store, s3):
    s3.put_object(Bucket=BUCKET, Key="subsets/2016-06-1.txt", Body=b"60WWV\n01GBH\n")
    seed(store, [Segment(JUNE_2016, 0, 2, tiles_key="subsets/2016-06-1.txt")])
    submitter = StubSubmitter()

    feed(store, submitter, submit_count=5)

    assert submitter.units == [("60WWV", JUNE_2016), ("01GBH", JUNE_2016)]


def test_rejects_a_segment_whose_tile_list_is_the_wrong_length(store):
    seed(store, [Segment(APRIL, 0, 99)])
    submitter = StubSubmitter()

    with pytest.raises(ValueError, match="99"):
        feed(store, submitter)


def test_raises_when_no_plan_has_been_initialized(store):
    submitter = StubSubmitter()
    with pytest.raises(PlanNotFoundError):
        feed(store, submitter)


def test_result_serializes_to_a_stable_json_shape(store):
    """Lambda returns JSON, so every key is present even on a throttled tick."""
    seed(store, [Segment(APRIL, 0, 5)])

    submitted = feed(store, StubSubmitter(), submit_count=2).to_dict()
    assert submitted == {
        "status": "submitted",
        "submitted": 2,
        "year_month": "2013-04",
        "submitted_count": 2,
    }

    throttled = feed(store, StubSubmitter(below_threshold=False)).to_dict()
    assert throttled == {
        "status": "throttled",
        "submitted": 0,
        "year_month": None,
        "submitted_count": None,
    }
    assert throttled.keys() == submitted.keys()


def test_sparse_segment_is_not_gated_by_the_plan_version(store, s3):
    """A self-describing segment must survive an edit to the plan-level list.

    This is what lets forward processing keep running while the shared tile
    list is revised, and while a long backfill holds its own list frozen.
    """
    s3.put_object(Bucket=BUCKET, Key="tiles/2016-06.txt", Body=b"60WWV\n01GBH\n")
    seed(
        store,
        [Segment(JUNE_2016, 0, 2, tiles_key="tiles/2016-06.txt")],
        plan_version="sha256:no-longer-matches-anything",
    )
    submitter = StubSubmitter()

    result = feed(store, submitter, submit_count=5)

    assert result.submitted == 2
    assert submitter.units == [("60WWV", JUNE_2016), ("01GBH", JUNE_2016)]


def test_dense_segment_is_still_gated_by_the_plan_version(store):
    """The guard stays for segments indexed against the plan-level list."""
    seed(store, [Segment(APRIL, 0, 5)], plan_version="sha256:stale")

    with pytest.raises(TileListMismatchError, match="sha256:stale"):
        feed(store, StubSubmitter())
