from datetime import date

import boto3
import pytest
from moto import mock_aws

from hls_composites.backfill.plan import BackfillPlan, Segment, tile_list_digest
from hls_composites.backfill.state import PlanStore, TileListMismatchError
from hls_composites.models import YearMonth
from month_opener.handler import open_month, target_month

BUCKET = "hls-composites-processing-test"
TILES = b"01FBE\n01FBF\n01GBH\n"
TILE_KEY = "tiles.txt"
PLAN_KEY = "plans/forward.json"
JULY = YearMonth(2026, 7)
AUGUST = YearMonth(2026, 8)


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
    return PlanStore(bucket=BUCKET, plan_key=PLAN_KEY, client=s3)


def open_for(store, year_month):
    return open_month(store=store, tile_list_key=TILE_KEY, year_month=year_month)


def test_target_month_is_the_month_that_just_ended():
    assert target_month(date(2026, 9, 10)) == AUGUST


def test_target_month_crosses_the_year_boundary():
    assert target_month(date(2026, 1, 10)) == YearMonth(2025, 12)


def test_target_month_is_correct_on_the_first_of_the_month():
    assert target_month(date(2026, 3, 1)) == YearMonth(2026, 2)


def test_target_month_does_not_depend_on_which_day_it_runs():
    """The schedule day is a lag knob; it must not change which month is opened."""
    assert target_month(date(2026, 9, 2)) == target_month(date(2026, 9, 28))


def test_creates_the_forward_plan_on_first_run(store):
    result = open_for(store, AUGUST)

    assert result.status == "opened"
    plan = store.get().plan
    assert plan.plan_version == tile_list_digest(TILES)
    assert [s.year_month for s in plan.segments] == [AUGUST]


def test_opened_segment_covers_every_tile(store):
    open_for(store, AUGUST)

    segment = store.get().plan.segments[0]
    assert segment.total_count == 3
    assert segment.submitted_count == 0
    assert segment.tiles_key is None


def test_opening_the_same_month_twice_is_a_no_op(store):
    """EventBridge retries; a second open must not queue 19,000 duplicate jobs."""
    open_for(store, AUGUST)
    result = open_for(store, AUGUST)

    assert result.status == "already_open"
    assert len(store.get().plan.segments) == 1


def test_months_accumulate_in_arrival_order(store):
    open_for(store, JULY)
    open_for(store, AUGUST)

    assert [s.year_month for s in store.get().plan.segments] == [JULY, AUGUST]


def test_does_not_disturb_an_unfinished_earlier_month(store):
    open_for(store, JULY)
    stored = store.get()
    stored.plan.segments[0].submitted_count = 2
    store.update(stored)

    open_for(store, AUGUST)

    segments = store.get().plan.segments
    assert segments[0].submitted_count == 2
    assert store.get().plan.next_segment().year_month == JULY


def test_refuses_to_extend_a_plan_built_on_a_different_tile_list(store):
    store.create(
        BackfillPlan(
            plan_version="sha256:stale",
            tile_list_source="src",
            segments=[Segment(JULY, 3, 3)],
        )
    )

    with pytest.raises(TileListMismatchError, match="sha256:stale"):
        open_for(store, AUGUST)

    assert len(store.get().plan.segments) == 1


def test_result_serializes_for_lambda(store):
    assert open_for(store, AUGUST).to_dict() == {
        "status": "opened",
        "year_month": "2026-08",
        "total_count": 3,
    }
