import json

import pytest

from hls_composites.backfill.plan import (
    BackfillPlan,
    Segment,
    months,
    parse_tile_list,
    tile_list_digest,
)
from hls_composites.models import YearMonth


def test_months_is_inclusive_of_both_ends():
    assert months(YearMonth(2013, 4), YearMonth(2013, 6)) == [
        YearMonth(2013, 4),
        YearMonth(2013, 5),
        YearMonth(2013, 6),
    ]


def test_months_rolls_over_years():
    assert months(YearMonth(2013, 11), YearMonth(2014, 2)) == [
        YearMonth(2013, 11),
        YearMonth(2013, 12),
        YearMonth(2014, 1),
        YearMonth(2014, 2),
    ]


def test_months_single_month():
    assert months(YearMonth(2020, 7), YearMonth(2020, 7)) == [YearMonth(2020, 7)]


def test_months_rejects_reversed_range():
    with pytest.raises(ValueError, match="is after"):
        months(YearMonth(2014, 1), YearMonth(2013, 1))


def test_parse_tile_list_strips_blanks_and_whitespace():
    assert parse_tile_list(b"01FBE\n01FBF\n\n  60WWV  \n") == [
        "01FBE",
        "01FBF",
        "60WWV",
    ]


def test_tile_list_digest_is_stable_and_prefixed():
    digest = tile_list_digest(b"01FBE\n")
    assert digest.startswith("sha256:")
    assert digest == tile_list_digest(b"01FBE\n")
    assert digest != tile_list_digest(b"01FBF\n")


def test_segment_completion():
    assert Segment(YearMonth(2013, 4), 19000, 19000).is_complete
    assert not Segment(YearMonth(2013, 4), 18999, 19000).is_complete
    assert Segment(YearMonth(2013, 4), 18999, 19000).remaining == 1


def test_new_plan_builds_one_segment_per_month():
    plan = BackfillPlan.new(
        plan_version="sha256:abc",
        tile_list_source="https://example.invalid/tiles@deadbee",
        year_months=[YearMonth(2013, 4), YearMonth(2013, 5)],
        tile_count=100,
    )
    assert [s.year_month for s in plan.segments] == [
        YearMonth(2013, 4),
        YearMonth(2013, 5),
    ]
    assert all(s.total_count == 100 and s.submitted_count == 0 for s in plan.segments)


def test_next_segment_skips_complete_segments():
    plan = BackfillPlan(
        plan_version="sha256:abc",
        tile_list_source="src",
        segments=[
            Segment(YearMonth(2013, 4), 10, 10),
            Segment(YearMonth(2013, 5), 3, 10),
            Segment(YearMonth(2013, 6), 0, 10),
        ],
    )
    assert plan.next_segment().year_month == YearMonth(2013, 5)


def test_next_segment_is_none_when_complete():
    plan = BackfillPlan("sha256:abc", "src", [Segment(YearMonth(2013, 4), 10, 10)])
    assert plan.next_segment() is None
    assert plan.is_complete


def test_advance_moves_the_cursor_and_clamps_at_total():
    plan = BackfillPlan("sha256:abc", "src", [Segment(YearMonth(2013, 4), 0, 10)])
    segment = plan.next_segment()
    assert plan.advance(segment, 4) == 4
    assert plan.segments[0].submitted_count == 4
    assert plan.advance(segment, 99) == 10


def test_advance_rejects_negative_counts():
    plan = BackfillPlan("sha256:abc", "src", [Segment(YearMonth(2013, 4), 0, 10)])
    with pytest.raises(ValueError, match="negative"):
        plan.advance(plan.segments[0], -1)


def test_json_round_trip_preserves_everything():
    plan = BackfillPlan(
        plan_version="sha256:abc",
        tile_list_source="https://example.invalid/tiles@deadbee",
        segments=[
            Segment(YearMonth(2013, 4), 5, 10),
            Segment(YearMonth(2016, 6), 0, 3, tiles_key="subsets/2016-06-1.txt"),
        ],
    )
    assert BackfillPlan.from_json(plan.to_json()) == plan


def test_json_omits_tiles_key_for_dense_segments():
    plan = BackfillPlan("sha256:abc", "src", [Segment(YearMonth(2013, 4), 0, 10)])
    assert "tiles_key" not in json.loads(plan.to_json())["segments"][0]


def test_json_is_indented_for_hand_editing():
    plan = BackfillPlan("sha256:abc", "src", [Segment(YearMonth(2013, 4), 0, 10)])
    assert "\n" in plan.to_json()
