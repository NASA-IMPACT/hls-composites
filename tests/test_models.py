from datetime import date

import pytest

from hls_composites.models import DateRange, Granule


def test_granule_is_frozen():
    g = Granule(path="s3://bucket/x", satellite="L30", date=date(2020, 1, 15))
    assert g.path == "s3://bucket/x"
    assert g.satellite == "L30"
    assert g.date == date(2020, 1, 15)
    with pytest.raises(AttributeError):
        g.path = "changed"  # type: ignore[misc]


def test_date_range_valid_construction():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
    assert r.start == date(2020, 1, 1)
    assert r.end == date(2020, 1, 31)


def test_date_range_start_after_end_raises():
    with pytest.raises(ValueError):
        DateRange(start=date(2020, 2, 1), end=date(2020, 1, 1))


def test_date_range_contains_is_inclusive_on_both_ends():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
    assert date(2020, 1, 1) in r
    assert date(2020, 1, 31) in r
    assert date(2020, 1, 15) in r


def test_date_range_contains_excludes_outside_dates():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
    assert date(2019, 12, 31) not in r
    assert date(2020, 2, 1) not in r


def test_date_range_single_day():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 1))
    assert date(2020, 1, 1) in r
    assert date(2020, 1, 2) not in r
