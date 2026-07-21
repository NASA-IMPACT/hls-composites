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


def _key(d: date) -> str:
    return f"{d.year:04d}{d.timetuple().tm_yday:03d}"


def _assert_full_coverage(date_range: DateRange) -> None:
    """Every day in the range must match at least one returned prefix."""
    prefixes = date_range.key_prefixes()
    day = date_range.start
    one_day = date.resolution
    while day <= date_range.end:
        key = _key(day)
        assert any(key.startswith(p) for p in prefixes), f"{day} ({key}) not covered by {prefixes}"
        day += one_day


def test_key_prefixes_full_year_is_single_year_prefix():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31))
    assert r.key_prefixes() == ["2020"]


def test_key_prefixes_full_leap_year_covers_feb_29():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31))
    _assert_full_coverage(r)


def test_key_prefixes_january_covers_full_month():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
    _assert_full_coverage(r)


def test_key_prefixes_boundary_crossing_month_covers_full_month():
    # April 2020: DOY 092-121, crosses the 099->100 boundary.
    r = DateRange(start=date(2020, 4, 1), end=date(2020, 4, 30))
    _assert_full_coverage(r)


def test_key_prefixes_boundary_crossing_month_does_not_regress_to_whole_year():
    # Regression guard: a naive "stop at first nonempty common prefix" rule
    # would return just ["2020"] here, i.e. the entire year, for an April-only
    # request. None of the returned prefixes may be that short.
    r = DateRange(start=date(2020, 4, 1), end=date(2020, 4, 30))
    for prefix in r.key_prefixes():
        assert len(prefix) > 4, f"prefix {prefix!r} overcovers to (nearly) the whole year"


def test_key_prefixes_multi_year_window_covers_full_range():
    r = DateRange(start=date(2020, 6, 15), end=date(2021, 6, 14))
    _assert_full_coverage(r)


def test_key_prefixes_single_day():
    r = DateRange(start=date(2020, 3, 5), end=date(2020, 3, 5))
    assert r.key_prefixes() == ["2020065"]
