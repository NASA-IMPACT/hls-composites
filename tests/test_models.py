from datetime import date

import pytest

from hls_composites.models import JOB_TYPE, DateRange, YearMonth, composite_id


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


def _key(d: date) -> str:
    return f"{d.year:04d}{d.timetuple().tm_yday:03d}"


def _assert_full_coverage(date_range: DateRange) -> None:
    """Every day in the range must match at least one returned prefix."""
    prefixes = date_range.key_prefixes()
    day = date_range.start
    one_day = date.resolution
    while day <= date_range.end:
        key = _key(day)
        assert any(key.startswith(p) for p in prefixes), (
            f"{day} ({key}) not covered by {prefixes}"
        )
        day += one_day


def test_key_prefixes_full_leap_year_covers_feb_29():
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31))
    _assert_full_coverage(r)


def test_key_prefixes_january_is_also_just_the_year_prefix():
    # One-prefix-per-year is intentionally coarse: even a single month gets
    # the whole-year prefix, since a full year of HLS granules (<=366/sat)
    # comfortably fits one list_objects_v2 page. Overcoverage is filtered
    # client-side, so this is fine -- see key_prefixes' docstring.
    r = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
    assert r.key_prefixes() == ["2020"]
    _assert_full_coverage(r)


def test_key_prefixes_multi_year_window_returns_one_prefix_per_year():
    r = DateRange(start=date(2020, 6, 15), end=date(2021, 6, 14))
    assert r.key_prefixes() == ["2020", "2021"]
    _assert_full_coverage(r)


def test_composite_id_follows_prototype_naming():
    date_range = DateRange(date(2020, 7, 1), date(2020, 7, 31))
    assert composite_id("14TPN", date_range) == "HLS.M30.T14TPN.2020183.2020213.v2.0"


def test_job_type_is_the_monitored_job_type():
    assert JOB_TYPE == "monthly-composite"


def test_models_does_not_pull_in_geospatial_stack():
    """The backfill Lambda imports this module and must not need GDAL."""
    import sys

    for module in ("rasterio", "rioxarray", "xarray", "dask"):
        assert module not in sys.modules or "hls_composites.io" in sys.modules


class TestYearMonth:
    """The canonical in-memory month. Strings appear only at boundaries."""

    def test_parses_the_canonical_form(self):
        assert YearMonth.parse("2015-07") == YearMonth(2015, 7)

    @pytest.mark.parametrize(
        "bad", ["2013", "2013-13", "2013-00", "not-a-month", "2013-07-01", ""]
    )
    def test_rejects_malformed_input(self, bad):
        with pytest.raises(ValueError):
            YearMonth.parse(bad)

    def test_rejects_an_unpadded_month(self):
        """`year_month` is an Athena partition key and part of every entity ID.

        `strptime("%Y-%m")` would accept this and yield a `2020-7` partition
        that the date projection cannot match.
        """
        with pytest.raises(ValueError):
            YearMonth.parse("2020-7")

    def test_rejects_an_out_of_range_month_on_construction(self):
        with pytest.raises(ValueError):
            YearMonth(2020, 13)

    def test_str_round_trips_through_parse(self):
        ym = YearMonth(2015, 7)
        assert str(ym) == "2015-07"
        assert YearMonth.parse(str(ym)) == ym

    def test_from_date_takes_the_containing_month(self):
        assert YearMonth.from_date(date(2020, 7, 23)) == YearMonth(2020, 7)

    def test_next_and_previous_cross_the_year_boundary(self):
        assert YearMonth(2013, 12).next() == YearMonth(2014, 1)
        assert YearMonth(2014, 1).previous() == YearMonth(2013, 12)

    def test_orders_chronologically(self):
        assert YearMonth(2013, 4) < YearMonth(2013, 5) < YearMonth(2014, 1)
        assert sorted([YearMonth(2014, 1), YearMonth(2013, 5)]) == [
            YearMonth(2013, 5),
            YearMonth(2014, 1),
        ]

    def test_date_range_covers_the_whole_month(self):
        assert YearMonth(2020, 7).date_range() == DateRange(
            date(2020, 7, 1), date(2020, 7, 31)
        )

    def test_date_range_handles_february_in_a_leap_year(self):
        assert YearMonth(2020, 2).date_range().end == date(2020, 2, 29)

    def test_for_month_agrees_with_year_month(self):
        assert DateRange.for_month("2020-07") == YearMonth(2020, 7).date_range()
