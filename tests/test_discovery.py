from datetime import date

from hls_composites.discovery import (
    list_common_prefixes,
    parse_granule_common_prefix,
    scan_bucket_for_granules,
)
from hls_composites.models import DateRange


def test_parse_granule_common_prefix_l30():
    prefix = "HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/"
    granule = parse_granule_common_prefix(prefix, bucket="lp-prod-protected")
    assert granule is not None
    assert granule.satellite == "L30"
    assert granule.date == date(2026, 5, 31)  # DOY 151 of 2026
    assert granule.path == (
        "s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/"
        "HLS.L30.T55HDT.2026151T235621.v2.0"
    )


def test_parse_granule_common_prefix_s30():
    prefix = "HLSS30.020/HLS.S30.T18SUJ.2020001T151911.v2.0/"
    granule = parse_granule_common_prefix(prefix, bucket="lp-prod-protected")
    assert granule is not None
    assert granule.satellite == "S30"
    assert granule.date == date(2020, 1, 1)  # DOY 001 of 2020


def test_parse_granule_common_prefix_rejects_non_granule_keys():
    assert parse_granule_common_prefix("HLSL30.020/not-a-granule/", bucket="b") is None


def test_parse_granule_common_prefix_without_trailing_slash():
    prefix = "HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0"
    granule = parse_granule_common_prefix(prefix, bucket="lp-prod-protected")
    assert granule is not None
    assert granule.date == date(2026, 5, 31)


def _page(prefixes: list[str]) -> dict:
    return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}


class _FakePaginator:
    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.calls = 0

    def paginate(self, **kwargs):
        self.calls += 1
        return self._pages


class _FakeS3Client:
    def __init__(self, paginator: _FakePaginator):
        self._paginator = paginator

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self._paginator


def test_list_common_prefixes_returns_prefixes_across_pages():
    paginator = _FakePaginator([_page(["a/", "b/"]), _page(["c/"])])
    client = _FakeS3Client(paginator)
    result = list_common_prefixes(client, "bucket", "prefix")
    assert result == ["a/", "b/", "c/"]
    assert paginator.calls == 1


def test_list_common_prefixes_returns_empty_list_when_nothing_found():
    paginator = _FakePaginator([_page([])])
    client = _FakeS3Client(paginator)
    result = list_common_prefixes(client, "bucket", "prefix")
    assert result == []


class _MultiCallFakePaginator:
    """Returns a fixed page list per Prefix argument, regardless of call order."""

    def __init__(self, pages_by_prefix: dict[str, list[str]]):
        self._pages_by_prefix = pages_by_prefix
        self.prefixes_seen: list[str] = []

    def paginate(self, **kwargs):
        prefix = kwargs["Prefix"]
        self.prefixes_seen.append(prefix)
        return [_page(self._pages_by_prefix.get(prefix, []))]


class _MultiCallFakeS3Client:
    def __init__(self, paginator: _MultiCallFakePaginator):
        self._paginator = paginator

    def get_paginator(self, name: str):
        return self._paginator


def test_scan_bucket_for_granules_filters_to_exact_month():
    # A January scan uses the single whole-year prefix "2020" (see
    # DateRange.key_prefixes). Feb 14 shares that prefix too -- realistic
    # overcoverage that must get filtered out client-side.
    date_range = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))
    prefixes = date_range.key_prefixes()
    assert prefixes == ["2020"]
    pages_by_prefix = {
        "HLSL30.020/HLS.L30.T18SUJ.2020": [
            "HLSL30.020/HLS.L30.T18SUJ.2020001T151911.v2.0/",  # Jan 1 -- in range
            "HLSL30.020/HLS.L30.T18SUJ.2020031T151911.v2.0/",  # Jan 31 -- in range
            "HLSL30.020/HLS.L30.T18SUJ.2020045T101911.v2.0/",  # Feb 14 -- overcoverage, must be dropped
        ],
    }
    paginator = _MultiCallFakePaginator(pages_by_prefix)
    client = _MultiCallFakeS3Client(paginator)

    granules = scan_bucket_for_granules(
        client, "lp-prod-protected", "18SUJ", date_range, satellites=("L30",)
    )

    assert sorted(g.date for g in granules) == [date(2020, 1, 1), date(2020, 1, 31)]
    assert all(g.satellite == "L30" for g in granules)
    assert paginator.prefixes_seen == ["HLSL30.020/HLS.L30.T18SUJ.2020"]


def test_scan_bucket_for_granules_returns_empty_list_when_nothing_found():
    date_range = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 5))
    client = _MultiCallFakeS3Client(_MultiCallFakePaginator({}))

    granules = scan_bucket_for_granules(
        client, "lp-prod-protected", "18SUJ", date_range
    )

    assert granules == []


def test_scan_bucket_for_granules_returns_chronological_order():
    # Interleaved dates across satellites. Order matters: an even-length stack
    # makes the median a tie that select_best_index resolves by stack position,
    # so discovery must not group by satellite.
    date_range = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 10))
    pages_by_prefix = {
        "HLSL30.020/HLS.L30.T18SUJ.2020": [
            "HLSL30.020/HLS.L30.T18SUJ.2020002T151911.v2.0/",  # Jan 2
            "HLSL30.020/HLS.L30.T18SUJ.2020008T151911.v2.0/",  # Jan 8
        ],
        "HLSS30.020/HLS.S30.T18SUJ.2020": [
            "HLSS30.020/HLS.S30.T18SUJ.2020004T101911.v2.0/",  # Jan 4
            "HLSS30.020/HLS.S30.T18SUJ.2020006T101911.v2.0/",  # Jan 6
        ],
    }
    client = _MultiCallFakeS3Client(_MultiCallFakePaginator(pages_by_prefix))

    granules = scan_bucket_for_granules(
        client, "lp-prod-protected", "18SUJ", date_range
    )

    assert [(g.date, g.satellite) for g in granules] == [
        (date(2020, 1, 2), "L30"),
        (date(2020, 1, 4), "S30"),
        (date(2020, 1, 6), "S30"),
        (date(2020, 1, 8), "L30"),
    ]


def test_scan_bucket_for_granules_orders_same_day_granules_by_path():
    # Both satellites can observe a tile on the same day; the order between
    # them still has to be deterministic.
    date_range = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 10))
    pages_by_prefix = {
        "HLSL30.020/HLS.L30.T18SUJ.2020": [
            "HLSL30.020/HLS.L30.T18SUJ.2020005T151911.v2.0/",
        ],
        "HLSS30.020/HLS.S30.T18SUJ.2020": [
            "HLSS30.020/HLS.S30.T18SUJ.2020005T101911.v2.0/",
        ],
    }
    client = _MultiCallFakeS3Client(_MultiCallFakePaginator(pages_by_prefix))

    granules = scan_bucket_for_granules(
        client, "lp-prod-protected", "18SUJ", date_range
    )

    assert [g.date for g in granules] == [date(2020, 1, 5), date(2020, 1, 5)]
    assert [g.path for g in granules] == sorted(g.path for g in granules)
