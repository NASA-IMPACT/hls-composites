from datetime import date

from botocore.exceptions import ClientError

from hls_composites.discovery import list_common_prefixes_with_retry, parse_granule_common_prefix


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


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "ListObjectsV2")


def _page(prefixes: list[str]) -> dict:
    return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}


class _FakePaginator:
    def __init__(self, sequence: list):
        # Each element is either an Exception to raise, or a list of pages to return.
        self._sequence = sequence
        self.calls = 0

    def paginate(self, **kwargs):
        item = self._sequence[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class _FakeS3Client:
    def __init__(self, paginator: _FakePaginator):
        self._paginator = paginator

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self._paginator


def test_list_common_prefixes_returns_results_on_first_try():
    paginator = _FakePaginator([[_page(["a/", "b/"])]])
    client = _FakeS3Client(paginator)
    result = list_common_prefixes_with_retry(client, "bucket", "prefix", max_retries=3)
    assert result == ["a/", "b/"]
    assert paginator.calls == 1


def test_list_common_prefixes_retries_on_throttling(monkeypatch):
    monkeypatch.setattr("hls_composites.discovery.time.sleep", lambda _: None)
    paginator = _FakePaginator(
        [_client_error("SlowDown"), _client_error("RequestLimitExceeded"), [_page(["a/"])]]
    )
    client = _FakeS3Client(paginator)
    result = list_common_prefixes_with_retry(client, "bucket", "prefix", max_retries=5)
    assert result == ["a/"]
    assert paginator.calls == 3


def test_list_common_prefixes_raises_immediately_on_non_throttling_error():
    paginator = _FakePaginator([_client_error("AccessDenied")])
    client = _FakeS3Client(paginator)
    try:
        list_common_prefixes_with_retry(client, "bucket", "prefix", max_retries=5)
        assert False, "expected ClientError"
    except ClientError as e:
        assert e.response["Error"]["Code"] == "AccessDenied"
    assert paginator.calls == 1


def test_list_common_prefixes_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("hls_composites.discovery.time.sleep", lambda _: None)
    paginator = _FakePaginator([_client_error("SlowDown"), _client_error("SlowDown")])
    client = _FakeS3Client(paginator)
    try:
        list_common_prefixes_with_retry(client, "bucket", "prefix", max_retries=2)
        assert False, "expected ClientError"
    except ClientError as e:
        assert e.response["Error"]["Code"] == "SlowDown"
    assert paginator.calls == 2
