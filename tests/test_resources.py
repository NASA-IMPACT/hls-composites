import dask
import pytest

from hls_composites import resources
from hls_composites.resources import (
    BASELINE_BYTES,
    block_bytes,
    container_memory_limit,
    default_max_workers,
    workers_for,
)

MiB = 2**20
GiB = 2**30


def test_block_bytes_scales_linearly_with_granules():
    assert block_bytes(200) == 2 * block_bytes(100)


def test_block_bytes_scales_with_block_area():
    assert block_bytes(10, block_size=256) * 4 == block_bytes(10, block_size=512)


def test_workers_for_no_limit_uses_max():
    assert workers_for(500, memory_limit=None, max_workers=4) == 4


def test_workers_for_small_stack_uses_max():
    assert workers_for(30, memory_limit=3600 * MiB, max_workers=4) == 4


def test_workers_for_large_stack_reduces_threads():
    workers = workers_for(183, memory_limit=3600 * MiB, max_workers=4)
    assert 1 <= workers < 4
    assert BASELINE_BYTES + workers * block_bytes(183) <= 3600 * MiB


def test_workers_for_never_below_one():
    assert workers_for(10_000, memory_limit=1 * GiB, max_workers=4) == 1


def test_workers_for_limit_below_baseline():
    assert workers_for(10, memory_limit=BASELINE_BYTES // 2, max_workers=4) == 1


def test_workers_for_more_memory_allows_more_threads():
    small = workers_for(183, memory_limit=4 * GiB, max_workers=8)
    big = workers_for(183, memory_limit=16 * GiB, max_workers=8)
    assert big > small


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"memory.max": "3774873600\n"}, 3774873600),
        ({"memory.max": "max\n"}, None),
        ({"memory/memory.limit_in_bytes": "3774873600\n"}, 3774873600),
        # cgroup v1 reports "unlimited" as a page-aligned near-max int64.
        ({"memory/memory.limit_in_bytes": "9223372036854771712\n"}, None),
        ({}, None),
    ],
)
def test_container_memory_limit(tmp_path, files, expected):
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    assert container_memory_limit(cgroup_root=tmp_path) == expected


def test_container_memory_limit_prefers_v2(tmp_path):
    (tmp_path / "memory.max").write_text("100\n")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("200\n")
    assert container_memory_limit(cgroup_root=tmp_path) == 100


def test_default_max_workers_from_dask_config():
    with dask.config.set(num_workers=3):
        assert default_max_workers() == 3


def test_default_max_workers_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 6)
    with dask.config.set(num_workers=None):
        assert default_max_workers() == 6
