"""Sizing dask's thread pool to fit a composite in the container's memory.

Each dask thread composites one spatial block holding the full temporal stack,
so peak memory grows with threads x granules. Tiles where adjacent orbital
paths overlap (common at high latitudes) can see several times the usual
granule count, and running the usual number of threads on them gets the
container killed for exceeding its memory limit.
"""

from __future__ import annotations

import os
from pathlib import Path

import dask

from hls_composites.composite import BLOCK_SIZE

BYTES_PER_OBSERVATION_PIXEL = 59
"""Peak bytes held per pixel per granule while one block is composited.

Measured from `composite._composite_block` with the default bands and
indices (~46 B), plus the inputs' copy (~13 B: six int16 reflectance bands
and a uint8 Fmask) that dask holds while concatenating per-granule chunks
into the block's temporal stack. Revisit if bands, indices, or the kernel
change.
"""

BASELINE_BYTES = 512 * 2**20
"""Memory the process holds outside the per-block work: interpreter,
libraries, GDAL's block cache, and the composited outputs."""

CGROUP_ROOT = Path("/sys/fs/cgroup")

_CGROUP_V1_UNLIMITED = 2**62
"""cgroup v1 reports no limit as a page-aligned value near the int64 max."""


def block_bytes(n_granules: int, block_size: int = BLOCK_SIZE) -> int:
    """Estimate peak memory one dask thread needs to composite one block.

    Parameters
    ----------
    n_granules : int
        Number of granules in the temporal stack.
    block_size : int, optional
        Spatial block edge length in pixels, by default `BLOCK_SIZE`.

    Returns
    -------
    int
        Estimated peak bytes.
    """
    return BYTES_PER_OBSERVATION_PIXEL * block_size * block_size * n_granules


def container_memory_limit(cgroup_root: Path = CGROUP_ROOT) -> int | None:
    """Read this container's memory limit from its cgroup.

    Parameters
    ----------
    cgroup_root : pathlib.Path, optional
        Root of the cgroup filesystem, by default ``/sys/fs/cgroup``.

    Returns
    -------
    int or None
        The limit in bytes, or None when unlimited or not discoverable
        (e.g. outside a container, or on a platform without cgroups).
    """
    v2 = cgroup_root / "memory.max"
    if v2.is_file():
        value = v2.read_text().strip()
        return None if value == "max" else int(value)
    v1 = cgroup_root / "memory" / "memory.limit_in_bytes"
    if v1.is_file():
        limit = int(v1.read_text().strip())
        return None if limit >= _CGROUP_V1_UNLIMITED else limit
    return None


def default_max_workers() -> int:
    """The thread count dask would use if left to itself.

    Returns
    -------
    int
        dask's configured ``num_workers`` (e.g. from ``DASK_NUM_WORKERS``),
        otherwise the host's CPU count.
    """
    configured = dask.config.get("num_workers", None)
    if configured:
        return int(configured)
    return os.cpu_count() or 1


def workers_for(n_granules: int, memory_limit: int | None, max_workers: int) -> int:
    """Choose how many dask threads fit a composite into `memory_limit`.

    Parameters
    ----------
    n_granules : int
        Number of granules being composited.
    memory_limit : int or None
        Bytes available to the process. None means unknown or unlimited,
        in which case `max_workers` is used as-is.
    max_workers : int
        Upper bound on threads.

    Returns
    -------
    int
        Thread count between 1 and `max_workers`. Never below 1, even when
        a single thread is estimated to exceed `memory_limit`.
    """
    if memory_limit is None:
        return max_workers
    per_thread = block_bytes(n_granules)
    fits = (memory_limit - BASELINE_BYTES) // per_thread if per_thread else max_workers
    return max(1, min(max_workers, fits))
