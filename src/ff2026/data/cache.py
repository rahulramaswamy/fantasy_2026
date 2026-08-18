"""A small on-disk parquet cache with TTLs.

Upstream sources have very different refresh rates: nflverse stats update a few
times a week in season, Sleeper's player dump changes daily, and a live draft
changes every few seconds. Rather than sprinkle ad-hoc caching around, every
fetch goes through `cached()` with an explicit TTL.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from ..config import get_settings

# Convenience TTLs (seconds).
TTL_MINUTE = 60
TTL_HOUR = 3600
TTL_DAY = 86_400
TTL_WEEK = 7 * 86_400


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def is_fresh(path: Path, ttl: float) -> bool:
    if not path.exists():
        return False
    if ttl <= 0:
        return False
    return (time.time() - path.stat().st_mtime) < ttl


def cached(
    name: str,
    fetch: Callable[[], pl.DataFrame],
    ttl: float = TTL_DAY,
    force: bool = False,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Return a cached dataframe, refetching when stale or forced."""
    directory = cache_dir or get_settings().cache_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"

    if not force and is_fresh(path, ttl):
        return pl.read_parquet(path)

    df = fetch()
    try:
        df.write_parquet(path)
        _meta_path(path).write_text(
            json.dumps({"name": name, "rows": df.height, "fetched_at": time.time()})
        )
    except Exception:  # noqa: BLE001 - a cache write failure must not break a draft
        pass
    return df


def cached_json(
    name: str,
    fetch: Callable[[], Any],
    ttl: float = TTL_DAY,
    force: bool = False,
    cache_dir: Path | None = None,
) -> Any:
    """Same as `cached` but for arbitrary JSON payloads (Sleeper responses)."""
    directory = cache_dir or get_settings().cache_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"

    if not force and is_fresh(path, ttl):
        return json.loads(path.read_text())

    payload = fetch()
    try:
        path.write_text(json.dumps(payload))
    except Exception:  # noqa: BLE001
        pass
    return payload


def clear(pattern: str = "*", cache_dir: Path | None = None) -> int:
    directory = cache_dir or get_settings().cache_dir
    n = 0
    for p in directory.glob(pattern):
        if p.is_file():
            p.unlink()
            n += 1
    return n
