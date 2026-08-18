"""Market data: ADP and trade values.

The model says what a player is worth. The market says what he will *cost* --
in draft capital, or in trade. You need both: value minus price is the only
edge that exists, and you cannot compute it from projections alone.

Sources:
  * Fantasy Football Calculator -- real mock-draft ADP, free REST API, asks for
    attribution and light request volume.
  * FantasyCalc -- trade values derived from actual completed trades across
    thousands of leagues, and it publishes `sleeperId` so joins are exact.
"""

from __future__ import annotations

from typing import Any

import httpx
import polars as pl

from ..ids import join_key
from .cache import TTL_HOUR, cached

FFC_ADP_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}"
FANTASYCALC_URL = "https://api.fantasycalc.com/values/current"

USER_AGENT = "ff2026/0.1 (personal fantasy tooling)"

# League scoring -> the ADP format slug Fantasy Football Calculator publishes.
FFC_FORMATS = {"standard", "ppr", "half-ppr", "2qb", "dynasty", "rookie"}


def ffc_format_for(ppr: float, superflex: bool = False) -> str:
    if superflex:
        return "2qb"
    if ppr >= 0.75:
        return "ppr"
    if ppr >= 0.25:
        return "half-ppr"
    return "standard"


def fetch_adp(
    season: int,
    fmt: str = "ppr",
    teams: int = 12,
    position: str = "all",
    timeout: float = 15.0,
) -> pl.DataFrame:
    """Average draft position from live mock drafts.

    Returns columns: name, position, team, adp, adp_stdev, times_drafted, bye,
    plus a `join_key` for matching against the Sleeper crosswalk.
    """
    if fmt not in FFC_FORMATS:
        raise ValueError(f"Unknown ADP format {fmt!r}; expected one of {sorted(FFC_FORMATS)}")

    url = FFC_ADP_URL.format(fmt=fmt)
    params = {"teams": teams, "year": season, "position": position}
    resp = httpx.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()

    players = payload.get("players") or []
    if not players:
        return pl.DataFrame(
            schema={
                "name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8, "adp": pl.Float64,
                "adp_stdev": pl.Float64, "times_drafted": pl.Int64, "bye": pl.Int64,
                "join_key": pl.Utf8,
            }
        )

    rows = [
        {
            "name": p.get("name"),
            "position": p.get("position"),
            "team": p.get("team"),
            "adp": float(p["adp"]) if p.get("adp") is not None else None,
            "adp_stdev": float(p["stdev"]) if p.get("stdev") is not None else None,
            "times_drafted": p.get("times_drafted"),
            "bye": p.get("bye"),
        }
        for p in players
    ]
    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.with_columns(
        pl.struct(["name", "position"])
        .map_elements(lambda s: join_key(s["name"], s["position"]), return_dtype=pl.Utf8)
        .alias("join_key")
    )


def fetch_trade_values(
    is_dynasty: bool = False,
    num_qbs: int = 1,
    num_teams: int = 12,
    ppr: float = 1.0,
    timeout: float = 15.0,
) -> pl.DataFrame:
    """Market trade values derived from real completed trades.

    Includes `sleeper_id`, so this joins to the roster data exactly rather than
    by name.
    """
    params = {
        "isDynasty": str(is_dynasty).lower(),
        "numQbs": num_qbs,
        "numTeams": num_teams,
        "ppr": ppr,
    }
    resp = httpx.get(
        FANTASYCALC_URL, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or not payload:
        return pl.DataFrame(
            schema={
                "sleeper_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8,
                "market_value": pl.Float64, "overall_rank": pl.Int64,
                "position_rank": pl.Int64, "trend_30day": pl.Float64,
                "market_adp": pl.Float64, "join_key": pl.Utf8,
            }
        )

    rows = []
    for entry in payload:
        player = entry.get("player") or {}
        rows.append(
            {
                "sleeper_id": str(player["sleeperId"]) if player.get("sleeperId") else None,
                "name": player.get("name"),
                "position": player.get("position"),
                "team": player.get("maybeTeam"),
                "market_value": float(entry.get("value") or 0),
                "overall_rank": entry.get("overallRank"),
                "position_rank": entry.get("positionRank"),
                "trend_30day": entry.get("trend30Day"),
                "market_adp": entry.get("maybeAdp"),
            }
        )
    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.with_columns(
        pl.struct(["name", "position"])
        .map_elements(lambda s: join_key(s["name"], s["position"]), return_dtype=pl.Utf8)
        .alias("join_key")
    )


def cached_adp(season: int, fmt: str = "ppr", teams: int = 12, force: bool = False):
    """ADP with a one-hour TTL -- it moves during draft season but not by the minute."""
    return cached(
        f"adp_{fmt}_{teams}_{season}",
        lambda: fetch_adp(season=season, fmt=fmt, teams=teams),
        ttl=TTL_HOUR,
        force=force,
    )


def cached_trade_values(
    is_dynasty: bool = False, num_qbs: int = 1, num_teams: int = 12, ppr: float = 1.0,
    force: bool = False,
):
    return cached(
        f"tradevalues_{'dyn' if is_dynasty else 'red'}_{num_qbs}qb_{num_teams}tm_{ppr}ppr",
        lambda: fetch_trade_values(is_dynasty, num_qbs, num_teams, ppr),
        ttl=TTL_HOUR,
        force=force,
    )
