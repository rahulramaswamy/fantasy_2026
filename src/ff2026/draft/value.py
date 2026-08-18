"""Turning projections into draft value.

Raw projected points are the wrong unit for a draft. 300 points from a QB in a
1-QB league is worth much less than 300 from a RB, because the QB you would
otherwise have started scores nearly as much and the RB you would otherwise have
started does not. Value over replacement fixes the units.

Replacement level here is derived from the league's actual roster rules -- team
count, dedicated slots, and how flex slots get filled in practice -- rather than
from a rule of thumb.
"""

from __future__ import annotations

import polars as pl

from ..config import SLOT_ELIGIBILITY, LeagueConfig

PROJECTABLE = ("QB", "RB", "WR", "TE")


def starter_demand(league: LeagueConfig, projections: pl.DataFrame) -> dict[str, int]:
    """How many players at each position the league starts in a typical week.

    Dedicated slots are simple multiplication. Flex slots are allocated greedily
    to whichever eligible players actually project best, which is what managers
    do and therefore where replacement level really sits.
    """
    demand = {pos: league.starters_at(pos) * league.teams for pos in PROJECTABLE}

    ranked = projections.filter(pl.col("position").is_in(PROJECTABLE)).sort(
        "proj_points", descending=True
    )

    for slot, count in league.flex_slots().items():
        eligible = [p for p in SLOT_ELIGIBILITY.get(slot, ()) if p in PROJECTABLE]
        if not eligible:
            continue
        total = count * league.teams

        # Players already accounted for by dedicated slots are off the table.
        pool = []
        seen: dict[str, int] = dict.fromkeys(eligible, 0)
        for row in ranked.filter(pl.col("position").is_in(eligible)).iter_rows(named=True):
            pos = row["position"]
            seen[pos] += 1
            if seen[pos] > demand.get(pos, 0):
                pool.append(pos)
            if len(pool) >= total:
                break

        for pos in pool:
            demand[pos] = demand.get(pos, 0) + 1

    return demand


def replacement_levels(
    league: LeagueConfig, projections: pl.DataFrame
) -> dict[str, float]:
    """Projected points of the first player at each position who won't start."""
    demand = starter_demand(league, projections)
    levels: dict[str, float] = {}

    for pos in PROJECTABLE:
        pool = (
            projections.filter(pl.col("position") == pos)
            .sort("proj_points", descending=True)["proj_points"]
            .to_list()
        )
        if not pool:
            levels[pos] = 0.0
            continue
        idx = min(demand.get(pos, 0), len(pool) - 1)
        levels[pos] = float(pool[idx])

    return levels


def add_value(
    projections: pl.DataFrame,
    league: LeagueConfig,
    adp: pl.DataFrame | None = None,
    market: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach VORP, positional rank, and (where available) market price.

    `value_vs_adp` is the number that finds bargains: a player the model likes
    far more than the room does.
    """
    levels = replacement_levels(league, projections)

    df = projections.with_columns(
        pl.col("position")
        .replace_strict(levels, default=0.0)
        .cast(pl.Float64)
        .alias("replacement_points")
    ).with_columns(
        (pl.col("proj_points") - pl.col("replacement_points")).alias("vorp")
    )

    df = df.with_columns(
        pl.col("proj_points").rank("ordinal", descending=True).over("position")
        .cast(pl.Int64).alias("pos_rank"),
        pl.col("vorp").rank("ordinal", descending=True).cast(pl.Int64).alias("value_rank"),
    )

    if adp is not None and adp.height and "join_key" in df.columns:
        df = df.join(
            adp.select(["join_key", "adp", "adp_stdev", "times_drafted"]).unique(
                subset=["join_key"], keep="first"
            ),
            on="join_key",
            how="left",
        )
    if "adp" not in df.columns:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("adp"),
            pl.lit(None, dtype=pl.Float64).alias("adp_stdev"),
        )

    if market is not None and market.height and "sleeper_id" in df.columns:
        df = df.join(
            market.select(["sleeper_id", "market_value", "overall_rank", "trend_30day"])
            .filter(pl.col("sleeper_id").is_not_null())
            .unique(subset=["sleeper_id"], keep="first"),
            on="sleeper_id",
            how="left",
        )

    # How far the model disagrees with the room, in draft slots.
    return df.with_columns(
        pl.when(pl.col("adp").is_not_null())
        .then(pl.col("adp") - pl.col("value_rank"))
        .otherwise(None)
        .alias("value_vs_adp")
    ).sort("vorp", descending=True)


def tier_breaks(
    projections: pl.DataFrame, position: str, max_players: int = 40, gap_sd: float = 1.0
) -> list[int]:
    """Find positional tier boundaries by looking for unusually large VORP gaps.

    Tiers are what make draft urgency legible: it doesn't matter that six backs
    are similar, it matters that the seventh is a step down.
    """
    pool = (
        projections.filter(pl.col("position") == position)
        .sort("proj_points", descending=True)
        .head(max_players)
    )
    if pool.height < 3:
        return []
    points = pool["proj_points"].to_list()
    gaps = [points[i] - points[i + 1] for i in range(len(points) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    sd = var**0.5
    threshold = mean_gap + gap_sd * sd
    return [i + 1 for i, g in enumerate(gaps) if g > threshold]
