"""Feature construction for the projection model."""

from __future__ import annotations

import datetime as dt

import polars as pl

from ..scoring import ScoringEngine

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")


def season_totals(weekly: pl.DataFrame, engine: ScoringEngine) -> pl.DataFrame:
    """Collapse weekly stats into player-season totals under league scoring.

    Games are counted as weeks the player actually recorded a stat line, which is
    the closest thing nflverse gives us to "was available and used".
    """
    scored = engine.score_weekly(weekly)
    usage_cols = [
        c
        for c in ("targets", "carries", "attempts", "receptions", "receiving_yards",
                  "rushing_yards", "passing_yards")
        if c in scored.columns
    ]
    agg = [
        pl.col("fantasy_points").sum().alias("points"),
        pl.len().alias("games"),
        pl.col("fantasy_points").mean().alias("ppg"),
        pl.col("fantasy_points").std().alias("ppg_sd"),
        pl.col("position").drop_nulls().first().alias("position"),
        pl.col("player_display_name").drop_nulls().first().alias("name"),
        pl.col("team").drop_nulls().last().alias("team"),
    ]
    agg += [pl.col(c).sum().alias(f"{c}_total") for c in usage_cols]

    return (
        scored.filter(pl.col("position").is_in(FANTASY_POSITIONS))
        .group_by(["player_id", "season"])
        .agg(agg)
        .rename({"player_id": "gsis_id"})
        .sort(["gsis_id", "season"])
    )


def attach_age(totals: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Add each player's age as of Sept 1 of that season, plus draft capital."""
    cols = ["gsis_id", "birth_date", "draft_year", "draft_round", "draft_pick", "rookie_season"]
    have = [c for c in cols if c in players.columns]
    master = players.select(have).unique(subset=["gsis_id"], keep="first")

    out = totals.join(master, on="gsis_id", how="left")
    if "birth_date" in out.columns:
        out = out.with_columns(
            pl.col("birth_date").cast(pl.Date, strict=False).alias("birth_date")
        ).with_columns(
            (
                (
                    pl.date(pl.col("season"), 9, 1).cast(pl.Date)
                    - pl.col("birth_date")
                ).dt.total_days()
                / 365.25
            ).alias("age")
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("age"))

    if "rookie_season" in out.columns:
        out = out.with_columns(
            (pl.col("season") - pl.col("rookie_season")).alias("experience")
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("experience"))
    return out


def attach_expected_points(
    totals: pl.DataFrame, opportunity: pl.DataFrame | None
) -> pl.DataFrame:
    """Add expected-points-from-usage totals (ffopportunity), when available.

    Expected points strip out TD luck, so a player whose actual points ran far
    above expected is a regression candidate and vice versa.
    """
    if opportunity is None or opportunity.height == 0:
        return totals.with_columns(pl.lit(None, dtype=pl.Float64).alias("exp_points"))

    exp_col = next(
        (c for c in ("total_fantasy_points_exp", "total_fantasy_points_ppr_exp")
         if c in opportunity.columns),
        None,
    )
    if exp_col is None:
        return totals.with_columns(pl.lit(None, dtype=pl.Float64).alias("exp_points"))

    agg = (
        opportunity.group_by(["player_id", "season"])
        .agg(pl.col(exp_col).sum().alias("exp_points"))
        .rename({"player_id": "gsis_id"})
    )
    return totals.join(agg, on=["gsis_id", "season"], how="left")


def current_age(birth_date: dt.date | None, season: int) -> float | None:
    if birth_date is None:
        return None
    return (dt.date(season, 9, 1) - birth_date).days / 365.25
