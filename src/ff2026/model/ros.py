"""Rest-of-season projections.

Preseason projections answer "how good is this player over a full year". By week
6 that is the wrong question twice over: some of those points are already banked
and cannot be traded for, and eight weeks of new evidence exist that the
preseason number never saw.

Rest-of-season fixes both:

  1. **Re-estimate the rate** by treating the preseason projection as a prior and
     this season's games as evidence, in exactly the shrinkage form the preseason
     model already uses. Week 1 leans almost entirely on the prior; by week 10
     the current season dominates. No arbitrary "hot streak" logic -- just
     accumulating evidence.

  2. **Count the games that actually remain**, from the real schedule, so byes
     and a short remaining slate are priced correctly. A player with 5 games left
     is worth less than an equal player with 9, and nothing in a full-season
     projection captures that.

This is the number every in-season decision should use: trades, waivers, and
start/sit. `evaluate_trade` takes `value_col`, so pointing it at `ros_points`
switches the whole trade engine onto rest-of-season footing.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..scoring import ScoringEngine

# Sleeper injury designations that mean a player is unlikely to play soon.
OUT_STATUSES = frozenset({"IR", "PUP", "Out", "Suspended", "NA", "Doubtful"})


@dataclass
class ROSConfig:
    """Knobs for the in-season update."""

    # Games of preseason prior to carry. Higher = slower to believe a hot start.
    # At `prior_games` games played, the projection is a 50/50 blend.
    # Validated on 2025: 2.0 beats 5.0 and 8.0 at every week tested.
    prior_games: float = 2.0
    # Availability prior. A player who has missed games is likelier to keep
    # missing them -- absence is evidence, not noise. `avail_prior_games` is how
    # many games of benefit-of-the-doubt to extend, at `avail_prior_rate`.
    avail_prior_games: float = 2.0
    avail_prior_rate: float = 0.9
    # Multiplier applied to players carrying an out-for-now designation.
    injury_discount: float = 0.35
    # Regular season length, used when a schedule is unavailable.
    default_total_weeks: int = 18


def team_games_played(
    schedule: pl.DataFrame, current_week: int, season: int | None = None
) -> dict[str, int]:
    """Games each team has already played, so availability has a denominator."""
    games = schedule
    if season is not None and "season" in games.columns:
        games = games.filter(pl.col("season") == season)
    if "game_type" in games.columns:
        games = games.filter(pl.col("game_type") == "REG")
    games = games.filter(pl.col("week") <= current_week)
    if games.is_empty():
        return {}
    stacked = pl.concat(
        [games.select(pl.col("home_team").alias("team")),
         games.select(pl.col("away_team").alias("team"))],
        how="vertical",
    )
    return {r["team"]: int(r["len"]) for r in stacked.group_by("team").len().to_dicts()}


def remaining_games_by_team(
    schedule: pl.DataFrame, current_week: int, season: int | None = None
) -> dict[str, int]:
    """Games each team still has to play, byes excluded by construction."""
    games = schedule
    if season is not None and "season" in games.columns:
        games = games.filter(pl.col("season") == season)
    if "game_type" in games.columns:
        games = games.filter(pl.col("game_type") == "REG")
    games = games.filter(pl.col("week") > current_week)

    if games.is_empty():
        return {}

    stacked = pl.concat(
        [
            games.select(pl.col("home_team").alias("team")),
            games.select(pl.col("away_team").alias("team")),
        ],
        how="vertical",
    )
    counts = stacked.group_by("team").len()
    return {row["team"]: int(row["len"]) for row in counts.to_dicts()}


def season_to_date(
    weekly: pl.DataFrame, engine: ScoringEngine, through_week: int | None = None
) -> pl.DataFrame:
    """This season's scored production per player, up to `through_week`."""
    df = weekly
    if "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    if through_week is not None:
        df = df.filter(pl.col("week") <= through_week)

    scored = engine.score_weekly(df)
    return (
        scored.group_by("player_id")
        .agg(
            pl.col("fantasy_points").sum().alias("std_points"),
            pl.len().alias("games_played"),
            pl.col("fantasy_points").mean().alias("std_ppg"),
            pl.col("team").drop_nulls().last().alias("current_team"),
        )
        .rename({"player_id": "gsis_id"})
    )


def rest_of_season(
    board: pl.DataFrame,
    weekly: pl.DataFrame,
    schedule: pl.DataFrame,
    engine: ScoringEngine,
    current_week: int,
    season: int | None = None,
    config: ROSConfig | None = None,
) -> pl.DataFrame:
    """Project the points each player has left to score this season.

    `board` is a preseason board (needs `proj_ppg`); `weekly` is this season's
    stats so far. Adds: games_played, std_points, std_ppg, ros_ppg, ros_games,
    ros_points.
    """
    config = config or ROSConfig()

    std = season_to_date(weekly, engine, through_week=current_week)
    df = board.join(std, on="gsis_id", how="left").with_columns(
        pl.col("std_points").fill_null(0.0),
        pl.col("games_played").fill_null(0).cast(pl.Float64),
    )

    # Prefer the team a player has actually been playing for this season.
    if "current_team" in df.columns:
        df = df.with_columns(
            pl.coalesce([pl.col("current_team"), pl.col("team")]).alias("team")
        )

    remaining = remaining_games_by_team(schedule, current_week, season)
    played = team_games_played(schedule, current_week, season)
    fallback = max(0, config.default_total_weeks - current_week)
    df = df.with_columns(
        pl.col("team").replace_strict(remaining, default=fallback)
        .cast(pl.Float64).alias("team_games_left"),
        pl.col("team").replace_strict(played, default=float(current_week))
        .cast(pl.Float64).alias("team_games_played"),
    )

    # Availability: what share of his team's games has he actually played?
    # Without this, a player who has missed every game so far is projected as if
    # he will play every remaining one -- which validation showed is badly wrong,
    # because missing time is the single strongest predictor of missing more.
    ka, rate = config.avail_prior_games, config.avail_prior_rate
    df = df.with_columns(
        (
            (pl.col("games_played") + ka * rate)
            / (pl.col("team_games_played") + ka)
        ).clip(0.0, 1.0).alias("availability")
    ).with_columns(
        (pl.col("team_games_left") * pl.col("availability")).alias("ros_games")
    )

    # Bayesian-ish update: preseason rate as prior, this season's games as data.
    k = config.prior_games
    df = df.with_columns(
        (
            (pl.col("std_points") + k * pl.col("proj_ppg"))
            / (pl.col("games_played") + k)
        ).alias("ros_ppg")
    )

    # Players currently unavailable are discounted rather than dropped -- they
    # still hold trade value, just less of it.
    if "injury_status" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("injury_status").is_in(list(OUT_STATUSES)))
            .then(pl.col("ros_ppg") * config.injury_discount)
            .otherwise(pl.col("ros_ppg"))
            .alias("ros_ppg")
        )

    return df.with_columns(
        (pl.col("ros_ppg") * pl.col("ros_games")).alias("ros_points")
    ).sort("ros_points", descending=True)


def blend_weight(games_played: float, prior_games: float = 5.0) -> float:
    """Share of the projection coming from this season, for explaining output."""
    if games_played <= 0:
        return 0.0
    return games_played / (games_played + prior_games)
