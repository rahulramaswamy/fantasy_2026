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

from dataclasses import dataclass, field

import polars as pl

from ..scoring import ScoringEngine

# Sleeper injury designations -> games the player should be expected to miss
# from now, on top of his base availability rate. These are Sleeper's literal
# strings (it abbreviates "Suspended" to "Sus", for instance).
#
#   * Out / Doubtful: this week's game, near-certainly.
#   * Questionable: roughly one in four questionable players sits.
#   * IR / PUP / NA (non-football injury): the league minimum stint is four
#     games, and most stays run longer.
#   * Sus: suspensions range from one game to a season; three is the median
#     of the ones fantasy-relevant players actually serve.
#
# The status itself says nothing about *how long* after the minimum a player
# is out; the availability rate (games played so far) carries that, because a
# player who has missed half his games is likelier to keep missing them.
EXPECTED_GAMES_MISSED: dict[str, float] = {
    "Out": 1.0,
    "Doubtful": 1.0,
    "Questionable": 0.25,
    "IR": 4.0,
    "PUP": 4.0,
    "NA": 4.0,
    "Sus": 3.0,
    "COV": 1.0,
    "DNR": 4.0,
}

# Designations under which a player will not take the field this week.
OUT_STATUSES = frozenset({"IR", "PUP", "Out", "Sus", "NA", "Doubtful", "COV", "DNR"})


@dataclass
class ROSConfig:
    """Knobs for the in-season update."""

    # Games of preseason prior to carry. Higher = slower to believe a hot start.
    # At `prior_games` games played, the projection is a 50/50 blend.
    # Validated on 2025: 2.0 beats 5.0 and 8.0 at every week tested.
    prior_games: float = 2.0
    # Availability prior. A player who has missed games is likelier to keep
    # missing them -- absence is evidence, not noise. `avail_prior_games` is how
    # many games of benefit-of-the-doubt to extend before observed availability
    # takes over. The prior *rate* is the player's own preseason durability
    # (proj_games / games_per_team) where known, so that with zero games played
    # the rest-of-season projection reduces exactly to the preseason one.
    avail_prior_games: float = 2.0
    avail_prior_rate: float = 0.9
    # Games each team plays in a full regular season.
    games_per_team: int = 17
    # Expected games missed per injury designation; see EXPECTED_GAMES_MISSED.
    games_missed: dict[str, float] = field(default_factory=lambda: dict(EXPECTED_GAMES_MISSED))
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
    ka = config.avail_prior_games
    if "proj_games" in df.columns:
        prior_rate = (
            (pl.col("proj_games") / config.games_per_team)
            .clip(0.0, 1.0)
            .fill_null(config.avail_prior_rate)
        )
    else:
        prior_rate = pl.lit(config.avail_prior_rate)

    df = df.with_columns(prior_rate.alias("_avail_prior")).with_columns(
        (
            (pl.col("games_played") + ka * pl.col("_avail_prior"))
            / (pl.col("team_games_played") + ka)
        ).clip(0.0, 1.0).alias("availability")
    ).drop("_avail_prior")

    # Known absences: a current injury designation is information about the
    # *next* games specifically, so it comes off the games remaining rather
    # than off the scoring rate. A player on IR is not a worse player; he is a
    # player with fewer games left.
    if "injury_status" in df.columns:
        missed = pl.col("injury_status").replace_strict(
            config.games_missed, default=0.0, return_dtype=pl.Float64
        )
    else:
        missed = pl.lit(0.0)
    df = df.with_columns(missed.alias("exp_games_missed")).with_columns(
        (
            (pl.col("team_games_left") - pl.col("exp_games_missed")).clip(lower_bound=0.0)
            * pl.col("availability")
        ).alias("ros_games")
    )

    # Bayesian-ish update: preseason rate as prior, this season's games as data.
    k = config.prior_games
    df = df.with_columns(
        (
            (pl.col("std_points") + k * pl.col("proj_ppg"))
            / (pl.col("games_played") + k)
        ).alias("ros_ppg")
    )

    return df.with_columns(
        (pl.col("ros_ppg") * pl.col("ros_games")).alias("ros_points")
    ).sort("ros_points", descending=True)


def blend_weight(games_played: float, prior_games: float = 5.0) -> float:
    """Share of the projection coming from this season, for explaining output."""
    if games_played <= 0:
        return 0.0
    return games_played / (games_played + prior_games)
