"""This week's lineup.

Rest-of-season points are the right currency for trades and waivers, but the
wrong one for Sunday: a player on bye is worth nothing this week however good
he is, and a player listed Out is worth nothing this week however good his
rest-of-season number looks. So start/sit runs on a separate, one-week value:

  * the player's current scoring rate (`ros_ppg`, which already folds in this
    season's evidence),
  * zeroed if his team is on bye or he carries an Out-type designation,
  * and, during the season, pulled toward the FantasyPros weekly consensus
    projection where one exists -- the experts re-rank after Friday injury
    reports, which is exactly the information a Sunday lineup needs.

The lineup itself is filled with the same most-constrained-first optimizer the
trade evaluator uses, so a FLEX never steals a player a dedicated slot needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from ..config import LeagueConfig
from ..model.ros import OUT_STATUSES
from ..trades.evaluate import optimal_lineup

# Questionable players play roughly three weeks in four.
QUESTIONABLE_FACTOR = 0.75

# Share of the weekly number taken from expert consensus when it is fresh. The
# rest comes from the player's own rate, which guards against a bad id match
# or a stale page swinging a lineup on its own.
DEFAULT_EXPERT_WEIGHT = 0.75

# A weekly scrape older than this is for a week that has already been played.
MAX_EXPERT_AGE_DAYS = 8


@dataclass
class WeeklyLineup:
    starters: pl.DataFrame
    bench: pl.DataFrame
    points: float
    used_expert: bool


def teams_on_bye(schedule: pl.DataFrame, week: int, season: int | None = None) -> set[str]:
    """Teams with no game in `week`. Empty set if the schedule has no such week."""
    games = schedule
    if season is not None and "season" in games.columns:
        games = games.filter(pl.col("season") == season)
    if "game_type" in games.columns:
        games = games.filter(pl.col("game_type") == "REG")
    week_games = games.filter(pl.col("week") == week)
    if week_games.is_empty():
        return set()
    playing = set(week_games["home_team"].to_list()) | set(week_games["away_team"].to_list())
    every_team = set(games["home_team"].to_list()) | set(games["away_team"].to_list())
    return every_team - playing


def expert_is_fresh(
    expert: pl.DataFrame | None, season: int, today: date | None = None
) -> bool:
    """Is the weekly expert feed from this season and this week?"""
    if expert is None or expert.is_empty() or "scrape_date" not in expert.columns:
        return False
    latest = expert["scrape_date"].drop_nulls().cast(pl.Utf8).max()
    if latest is None:
        return False
    try:
        scraped = date.fromisoformat(str(latest)[:10])
    except ValueError:
        return False
    if scraped < date(season, 8, 25):
        return False
    today = today or date.today()
    return (today - scraped).days <= MAX_EXPERT_AGE_DAYS


def weekly_values(
    board: pl.DataFrame,
    week: int,
    schedule: pl.DataFrame,
    season: int | None = None,
    expert: pl.DataFrame | None = None,
    expert_weight: float = DEFAULT_EXPERT_WEIGHT,
    today: date | None = None,
) -> tuple[pl.DataFrame, bool]:
    """Attach `week_points`: what each player is worth in this one week.

    Returns the frame and whether expert weekly projections were folded in.
    """
    rate_col = "ros_ppg" if "ros_ppg" in board.columns else "proj_ppg"
    byes = teams_on_bye(schedule, week, season)

    df = board.with_columns(pl.col("team").is_in(list(byes)).alias("on_bye"))
    status = pl.col("injury_status") if "injury_status" in df.columns else pl.lit(None)
    plays = (
        pl.when(pl.col("on_bye")).then(0.0)
        .when(status.is_in(list(OUT_STATUSES))).then(0.0)
        .when(status == "Questionable").then(QUESTIONABLE_FACTOR)
        .otherwise(1.0)
    )
    df = df.with_columns((pl.col(rate_col).fill_null(0.0) * plays).alias("_own_week"))

    fresh = expert_is_fresh(expert, season or 0, today) if season else False
    if fresh and expert is not None and "gsis_id" in df.columns:
        cols = [c for c in ("gsis_id", "week_pts", "week_opp", "week_note", "week_ecr")
                if c in expert.columns]
        df = df.join(expert.select(cols), on="gsis_id", how="left")
        if "week_pts" in df.columns:
            # Experts list a player on bye/out at 0 or not at all; either way the
            # zero from our own side wins, so a stale non-zero cannot start him.
            df = df.with_columns(
                pl.when(pl.col("week_pts").is_not_null() & (pl.col("_own_week") > 0))
                .then(
                    expert_weight * pl.col("week_pts")
                    + (1 - expert_weight) * pl.col("_own_week")
                )
                .otherwise(pl.col("_own_week"))
                .alias("week_points")
            )
        else:
            df = df.with_columns(pl.col("_own_week").alias("week_points"))
            fresh = False
    else:
        df = df.with_columns(pl.col("_own_week").alias("week_points"))
        fresh = False

    return df.drop("_own_week"), fresh


def set_lineup(
    my_roster: pl.DataFrame, league: LeagueConfig, value_col: str = "week_points"
) -> WeeklyLineup:
    """Best legal lineup for the week, plus who sits."""
    points, starters = optimal_lineup(my_roster, league, value_col)
    if starters.is_empty():
        return WeeklyLineup(starters, my_roster, 0.0, False)
    key = "sleeper_id" if "sleeper_id" in my_roster.columns else "gsis_id"
    started = set(starters[key].to_list())
    bench = my_roster.filter(~pl.col(key).is_in(list(started))).sort(
        value_col, descending=True, nulls_last=True
    )
    return WeeklyLineup(starters, bench, points, "week_pts" in my_roster.columns)


def flags(row: dict) -> str:
    """Short human-readable reasons a player is worth less than usual this week."""
    out = []
    if row.get("on_bye"):
        out.append("BYE")
    status = row.get("injury_status")
    if status:
        out.append(str(status))
    return " ".join(out)
