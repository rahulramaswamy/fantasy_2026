"""This week's lineup.

Rest-of-season points are the right currency for trades and waivers, but the
wrong one for Sunday: a player on bye is worth nothing this week however good
he is, and a player listed Out is worth nothing this week however good his
rest-of-season number looks. So start/sit runs on a separate, one-week value:

  * the player's current scoring rate (`ros_ppg`, which already folds in this
    season's evidence),
  * zeroed if his team is on bye or he carries an Out-type designation,
  * and, during the season, pulled toward outside weekly projections -- the
    FantasyPros consensus and Sleeper's own projection, averaged where both
    exist. Both are built for this week's opponent and re-rank after injury
    news, which is exactly what a Sunday lineup needs and what a season-long
    rate cannot see.

The lineup itself is filled with the same most-constrained-first optimizer the
trade evaluator uses, so a FLEX never steals a player a dedicated slot needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

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


def sleeper_week_points(
    rows: list[dict[str, Any]], scoring_settings: dict[str, float]
) -> pl.DataFrame:
    """Score Sleeper's projected stat lines under this league's rules.

    Sleeper projects stats under its own scoring keys (`rec`, `rec_yd`,
    `bonus_rec_te`, ...), the same keys a league's `scoring_settings` uses, so
    the league score is a straight dot product -- no stat mapping to go wrong.

    Returns: sleeper_id, sleeper_pts, sleeper_opp.
    """
    out = []
    for row in rows:
        stats = row.get("stats") or {}
        pid = row.get("player_id")
        if not pid or not stats:
            continue
        pts = sum(
            float(stats[key]) * value
            for key, value in scoring_settings.items()
            if value and isinstance(stats.get(key), int | float)
        )
        out.append({
            "sleeper_id": str(pid),
            "sleeper_pts": round(pts, 2),
            "sleeper_opp": row.get("opponent"),
        })
    schema = {"sleeper_id": pl.Utf8, "sleeper_pts": pl.Float64, "sleeper_opp": pl.Utf8}
    if not out:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(out, schema=schema).unique(subset=["sleeper_id"], keep="first")


def weekly_values(
    board: pl.DataFrame,
    week: int,
    schedule: pl.DataFrame,
    season: int | None = None,
    expert: pl.DataFrame | None = None,
    expert_weight: float = DEFAULT_EXPERT_WEIGHT,
    today: date | None = None,
    sleeper: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, bool]:
    """Attach `week_points`: what each player is worth in this one week.

    `expert` is the FantasyPros weekly feed (gated on freshness); `sleeper` is
    `sleeper_week_points` for this week, which is fresh by construction since
    it is fetched for the week asked about. Where both exist they are averaged,
    and that average takes `expert_weight` of the number.

    Returns the frame and whether any outside weekly projection was folded in.
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

    outside: list[str] = []
    fresh = expert_is_fresh(expert, season or 0, today) if season else False
    if fresh and expert is not None and "gsis_id" in df.columns:
        cols = [c for c in ("gsis_id", "week_pts", "week_opp", "week_note", "week_ecr")
                if c in expert.columns]
        df = df.join(expert.select(cols), on="gsis_id", how="left")
        if "week_pts" in df.columns:
            outside.append("week_pts")

    if (sleeper is not None and not sleeper.is_empty()
            and "sleeper_id" in df.columns and "sleeper_pts" in sleeper.columns):
        cols = [c for c in ("sleeper_id", "sleeper_pts", "sleeper_opp") if c in sleeper.columns]
        df = df.with_columns(pl.col("sleeper_id").cast(pl.Utf8)).join(
            sleeper.select(cols).with_columns(pl.col("sleeper_id").cast(pl.Utf8)),
            on="sleeper_id", how="left",
        )
        outside.append("sleeper_pts")
        if "week_opp" not in df.columns and "sleeper_opp" in df.columns:
            df = df.with_columns(pl.col("sleeper_opp").alias("week_opp"))

    if outside:
        # Nulls are skipped, so a player only one source covers gets that one.
        consensus = pl.mean_horizontal([pl.col(c) for c in outside])
        # Outside sources list a player on bye/out at 0 or not at all; either way
        # the zero from our own side wins, so a stale non-zero cannot start him.
        df = df.with_columns(
            pl.when(consensus.is_not_null() & (pl.col("_own_week") > 0))
            .then(expert_weight * consensus + (1 - expert_weight) * pl.col("_own_week"))
            .otherwise(pl.col("_own_week"))
            .alias("week_points")
        )
    else:
        df = df.with_columns(pl.col("_own_week").alias("week_points"))

    return df.drop("_own_week"), bool(outside)


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
    used = "week_pts" in my_roster.columns or "sleeper_pts" in my_roster.columns
    return WeeklyLineup(starters, bench, points, used)


def flags(row: dict) -> str:
    """Short human-readable reasons a player is worth less than usual this week."""
    out = []
    if row.get("on_bye"):
        out.append("BYE")
    status = row.get("injury_status")
    if status:
        out.append(str(status))
    return " ".join(out)
