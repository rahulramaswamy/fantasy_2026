"""League power rankings: roster strength vs. actual results.

Standings tell you who has won. They do not tell you who is *good*. In a
12-game season with weekly head-to-head matchups, a large share of any record is
schedule luck -- you can score the second-most points in the league and sit at
4-6 because you kept drawing whoever went off that week.

This separates the two by ranking each team three ways:

  * **Preseason** -- what the roster projected to score before a snap was played.
  * **Rest of season** -- what it projects to score from here, which is the only
    one that affects decisions you can still make.
  * **Actual** -- points scored and record to date.

The gaps are the interesting part:

  * Strong roster, bad record  -> unlucky. Their owner may be ready to sell.
    The best trade partner in the league.
  * Weak roster, good record   -> lucky, and likely to regress. A dangerous
    team to bet on and a good one to sell to, because they feel like winners.

`ff league power` prints all three side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from .config import LeagueConfig
from .trades.evaluate import optimal_lineup


@dataclass
class TeamStrength:
    roster_id: int
    manager: str
    team_name: str
    preseason: float
    ros: float
    points_for: float
    wins: int
    losses: int
    ties: int
    players: int

    @property
    def record(self) -> str:
        base = f"{self.wins}-{self.losses}"
        return f"{base}-{self.ties}" if self.ties else base


def _display_name(user: dict[str, Any]) -> tuple[str, str]:
    manager = user.get("display_name") or user.get("username") or "unknown"
    meta = user.get("metadata") or {}
    return manager, (meta.get("team_name") or manager)


def team_strengths(
    rosters: list[dict[str, Any]],
    users: list[dict[str, Any]],
    board: pl.DataFrame,
    league: LeagueConfig,
    ros_col: str = "ros_points",
    preseason_col: str = "proj_points",
) -> list[TeamStrength]:
    """Score every roster in the league on all three axes."""
    by_user = {u.get("user_id"): u for u in users}
    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"
    has_ros = ros_col in board.columns

    out: list[TeamStrength] = []
    for roster in rosters:
        player_ids = [str(p) for p in (roster.get("players") or [])]
        held = board.filter(pl.col(id_col).is_in(player_ids))

        preseason, _ = optimal_lineup(held, league, preseason_col)
        ros = optimal_lineup(held, league, ros_col)[0] if has_ros else 0.0

        settings = roster.get("settings") or {}
        points = float(settings.get("fpts") or 0) + float(
            settings.get("fpts_decimal") or 0
        ) / 100.0

        user = by_user.get(roster.get("owner_id")) or {}
        manager, team_name = _display_name(user)

        out.append(
            TeamStrength(
                roster_id=int(roster.get("roster_id") or 0),
                manager=manager,
                team_name=team_name,
                preseason=preseason,
                ros=ros,
                points_for=points,
                wins=int(settings.get("wins") or 0),
                losses=int(settings.get("losses") or 0),
                ties=int(settings.get("ties") or 0),
                players=len(player_ids),
            )
        )
    return out


def power_table(strengths: list[TeamStrength]) -> pl.DataFrame:
    """Rank teams on each axis and expose the gaps between them."""
    if not strengths:
        return pl.DataFrame()

    df = pl.DataFrame(
        [
            {
                "manager": s.manager,
                "team": s.team_name,
                "record": s.record,
                "wins": s.wins,
                "preseason": round(s.preseason, 1),
                "ros": round(s.ros, 1),
                "points_for": round(s.points_for, 1),
                "players": s.players,
            }
            for s in strengths
        ]
    )

    df = df.with_columns(
        pl.col("preseason").rank("ordinal", descending=True).cast(pl.Int64)
        .alias("rank_pre"),
        pl.col("ros").rank("ordinal", descending=True).cast(pl.Int64).alias("rank_ros"),
        pl.col("points_for").rank("ordinal", descending=True).cast(pl.Int64)
        .alias("rank_actual"),
        pl.col("wins").rank("ordinal", descending=True).cast(pl.Int64).alias("rank_record"),
    )

    # Positive luck = better record than the points scored deserve.
    return df.with_columns(
        (pl.col("rank_actual") - pl.col("rank_record")).alias("luck"),
        (pl.col("rank_pre") - pl.col("rank_ros")).alias("trend"),
    ).sort("rank_ros")


def read_the_table(df: pl.DataFrame, top_n: int = 3) -> list[str]:
    """Plain-English takeaways, so the numbers do not need interpreting."""
    if df.is_empty():
        return []

    notes: list[str] = []
    unlucky = df.filter(pl.col("luck") < 0).sort("luck").head(top_n)
    lucky = df.filter(pl.col("luck") > 0).sort("luck", descending=True).head(top_n)
    risers = df.filter(pl.col("trend") > 1).sort("trend", descending=True).head(top_n)

    for row in unlucky.to_dicts():
        notes.append(
            f"{row['manager']}: scores like #{row['rank_actual']} but sits #"
            f"{row['rank_record']} on record -- unlucky, and the best team to "
            f"try to buy from."
        )
    for row in lucky.to_dicts():
        notes.append(
            f"{row['manager']}: record of #{row['rank_record']} on #"
            f"{row['rank_actual']} scoring -- riding luck, likely to regress."
        )
    for row in risers.to_dicts():
        notes.append(
            f"{row['manager']}: roster has improved from #{row['rank_pre']} "
            f"preseason to #{row['rank_ros']} rest-of-season."
        )
    return notes
