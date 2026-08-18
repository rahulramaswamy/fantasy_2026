"""nflverse loaders.

nflverse is the backbone of this project: free, well-maintained, and it carries
the actual box-score inputs a projection needs. `nflreadpy` handles its own HTTP
caching, so these wrappers exist mostly to pin the season ranges we care about
and to keep loader names in one place.
"""

from __future__ import annotations

import warnings

import polars as pl

# nflreadpy is imported lazily inside functions so that importing this module
# (and therefore the CLI) stays fast and works offline.

DEFAULT_LOOKBACK = 4  # seasons of history used to fit the model


def _nfl():  # noqa: ANN202
    import nflreadpy as nfl

    return nfl


def seasons_back(current_season: int, n: int = DEFAULT_LOOKBACK) -> list[int]:
    """The n completed seasons before `current_season`."""
    return list(range(current_season - n, current_season))


def weekly_stats(seasons: list[int]) -> pl.DataFrame:
    """Per-player, per-week box score stats (the model's dependent variable)."""
    df = _nfl().load_player_stats(seasons=seasons, summary_level="week")
    return df.filter(pl.col("season_type") == "REG")


def weekly_stats_if_available(seasons: list[int]) -> pl.DataFrame:
    """Weekly stats, or an empty frame if nflverse has no file for that season.

    Before week 1 the current season's stats release does not exist yet, and a
    404 there is expected rather than exceptional -- rest-of-season projections
    must still work in the preseason, they just have no new evidence to fold in.
    """
    try:
        return weekly_stats(seasons)
    except Exception as exc:  # noqa: BLE001 - any transport error means "no data yet"
        warnings.warn(
            f"No weekly stats for {seasons} ({type(exc).__name__}); "
            "treating the season as not yet started.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "season_type": pl.Utf8, "team": pl.Utf8, "position": pl.Utf8,
            }
        )


def player_master() -> pl.DataFrame:
    """nflverse player master: ids, position, birth date, draft capital."""
    return _nfl().load_players()


def rosters(seasons: list[int]) -> pl.DataFrame:
    """Season-level rosters -- used for team assignment and depth context."""
    return _nfl().load_rosters(seasons=seasons)


def schedules(seasons: list[int]) -> pl.DataFrame:
    """Game schedule with Vegas lines, roof/surface, and rest days."""
    return _nfl().load_schedules(seasons=seasons)


def snap_counts(seasons: list[int]) -> pl.DataFrame:
    """Snap share -- the cleanest available proxy for role."""
    return _nfl().load_snap_counts(seasons=seasons)


def depth_charts(seasons: list[int]) -> pl.DataFrame:
    """Weekly depth charts."""
    return _nfl().load_depth_charts(seasons=seasons)


def injuries(seasons: list[int]) -> pl.DataFrame:
    """Official injury reports -- practice and game status by week."""
    return _nfl().load_injuries(seasons=seasons)


def ff_opportunity(seasons: list[int]) -> pl.DataFrame:
    """Expected fantasy points from ffopportunity.

    Models expected points from opportunity alone (air yards, carries, usage),
    which is far stickier season-to-season than realized points. The gap between
    actual and expected is the single most useful regression signal available
    for free.
    """
    return _nfl().load_ff_opportunity(seasons=seasons)


def ff_playerids() -> pl.DataFrame:
    """DynastyProcess cross-platform id map (sleeper/mfl/gsis/espn/...)."""
    return _nfl().load_ff_playerids()


def ff_rankings(kind: str = "draft") -> pl.DataFrame:
    """FantasyPros expert consensus rankings (ECR) via ffverse."""
    return _nfl().load_ff_rankings(type=kind)


def nextgen_stats(seasons: list[int], stat_type: str = "receiving") -> pl.DataFrame:
    """NFL Next Gen Stats: separation, cushion, time to throw, rush over expected."""
    return _nfl().load_nextgen_stats(seasons=seasons, stat_type=stat_type)


def pfr_advstats(seasons: list[int], stat_type: str = "rec") -> pl.DataFrame:
    """Pro-Football-Reference advanced stats: broken tackles, drops, YAC."""
    return _nfl().load_pfr_advstats(seasons=seasons, stat_type=stat_type)


def draft_picks() -> pl.DataFrame:
    """NFL draft history -- draft capital is the best prior for rookies."""
    return _nfl().load_draft_picks()


def team_stats(seasons: list[int]) -> pl.DataFrame:
    """Team-level weekly stats, used for defense scoring and team context."""
    return _nfl().load_team_stats(seasons=seasons, summary_level="week")


#: Everything `ff data sync` pulls, with the reason each one is worth the bytes.
SYNC_TARGETS: dict[str, str] = {
    "weekly_stats": "per-week box scores; the projection target",
    "player_master": "ids, position, birth date, draft capital",
    "rosters": "team assignment",
    "schedules": "opponent, Vegas lines, bye weeks",
    "snap_counts": "role/usage share",
    "depth_charts": "starter vs backup",
    "injuries": "availability history",
    "ff_opportunity": "expected fantasy points from usage",
    "ff_playerids": "sleeper <-> gsis id crosswalk",
    "draft_picks": "rookie draft capital prior",
}
