"""End-to-end build: raw sources -> a draft-ready board.

This is the one function the CLI and any notebook should call. It is written to
degrade rather than fail: if the market feeds are unreachable you still get
projections and VORP, just without ADP-based survival probabilities.
"""

from __future__ import annotations

import warnings

import polars as pl

from .config import LeagueConfig, get_settings
from .data import expert as expert_mod
from .data import market as market_mod
from .data import nflverse
from .data.cache import TTL_HOUR, cached
from .data.sleeper import SleeperClient
from .draft.value import add_value
from .ids import build_crosswalk, join_key, sleeper_players_to_frame
from .model.blend import DEFAULT_EXPERT_WEIGHT, blend_rankings
from .model.features import attach_age, attach_expected_points, season_totals
from .model.projections import (
    ProjectionConfig,
    attach_uncertainty,
    calibrate_uncertainty,
    project_season,
)
from .scoring import ScoringEngine


def build_totals(
    league: LeagueConfig,
    season: int,
    lookback: int = 4,
    use_opportunity: bool = True,
) -> pl.DataFrame:
    """Historical player-seasons scored under this league's rules."""
    seasons = nflverse.seasons_back(season, lookback)
    weekly = nflverse.weekly_stats(seasons)
    engine = ScoringEngine(league)
    totals = season_totals(weekly, engine)
    totals = attach_age(totals, nflverse.player_master())

    if use_opportunity:
        try:
            totals = attach_expected_points(totals, nflverse.ff_opportunity(seasons))
        except Exception:  # noqa: BLE001 - optional signal; never block a draft on it
            totals = attach_expected_points(totals, None)
    else:
        totals = attach_expected_points(totals, None)

    return totals


def build_universe(season: int) -> pl.DataFrame:
    """Everyone who could plausibly score this season, including rookies."""
    rosters = nflverse.rosters([season])
    players = nflverse.player_master()

    keep = [c for c in ("gsis_id", "position", "team", "full_name") if c in rosters.columns]
    universe = rosters.select(keep).filter(pl.col("gsis_id").is_not_null())
    universe = universe.unique(subset=["gsis_id"], keep="first")

    master_cols = [
        c
        for c in ("gsis_id", "display_name", "birth_date", "draft_round", "draft_pick",
                  "rookie_season")
        if c in players.columns
    ]
    universe = universe.join(players.select(master_cols).unique(subset=["gsis_id"]),
                             on="gsis_id", how="left")

    if "birth_date" in universe.columns:
        universe = universe.with_columns(
            pl.col("birth_date").cast(pl.Date, strict=False)
        ).with_columns(
            (
                (pl.date(season, 9, 1).cast(pl.Date) - pl.col("birth_date")).dt.total_days()
                / 365.25
            ).alias("age")
        )
    else:
        universe = universe.with_columns(pl.lit(None, dtype=pl.Float64).alias("age"))

    if "rookie_season" in universe.columns:
        universe = universe.with_columns(
            (season - pl.col("rookie_season")).alias("experience")
        )
    else:
        universe = universe.with_columns(pl.lit(None, dtype=pl.Int64).alias("experience"))

    name_col = "display_name" if "display_name" in universe.columns else "full_name"
    return universe.with_columns(pl.col(name_col).alias("name"))


def build_crosswalk_table(force: bool = False) -> pl.DataFrame:
    """Sleeper players joined to gsis ids, cached for the day."""

    def _fetch() -> pl.DataFrame:
        with SleeperClient() as client:
            sleeper_df = sleeper_players_to_frame(client.players())
        try:
            ff_ids = nflverse.ff_playerids()
        except Exception:  # noqa: BLE001
            ff_ids = None
        try:
            nfl_players = nflverse.player_master()
        except Exception:  # noqa: BLE001
            nfl_players = None
        return build_crosswalk(sleeper_df, ff_ids, nfl_players)

    return cached("crosswalk", _fetch, ttl=86_400, force=force)


def build_board(
    league: LeagueConfig,
    season: int | None = None,
    lookback: int = 4,
    config: ProjectionConfig | None = None,
    with_market: bool = True,
    expert_weight: float = DEFAULT_EXPERT_WEIGHT,
    force: bool = False,
) -> pl.DataFrame:
    """Projections + uncertainty + VORP + market price, keyed by Sleeper id.

    `expert_weight` controls how much the board's ordering defers to FantasyPros
    expert consensus, which benchmarks better than the model alone. Set 0.0 for
    a pure-model board.
    """
    season = season or league.season
    config = config or ProjectionConfig(lookback=lookback)

    totals = build_totals(league, season, lookback)
    universe = build_universe(season)

    projections = project_season(totals, universe, season, config)
    calibration = calibrate_uncertainty(totals, season, config)
    projections = attach_uncertainty(projections, calibration, config)

    # Expert consensus orders players better than the model does (see
    # `ff model benchmark`), so defer to it for ordering while keeping the
    # model's point magnitudes, which VORP and opportunity cost need.
    if expert_weight > 0:
        try:
            ecr = expert_mod.current_ecr(
                page=expert_mod.draft_page_for(league.ppr, league.superflex)
            )
            projections = blend_rankings(projections, ecr, weight=expert_weight)
        except Exception as exc:  # noqa: BLE001 - never block a board build on a feed
            # Warn loudly rather than silently shipping a pure-model board: the
            # expert blend is the single biggest accuracy input, and a board that
            # quietly lost it looks identical to one that never asked for it.
            warnings.warn(
                f"Expert rankings unavailable ({type(exc).__name__}: {exc}); "
                "board is model-only and will rank worse. "
                "Check network access or set FF_DP_LOCAL_DIR.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Attach Sleeper ids so the board can talk to the draft feed.
    try:
        crosswalk = build_crosswalk_table(force=force)
    except Exception:  # noqa: BLE001 - offline: keep gsis-only board
        crosswalk = pl.DataFrame()

    if not crosswalk.is_empty():
        cw = (
            crosswalk.filter(pl.col("gsis_id").is_not_null())
            .select(["gsis_id", "sleeper_id", "join_key", "injury_status", "search_rank"])
            .unique(subset=["gsis_id"], keep="first")
        )
        projections = projections.join(cw, on="gsis_id", how="left")

    if "join_key" not in projections.columns:
        projections = projections.with_columns(
            pl.struct(["name", "position"])
            .map_elements(lambda s: join_key(s["name"], s["position"]), return_dtype=pl.Utf8)
            .alias("join_key")
        )

    adp = None
    market = None
    if with_market:
        fmt = market_mod.ffc_format_for(league.ppr, league.superflex)
        try:
            adp = cached(
                f"adp_{fmt}_{league.teams}_{season}",
                lambda: market_mod.fetch_adp(season, fmt, league.teams),
                ttl=TTL_HOUR,
                force=force,
            )
        except Exception:  # noqa: BLE001
            adp = None
        try:
            market = cached(
                f"market_{league.teams}_{league.ppr}",
                lambda: market_mod.fetch_trade_values(
                    is_dynasty=False,
                    num_qbs=2 if league.superflex else 1,
                    num_teams=league.teams,
                    ppr=league.ppr,
                ),
                ttl=TTL_HOUR,
                force=force,
            )
        except Exception:  # noqa: BLE001
            market = None

    return add_value(projections, league, adp=adp, market=market)


def save_board(board: pl.DataFrame, name: str = "board") -> str:
    path = get_settings().artifacts_dir / f"{name}.parquet"
    board.write_parquet(path)
    return str(path)


def load_board(name: str = "board") -> pl.DataFrame:
    path = get_settings().artifacts_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No saved board at {path}. Run `ff board build` first.")
    return pl.read_parquet(path)
