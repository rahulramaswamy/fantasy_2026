"""Expert consensus rankings (FantasyPros ECR, via ffverse/DynastyProcess).

Why this module exists
----------------------
The projection model in `ff2026.model` is purely backward-looking: it sees a
player's own statistics, age and draft capital, and nothing else. It cannot see
that he changed teams, that his team drafted a replacement, that the coaching
staff turned over, or that he is holding out. In August, that information is
worth a lot.

Expert consensus captures exactly that, and benchmarking (`ff model benchmark`)
shows it beats the model's own ordering at every position. So ECR is not a
nice-to-have here -- it is the better ranking, and the model's job is to supply
what a ranking cannot: point magnitudes, replacement levels, and opportunity
cost.

Two feeds are used:
  * `db_fpecr_latest` -- current rankings, for building this season's board.
  * `db_fpecr`        -- the historical archive, which carries a `scrape_date`
                         so a backtest can ask "what did the experts think
                         *before* this season started?" without leaking hindsight.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

# The FantasyPros page whose consensus is a PPR redraft draft-day ranking.
PPR_DRAFT_PAGE = "ppr-cheatsheets"
SUPERFLEX_DRAFT_PAGE = "ppr-superflex-cheatsheets"
STANDARD_DRAFT_PAGE = "consensus-cheatsheets"


def draft_page_for(ppr: float, superflex: bool = False) -> str:
    """Pick the FantasyPros page matching a league's format."""
    if superflex:
        return SUPERFLEX_DRAFT_PAGE
    return PPR_DRAFT_PAGE if ppr >= 0.5 else STANDARD_DRAFT_PAGE


def default_local_dir() -> str | None:
    """Optional local checkout of dynastyprocess/data, via FF_DP_LOCAL_DIR.

    Read through Settings rather than os.environ directly, so a value set in
    `.env` works the same as one exported in the shell -- pydantic-settings
    loads the file, it does not populate the process environment.
    """
    from ..config import get_settings

    try:
        configured = get_settings().ff_dp_local_dir
    except Exception:  # noqa: BLE001 - settings must never break a data read
        configured = None
    return configured or os.environ.get("FF_DP_LOCAL_DIR") or None


def _local_or_remote(filename: str, local_dir: str | Path | None) -> pl.DataFrame:
    """Read a DynastyProcess file from a local checkout, else over the network."""
    local_dir = local_dir or default_local_dir()
    if local_dir:
        for suffix in (".parquet", ".csv"):
            path = Path(local_dir) / f"{filename}{suffix}"
            if path.exists():
                if suffix == ".parquet":
                    return pl.read_parquet(path)
                return pl.read_csv(path, infer_schema_length=None)

    import nflreadpy as nfl

    return nfl.load_ff_rankings(type="all" if filename == "db_fpecr" else "draft")


def load_player_ids(local_dir: str | Path | None = None) -> pl.DataFrame:
    """FantasyPros id -> gsis_id crosswalk."""
    local_dir = local_dir or default_local_dir()
    if local_dir:
        path = Path(local_dir) / "db_playerids.csv"
        if path.exists():
            ids = pl.read_csv(path, infer_schema_length=None)
        else:
            import nflreadpy as nfl

            ids = nfl.load_ff_playerids()
    else:
        import nflreadpy as nfl

        ids = nfl.load_ff_playerids()

    return (
        ids.select(
            pl.col("fantasypros_id").cast(pl.Utf8),
            pl.col("gsis_id").cast(pl.Utf8),
        )
        .filter(pl.col("fantasypros_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .unique(subset=["fantasypros_id"], keep="first")
    )


def _tidy(rankings: pl.DataFrame, crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Normalize a raw ECR slice to gsis_id + ecr (+ spread, when present)."""
    cols = [pl.col("id").cast(pl.Utf8).alias("fantasypros_id"), pl.col("ecr")]
    if "sd" in rankings.columns:
        cols.append(pl.col("sd").alias("ecr_sd"))
    if "player" in rankings.columns:
        cols.append(pl.col("player").alias("ecr_name"))

    return (
        rankings.select(cols)
        .join(crosswalk, on="fantasypros_id", how="inner")
        .filter(pl.col("ecr").is_not_null())
        .unique(subset=["gsis_id"], keep="first")
    )


def preseason_ecr(
    season: int,
    page: str = PPR_DRAFT_PAGE,
    local_dir: str | Path | None = None,
    month: str = "08",
) -> pl.DataFrame:
    """Expert rankings as they stood just before `season` kicked off.

    Takes the last scrape in August, which is after camp and roughly when most
    drafts happen. Using a later date would leak information a drafter could not
    have had.
    """
    archive = _local_or_remote("db_fpecr", local_dir)
    subset = archive.filter(
        pl.col("fp_page").str.contains(page)
        & pl.col("scrape_date").cast(pl.Utf8).str.starts_with(f"{season}-{month}")
    )
    if subset.is_empty():
        return pl.DataFrame(schema={"gsis_id": pl.Utf8, "ecr": pl.Float64})

    latest = sorted(subset["scrape_date"].cast(pl.Utf8).unique().to_list())[-1]
    subset = subset.filter(pl.col("scrape_date").cast(pl.Utf8) == latest)
    return _tidy(subset, load_player_ids(local_dir))


def current_ecr(
    page: str = PPR_DRAFT_PAGE, local_dir: str | Path | None = None
) -> pl.DataFrame:
    """Today's expert rankings, for building this season's board."""
    latest = _local_or_remote("db_fpecr_latest", local_dir)
    if "fp_page" in latest.columns:
        filtered = latest.filter(pl.col("fp_page").str.contains(page))
        if not filtered.is_empty():
            latest = filtered
    return _tidy(latest, load_player_ids(local_dir))
