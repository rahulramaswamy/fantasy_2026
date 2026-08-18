"""Player identity crosswalk.

Sleeper speaks `player_id` (its own numeric ids), nflverse speaks `gsis_id`, and
the market feeds speak names. Every join in this project runs through here.

Resolution order, best evidence first:
  1. Sleeper's own player dump, which carries a `gsis_id` for most NFL players.
  2. DynastyProcess's id map (nflreadpy.load_ff_playerids), which is maintained
     specifically as a cross-platform crosswalk.
  3. Normalized name + position matching, as a last resort.

Coverage is reported rather than assumed -- an unmatched star is a silent,
expensive bug on draft day, so `crosswalk_report()` exists to surface them.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import polars as pl

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Names where the common fantasy spelling differs from the NFL's official one.
NAME_ALIASES: dict[str, str] = {
    "hollywood brown": "marquise brown",
    "gabe davis": "gabriel davis",
    "josh palmer": "joshua palmer",
    "cam ward": "cameron ward",
    "chig okonkwo": "chigoziem okonkwo",
    "tank dell": "nathaniel dell",
    "scotty miller": "scott miller",
    "mike thomas": "michael thomas",
    "dj moore": "d j moore",
    "aj brown": "a j brown",
    "cd lamb": "ceedee lamb",
}


def normalize_name(name: str | None) -> str:
    """Lowercase, strip accents/punctuation/suffixes so names join reliably."""
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("'", "").replace(".", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    parts = [p for p in text.split() if p and p not in SUFFIXES]
    cleaned = " ".join(parts)
    return NAME_ALIASES.get(cleaned, cleaned)


def join_key(name: str | None, position: str | None) -> str:
    """A name+position key. Position disambiguates the handful of shared names."""
    return f"{normalize_name(name)}|{(position or '').upper()}"


def sleeper_players_to_frame(players: dict[str, Any]) -> pl.DataFrame:
    """Flatten Sleeper's /players/nfl dict-of-dicts into a tidy frame."""
    rows = []
    for pid, p in (players or {}).items():
        if not isinstance(p, dict):
            continue
        position = p.get("position")
        # Keep fantasy-relevant positions only; the dump includes every NFL body.
        if position not in ("QB", "RB", "WR", "TE", "K", "DEF"):
            continue
        full_name = p.get("full_name") or (
            f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        )
        rows.append(
            {
                "sleeper_id": str(pid),
                "sleeper_name": full_name,
                "position": position,
                "team": p.get("team"),
                "gsis_id": p.get("gsis_id"),
                "espn_id": str(p.get("espn_id")) if p.get("espn_id") else None,
                "yahoo_id": str(p.get("yahoo_id")) if p.get("yahoo_id") else None,
                "age": p.get("age"),
                "years_exp": p.get("years_exp"),
                "status": p.get("status"),
                "injury_status": p.get("injury_status"),
                "depth_chart_order": p.get("depth_chart_order"),
                "search_rank": p.get("search_rank"),
                "number": p.get("number"),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "sleeper_id": pl.Utf8, "sleeper_name": pl.Utf8, "position": pl.Utf8,
                "team": pl.Utf8, "gsis_id": pl.Utf8, "espn_id": pl.Utf8,
                "yahoo_id": pl.Utf8, "age": pl.Float64, "years_exp": pl.Int64,
                "status": pl.Utf8, "injury_status": pl.Utf8,
                "depth_chart_order": pl.Int64, "search_rank": pl.Int64, "number": pl.Int64,
            }
        )
    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.with_columns(
        pl.struct(["sleeper_name", "position"])
        .map_elements(
            lambda s: join_key(s["sleeper_name"], s["position"]), return_dtype=pl.Utf8
        )
        .alias("join_key")
    )


def build_crosswalk(
    sleeper_df: pl.DataFrame,
    ff_ids: pl.DataFrame | None = None,
    nfl_players: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach a `gsis_id` to every Sleeper player we can identify.

    Returns the Sleeper frame plus `gsis_id` and `id_source` (how it matched).
    """
    df = sleeper_df.with_columns(
        pl.when(pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != ""))
        .then(pl.lit("sleeper"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("id_source")
    )

    # 2) DynastyProcess id map, keyed on sleeper_id.
    if ff_ids is not None and ff_ids.height and "sleeper_id" in ff_ids.columns:
        dp = (
            ff_ids.select(
                pl.col("sleeper_id").cast(pl.Utf8),
                pl.col("gsis_id").cast(pl.Utf8).alias("dp_gsis_id"),
            )
            .filter(pl.col("sleeper_id").is_not_null() & pl.col("dp_gsis_id").is_not_null())
            .unique(subset=["sleeper_id"])
        )
        df = df.join(dp, on="sleeper_id", how="left").with_columns(
            [
                pl.when(pl.col("gsis_id").is_null() | (pl.col("gsis_id") == ""))
                .then(pl.col("dp_gsis_id"))
                .otherwise(pl.col("gsis_id"))
                .alias("gsis_id"),
                pl.when(
                    (pl.col("id_source").is_null()) & pl.col("dp_gsis_id").is_not_null()
                )
                .then(pl.lit("dynastyprocess"))
                .otherwise(pl.col("id_source"))
                .alias("id_source"),
            ]
        ).drop("dp_gsis_id")

    # 3) Name + position fallback against the nflverse player master.
    if nfl_players is not None and nfl_players.height:
        name_col = "display_name" if "display_name" in nfl_players.columns else "full_name"
        master = (
            nfl_players.select(
                pl.col("gsis_id").cast(pl.Utf8).alias("nv_gsis_id"),
                pl.col(name_col).alias("nv_name"),
                pl.col("position").alias("nv_position"),
            )
            .filter(pl.col("nv_gsis_id").is_not_null())
            .with_columns(
                pl.struct(["nv_name", "nv_position"])
                .map_elements(
                    lambda s: join_key(s["nv_name"], s["nv_position"]), return_dtype=pl.Utf8
                )
                .alias("join_key")
            )
            .unique(subset=["join_key"], keep="first")
            .select(["join_key", "nv_gsis_id"])
        )
        df = df.join(master, on="join_key", how="left").with_columns(
            [
                pl.when(pl.col("gsis_id").is_null() | (pl.col("gsis_id") == ""))
                .then(pl.col("nv_gsis_id"))
                .otherwise(pl.col("gsis_id"))
                .alias("gsis_id"),
                pl.when((pl.col("id_source").is_null()) & pl.col("nv_gsis_id").is_not_null())
                .then(pl.lit("name_match"))
                .otherwise(pl.col("id_source"))
                .alias("id_source"),
            ]
        ).drop("nv_gsis_id")

    return df.with_columns(pl.col("id_source").fill_null("unmatched"))


def crosswalk_report(crosswalk: pl.DataFrame, top_n: int = 20) -> dict[str, Any]:
    """Summarize match quality and list the most notable unmatched players.

    Sleeper's `search_rank` is a decent proxy for player prominence, so an
    unmatched player with a low search_rank is the kind of miss that matters.
    """
    by_source = crosswalk.group_by("id_source").len().sort("len", descending=True)
    unmatched = crosswalk.filter(pl.col("id_source") == "unmatched")
    notable = (
        unmatched.filter(pl.col("search_rank").is_not_null())
        .sort("search_rank")
        .head(top_n)
        .select(["sleeper_name", "position", "team", "search_rank"])
    )
    total = crosswalk.height
    matched = total - unmatched.height
    return {
        "total": total,
        "matched": matched,
        "match_rate": (matched / total) if total else 0.0,
        "by_source": by_source,
        "notable_unmatched": notable,
    }
