"""Three-way benchmark: this model vs expert consensus vs a naive baseline.

Beating a naive baseline proves the machinery works. It does not prove the model
is worth using, because the real alternative is a free expert ranking. This
module runs that comparison honestly, on identical player sets, using rankings
frozen before each season started.

Only rank metrics are reported, because expert consensus is a rank -- it has no
points, so MAE and RMSE are undefined for it. Rank accuracy is the metric that
matters for drafting anyway.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..data import expert as expert_mod
from .evaluate import _spearman, _top_n_hit_rate, naive_baseline
from .projections import ProjectionConfig, project_season

POSITIONS = ("QB", "RB", "WR", "TE")
DEFAULT_BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _score(preds: pl.DataFrame, actual: pl.DataFrame, label: str, season: int) -> list[dict]:
    joined = preds.join(actual, on="gsis_id", how="inner")
    rows = []
    for position in POSITIONS:
        subset = joined.filter(pl.col("position") == position)
        if subset.height < 10:
            continue
        rows.append(
            {
                "season": season,
                "position": position,
                "model": label,
                "n": subset.height,
                "spearman": _spearman(subset, "score", "actual"),
                "top12": _top_n_hit_rate(subset, "score", "actual"),
            }
        )
    return rows


def benchmark_season(
    totals: pl.DataFrame,
    season: int,
    config: ProjectionConfig | None = None,
    local_dir: str | Path | None = None,
    page: str = expert_mod.PPR_DRAFT_PAGE,
    min_games: int = 6,
    blend_weights: tuple[float, ...] = DEFAULT_BLEND_WEIGHTS,
) -> list[dict]:
    """Compare all three approaches (plus blends) for one season."""
    config = config or ProjectionConfig()

    ecr = expert_mod.preseason_ecr(season, page=page, local_dir=local_dir)
    if ecr.is_empty():
        return []

    actual = totals.filter(
        (pl.col("season") == season) & (pl.col("games") >= min_games)
    ).select(["gsis_id", "position", pl.col("points").alias("actual")])

    history = totals.filter(pl.col("season") < season)
    universe = totals.filter(pl.col("season") == season).select(
        [c for c in ("gsis_id", "position", "team", "age", "experience", "draft_round")
         if c in totals.columns]
    )

    model = project_season(history, universe, season, config).select(
        ["gsis_id", "position", "proj_points"]
    )
    naive = naive_baseline(history, season)

    # Score everyone on the same players, or the comparison is meaningless.
    common = (
        set(model["gsis_id"]) & set(naive["gsis_id"])
        & set(ecr["gsis_id"]) & set(actual["gsis_id"])
    )
    if len(common) < 50:
        return []
    keep = pl.Series("gsis_id", sorted(common))

    model = model.filter(pl.col("gsis_id").is_in(keep))
    naive = naive.filter(pl.col("gsis_id").is_in(keep))
    actual = actual.filter(pl.col("gsis_id").is_in(keep))

    rows: list[dict] = []
    rows += _score(model.with_columns(pl.col("proj_points").alias("score"))
                   .select(["gsis_id", "score"]), actual, "model", season)
    rows += _score(naive.with_columns(pl.col("proj_points").alias("score"))
                   .select(["gsis_id", "score"]), actual, "naive", season)
    rows += _score(
        ecr.filter(pl.col("gsis_id").is_in(keep))
        .with_columns((-pl.col("ecr")).alias("score"))
        .select(["gsis_id", "score"]),
        actual, "expert_ecr", season,
    )

    # Blends, scored on rank so the two sources are commensurable.
    paired = model.join(ecr.select(["gsis_id", "ecr"]), on="gsis_id", how="inner")
    paired = paired.with_columns(
        pl.col("proj_points").rank("average", descending=True).over("position").alias("rm"),
        pl.col("ecr").rank("average").over("position").alias("re"),
    )
    for weight in blend_weights:
        if weight in (0.0, 1.0):
            continue  # already covered by the pure model / pure expert rows
        blended = paired.with_columns(
            (-((1 - weight) * pl.col("rm") + weight * pl.col("re"))).alias("score")
        ).select(["gsis_id", "score"])
        rows += _score(blended, actual, f"blend_{weight:g}", season)

    return rows


def benchmark(
    totals: pl.DataFrame,
    seasons: list[int],
    config: ProjectionConfig | None = None,
    local_dir: str | Path | None = None,
    page: str = expert_mod.PPR_DRAFT_PAGE,
) -> pl.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        rows.extend(benchmark_season(totals, season, config, local_dir, page))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def summarize(results: pl.DataFrame, by_position: bool = True) -> pl.DataFrame:
    if results.is_empty():
        return results
    keys = ["position", "model"] if by_position else ["model"]
    return (
        results.group_by(keys)
        .agg(
            pl.col("spearman").mean().round(3),
            pl.col("top12").mean().round(3),
            pl.col("n").sum(),
        )
        .sort([*keys[:-1], "spearman"], descending=[False] * (len(keys) - 1) + [True])
    )
