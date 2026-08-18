"""Backtesting.

A projection is only worth having if it beats the obvious alternative. The
obvious alternative in fantasy football is "last season's points per game", so
that is the baseline every change has to clear.

Metrics reported:
  * MAE / RMSE  -- absolute accuracy in league points.
  * Spearman    -- rank accuracy, which is what actually drives draft decisions.
  * Hit rate    -- share of a position's true top-12 that the model ranked top-12.

Rank metrics matter more than absolute error here: you never need to know that a
receiver scores 214.3 points, you need to know he goes before the other guy.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .projections import ProjectionConfig, project_season


@dataclass
class BacktestResult:
    season: int
    position: str
    model: str
    n: int
    mae: float
    rmse: float
    spearman: float
    top12_hit_rate: float

    def as_row(self) -> dict[str, object]:
        return {
            "season": self.season,
            "position": self.position,
            "model": self.model,
            "n": self.n,
            "mae": round(self.mae, 2),
            "rmse": round(self.rmse, 2),
            "spearman": round(self.spearman, 3),
            "top12_hit": round(self.top12_hit_rate, 3),
        }


def _spearman(df: pl.DataFrame, pred: str, actual: str) -> float:
    if df.height < 3:
        return float("nan")
    ranked = df.select(
        pl.col(pred).rank("average").alias("rp"),
        pl.col(actual).rank("average").alias("ra"),
    )
    corr = ranked.select(pl.corr("rp", "ra")).item()
    return float(corr) if corr is not None else float("nan")


def _top_n_hit_rate(df: pl.DataFrame, pred: str, actual: str, n: int = 12) -> float:
    if df.height < n:
        return float("nan")
    pred_top = set(df.sort(pred, descending=True).head(n)["gsis_id"].to_list())
    actual_top = set(df.sort(actual, descending=True).head(n)["gsis_id"].to_list())
    return len(pred_top & actual_top) / n


def naive_baseline(totals: pl.DataFrame, target_season: int) -> pl.DataFrame:
    """Last season's points per game, carried forward over a 16-game season."""
    prev = totals.filter(pl.col("season") == target_season - 1)
    return prev.select(
        pl.col("gsis_id"),
        pl.col("position"),
        (pl.col("ppg") * 16.0).alias("proj_points"),
    )


def evaluate_season(
    totals: pl.DataFrame,
    target_season: int,
    config: ProjectionConfig | None = None,
    min_games: int = 6,
) -> list[BacktestResult]:
    """Fit on everything before `target_season`, score against what happened."""
    config = config or ProjectionConfig()

    actual = totals.filter(
        (pl.col("season") == target_season) & (pl.col("games") >= min_games)
    ).select(
        ["gsis_id", "position", pl.col("points").alias("actual_points")]
    )
    if actual.height < 30:
        return []

    history = totals.filter(pl.col("season") < target_season)
    universe = totals.filter(pl.col("season") == target_season).select(
        [
            c
            for c in ("gsis_id", "position", "team", "age", "experience", "draft_round")
            if c in totals.columns
        ]
    )

    model_preds = project_season(history, universe, target_season, config).select(
        ["gsis_id", "position", "proj_points"]
    )
    baseline_preds = naive_baseline(history, target_season)

    # Score both models on exactly the same players. The naive baseline only has
    # an opinion about players who played last season, so comparing it on its own
    # (easier, survivorship-filtered) subset would flatter it badly.
    common = set(model_preds["gsis_id"].to_list()) & set(baseline_preds["gsis_id"].to_list())
    common &= set(actual["gsis_id"].to_list())
    if len(common) < 30:
        return []
    keep = pl.Series("gsis_id", sorted(common))

    results: list[BacktestResult] = []
    for label, preds in (("model", model_preds), ("naive_last_ppg", baseline_preds)):
        joined = (
            preds.filter(pl.col("gsis_id").is_in(keep))
            .join(actual.drop("position"), on="gsis_id", how="inner")
        )
        for position in ("QB", "RB", "WR", "TE"):
            sub = joined.filter(pl.col("position") == position)
            if sub.height < 10:
                continue
            err = sub.select(
                (pl.col("proj_points") - pl.col("actual_points")).alias("e")
            )["e"]
            results.append(
                BacktestResult(
                    season=target_season,
                    position=position,
                    model=label,
                    n=sub.height,
                    mae=float(err.abs().mean()),
                    rmse=float((err**2).mean() ** 0.5),
                    spearman=_spearman(sub, "proj_points", "actual_points"),
                    top12_hit_rate=_top_n_hit_rate(sub, "proj_points", "actual_points"),
                )
            )
    return results


def coverage(
    totals: pl.DataFrame, target_season: int, config: ProjectionConfig | None = None,
    min_games: int = 6,
) -> dict[str, int]:
    """How many of the season's real contributors each model can even rank.

    The naive baseline is structurally blind to rookies and to anyone who missed
    the prior season, which is a real cost that accuracy-on-the-overlap hides.
    """
    config = config or ProjectionConfig()
    actual = totals.filter(
        (pl.col("season") == target_season) & (pl.col("games") >= min_games)
    )
    history = totals.filter(pl.col("season") < target_season)
    universe = totals.filter(pl.col("season") == target_season).select(
        [c for c in ("gsis_id", "position", "team", "age", "experience", "draft_round")
         if c in totals.columns]
    )
    model_ids = set(project_season(history, universe, target_season, config)["gsis_id"])
    naive_ids = set(naive_baseline(history, target_season)["gsis_id"])
    actual_ids = set(actual["gsis_id"])
    return {
        "season": target_season,
        "actual_contributors": len(actual_ids),
        "model_covers": len(model_ids & actual_ids),
        "naive_covers": len(naive_ids & actual_ids),
    }


def backtest(
    totals: pl.DataFrame,
    seasons: list[int],
    config: ProjectionConfig | None = None,
) -> pl.DataFrame:
    """Run `evaluate_season` across several seasons and stack the results."""
    rows: list[dict[str, object]] = []
    for season in seasons:
        rows.extend(r.as_row() for r in evaluate_season(totals, season, config))
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["season", "position", "model"])


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Average each model's metrics across seasons, by position."""
    if results.is_empty():
        return results
    return (
        results.group_by(["position", "model"])
        .agg(
            pl.col("mae").mean().round(2),
            pl.col("rmse").mean().round(2),
            pl.col("spearman").mean().round(3),
            pl.col("top12_hit").mean().round(3),
            pl.col("n").sum(),
        )
        .sort(["position", "model"])
    )
