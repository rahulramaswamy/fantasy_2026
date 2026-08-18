"""Blending model projections with expert consensus rankings.

The benchmark (`ff model benchmark`) is unambiguous: expert consensus orders
players better than this model does, at every position. But a ranking is only an
*order* -- it has no points in it, and the draft machinery needs points, because
value over replacement and opportunity cost are both differences in points.

So the two are combined by role rather than averaged naively:

  * **Order** comes from a weighted blend of expert rank and model rank.
  * **Magnitude** comes from the model, by reassigning its projected point
    distribution along the blended order.

Concretely: if the blend decides receiver X is now the 5th-best receiver, he
inherits the points the model had assigned to the 5th-best receiver. The shape
of the positional curve -- which is what drives replacement level and tier
breaks -- is preserved, while the ordering follows the better source.

Players the experts do not rank keep their model ordering, so coverage of deep
rookies and returning players is not lost.
"""

from __future__ import annotations

import polars as pl

# Benchmarked over 2022-2025. Expert-only edges out the blend overall, but 0.75
# is at or near optimal for RB and TE and within noise elsewhere, while keeping
# the model's coverage of unranked players. See docs/MODEL.md.
DEFAULT_EXPERT_WEIGHT = 0.75

VALUE_COLUMNS = ("proj_points", "proj_ppg", "floor", "ceiling")


def blend_rankings(
    projections: pl.DataFrame,
    ecr: pl.DataFrame,
    weight: float = DEFAULT_EXPERT_WEIGHT,
    value_columns: tuple[str, ...] = VALUE_COLUMNS,
) -> pl.DataFrame:
    """Reorder projections by a blend of model rank and expert rank.

    `weight` is the weight on expert consensus: 0.0 keeps the model's ordering
    untouched, 1.0 adopts the experts' ordering wherever they have an opinion.
    """
    if ecr.is_empty() or not 0.0 <= weight <= 1.0:
        return projections.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ecr"),
            pl.lit(False).alias("expert_ranked"),
        )

    df = projections.join(ecr.select(["gsis_id", "ecr"]), on="gsis_id", how="left")

    # Rank within position: 1 = best, for both sources.
    df = df.with_columns(
        pl.col("proj_points").rank("average", descending=True).over("position")
        .alias("_r_model"),
        pl.col("ecr").rank("average").over("position").alias("_r_expert"),
    )

    # Unranked players fall back entirely to the model's ordering.
    df = df.with_columns(
        pl.when(pl.col("_r_expert").is_null())
        .then(pl.col("_r_model"))
        .otherwise((1 - weight) * pl.col("_r_model") + weight * pl.col("_r_expert"))
        .alias("_r_blend"),
        pl.col("ecr").is_not_null().alias("expert_ranked"),
    )

    # Reassign each position's point distribution along the blended order, so the
    # positional value curve survives intact.
    out_frames = []
    for _position, group in df.group_by(["position"], maintain_order=True):
        ordered = group.sort("_r_blend")
        reassigned = {}
        for col in value_columns:
            if col in group.columns:
                reassigned[col] = group[col].sort(descending=True, nulls_last=True)
        if reassigned:
            ordered = ordered.with_columns(
                [pl.Series(name, values) for name, values in reassigned.items()]
            )
        out_frames.append(ordered)

    if not out_frames:
        return df.drop(["_r_model", "_r_expert", "_r_blend"])

    return (
        pl.concat(out_frames, how="vertical")
        .drop(["_r_model", "_r_expert", "_r_blend"])
        .sort("proj_points", descending=True)
    )
