"""Season projection model.

Design notes
------------
The model is deliberately a *shrunk, opportunity-aware, age-adjusted rate
model* rather than a black box:

  1. Score every historical week under the league's own rules, so the target
     variable is the points this league actually pays out.
  2. Estimate a per-game rate from weighted recent seasons, shrunk toward a
     positional prior. Shrinkage is expressed in games, which makes it an
     honest empirical-Bayes weight: a player with 4 career games is mostly
     prior, a player with 50 is mostly himself.
  3. Blend realized points with *expected* points from usage (ffopportunity).
     Touchdown rates regress hard; opportunity does not.
  4. Apply an age curve fitted from year-over-year deltas of the same player,
     which controls for player quality in a way cross-sectional age means don't.
  5. Project games played separately from points per game, because durability
     and productivity are different questions.
  6. Calibrate uncertainty on a holdout season instead of assuming a spread.

Every one of those steps is measurable by `ff model backtest`, which compares
this against the naive "last season's points per game" baseline. If a change
doesn't beat the baseline by more than noise, it isn't an improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")

# Shrinkage constant per position, in games. Larger = trust the prior longer.
# Quarterback rates stabilize slowest in points terms because of TD variance.
DEFAULT_SHRINKAGE_GAMES: dict[str, float] = {"QB": 8.0, "RB": 7.0, "WR": 8.0, "TE": 9.0}

# Age-curve peaks and curvature by position. Validate with `ff model age-curve`,
# which prints the empirical year-over-year deltas these are meant to track.
FALLBACK_PEAK_AGE: dict[str, float] = {"QB": 30.0, "RB": 25.0, "WR": 26.5, "TE": 27.0}
FALLBACK_CURVATURE: dict[str, float] = {"QB": 0.004, "RB": 0.009, "WR": 0.006, "TE": 0.006}

# Bounds on any one-year age multiplier, to stop a thin cell producing nonsense.
AGE_MULT_BOUNDS = (0.80, 1.20)


@dataclass
class ProjectionConfig:
    """Knobs for the projection. Defaults are the ones that backtest best."""

    lookback: int = 4
    # Weight applied per season of recency: season T-1 gets 1.0, T-2 gets decay, ...
    decay: float = 0.55
    shrinkage_games: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SHRINKAGE_GAMES)
    )
    # How much to lean on expected points vs realized points (0 = ignore expected).
    expected_weight: float = 0.35
    # Expected games for a fully healthy season, and shrinkage for availability.
    full_season_games: float = 16.0
    games_shrinkage: float = 1.0
    min_games_for_prior: int = 4
    # Quantiles reported as floor/ceiling.
    floor_q: float = 0.20
    ceiling_q: float = 0.80


def positional_priors(
    totals: pl.DataFrame, seasons: list[int], min_games: int = 4
) -> dict[str, float]:
    """Mean points per game by position among players with a real sample.

    This is the "average startable player at this position" anchor that thin
    samples get pulled toward.
    """
    subset = totals.filter(
        pl.col("season").is_in(seasons) & (pl.col("games") >= min_games)
    )
    if subset.height == 0:
        return dict.fromkeys(FANTASY_POSITIONS, 8.0)
    rows = subset.group_by("position").agg(pl.col("ppg").mean().alias("prior"))
    priors = {r["position"]: float(r["prior"]) for r in rows.to_dicts()}
    return {pos: priors.get(pos, 8.0) for pos in FANTASY_POSITIONS}


def fit_age_curve(totals: pl.DataFrame, min_games: int = 6) -> pl.DataFrame:
    """Fit a one-year-ahead age multiplier from within-player deltas.

    For every player with two consecutive qualifying seasons we take the change
    in points per game, and summarize it by (position, age). Using the same
    player on both sides removes the biggest confound -- that good players and
    young players are not the same population.

    Caveat worth remembering: players who fall off get cut rather than playing a
    bad season, so surviving-pair deltas understate real decline at the tail.
    """
    qualifying = totals.filter(
        (pl.col("games") >= min_games) & pl.col("age").is_not_null()
    ).sort(["gsis_id", "season"])

    pairs = qualifying.join(
        qualifying.select(
            pl.col("gsis_id"),
            (pl.col("season") - 1).alias("season"),
            pl.col("ppg").alias("next_ppg"),
            pl.col("age").alias("next_age"),
        ),
        on=["gsis_id", "season"],
        how="inner",
    )
    if pairs.height < 50:
        return pl.DataFrame(schema={"position": pl.Utf8, "age_bucket": pl.Int64,
                                    "delta": pl.Float64, "n": pl.UInt32})

    return (
        pairs.with_columns(
            (pl.col("next_ppg") - pl.col("ppg")).alias("delta_ppg"),
            pl.col("next_age").round(0).cast(pl.Int64).alias("age_bucket"),
        )
        .group_by(["position", "age_bucket"])
        .agg(
            pl.col("delta_ppg").median().alias("delta"),
            pl.len().alias("n"),
        )
        .sort(["position", "age_bucket"])
    )


def age_level(position: str, age: float | None) -> float:
    """Ability level at a given age, relative to that position's peak (peak = 1.0).

    A single quadratic around a position-specific peak. It is deliberately a
    *level* curve, not a year-over-year delta: the two are easy to conflate and
    conflating them is wrong. A 22-year-old is below his own peak but rising, and
    a delta curve applied as a level would penalize him for being young.
    """
    if age is None:
        return 1.0
    peak = FALLBACK_PEAK_AGE.get(position, 27.0)
    curvature = FALLBACK_CURVATURE.get(position, 0.01)
    # Ageing past peak costs more than being equally far short of it.
    distance = age - peak
    if distance < 0:
        distance *= 0.6
    return max(0.5, min(1.0, 1.0 - curvature * distance**2))


def age_multiplier(
    position: str,
    age: float | None,
    history_age: float | None = None,
) -> float:
    """Adjust a player's historical rate for where he is on the age curve now.

    His history was produced at `history_age`; we are projecting at `age`. The
    adjustment is the ratio of curve levels between those two points, so it only
    ever prices the *movement* along the curve rather than re-charging a player
    for his age every season.

    Rookies (no history) return 1.0 -- the rookie draft-capital prior already
    encodes their level, so an age adjustment on top would double-count.
    """
    if age is None or history_age is None:
        return 1.0
    now = age_level(position, age)
    before = age_level(position, history_age)
    if before <= 0:
        return 1.0
    return max(AGE_MULT_BOUNDS[0], min(AGE_MULT_BOUNDS[1], now / before))


def rookie_priors(totals: pl.DataFrame) -> pl.DataFrame:
    """Expected rookie points per game by position and draft round.

    Draft capital is the only strong public signal for a player with no NFL
    snaps: teams give early picks the touches that fantasy points come from.
    """
    rookies = totals.filter(
        (pl.col("experience") == 0) & pl.col("draft_round").is_not_null()
    )
    if rookies.height < 30:
        return pl.DataFrame(schema={"position": pl.Utf8, "draft_round": pl.Int64,
                                    "rookie_ppg": pl.Float64, "n": pl.UInt32})
    return (
        rookies.group_by(["position", "draft_round"])
        .agg(pl.col("ppg").median().alias("rookie_ppg"), pl.len().alias("n"))
        .sort(["position", "draft_round"])
    )


def _weighted_history(
    totals: pl.DataFrame, target_season: int, config: ProjectionConfig
) -> pl.DataFrame:
    """Collapse each player's prior seasons into weighted points and games."""
    seasons = list(range(target_season - config.lookback, target_season))
    hist = totals.filter(pl.col("season").is_in(seasons))
    if hist.height == 0:
        return pl.DataFrame(
            schema={"gsis_id": pl.Utf8, "w_points": pl.Float64, "w_games": pl.Float64,
                    "w_sum": pl.Float64, "last_season": pl.Int64, "recent_ppg_sd": pl.Float64,
                    "history_age": pl.Float64, "mean_games": pl.Float64}
        )

    hist = hist.with_columns(
        (pl.lit(config.decay) ** (target_season - 1 - pl.col("season"))).alias("w")
    )

    # Blend realized points with expected-from-usage points where we have them.
    if "exp_points" in hist.columns:
        lam = config.expected_weight
        hist = hist.with_columns(
            pl.when(pl.col("exp_points").is_not_null())
            .then((1 - lam) * pl.col("points") + lam * pl.col("exp_points"))
            .otherwise(pl.col("points"))
            .alias("blend_points")
        )
    else:
        hist = hist.with_columns(pl.col("points").alias("blend_points"))

    return hist.group_by("gsis_id").agg(
        (pl.col("w") * pl.col("blend_points")).sum().alias("w_points"),
        (pl.col("w") * pl.col("games")).sum().alias("w_games"),
        pl.col("w").sum().alias("w_sum"),
        pl.col("season").max().alias("last_season"),
        pl.col("ppg_sd").mean().alias("recent_ppg_sd"),
        (
            (pl.col("w") * pl.col("age")).sum() / pl.col("w").sum()
            if "age" in hist.columns
            else pl.lit(None, dtype=pl.Float64)
        ).alias("history_age"),
        (pl.col("w") * pl.col("games")).sum().truediv(pl.col("w").sum()).alias("mean_games"),
    )


def project_season(
    totals: pl.DataFrame,
    universe: pl.DataFrame,
    target_season: int,
    config: ProjectionConfig | None = None,
) -> pl.DataFrame:
    """Project points for `target_season` for everyone in `universe`.

    `universe` must carry gsis_id, position, and (ideally) team, age, draft_round
    and experience -- typically built from the target season's rosters so that
    rookies and players returning from a missed year are included.
    """
    config = config or ProjectionConfig()
    prior_seasons = list(range(target_season - config.lookback, target_season))

    priors = positional_priors(totals, prior_seasons, config.min_games_for_prior)
    rookie_table = rookie_priors(totals.filter(pl.col("season") < target_season))
    history = _weighted_history(totals, target_season, config)

    df = (
        universe.filter(pl.col("position").is_in(FANTASY_POSITIONS))
        .join(history, on="gsis_id", how="left")
        .with_columns(
            [
                pl.col("w_points").fill_null(0.0),
                pl.col("w_games").fill_null(0.0),
                pl.col("w_sum").fill_null(0.0),
            ]
        )
    )

    # Positional prior and shrinkage weight, per row.
    df = df.with_columns(
        [
            pl.col("position")
            .replace_strict(priors, default=8.0)
            .cast(pl.Float64)
            .alias("prior_ppg"),
            pl.col("position")
            .replace_strict(config.shrinkage_games, default=8.0)
            .cast(pl.Float64)
            .alias("k_games"),
        ]
    )

    # Rookies (no NFL history) fall back to a draft-capital prior.
    if rookie_table.height and "draft_round" in df.columns:
        df = df.join(rookie_table.drop("n"), on=["position", "draft_round"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("rookie_ppg"))

    df = df.with_columns(
        pl.when((pl.col("w_games") < 1) & pl.col("rookie_ppg").is_not_null())
        .then(pl.col("rookie_ppg"))
        .otherwise(pl.col("prior_ppg"))
        .alias("anchor_ppg")
    )

    # Empirical-Bayes rate: observed points and prior, weighted in games.
    df = df.with_columns(
        (
            (pl.col("w_points") + pl.col("k_games") * pl.col("anchor_ppg"))
            / (pl.col("w_games") + pl.col("k_games"))
        ).alias("raw_ppg")
    )

    # Age adjustment: move the player's own history along the age curve.
    if "age" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("age"))
    if "history_age" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("history_age"))
    mults = [
        age_multiplier(row["position"], row["age"], row["history_age"])
        for row in df.select(["position", "age", "history_age"]).to_dicts()
    ]
    df = df.with_columns(pl.Series("age_mult", mults, dtype=pl.Float64))

    # Availability: weighted games per season, shrunk toward a full season.
    # `games_shrinkage` is measured in season-equivalents of prior, so a player
    # with one weighted season of history sits halfway between his own
    # durability and a healthy baseline.
    df = df.with_columns(
        (
            (pl.col("w_games") + config.games_shrinkage * config.full_season_games)
            / (pl.col("w_sum") + config.games_shrinkage)
        )
        .clip(1.0, 17.0)
        .fill_null(config.full_season_games)
        .alias("proj_games")
    )

    df = df.with_columns(
        (pl.col("raw_ppg") * pl.col("age_mult")).alias("proj_ppg"),
    ).with_columns(
        (pl.col("proj_ppg") * pl.col("proj_games")).alias("proj_points"),
    )

    keep = [
        c
        for c in (
            "gsis_id", "sleeper_id", "name", "position", "team", "age", "experience",
            "draft_round", "proj_ppg", "proj_games", "proj_points", "raw_ppg",
            "age_mult", "prior_ppg", "w_games",
        )
        if c in df.columns
    ]
    return df.select(keep).sort("proj_points", descending=True)


def calibrate_uncertainty(
    totals: pl.DataFrame,
    target_season: int,
    config: ProjectionConfig | None = None,
    n_tiers: int = 5,
) -> pl.DataFrame:
    """Measure real forecast error on a holdout season, by position and tier.

    Running the model as-of `target_season - 1` and scoring it against what
    actually happened gives an honest spread instead of an assumed one.
    """
    config = config or ProjectionConfig()
    holdout = target_season - 1

    actual = totals.filter(pl.col("season") == holdout)
    if actual.height < 50:
        return pl.DataFrame(
            schema={"position": pl.Utf8, "tier": pl.Int64, "resid_sd": pl.Float64,
                    "n": pl.UInt32}
        )

    universe = actual.select(
        [c for c in ("gsis_id", "position", "team", "age", "experience", "draft_round")
         if c in actual.columns]
    )
    preds = project_season(totals, universe, holdout, config)

    joined = preds.join(
        actual.select(["gsis_id", pl.col("points").alias("actual_points")]),
        on="gsis_id",
        how="inner",
    )
    if joined.height < 30:
        return pl.DataFrame(
            schema={"position": pl.Utf8, "tier": pl.Int64, "resid_sd": pl.Float64,
                    "n": pl.UInt32}
        )

    return (
        joined.with_columns(
            (pl.col("actual_points") - pl.col("proj_points")).alias("resid"),
            (
                pl.col("proj_points").rank("ordinal", descending=True).over("position")
                * n_tiers
                / pl.len().over("position")
            )
            .ceil()
            .clip(1, n_tiers)
            .cast(pl.Int64)
            .alias("tier"),
        )
        .group_by(["position", "tier"])
        .agg(pl.col("resid").std().alias("resid_sd"), pl.len().alias("n"))
        .sort(["position", "tier"])
    )


def attach_uncertainty(
    projections: pl.DataFrame,
    calibration: pl.DataFrame,
    config: ProjectionConfig | None = None,
    n_tiers: int = 5,
) -> pl.DataFrame:
    """Add sd/floor/ceiling to projections using calibrated residual spread."""
    config = config or ProjectionConfig()
    # 80/20 quantiles of a normal are +/- 0.8416 sd.
    z = 0.8416

    df = projections.with_columns(
        (
            pl.col("proj_points").rank("ordinal", descending=True).over("position")
            * n_tiers
            / pl.len().over("position")
        )
        .ceil()
        .clip(1, n_tiers)
        .cast(pl.Int64)
        .alias("tier")
    )

    if calibration.height:
        df = df.join(calibration.drop("n"), on=["position", "tier"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("resid_sd"))

    # Fall back to a coefficient-of-variation style spread where uncalibrated.
    return (
        df.with_columns(
            pl.col("resid_sd").fill_null(pl.col("proj_points") * 0.45).alias("proj_sd")
        )
        .with_columns(
            (pl.col("proj_points") - z * pl.col("proj_sd")).clip(0.0).alias("floor"),
            (pl.col("proj_points") + z * pl.col("proj_sd")).alias("ceiling"),
        )
        .drop("resid_sd")
    )
