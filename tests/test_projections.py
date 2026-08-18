import polars as pl

from ff2026.model.projections import (
    ProjectionConfig,
    age_level,
    age_multiplier,
    positional_priors,
    project_season,
)


def _totals():
    """Two seasons of history for three players of differing quality."""
    rows = []
    for season in (2024, 2025):
        rows += [
            {"gsis_id": "a", "season": season, "position": "RB", "points": 250.0,
             "games": 16, "ppg": 15.6, "ppg_sd": 6.0, "name": "Stud", "team": "SF",
             "age": 24.0 + (season - 2024), "experience": 2, "draft_round": 1},
            {"gsis_id": "b", "season": season, "position": "RB", "points": 60.0,
             "games": 16, "ppg": 3.8, "ppg_sd": 2.0, "name": "Scrub", "team": "NYJ",
             "age": 27.0 + (season - 2024), "experience": 5, "draft_round": 5},
            {"gsis_id": "c", "season": season, "position": "WR", "points": 200.0,
             "games": 16, "ppg": 12.5, "ppg_sd": 5.0, "name": "Wideout", "team": "LA",
             "age": 26.0 + (season - 2024), "experience": 4, "draft_round": 2},
        ]
    return pl.DataFrame(rows)


def _universe():
    return pl.DataFrame(
        [
            {"gsis_id": "a", "position": "RB", "team": "SF", "age": 26.0,
             "experience": 3, "draft_round": 1, "name": "Stud"},
            {"gsis_id": "b", "position": "RB", "team": "NYJ", "age": 29.0,
             "experience": 6, "draft_round": 5, "name": "Scrub"},
            {"gsis_id": "c", "position": "WR", "team": "LA", "age": 28.0,
             "experience": 5, "draft_round": 2, "name": "Wideout"},
            {"gsis_id": "rook", "position": "RB", "team": "DEN", "age": 22.0,
             "experience": 0, "draft_round": 1, "name": "Rookie"},
        ]
    )


def test_age_level_peaks_at_the_position_peak():
    assert age_level("RB", 25) > age_level("RB", 31)
    assert age_level("RB", 25) >= age_level("RB", 22)
    assert age_level("QB", 30) > age_level("QB", 22)


def test_age_multiplier_is_neutral_without_history():
    """Rookies must not be penalised for being young -- their prior already covers it."""
    assert age_multiplier("RB", 22.0, None) == 1.0


def test_age_multiplier_penalises_ageing_past_peak():
    assert age_multiplier("RB", 30.0, 29.0) < 1.0
    assert age_multiplier("RB", 24.0, 23.0) >= 1.0


def test_age_multiplier_is_bounded():
    assert 0.79 < age_multiplier("RB", 40.0, 25.0) <= 1.0


def test_projection_ranks_the_better_player_higher():
    proj = project_season(_totals(), _universe(), 2026, ProjectionConfig(lookback=2))
    ranks = dict(zip(proj["gsis_id"].to_list(), range(proj.height), strict=True))
    assert ranks["a"] < ranks["b"]


def test_shrinkage_pulls_extremes_toward_the_prior():
    totals = _totals()
    proj = project_season(totals, _universe(), 2026, ProjectionConfig(lookback=2))
    stud = proj.filter(pl.col("gsis_id") == "a").to_dicts()[0]
    # Projected below his own 15.6 ppg history: shrinkage and ageing both bite.
    assert stud["proj_ppg"] < 15.6


def test_rookies_are_projected_from_draft_capital():
    proj = project_season(_totals(), _universe(), 2026, ProjectionConfig(lookback=2))
    rookie = proj.filter(pl.col("gsis_id") == "rook")
    assert rookie.height == 1
    assert rookie["proj_points"][0] > 0


def test_positional_priors_are_computed_per_position():
    priors = positional_priors(_totals(), [2024, 2025])
    assert priors["RB"] != priors["WR"]
    assert priors["WR"] > priors["RB"]  # one strong WR vs a stud and a scrub


def test_projected_games_stay_in_a_real_range():
    proj = project_season(_totals(), _universe(), 2026, ProjectionConfig(lookback=2))
    assert proj["proj_games"].min() >= 1.0
    assert proj["proj_games"].max() <= 17.0
