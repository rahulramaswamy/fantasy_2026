import polars as pl

from ff2026.config import LeagueConfig
from ff2026.scoring import ScoringEngine, default_ppr_settings


def _line(**kwargs):
    base = {
        "position": "WR", "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
        "rushing_yards": 0, "rushing_tds": 0, "receptions": 0, "receiving_yards": 0,
        "receiving_tds": 0, "fumbles_lost_total": 0,
    }
    base.update(kwargs)
    return pl.DataFrame([base])


def test_ppr_reception_and_yardage(ppr_league):
    engine = ScoringEngine(ppr_league)
    df = engine.score_weekly(_line(receptions=8, receiving_yards=100, receiving_tds=1))
    # 8 receptions + 10 yards points + 6 TD
    assert df["fantasy_points"][0] == 24.0


def test_half_ppr_halves_receptions():
    league = LeagueConfig(scoring_settings=default_ppr_settings(0.5))
    engine = ScoringEngine(league)
    df = engine.score_weekly(_line(receptions=8, receiving_yards=100))
    assert df["fantasy_points"][0] == 14.0


def test_passing_line(ppr_league):
    engine = ScoringEngine(ppr_league)
    df = engine.score_weekly(
        _line(position="QB", passing_yards=300, passing_tds=2, passing_interceptions=1)
    )
    # 12 + 8 - 1
    assert df["fantasy_points"][0] == 19.0


def test_te_premium_applies_only_to_tight_ends():
    league = LeagueConfig(scoring_settings={**default_ppr_settings(1.0), "bonus_rec_te": 0.5})
    engine = ScoringEngine(league)
    te = engine.score_weekly(_line(position="TE", receptions=10))["fantasy_points"][0]
    wr = engine.score_weekly(_line(position="WR", receptions=10))["fantasy_points"][0]
    assert te - wr == 5.0


def test_threshold_bonus_is_per_game():
    league = LeagueConfig(
        scoring_settings={**default_ppr_settings(1.0), "bonus_rush_yd_100": 3.0}
    )
    engine = ScoringEngine(league)
    at_threshold = engine.score_weekly(_line(rushing_yards=100))["fantasy_points"][0]
    below = engine.score_weekly(_line(rushing_yards=99))["fantasy_points"][0]
    assert round(at_threshold - below, 2) == 3.1  # 3 bonus + 0.1 for the extra yard


def test_kicker_distance_buckets():
    league = LeagueConfig(scoring_settings={"fgm_0_19": 3.0, "fgm_50p": 5.0, "xpm": 1.0})
    engine = ScoringEngine(league)
    df = pl.DataFrame(
        [{"position": "K", "fg_made_0_19": 1, "fg_made_50_59": 1, "fg_made_60_": 1,
          "pat_made": 2}]
    )
    # 3 + 5 + 5 + 2
    assert engine.score_weekly(df)["fantasy_points"][0] == 15.0


def test_unsupported_keys_are_reported_not_silently_dropped():
    league = LeagueConfig(
        scoring_settings={**default_ppr_settings(1.0), "pass_td_50p": 2.0}
    )
    engine = ScoringEngine(league)
    assert "pass_td_50p" in engine.static_unsupported()


def test_sleeper_default_interception_differs_from_nflverse():
    """Pins a real, easy-to-miss discrepancy.

    nflverse's built-in `fantasy_points` column uses the NFL-standard -2 per
    interception, while Sleeper's default is -1. Scoring must follow the
    league's own settings, not nflverse's column.
    """
    league = LeagueConfig(scoring_settings=default_ppr_settings(1.0))
    assert league.scoring_settings["pass_int"] == -1.0
    engine = ScoringEngine(league)
    df = engine.score_weekly(_line(position="QB", passing_interceptions=1))
    assert df["fantasy_points"][0] == -1.0
