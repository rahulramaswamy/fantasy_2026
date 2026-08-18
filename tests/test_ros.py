import polars as pl

from ff2026.config import LeagueConfig
from ff2026.model.ros import (
    ROSConfig,
    blend_weight,
    remaining_games_by_team,
    rest_of_season,
    team_games_played,
)
from ff2026.scoring import ScoringEngine, default_ppr_settings


def _schedule():
    rows = []
    for week in range(1, 5):
        rows.append({"season": 2025, "game_type": "REG", "week": week,
                     "home_team": "AAA", "away_team": "BBB"})
    # CCC plays only in weeks 3-4 -- byes must be counted, not assumed.
    for week in (3, 4):
        rows.append({"season": 2025, "game_type": "REG", "week": week,
                     "home_team": "CCC", "away_team": "DDD"})
    return pl.DataFrame(rows)


def _board():
    return pl.DataFrame({
        "gsis_id": ["p1", "p2", "p3"],
        "name": ["Steady", "Breakout", "Missing"],
        "position": ["RB", "WR", "TE"],
        "team": ["AAA", "AAA", "AAA"],
        "proj_ppg": [10.0, 10.0, 10.0],
        "proj_points": [170.0, 170.0, 170.0],
    })


def _weekly(points_by_player):
    rows = []
    for pid, weekly_points in points_by_player.items():
        for week, pts in enumerate(weekly_points, start=1):
            rows.append({
                "player_id": pid, "season": 2025, "week": week, "season_type": "REG",
                "team": "AAA", "position": "RB", "receptions": 0, "receiving_yards": 0,
                "rushing_yards": pts * 10, "rushing_tds": 0,
            })
    return pl.DataFrame(rows)


def _engine():
    return ScoringEngine(LeagueConfig(scoring_settings=default_ppr_settings(1.0)))


def test_remaining_games_excludes_byes():
    remaining = remaining_games_by_team(_schedule(), current_week=2, season=2025)
    assert remaining["AAA"] == 2
    # CCC has both its games still to come.
    assert remaining["CCC"] == 2


def test_team_games_played():
    played = team_games_played(_schedule(), current_week=2, season=2025)
    assert played["AAA"] == 2
    assert played.get("CCC", 0) == 0


def test_hot_start_is_shrunk_toward_preseason():
    """A player scoring double his projection should not be projected at double."""
    weekly = _weekly({"p1": [20.0, 20.0]})
    out = rest_of_season(_board(), weekly, _schedule(), _engine(), 2, 2025,
                         ROSConfig(prior_games=2.0))
    row = out.filter(pl.col("gsis_id") == "p1").to_dicts()[0]
    # (40 + 2*10) / (2 + 2) = 15, between the 10 projected and the 20 scored.
    assert 14.0 < row["ros_ppg"] < 16.0


def test_missing_player_is_penalised_through_availability():
    """Absence is evidence. A player who has not played must not be projected
    as if he will play every remaining game."""
    weekly = _weekly({"p1": [10.0, 10.0]})  # p3 never appears
    out = rest_of_season(_board(), weekly, _schedule(), _engine(), 2, 2025)
    played = out.filter(pl.col("gsis_id") == "p1").to_dicts()[0]
    missing = out.filter(pl.col("gsis_id") == "p3").to_dicts()[0]
    assert missing["availability"] < played["availability"]
    assert missing["ros_points"] < played["ros_points"]


def test_ros_points_is_rate_times_games():
    weekly = _weekly({"p1": [10.0, 10.0]})
    out = rest_of_season(_board(), weekly, _schedule(), _engine(), 2, 2025)
    row = out.filter(pl.col("gsis_id") == "p1").to_dicts()[0]
    assert abs(row["ros_points"] - row["ros_ppg"] * row["ros_games"]) < 1e-6


def test_blend_weight_moves_toward_current_season():
    assert blend_weight(0, 2.0) == 0.0
    assert blend_weight(2, 2.0) == 0.5
    assert blend_weight(10, 2.0) > 0.8
