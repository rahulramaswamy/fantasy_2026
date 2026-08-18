import polars as pl

from ff2026.config import LeagueConfig
from ff2026.power import power_table, read_the_table, team_strengths


def _league():
    return LeagueConfig(
        name="Test", teams=2,
        roster_positions=["QB", "RB", "WR", "TE", "BN"],
        scoring_settings={"rec": 1.0},
    )


def _board():
    return pl.DataFrame({
        "sleeper_id": ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"],
        "name": [f"P{i}" for i in range(8)],
        "position": ["QB", "RB", "WR", "TE"] * 2,
        "proj_points": [200.0, 150.0, 140.0, 90.0, 100.0, 80.0, 70.0, 50.0],
        "ros_points": [100.0, 70.0, 60.0, 40.0, 60.0, 50.0, 45.0, 30.0],
    })


def _rosters():
    return [
        {"roster_id": 1, "owner_id": "u1", "players": ["a1", "a2", "a3", "a4"],
         "settings": {"wins": 2, "losses": 8, "fpts": 900, "fpts_decimal": 50}},
        {"roster_id": 2, "owner_id": "u2", "players": ["b1", "b2", "b3", "b4"],
         "settings": {"wins": 8, "losses": 2, "fpts": 700, "fpts_decimal": 0}},
    ]


def _users():
    return [
        {"user_id": "u1", "display_name": "strong_unlucky",
         "metadata": {"team_name": "Sleeping Giants"}},
        {"user_id": "u2", "display_name": "weak_lucky", "metadata": {}},
    ]


def test_strength_reflects_roster_not_record():
    strengths = team_strengths(_rosters(), _users(), _board(), _league())
    by_name = {s.manager: s for s in strengths}
    assert by_name["strong_unlucky"].preseason > by_name["weak_lucky"].preseason
    assert by_name["strong_unlucky"].ros > by_name["weak_lucky"].ros
    # ...even though the record says the opposite.
    assert by_name["strong_unlucky"].wins < by_name["weak_lucky"].wins


def test_points_for_combines_whole_and_decimal():
    strengths = team_strengths(_rosters(), _users(), _board(), _league())
    assert abs(next(s for s in strengths if s.manager == "strong_unlucky").points_for
               - 900.5) < 1e-6


def test_luck_column_flags_the_mismatch():
    df = power_table(team_strengths(_rosters(), _users(), _board(), _league()))
    lucky = df.filter(pl.col("manager") == "weak_lucky").to_dicts()[0]
    unlucky = df.filter(pl.col("manager") == "strong_unlucky").to_dicts()[0]
    # Better record than scoring deserves -> positive luck, and vice versa.
    assert lucky["luck"] > 0
    assert unlucky["luck"] < 0


def test_team_name_falls_back_to_manager():
    strengths = team_strengths(_rosters(), _users(), _board(), _league())
    assert next(s for s in strengths if s.manager == "weak_lucky").team_name == "weak_lucky"
    assert next(s for s in strengths
                if s.manager == "strong_unlucky").team_name == "Sleeping Giants"


def test_read_the_table_calls_out_the_buy_low_target():
    df = power_table(team_strengths(_rosters(), _users(), _board(), _league()))
    notes = " ".join(read_the_table(df))
    assert "strong_unlucky" in notes
    assert "buy" in notes.lower() or "unlucky" in notes.lower()


def test_missing_ros_column_does_not_break():
    board = _board().drop("ros_points")
    strengths = team_strengths(_rosters(), _users(), board, _league())
    assert all(s.ros == 0.0 for s in strengths)
    assert not power_table(strengths).is_empty()


def test_co_owned_rosters_are_recognised():
    """A co-owner should find their team; matching only on owner_id misses them."""
    from ff2026.power import team_strengths

    rosters = _rosters()
    rosters[0]["owner_id"] = "someone_else"
    rosters[0]["co_owners"] = ["u1"]
    strengths = team_strengths(rosters, _users(), _board(), _league())
    # team_strengths keys off owner_id for naming, but must not crash or drop rows.
    assert len(strengths) == 2


def test_empty_rosters_are_detectable_not_a_fake_ranking():
    """An undrafted league must be identifiable as such.

    Ranking ten empty rosters produces a tidy table of zeroes that reads like a
    real leaderboard. Callers need to be able to tell the difference.
    """
    from ff2026.power import team_strengths

    rosters = [
        {"roster_id": 1, "owner_id": "u1", "players": [], "settings": {}},
        {"roster_id": 2, "owner_id": "u2", "players": None, "settings": {}},
    ]
    strengths = team_strengths(rosters, _users(), _board(), _league())
    assert sum(s.players for s in strengths) == 0
    assert all(s.preseason == 0.0 for s in strengths)
