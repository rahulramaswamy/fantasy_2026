from datetime import date

import polars as pl

from ff2026.roster.lineup import (
    expert_is_fresh,
    set_lineup,
    teams_on_bye,
    weekly_values,
)
from ff2026.roster.waivers import (
    WaiverConfig,
    find_moves,
    free_agents,
    marginal_values,
    rostered_in_league,
    trending_counts,
)


def _schedule():
    rows = []
    for week in (1, 2):
        rows.append({"season": 2026, "game_type": "REG", "week": week,
                     "home_team": "AAA", "away_team": "BBB"})
    # CCC sits out week 2.
    rows.append({"season": 2026, "game_type": "REG", "week": 1,
                 "home_team": "CCC", "away_team": "DDD"})
    return pl.DataFrame(rows)


def _roster():
    return pl.DataFrame({
        "sleeper_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "gsis_id": [f"g{i}" for i in range(1, 9)],
        "name": ["QB1", "RB1", "RB2", "RB3", "WR1", "WR2", "WR3", "TE1"],
        "position": ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE"],
        "team": ["AAA", "AAA", "CCC", "AAA", "AAA", "AAA", "AAA", "AAA"],
        "ros_ppg": [20.0, 15.0, 14.0, 6.0, 16.0, 12.0, 8.0, 9.0],
        "ros_points": [300.0, 225.0, 210.0, 90.0, 240.0, 180.0, 120.0, 135.0],
        "injury_status": [None, None, None, None, None, "Out", None, None],
    })


def test_teams_on_bye():
    assert teams_on_bye(_schedule(), 2, 2026) == {"CCC", "DDD"}
    assert teams_on_bye(_schedule(), 1, 2026) == set()
    assert teams_on_bye(_schedule(), 9, 2026) == set()


def test_weekly_values_zero_bye_and_out():
    df, used = weekly_values(_roster(), 2, _schedule(), 2026)
    vals = dict(zip(df["name"].to_list(), df["week_points"].to_list(), strict=True))
    assert vals["RB2"] == 0.0  # bye
    assert vals["WR2"] == 0.0  # Out
    assert vals["RB1"] == 15.0
    assert used is False


def test_lineup_benches_bye_and_out(ppr_league):
    df, _ = weekly_values(_roster(), 2, _schedule(), 2026)
    lineup = set_lineup(df, ppr_league, "week_points")
    started = set(lineup.starters["name"].to_list())
    # Healthy depth starts ahead of the bye/Out players...
    assert {"RB1", "RB3", "WR1", "WR3"} <= started
    # ...and only one of the two zero-value players is forced into the FLEX.
    assert len({"RB2", "WR2"} & started) == 1
    assert lineup.points == 20 + 15 + 6 + 16 + 8 + 9


def test_expert_freshness_gate():
    stale = pl.DataFrame({"gsis_id": ["g1"], "scrape_date": ["2025-12-30"]})
    fresh = pl.DataFrame({"gsis_id": ["g1"], "scrape_date": ["2026-10-02"]})
    today = date(2026, 10, 4)
    assert not expert_is_fresh(stale, 2026, today)
    assert expert_is_fresh(fresh, 2026, today)
    assert not expert_is_fresh(fresh, 2026, date(2026, 11, 1))


def test_expert_weekly_points_blend_in_when_fresh():
    expert = pl.DataFrame({
        "gsis_id": ["g2", "g6"], "week_pts": [25.0, 30.0],
        "scrape_date": ["2026-09-10", "2026-09-10"],
    })
    df, used = weekly_values(_roster(), 1, _schedule(), 2026, expert, today=date(2026, 9, 12))
    assert used
    vals = dict(zip(df["name"].to_list(), df["week_points"].to_list(), strict=True))
    assert abs(vals["RB1"] - (0.75 * 25 + 0.25 * 15)) < 1e-6
    # An Out player stays at zero even if the expert page has a number.
    assert vals["WR2"] == 0.0


def test_marginal_values_identify_bench(ppr_league):
    mv = marginal_values(_roster(), ppr_league)
    by = dict(zip(mv["name"].to_list(), mv["marginal"].to_list(), strict=True))
    # QB1 is the only QB: losing him costs his whole value.
    assert by["QB1"] == 300.0
    # RB3 is pure bench in a 2RB + 1FLEX league with three better options.
    assert by["RB3"] == 0.0
    # The first row is the natural drop: zero marginal, lowest ROS.
    assert mv["name"][0] == "RB3"


def _pool():
    return pl.DataFrame({
        "sleeper_id": ["101", "102", "103"],
        "gsis_id": ["fa1", "fa2", "fa3"],
        "name": ["Stud RB", "Meh WR", "Backup QB"],
        "position": ["RB", "WR", "QB"],
        "team": ["BBB", "BBB", "BBB"],
        "ros_ppg": [15.5, 7.0, 12.0],
        "ros_points": [232.0, 105.0, 180.0],
        "injury_status": [None, None, None],
    })


def test_find_moves_prefers_lineup_gain(ppr_league):
    moves = find_moves(_roster(), _pool(), ppr_league, WaiverConfig(min_gain=1.0))
    assert moves
    best = moves[0]
    assert best.add_name == "Stud RB"
    assert best.lineup_gain > 0
    # Drops the zero-marginal, lowest-value player, not a starter.
    assert best.drop_name == "RB3"
    # Meh WR is not worth a lineup slot, but he does beat RB3 on depth.
    meh = next(m for m in moves if m.add_name == "Meh WR")
    assert meh.lineup_gain < 0.5 and meh.depth_gain > 0


def test_find_moves_respects_protect_and_open_spot(ppr_league):
    moves = find_moves(
        _roster(), _pool(), ppr_league,
        WaiverConfig(min_gain=1.0, protect=frozenset({"4"})),
    )
    assert all(m.drop_id != "4" for m in moves)

    open_moves = find_moves(_roster(), _pool(), ppr_league, roster_full=False)
    assert open_moves and all(m.drop_id is None for m in open_moves)
    assert open_moves[0].add_name == "Stud RB"


def test_free_agents_and_rostered():
    rosters = [{"players": ["1", "2"], "reserve": ["9"]}, {"players": ["3"], "taxi": None}]
    assert rostered_in_league(rosters) == {"1", "2", "3", "9"}
    pool = free_agents(_roster(), {"1", "2", "3"})
    assert set(pool["sleeper_id"].to_list()) == {"4", "5", "6", "7", "8"}


def test_trending_counts():
    assert trending_counts([{"player_id": 12, "count": 300}]) == {"12": 300}


def test_regular_season_week_ignores_preseason_weeks():
    from ff2026.cli import regular_season_week

    assert regular_season_week({"season_type": "pre", "week": 2}) == 0
    assert regular_season_week({"season_type": "regular", "week": 7}) == 7
    assert regular_season_week({"season_type": "post", "week": 1}) == 18
    assert regular_season_week({}) == 1
