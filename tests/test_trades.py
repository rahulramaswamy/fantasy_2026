import polars as pl

from ff2026.trades.evaluate import apply_trade, evaluate_trade, optimal_lineup


def _players():
    rows = [
        ("qb1", "QB", 300.0), ("qb2", "QB", 250.0),
        ("rb1", "RB", 280.0), ("rb2", "RB", 200.0), ("rb3", "RB", 120.0),
        ("wr1", "WR", 260.0), ("wr2", "WR", 210.0), ("wr3", "WR", 150.0),
        ("te1", "TE", 180.0), ("k1", "K", 120.0), ("def1", "DEF", 110.0),
        ("rbX", "RB", 300.0), ("wrX", "WR", 90.0),
    ]
    return pl.DataFrame(
        [{"sleeper_id": i, "name": i, "position": p, "proj_points": v} for i, p, v in rows]
    )


def test_optimal_lineup_fills_every_starting_slot(ppr_league):
    players = _players()
    roster = players.filter(pl.col("sleeper_id") != "rbX")
    total, starters = optimal_lineup(roster, ppr_league)
    assert starters.height == len(ppr_league.starting_slots)
    assert total > 0


def test_flex_does_not_steal_from_a_dedicated_slot(ppr_league):
    """A single TE must start at TE, not be consumed by FLEX."""
    players = _players()
    roster = players.filter(pl.col("sleeper_id") != "rbX")
    _, starters = optimal_lineup(roster, ppr_league)
    slots = dict(
        zip(starters["sleeper_id"].to_list(), starters["slot"].to_list(), strict=True)
    )
    assert slots["te1"] == "TE"


def test_no_player_starts_twice(ppr_league):
    players = _players()
    _, starters = optimal_lineup(players, ppr_league)
    assert starters["sleeper_id"].n_unique() == starters.height


def test_apply_trade_swaps_the_right_players():
    players = _players()
    roster = players.filter(pl.col("sleeper_id").is_in(["rb1", "wr1"]))
    after = apply_trade(roster, players, sends=["rb1"], receives=["rbX"])
    assert set(after["sleeper_id"].to_list()) == {"wr1", "rbX"}


def test_upgrading_a_starter_is_scored_as_a_gain(ppr_league):
    players = _players()
    roster = players.filter(~pl.col("sleeper_id").is_in(["rbX", "wrX"]))
    result = evaluate_trade(roster, players, ppr_league, sends=["rb3"], receives=["rbX"])
    assert result["lineup_delta"] > 0
    assert result["verdict"].startswith("accept")


def test_trading_a_starter_for_a_bench_player_is_a_loss(ppr_league):
    players = _players()
    roster = players.filter(~pl.col("sleeper_id").is_in(["rbX", "wrX"]))
    result = evaluate_trade(roster, players, ppr_league, sends=["rb1"], receives=["wrX"])
    assert result["lineup_delta"] < 0
    assert result["verdict"] == "decline"


def test_bench_only_swap_barely_moves_lineup_value(ppr_league):
    """Points scored on your bench are worth nothing, and the model should know."""
    players = _players()
    roster = players.filter(~pl.col("sleeper_id").is_in(["rbX", "wrX"]))
    before, _ = optimal_lineup(roster, ppr_league)
    after_roster = apply_trade(roster, players, sends=["rb3"], receives=["wrX"])
    after, _ = optimal_lineup(after_roster, ppr_league)
    assert abs(after - before) < 1e-6
