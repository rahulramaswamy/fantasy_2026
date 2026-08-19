import polars as pl

from ff2026.config import LeagueConfig
from ff2026.trades.finder import FinderConfig, find_trades, to_frame


def _league():
    return LeagueConfig(
        name="Test", teams=2,
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN"],
        scoring_settings={"rec": 1.0},
    )


def _players():
    """Mirror-image rosters with genuine BENCH surplus.

    The surplus has to be benched to be tradeable: a player who is already in
    your starting lineup cannot be given away for free, so rosters where every
    player starts have no trades in them at all -- correctly.
    """
    rows = [
        # Mine: four startable RBs (one benched), thin at WR.
        ("m1", "My QB",  "QB", 200.0), ("m2", "My RB1", "RB", 180.0),
        ("m3", "My RB2", "RB", 170.0), ("m4", "My RB3", "RB", 160.0),
        ("m5", "My RB4", "RB", 150.0), ("m6", "My WR1", "WR", 90.0),
        ("m7", "My WR2", "WR", 85.0),  ("m8", "My WR3", "WR", 40.0),
        ("m9", "My TE",  "TE", 80.0),
        # Theirs: four startable WRs (one benched), thin at RB.
        ("t1", "Th QB",  "QB", 195.0), ("t2", "Th WR1", "WR", 175.0),
        ("t3", "Th WR2", "WR", 165.0), ("t4", "Th WR3", "WR", 155.0),
        ("t5", "Th WR4", "WR", 145.0), ("t6", "Th RB1", "RB", 85.0),
        ("t7", "Th RB2", "RB", 80.0),  ("t8", "Th RB3", "RB", 40.0),
        ("t9", "Th TE",  "TE", 75.0),
    ]
    return pl.DataFrame(
        {
            "sleeper_id": [r[0] for r in rows],
            "name": [r[1] for r in rows],
            "position": [r[2] for r in rows],
            "ros_points": [r[3] for r in rows],
        }
    )


def _split(players):
    mine = players.filter(pl.col("sleeper_id").str.starts_with("m"))
    theirs = players.filter(pl.col("sleeper_id").str.starts_with("t"))
    return mine, theirs


def test_finds_the_mutually_beneficial_swap():
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1.0))
    assert ideas, "surplus RB for surplus WR is the textbook trade; it must be found"
    top = ideas[0]
    assert top.my_gain > 0 and top.their_gain > 0


def test_only_surfaces_trades_both_sides_gain_from():
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1.0))
    assert all(i.my_gain > 0 for i in ideas)
    assert all(i.their_gain >= 0 for i in ideas)


def test_min_gain_threshold_is_respected():
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1000.0))
    assert ideas == []


def test_excluded_players_are_never_traded():
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1.0, exclude={"m5"}))
    assert all("m5" not in i.send for i in ideas)


def test_no_opponents_yields_nothing():
    players = _players()
    mine, _ = _split(players)
    assert find_trades(mine, {}, players, _league()) == []


def test_to_frame_shape():
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1.0))
    df = to_frame(ideas)
    assert {"partner", "send", "receive", "my_gain", "their_gain"} <= set(df.columns)
    assert df.height == len(ideas)


def test_lopsided_trades_are_filtered_out():
    """A trade the partner has no reason to accept is not a trade.

    Gaining 125 while the other manager gains 5 is technically mutual and will
    still be declined, so it should not reach the list.
    """
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1.0, min_their_gain_ratio=0.25))
    assert ideas
    assert all(i.their_gain >= 0.25 * i.my_gain for i in ideas)


def test_ratio_of_zero_allows_lopsided_trades():
    players = _players()
    mine, theirs = _split(players)
    ideas = find_trades(mine, {"rival": theirs}, players, _league(),
                        FinderConfig(min_my_gain=1.0, min_their_gain_ratio=0.0))
    assert max(i.my_gain for i in ideas) > 100


def test_unknown_next_pick_is_null_not_zero():
    """Regression: before slots are assigned there is no next pick.

    Reporting 0% survival there tells the drafter a player is certainly gone,
    which is the opposite of the truth (we simply don't know).
    """
    from ff2026.config import LeagueConfig
    from ff2026.draft.agent import recommend

    board = pl.DataFrame({
        "sleeper_id": ["a", "b"], "name": ["A", "B"], "position": ["RB", "WR"],
        "vorp": [50.0, 40.0], "proj_points": [200.0, 190.0],
        "replacement_points": [150.0, 150.0], "adp": [10.0, 20.0],
        "adp_stdev": [5.0, 5.0],
    })
    league = LeagueConfig(teams=10, roster_positions=["RB", "WR", "BN"],
                          scoring_settings={"rec": 1.0})
    out = recommend(board, league, {}, current_pick=5, next_pick=None)
    assert out["survives_to_next"].null_count() == out.height


def test_out_of_range_draft_slot_yields_no_picks():
    """Regression: an impossible slot silently produced nonsense pick numbers.

    Slot 16 in a 10-team draft mapped to [16, 5, 36, 25, ...] -- descending
    within a round -- and slot 0 or negative produced negative pick numbers.
    Those fed survival probabilities that looked plausible and meant nothing.
    """
    from ff2026.draft.board import DraftState

    def _state(slot):
        return DraftState(draft_id="x", draft_type="snake", teams=10, rounds=15,
                          status="pre_draft", my_slot=slot)

    for bad in (16, 0, -3, 11):
        assert not _state(bad).slot_is_valid
        assert _state(bad).my_pick_numbers() == []

    good = _state(5)
    assert good.slot_is_valid
    assert good.my_pick_numbers()[:3] == [5, 16, 25]


def test_likely_survivor_scores_near_zero_for_taking_him_early():
    """A player almost certain to last must not be recommended early.

    If a quarterback is ~98% to survive until your next pick, passing on him
    costs about 2% of his value, so his pick score should be near zero.

    This pins why the candidate stays in his own opportunity-cost pool. The
    baseline is the value of *not* drafting him, and in that world he is still
    on the board. Removing him scores him as though he would vanish, which
    recommends spending a premium pick on a player you would get for free.
    """
    from ff2026.config import LeagueConfig
    from ff2026.draft.agent import recommend

    board = pl.DataFrame({
        "sleeper_id": ["qb1", "qb2", "rb1"],
        "name": ["Safe QB", "Other QB", "Scarce RB"],
        "position": ["QB", "QB", "RB"],
        "vorp": [76.0, 30.0, 70.0],
        "proj_points": [350.0, 300.0, 250.0],
        "replacement_points": [274.0, 274.0, 150.0],
        "adp": [31.4, 60.0, 4.0],
        "adp_stdev": [11.0, 12.0, 3.0],
    })
    league = LeagueConfig(teams=10, roster_positions=["QB", "RB", "BN"],
                          scoring_settings={"rec": 1.0})
    out = recommend(board, league, {}, current_pick=5, next_pick=16, top_n=3)
    scores = dict(zip(out["name"].to_list(), out["pick_score"].to_list(), strict=False))
    assert scores["Safe QB"] < 10, "a near-certain survivor must not score high"
    assert scores["Scarce RB"] > scores["Safe QB"], "scarcity must outrank safety"
