from ff2026.draft.value import add_value, replacement_levels, starter_demand, tier_breaks


def test_dedicated_slots_drive_baseline_demand(ppr_league, board):
    demand = starter_demand(ppr_league, board)
    # 12 teams x 1 QB, and no flex slot accepts a QB in this league.
    assert demand["QB"] == 12
    # 12 teams x 2 RB dedicated, plus whatever share of the 12 FLEX slots RBs win.
    assert demand["RB"] > 24
    assert demand["WR"] > 24


def test_flex_slots_are_allocated_to_whoever_projects_best(ppr_league, board):
    demand = starter_demand(ppr_league, board)
    flex_extra = (demand["RB"] - 24) + (demand["WR"] - 24) + (demand["TE"] - 12)
    assert flex_extra == 12  # one FLEX per team, allocated across RB/WR/TE


def test_superflex_raises_quarterback_demand(superflex_league, board):
    demand = starter_demand(superflex_league, board)
    assert demand["QB"] > superflex_league.teams


def test_replacement_level_sits_below_the_starters(ppr_league, board):
    levels = replacement_levels(ppr_league, board)
    for pos in ("QB", "RB", "WR", "TE"):
        starters = board.filter(board["position"] == pos).sort("proj_points", descending=True)
        assert levels[pos] < starters["proj_points"][0]


def test_vorp_ranks_scarcity_not_raw_points(ppr_league, board):
    valued = add_value(board, ppr_league)
    top = valued.head(1).to_dicts()[0]
    best_qb = board.filter(board["position"] == "QB")["proj_points"].max()
    # The highest raw scorer is a QB, but VORP should not crown him.
    assert top["proj_points"] < best_qb
    assert top["position"] != "QB"


def test_vorp_is_points_minus_replacement(ppr_league, board):
    valued = add_value(board, ppr_league)
    row = valued.head(1).to_dicts()[0]
    assert abs(row["vorp"] - (row["proj_points"] - row["replacement_points"])) < 1e-6


def test_tier_breaks_find_gaps():
    import polars as pl

    df = pl.DataFrame(
        {
            "position": ["RB"] * 6,
            # A clear cliff after the third back.
            "proj_points": [300.0, 295.0, 290.0, 200.0, 197.0, 194.0],
        }
    )
    assert 3 in tier_breaks(df, "RB")
