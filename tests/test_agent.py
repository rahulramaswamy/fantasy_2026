import polars as pl

from ff2026.draft.agent import (
    AgentConfig,
    expected_best_available,
    positional_need,
    recommend,
    survival_probability,
)
from ff2026.draft.value import add_value


def test_survival_falls_as_the_pick_gets_later():
    early = survival_probability(adp=20, adp_sd=8, pick=10)
    at_adp = survival_probability(adp=20, adp_sd=8, pick=20)
    late = survival_probability(adp=20, adp_sd=8, pick=40)
    assert early > at_adp > late
    assert early > 0.85
    assert late < 0.05
    assert 0.4 < at_adp < 0.6


def test_survival_defaults_to_a_coin_flip_without_adp():
    assert survival_probability(None, None, 30) == 0.5


def test_expected_best_available_is_bounded_by_the_best_player():
    pool = pl.DataFrame(
        {
            "position": ["RB"] * 3,
            "vorp": [100.0, 80.0, 60.0],
            "adp": [5.0, 30.0, 50.0],
            "adp_stdev": [5.0, 5.0, 5.0],
        }
    )
    got = expected_best_available(pool, pick=40)
    assert 0 < got <= 100.0
    # The 100-VORP back is long gone by pick 40; expectation should sit near 60.
    assert got < 80.0


def test_empty_pool_has_no_value():
    empty = pl.DataFrame(schema={"position": pl.Utf8, "vorp": pl.Float64,
                                 "adp": pl.Float64, "adp_stdev": pl.Float64})
    assert expected_best_available(empty, pick=10) == 0.0


def test_need_is_full_for_unfilled_starting_slots(ppr_league):
    need = positional_need({}, ppr_league, AgentConfig())
    assert need["RB"] == 1.0
    assert need["QB"] == 1.0


def test_need_collapses_once_a_position_is_capped(ppr_league):
    cfg = AgentConfig()
    need = positional_need({"QB": 2}, ppr_league, cfg)
    assert need["QB"] == cfg.surplus_weight


def test_need_drops_to_depth_once_starters_and_flex_are_filled(ppr_league):
    cfg = AgentConfig()
    roster = {"RB": 3, "WR": 3, "TE": 1, "QB": 1}
    need = positional_need(roster, ppr_league, cfg)
    assert need["TE"] == cfg.depth_weight


def test_recommendation_prefers_the_scarcer_position(ppr_league, board):
    valued = add_value(board, ppr_league)
    recs = recommend(valued, ppr_league, roster={}, current_pick=1, next_pick=24, top_n=5)
    assert recs.height == 5
    assert recs["pick_score"].is_sorted(descending=True)


def test_recommendation_skips_positions_already_capped(ppr_league, board):
    valued = add_value(board, ppr_league)
    recs = recommend(
        valued, ppr_league, roster={"QB": 2}, current_pick=30, next_pick=50, top_n=10
    )
    assert "QB" not in recs["position"].to_list()


def test_last_pick_of_draft_has_no_opportunity_cost(ppr_league, board):
    valued = add_value(board, ppr_league)
    recs = recommend(valued, ppr_league, roster={}, current_pick=180, next_pick=None, top_n=3)
    assert (recs["vona_baseline"] == 0.0).all()
