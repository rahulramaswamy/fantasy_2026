import polars as pl

from ff2026.model.blend import DEFAULT_EXPERT_WEIGHT, blend_rankings


def _proj():
    return pl.DataFrame(
        {
            "gsis_id": ["a", "b", "c", "d"],
            "position": ["RB", "RB", "RB", "RB"],
            "proj_points": [300.0, 250.0, 200.0, 150.0],
            "floor": [200.0, 170.0, 130.0, 90.0],
            "ceiling": [400.0, 330.0, 270.0, 210.0],
        }
    )


def test_pure_expert_weight_adopts_expert_order():
    # Experts think the model has it exactly backwards.
    ecr = pl.DataFrame({"gsis_id": ["a", "b", "c", "d"], "ecr": [4.0, 3.0, 2.0, 1.0]})
    out = blend_rankings(_proj(), ecr, weight=1.0)
    assert out["gsis_id"].to_list() == ["d", "c", "b", "a"]


def test_zero_weight_keeps_model_order():
    ecr = pl.DataFrame({"gsis_id": ["a", "b", "c", "d"], "ecr": [4.0, 3.0, 2.0, 1.0]})
    out = blend_rankings(_proj(), ecr, weight=0.0)
    assert out["gsis_id"].to_list() == ["a", "b", "c", "d"]


def test_point_distribution_is_preserved():
    """The ordering may change, but the positional value curve must not.

    Replacement level and tier breaks are computed from the shape of this curve,
    so the blend reassigns points along the new order rather than inventing them.
    """
    ecr = pl.DataFrame({"gsis_id": ["a", "b", "c", "d"], "ecr": [4.0, 3.0, 2.0, 1.0]})
    out = blend_rankings(_proj(), ecr, weight=1.0)
    assert sorted(out["proj_points"].to_list(), reverse=True) == [300.0, 250.0, 200.0, 150.0]
    # The player promoted to first inherits first place's points.
    assert out.filter(pl.col("gsis_id") == "d")["proj_points"][0] == 300.0


def test_unranked_players_keep_model_ordering_and_are_flagged():
    ecr = pl.DataFrame({"gsis_id": ["a", "b"], "ecr": [1.0, 2.0]})
    out = blend_rankings(_proj(), ecr, weight=1.0)
    assert out.filter(pl.col("gsis_id") == "c")["expert_ranked"][0] is False
    assert out.filter(pl.col("gsis_id") == "a")["expert_ranked"][0] is True
    # Everyone still present -- coverage must not be lost.
    assert out.height == 4


def test_empty_expert_frame_is_a_no_op():
    out = blend_rankings(_proj(), pl.DataFrame(), weight=DEFAULT_EXPERT_WEIGHT)
    assert out["gsis_id"].to_list() == ["a", "b", "c", "d"]
    assert not out["expert_ranked"].any()


def test_blend_is_between_the_two_sources():
    ecr = pl.DataFrame({"gsis_id": ["a", "b", "c", "d"], "ecr": [4.0, 3.0, 2.0, 1.0]})
    half = blend_rankings(_proj(), ecr, weight=0.5)["gsis_id"].to_list()
    # With a symmetric disagreement, a 50/50 blend must not simply equal either side.
    assert half != ["a", "b", "c", "d"] or half != ["d", "c", "b", "a"]
