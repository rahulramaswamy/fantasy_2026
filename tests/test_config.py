from ff2026.config import LeagueConfig


def test_starting_slots_exclude_bench_and_ir(ppr_league):
    assert "BN" not in ppr_league.starting_slots
    assert len(ppr_league.starting_slots) == 9
    assert ppr_league.bench_size == 6


def test_starters_at_counts_dedicated_slots_only(ppr_league):
    assert ppr_league.starters_at("RB") == 2
    assert ppr_league.starters_at("QB") == 1


def test_flex_slots_detected(ppr_league, superflex_league):
    assert ppr_league.flex_slots() == {"FLEX": 1}
    assert superflex_league.flex_slots() == {"FLEX": 1, "SUPER_FLEX": 1}


def test_from_sleeper_reads_real_payload_shape():
    payload = {
        "name": "Real League",
        "league_id": "9999",
        "season": "2026",
        "settings": {"num_teams": 10},
        "roster_positions": ["QB", "RB", "WR", "TE", "SUPER_FLEX", "BN", "BN"],
        "scoring_settings": {"rec": 0.5, "pass_td": 6.0},
    }
    cfg = LeagueConfig.from_sleeper(payload)
    assert cfg.teams == 10
    assert cfg.season == 2026
    assert cfg.ppr == 0.5
    assert cfg.superflex is True
    assert cfg.scoring_settings["pass_td"] == 6.0


def test_two_quarterback_league_counts_as_superflex():
    payload = {
        "name": "2QB", "season": "2026", "settings": {"num_teams": 12},
        "roster_positions": ["QB", "QB", "RB", "WR", "BN"],
        "scoring_settings": {"rec": 1.0},
    }
    assert LeagueConfig.from_sleeper(payload).superflex is True


def test_roundtrip_through_yaml(tmp_path, ppr_league):
    path = ppr_league.save(tmp_path / "league.yaml")
    loaded = LeagueConfig.load(path)
    assert loaded.name == ppr_league.name
    assert loaded.scoring_settings == ppr_league.scoring_settings
