import polars as pl

from ff2026.ids import (
    build_crosswalk,
    crosswalk_report,
    join_key,
    normalize_name,
    sleeper_players_to_frame,
)


def test_normalize_strips_suffixes_and_punctuation():
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalize_name("Ke'Shawn Vaughn") == "keshawn vaughn"
    assert normalize_name("Amon-Ra St. Brown") == "amon ra st brown"


def test_normalize_handles_accents():
    assert normalize_name("José Ramírez") == "jose ramirez"


def test_join_key_separates_players_by_position():
    assert join_key("Josh Allen", "QB") != join_key("Josh Allen", "LB")


def test_sleeper_frame_keeps_only_fantasy_positions():
    payload = {
        "1": {"full_name": "A Back", "position": "RB", "team": "SF", "gsis_id": "00-1"},
        "2": {"full_name": "A Guard", "position": "OG", "team": "SF"},
    }
    df = sleeper_players_to_frame(payload)
    assert df.height == 1
    assert df["position"][0] == "RB"


def test_empty_payload_returns_typed_empty_frame():
    df = sleeper_players_to_frame({})
    assert df.height == 0
    assert "sleeper_id" in df.columns


def test_crosswalk_prefers_sleepers_own_gsis_id():
    sleeper = sleeper_players_to_frame(
        {"1": {"full_name": "Known Guy", "position": "WR", "gsis_id": "00-0011"}}
    )
    out = build_crosswalk(sleeper)
    assert out["gsis_id"][0] == "00-0011"
    assert out["id_source"][0] == "sleeper"


def test_crosswalk_falls_back_to_name_match():
    sleeper = sleeper_players_to_frame(
        {"9": {"full_name": "Nameless Guy", "position": "TE", "gsis_id": None}}
    )
    master = pl.DataFrame(
        {"gsis_id": ["00-0099"], "display_name": ["Nameless Guy"], "position": ["TE"]}
    )
    out = build_crosswalk(sleeper, None, master)
    assert out["gsis_id"][0] == "00-0099"
    assert out["id_source"][0] == "name_match"


def test_unmatched_players_are_reported_not_hidden():
    sleeper = sleeper_players_to_frame(
        {"9": {"full_name": "Ghost Player", "position": "WR", "search_rank": 5}}
    )
    out = build_crosswalk(sleeper)
    report = crosswalk_report(out)
    assert report["match_rate"] == 0.0
    assert report["notable_unmatched"].height == 1
