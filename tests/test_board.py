"""Snake draft ordering is the easiest thing in a draft tool to get subtly wrong."""

from ff2026.draft.board import DraftState


def _state(teams=12, rounds=15, slot=3, picks=None, draft_type="snake"):
    return DraftState(
        draft_id="d1", draft_type=draft_type, teams=teams, rounds=rounds,
        status="drafting", my_slot=slot, picks=picks or [],
    )


def test_snake_odd_rounds_go_forward():
    s = _state()
    assert s.pick_number(slot=1, rnd=1) == 1
    assert s.pick_number(slot=3, rnd=1) == 3
    assert s.pick_number(slot=12, rnd=1) == 12


def test_snake_even_rounds_reverse():
    s = _state()
    assert s.pick_number(slot=12, rnd=2) == 13
    assert s.pick_number(slot=1, rnd=2) == 24
    assert s.pick_number(slot=3, rnd=2) == 22


def test_linear_draft_never_reverses():
    s = _state(draft_type="linear")
    assert s.pick_number(slot=3, rnd=2) == 15
    assert s.pick_number(slot=3, rnd=3) == 27


def test_my_pick_numbers_snake_from_slot_3():
    s = _state(slot=3, rounds=4)
    # R1 pick 3, R2 pick 22, R3 pick 27, R4 pick 46
    assert s.my_pick_numbers() == [3, 22, 27, 46]


def test_turn_detection_and_wait():
    picks = [{"pick_no": i, "player_id": str(i), "draft_slot": i} for i in range(1, 3)]
    s = _state(slot=3, picks=picks)
    assert s.next_pick_overall == 3
    assert s.is_my_turn()
    assert s.picks_until_my_turn() == 0


def test_wait_counts_picks_until_turn():
    picks = [{"pick_no": 1, "player_id": "1", "draft_slot": 1}]
    s = _state(slot=5, picks=picks)
    assert s.next_pick_overall == 2
    assert not s.is_my_turn()
    assert s.picks_until_my_turn() == 3


def test_next_two_picks_spans_the_turn():
    """At slot 1 the gap between picks is longest -- this is why turn strategy matters."""
    s = _state(slot=1, rounds=3)
    first, second = s.my_next_two_picks()
    assert (first, second) == (1, 24)


def test_my_roster_only_counts_my_slot():
    picks = [
        {"pick_no": 1, "player_id": "a", "draft_slot": 1},
        {"pick_no": 3, "player_id": "b", "draft_slot": 3},
        {"pick_no": 22, "player_id": "c", "draft_slot": 3},
    ]
    s = _state(slot=3, picks=picks)
    assert s.my_roster() == ["b", "c"]


def test_from_sleeper_reads_slot_from_draft_order():
    draft = {
        "draft_id": "x", "type": "snake", "status": "drafting",
        "settings": {"teams": 10, "rounds": 16},
        "draft_order": {"user123": 7},
    }
    s = DraftState.from_sleeper(draft, [], my_user_id="user123")
    assert s.my_slot == 7
    assert s.teams == 10
    assert s.rounds == 16
    assert s.total_picks == 160


def test_complete_when_all_picks_in():
    picks = [{"pick_no": i, "player_id": str(i), "draft_slot": 1} for i in range(1, 25)]
    s = _state(teams=12, rounds=2, picks=picks)
    assert s.is_complete
