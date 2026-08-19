"""Live draft state derived from Sleeper.

Tracks who is gone, whose turn it is, and -- the part that actually drives
strategy -- exactly which pick numbers are yours and how long until they come
around. Snake order is where most homegrown draft tools quietly get this wrong,
so it is computed explicitly and unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class DraftState:
    """A snapshot of a Sleeper draft."""

    draft_id: str
    draft_type: str  # "snake" | "linear" | "auction"
    teams: int
    rounds: int
    status: str
    my_slot: int | None = None
    picks: list[dict[str, Any]] = field(default_factory=list)
    slot_to_roster_id: dict[str, Any] = field(default_factory=dict)
    draft_order: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ basics

    @property
    def picks_made(self) -> int:
        return len(self.picks)

    @property
    def total_picks(self) -> int:
        return self.teams * self.rounds

    @property
    def is_complete(self) -> bool:
        return self.status == "complete" or self.picks_made >= self.total_picks

    @property
    def next_pick_overall(self) -> int:
        """1-indexed overall pick number that is on the clock."""
        return self.picks_made + 1

    @property
    def current_round(self) -> int:
        return (self.next_pick_overall - 1) // self.teams + 1

    def drafted_player_ids(self) -> set[str]:
        return {str(p["player_id"]) for p in self.picks if p.get("player_id")}

    def picks_by_slot(self, slot: int) -> list[dict[str, Any]]:
        return [p for p in self.picks if p.get("draft_slot") == slot]

    def my_roster(self) -> list[str]:
        if self.my_slot is None:
            return []
        return [
            str(p["player_id"]) for p in self.picks_by_slot(self.my_slot) if p.get("player_id")
        ]

    # ------------------------------------------------------------- pick timing

    def pick_number(self, slot: int, rnd: int) -> int:
        """Overall pick number for a draft slot in a given round (1-indexed)."""
        if self.draft_type == "snake" and rnd % 2 == 0:
            return (rnd - 1) * self.teams + (self.teams - slot + 1)
        return (rnd - 1) * self.teams + slot

    @property
    def slot_is_valid(self) -> bool:
        """Is `my_slot` a real position in this draft?

        Snake ordering maps an out-of-range slot to nonsense -- slot 16 in a
        10-team draft yields picks [16, 5, 36, 25, ...], descending within a
        round, and a slot of 0 or less produces negative pick numbers. Those
        then feed survival probabilities and recommendations that look
        plausible and are meaningless, so the range is checked rather than
        assumed.
        """
        return self.my_slot is not None and 1 <= self.my_slot <= self.teams

    def my_pick_numbers(self) -> list[int]:
        if not self.slot_is_valid:
            return []
        assert self.my_slot is not None  # narrowed by slot_is_valid
        return [self.pick_number(self.my_slot, r) for r in range(1, self.rounds + 1)]

    def my_upcoming_picks(self) -> list[int]:
        return [p for p in self.my_pick_numbers() if p >= self.next_pick_overall]

    def is_my_turn(self) -> bool:
        return bool(self.my_upcoming_picks()) and (
            self.my_upcoming_picks()[0] == self.next_pick_overall
        )

    def picks_until_my_turn(self) -> int | None:
        upcoming = self.my_upcoming_picks()
        if not upcoming:
            return None
        return upcoming[0] - self.next_pick_overall

    def my_next_two_picks(self) -> tuple[int | None, int | None]:
        """My current (or next) pick and the one after it.

        The gap between these two is the whole basis of draft strategy: it is how
        long a player has to survive for you to still get him.
        """
        upcoming = self.my_upcoming_picks()
        first = upcoming[0] if upcoming else None
        second = upcoming[1] if len(upcoming) > 1 else None
        return first, second

    # ------------------------------------------------------------ construction

    @classmethod
    def from_sleeper(
        cls,
        draft: dict[str, Any],
        picks: list[dict[str, Any]],
        my_user_id: str | None = None,
        my_slot: int | None = None,
    ) -> DraftState:
        settings = draft.get("settings") or {}
        draft_order = draft.get("draft_order") or {}

        slot = my_slot
        if slot is None and my_user_id and draft_order:
            raw = draft_order.get(str(my_user_id))
            slot = int(raw) if raw is not None else None

        return cls(
            draft_id=str(draft.get("draft_id")),
            draft_type=str(draft.get("type") or "snake"),
            teams=int(settings.get("teams") or 12),
            rounds=int(settings.get("rounds") or 15),
            status=str(draft.get("status") or "unknown"),
            my_slot=slot,
            picks=list(picks or []),
            slot_to_roster_id=draft.get("slot_to_roster_id") or {},
            draft_order=draft_order,
        )


def roster_counts(player_ids: list[str], crosswalk: pl.DataFrame) -> dict[str, int]:
    """Count a roster by position using the Sleeper crosswalk."""
    if not player_ids or crosswalk.is_empty():
        return {}
    sub = crosswalk.filter(pl.col("sleeper_id").is_in(player_ids))
    return {r["position"]: r["len"] for r in sub.group_by("position").len().to_dicts()}


def picks_frame(picks: list[dict[str, Any]], crosswalk: pl.DataFrame) -> pl.DataFrame:
    """Tidy the raw pick feed into something printable."""
    if not picks:
        return pl.DataFrame(
            schema={"pick_no": pl.Int64, "round": pl.Int64, "draft_slot": pl.Int64,
                    "sleeper_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8}
        )
    rows = []
    for p in picks:
        meta = p.get("metadata") or {}
        name = " ".join(
            x for x in (meta.get("first_name"), meta.get("last_name")) if x
        ).strip()
        rows.append(
            {
                "pick_no": p.get("pick_no"),
                "round": p.get("round"),
                "draft_slot": p.get("draft_slot"),
                "sleeper_id": str(p.get("player_id")) if p.get("player_id") else None,
                "name": name or None,
                "position": meta.get("position"),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("pick_no")
