"""Finding trades worth proposing.

`evaluate_trade` judges a trade you already thought of. This searches for the
ones you didn't.

The search is built around one idea: **a trade only happens if both managers
think they won.** So a proposal is only interesting when it improves *your*
starting lineup and plausibly improves *theirs* too. Those trades exist because
rosters are unbalanced -- you have three startable running backs and one
receiver, someone else has the mirror image, and the surplus is worth more to
the other team than to its owner. That is real, findable arbitrage, and it is
completely different from trying to fleece someone.

Everything is valued on **rest-of-season** points, because points already scored
cannot be traded and a full-season number quietly includes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import polars as pl

from ..config import LeagueConfig
from .evaluate import optimal_lineup


@dataclass
class TradeIdea:
    """One proposal, scored from both sides."""

    partner: str
    send: list[str]
    receive: list[str]
    send_names: list[str]
    receive_names: list[str]
    my_gain: float
    their_gain: float
    my_before: float
    my_after: float
    depth_delta: int

    @property
    def joint_gain(self) -> float:
        return self.my_gain + self.their_gain

    def as_row(self) -> dict[str, object]:
        return {
            "partner": self.partner,
            "send": ", ".join(self.send_names),
            "receive": ", ".join(self.receive_names),
            "my_gain": round(self.my_gain, 1),
            "their_gain": round(self.their_gain, 1),
            "joint": round(self.joint_gain, 1),
            "spots": self.depth_delta,
        }


@dataclass
class FinderConfig:
    """Bounds on the search. Defaults keep a full league scan to a few seconds."""

    # Only consider each side's most valuable N players as trade chips.
    max_candidates: int = 12
    # Largest package on each side.
    max_send: int = 2
    max_receive: int = 2
    # A trade must clear this many rest-of-season points for me to surface it.
    min_my_gain: float = 5.0
    # And must not obviously hurt the other manager, or they will decline.
    min_their_gain: float = 0.0
    # The partner's gain must be at least this fraction of mine. A trade where I
    # gain 125 and they gain 5 is technically mutual and will still be rejected --
    # nobody accepts a deal that barely moves their lineup while transforming
    # yours. This is what keeps the list to proposals worth actually sending.
    min_their_gain_ratio: float = 0.25
    top_n: int = 15
    value_col: str = "ros_points"
    exclude: set[str] = field(default_factory=set)


def _packages(ids: list[str], max_size: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for size in range(1, max_size + 1):
        out.extend(combinations(ids, size))
    return out


def _swap(
    roster: pl.DataFrame, all_players: pl.DataFrame, out_ids: tuple[str, ...],
    in_ids: tuple[str, ...], id_col: str,
) -> pl.DataFrame:
    kept = roster.filter(~pl.col(id_col).is_in(list(out_ids)))
    incoming = all_players.filter(pl.col(id_col).is_in(list(in_ids)))
    if incoming.is_empty():
        return kept
    shared = [c for c in kept.columns if c in incoming.columns]
    return pl.concat([kept.select(shared), incoming.select(shared)], how="vertical")


def find_trades(
    my_roster: pl.DataFrame,
    opponent_rosters: dict[str, pl.DataFrame],
    all_players: pl.DataFrame,
    league: LeagueConfig,
    config: FinderConfig | None = None,
) -> list[TradeIdea]:
    """Search every opponent for mutually beneficial trades.

    `opponent_rosters` maps a display name to that manager's roster frame. All
    frames must carry the value column (rest-of-season points by default).
    """
    config = config or FinderConfig()
    value_col = config.value_col
    if my_roster.is_empty() or value_col not in my_roster.columns:
        return []

    id_col = "sleeper_id" if "sleeper_id" in my_roster.columns else "gsis_id"
    name_col = "name" if "name" in my_roster.columns else id_col

    def _chips(roster: pl.DataFrame) -> list[str]:
        pool = roster.filter(pl.col(id_col).is_not_null())
        if config.exclude:
            pool = pool.filter(~pl.col(id_col).is_in(list(config.exclude)))
        return (
            pool.sort(value_col, descending=True, nulls_last=True)
            .head(config.max_candidates)[id_col]
            .to_list()
        )

    lookup = dict(
        zip(
            all_players[id_col].to_list(),
            all_players[name_col].to_list(),
            strict=False,
        )
    )

    my_before, _ = optimal_lineup(my_roster, league, value_col)
    my_chips = _chips(my_roster)
    my_packages = _packages(my_chips, config.max_send)

    ideas: list[TradeIdea] = []

    for partner, their_roster in opponent_rosters.items():
        if their_roster.is_empty() or value_col not in their_roster.columns:
            continue
        their_before, _ = optimal_lineup(their_roster, league, value_col)
        their_packages = _packages(_chips(their_roster), config.max_receive)

        for send in my_packages:
            for receive in their_packages:
                # A trade whose two sides are wildly different in size is a
                # roster-crunch problem, not a trade.
                if abs(len(send) - len(receive)) > 1:
                    continue

                my_after, _ = optimal_lineup(
                    _swap(my_roster, all_players, send, receive, id_col), league, value_col
                )
                my_gain = my_after - my_before
                if my_gain < config.min_my_gain:
                    continue

                their_after, _ = optimal_lineup(
                    _swap(their_roster, all_players, receive, send, id_col),
                    league, value_col,
                )
                their_gain = their_after - their_before
                if their_gain < config.min_their_gain:
                    continue
                if their_gain < config.min_their_gain_ratio * my_gain:
                    continue

                ideas.append(
                    TradeIdea(
                        partner=partner,
                        send=list(send),
                        receive=list(receive),
                        send_names=[lookup.get(i, i) for i in send],
                        receive_names=[lookup.get(i, i) for i in receive],
                        my_gain=my_gain,
                        their_gain=their_gain,
                        my_before=my_before,
                        my_after=my_after,
                        depth_delta=len(receive) - len(send),
                    )
                )

    # Best for me first; joint gain breaks ties toward trades they will accept.
    # The ratio filter above has already removed proposals too lopsided to send.
    ideas.sort(key=lambda t: (t.my_gain, t.joint_gain), reverse=True)
    return _dedupe(ideas)[: config.top_n]


def _dedupe(ideas: list[TradeIdea]) -> list[TradeIdea]:
    """Drop near-duplicates that differ only by a throw-in.

    Proposing five variants of the same swap is noise; keep the best per
    (partner, primary player received) pair.
    """
    seen: set[tuple[str, str]] = set()
    out: list[TradeIdea] = []
    for idea in ideas:
        key = (idea.partner, idea.receive[0] if idea.receive else "")
        if key in seen:
            continue
        seen.add(key)
        out.append(idea)
    return out


def to_frame(ideas: list[TradeIdea]) -> pl.DataFrame:
    if not ideas:
        return pl.DataFrame(
            schema={"partner": pl.Utf8, "send": pl.Utf8, "receive": pl.Utf8,
                    "my_gain": pl.Float64, "their_gain": pl.Float64,
                    "joint": pl.Float64, "spots": pl.Int64}
        )
    return pl.DataFrame([i.as_row() for i in ideas])
