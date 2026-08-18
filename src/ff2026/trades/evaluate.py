"""Trade evaluation.

A trade is not a comparison of two piles of players. It is a comparison of two
*starting lineups over the remaining weeks*, because points scored by your bench
are worth nothing. Trading your RB3 for someone's RB1 is a big gain even if the
raw point totals look close, and trading two starters for one is usually a loss
even when the "value" columns say otherwise.

So we evaluate on lineup value: fill your best legal starting lineup before and
after the trade, and compare. Market value is reported alongside, not instead --
it tells you whether the other manager is likely to accept.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import BENCH_SLOTS, SLOT_ELIGIBILITY, LeagueConfig


@dataclass
class TradeSide:
    """One manager's half of a proposal."""

    label: str
    sends: list[str]  # sleeper_ids going out
    receives: list[str]  # sleeper_ids coming in


def optimal_lineup(
    roster: pl.DataFrame, league: LeagueConfig, value_col: str = "proj_points"
) -> tuple[float, pl.DataFrame]:
    """Fill the best legal starting lineup and return its value.

    Slots are filled most-constrained first (dedicated positions before flex), so
    a flex slot never steals a player that only a dedicated slot could have used.
    """
    if roster.is_empty():
        return 0.0, roster

    slots = [s for s in league.roster_positions if s not in BENCH_SLOTS]
    # Fewest eligible positions first => most constrained first.
    slots.sort(key=lambda s: len(SLOT_ELIGIBILITY.get(s, (s,))))

    pool = roster.sort(value_col, descending=True)
    used: set[str] = set()
    chosen: list[dict] = []

    for slot in slots:
        eligible = SLOT_ELIGIBILITY.get(slot, (slot,))
        for row in pool.iter_rows(named=True):
            key = str(row.get("sleeper_id") or row.get("gsis_id") or row.get("name"))
            if key in used:
                continue
            if row.get("position") in eligible:
                used.add(key)
                chosen.append({**row, "slot": slot})
                break

    if not chosen:
        return 0.0, pl.DataFrame()

    starters = pl.DataFrame(chosen, infer_schema_length=None)
    return float(starters[value_col].sum()), starters


def apply_trade(
    roster: pl.DataFrame, all_players: pl.DataFrame, sends: list[str], receives: list[str]
) -> pl.DataFrame:
    """Return the roster as it would look after the trade."""
    id_col = "sleeper_id" if "sleeper_id" in roster.columns else "gsis_id"
    kept = roster.filter(~pl.col(id_col).is_in(sends))
    incoming = all_players.filter(pl.col(id_col).is_in(receives))
    if incoming.is_empty():
        return kept
    shared = [c for c in kept.columns if c in incoming.columns]
    return pl.concat([kept.select(shared), incoming.select(shared)], how="vertical")


def evaluate_trade(
    my_roster: pl.DataFrame,
    all_players: pl.DataFrame,
    league: LeagueConfig,
    sends: list[str],
    receives: list[str],
    value_col: str = "proj_points",
    market_col: str = "market_value",
) -> dict[str, object]:
    """Score a proposed trade from your side.

    Returns lineup value before/after, the market read, and a verdict. The two
    can disagree, and when they do that disagreement is the actual information:
    a trade the market hates and your lineup loves is exactly the one to make.
    """
    before_value, before_lineup = optimal_lineup(my_roster, league, value_col)
    after_roster = apply_trade(my_roster, all_players, sends, receives)
    after_value, after_lineup = optimal_lineup(after_roster, league, value_col)

    id_col = "sleeper_id" if "sleeper_id" in all_players.columns else "gsis_id"
    out_df = all_players.filter(pl.col(id_col).is_in(sends))
    in_df = all_players.filter(pl.col(id_col).is_in(receives))

    def _sum(df: pl.DataFrame, col: str) -> float:
        if df.is_empty() or col not in df.columns:
            return 0.0
        return float(df[col].fill_null(0).sum())

    market_out = _sum(out_df, market_col)
    market_in = _sum(in_df, market_col)

    lineup_delta = after_value - before_value
    market_delta = market_in - market_out

    # Depth cost: giving up more bodies than you get back thins the roster, which
    # matters for bye weeks and injuries even when the starting lineup improves.
    depth_delta = len(receives) - len(sends)

    if lineup_delta > 5 and market_delta >= -0.1 * max(market_out, 1):
        verdict = "accept"
    elif lineup_delta > 5:
        verdict = "accept (you win on lineup, market disagrees -- they may too)"
    elif lineup_delta < -5:
        verdict = "decline"
    else:
        verdict = "roughly neutral"

    return {
        "lineup_before": round(before_value, 1),
        "lineup_after": round(after_value, 1),
        "lineup_delta": round(lineup_delta, 1),
        "market_out": round(market_out, 1),
        "market_in": round(market_in, 1),
        "market_delta": round(market_delta, 1),
        "depth_delta": depth_delta,
        "verdict": verdict,
        "starters_before": before_lineup,
        "starters_after": after_lineup,
        "sending": out_df,
        "receiving": in_df,
    }
