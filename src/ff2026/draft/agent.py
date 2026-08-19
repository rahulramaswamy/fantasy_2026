"""Pick recommendation.

The question at a draft pick is never "who is the best player available". It is
"which choice leaves me best off two picks from now", because the cost of taking
a receiver is the running back who won't survive until your next turn.

So the agent scores each available player by opportunity cost:

    marginal value = VORP(player) - E[VORP of the best player at his position
                                     still available at my next pick]

The expectation uses ADP and its standard deviation to model survival. A player
whose ADP is far past your next pick will probably still be there, so taking him
now is wasted urgency; a player right at the turn is the one to grab.

Roster need then scales that value: the tenth-best QB in a one-QB league is worth
very little to a team that already has two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from ..config import SLOT_ELIGIBILITY, LeagueConfig
from .value import PROJECTABLE


@dataclass
class AgentConfig:
    """Weights governing draft behaviour."""

    # Value placed on bench depth once starting slots at a position are filled.
    depth_weight: float = 0.55
    # Value once a position is at its practical roster cap.
    surplus_weight: float = 0.10
    # Practical maximum worth rostering, by position, in a standard league.
    position_caps: dict[str, int] = None  # type: ignore[assignment]
    # Weight on upside (ceiling) vs expected value. Rises in later rounds.
    late_round_upside: float = 0.35
    # Rounds after which upside starts to matter more than floor.
    upside_from_round: int = 8
    # If ADP is missing, assume this much uncertainty (in picks).
    default_adp_sd: float = 12.0

    def __post_init__(self) -> None:
        if self.position_caps is None:
            self.position_caps = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}


def survival_probability(adp: float | None, adp_sd: float | None, pick: int,
                         default_sd: float = 12.0) -> float:
    """P(player is still on the board at `pick`), from ADP and its spread.

    Treats draft position as roughly normal around ADP. Crude, but it is
    calibrated against thousands of real drafts, which beats a hand-wave.
    """
    if adp is None:
        return 0.5
    sd = adp_sd if adp_sd and adp_sd > 0 else default_sd
    # P(draft_position > pick) with a continuity correction.
    z = (pick - 0.5 - adp) / sd
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def expected_best_available(
    pool: pl.DataFrame, pick: int, value_col: str = "vorp", default_sd: float = 12.0
) -> float:
    """Expected value of the best player left in `pool` at pick number `pick`.

    Walks the pool best-first: player i is the best survivor if he survives and
    everyone ahead of him does not. Assumes independence across players, which
    slightly understates the tail -- real drafts have positional runs.

    The candidate being scored MUST stay in this pool. It is tempting to remove
    him ("you cannot draft him now and also find him later"), but the baseline
    represents the world in which you did *not* draft him -- and in that world
    he is still on the board. Removing him inflates the score of exactly the
    players who least need drafting now: a quarterback who is 98% to survive
    until your next pick should score ~0 for taking him early, because passing
    costs you almost nothing. Excluding him instead scores him as though he
    would vanish, which recommends spending a premium pick on a player you
    would get for free.
    """
    if pool.is_empty():
        return 0.0

    ranked = pool.sort(value_col, descending=True)
    expected = 0.0
    prob_all_gone = 1.0

    for row in ranked.iter_rows(named=True):
        surv = survival_probability(row.get("adp"), row.get("adp_stdev"), pick, default_sd)
        expected += float(row[value_col]) * surv * prob_all_gone
        prob_all_gone *= 1.0 - surv
        if prob_all_gone < 0.01:
            break

    return expected


def positional_need(
    roster: dict[str, int], league: LeagueConfig, config: AgentConfig
) -> dict[str, float]:
    """A 0-1 multiplier per position reflecting how badly the roster needs one.

    Starting slots come first, then flex, then bench depth, then surplus.
    """
    need: dict[str, float] = {}

    # Remaining dedicated starting slots.
    remaining_dedicated = {
        pos: max(0, league.starters_at(pos) - roster.get(pos, 0)) for pos in PROJECTABLE
    }

    # Flex slots any of the eligible positions could fill.
    flex_capacity = 0
    flex_eligible: set[str] = set()
    for slot, count in league.flex_slots().items():
        flex_capacity += count
        flex_eligible |= {p for p in SLOT_ELIGIBILITY.get(slot, ()) if p in PROJECTABLE}
    surplus_flex_players = sum(
        max(0, roster.get(pos, 0) - league.starters_at(pos)) for pos in flex_eligible
    )
    flex_open = max(0, flex_capacity - surplus_flex_players)

    for pos in PROJECTABLE:
        cap = config.position_caps.get(pos, 6)
        held = roster.get(pos, 0)

        if held >= cap:
            need[pos] = config.surplus_weight
        elif remaining_dedicated[pos] > 0 or pos in flex_eligible and flex_open > 0:
            need[pos] = 1.0
        else:
            need[pos] = config.depth_weight

    return need


def recommend(
    available: pl.DataFrame,
    league: LeagueConfig,
    roster: dict[str, int],
    current_pick: int,
    next_pick: int | None,
    config: AgentConfig | None = None,
    top_n: int = 10,
) -> pl.DataFrame:
    """Rank available players by opportunity-cost-adjusted value.

    `available` needs at least: position, vorp, proj_points, and ideally adp,
    adp_stdev and ceiling.
    """
    config = config or AgentConfig()
    if available.is_empty():
        return available

    rnd = ((current_pick - 1) // max(league.teams, 1)) + 1
    need = positional_need(roster, league, config)

    # What each position is expected to offer at my next turn.
    if next_pick is None:
        # Final pick of the draft: nothing to wait for, so cost is zero.
        vona_by_pos = dict.fromkeys(PROJECTABLE, 0.0)
    else:
        vona_by_pos = {
            pos: expected_best_available(
                available.filter(pl.col("position") == pos),
                next_pick,
                default_sd=config.default_adp_sd,
            )
            for pos in PROJECTABLE
        }

    df = available.with_columns(
        pl.col("position").replace_strict(vona_by_pos, default=0.0)
        .cast(pl.Float64).alias("vona_baseline"),
        pl.col("position").replace_strict(need, default=1.0)
        .cast(pl.Float64).alias("need_mult"),
    )

    # Opportunity cost: value now minus what the position is likely to offer later.
    df = df.with_columns(
        (pl.col("vorp") - pl.col("vona_baseline")).alias("marginal_value")
    )

    # Late in the draft, swing for upside -- a replacement-level bench player is
    # worth nothing, so variance is free.
    if "ceiling" in df.columns and rnd >= config.upside_from_round:
        w = config.late_round_upside
        df = df.with_columns(
            (
                (1 - w) * pl.col("marginal_value")
                + w * (pl.col("ceiling") - pl.col("replacement_points"))
            ).alias("marginal_value")
        )

    df = df.with_columns(
        (pl.col("marginal_value") * pl.col("need_mult")).alias("pick_score")
    )

    if next_pick is not None:
        df = df.with_columns(
            pl.struct(["adp", "adp_stdev"])
            .map_elements(
                lambda s: survival_probability(
                    s["adp"], s["adp_stdev"], next_pick, config.default_adp_sd
                ),
                return_dtype=pl.Float64,
            )
            .alias("survives_to_next")
        )
    else:
        # Null, not zero. Zero means "certainly gone"; this is "unknown", which
        # is what we have before draft slots are assigned. Rendering unknown as
        # 0% tells the drafter the exact opposite of the truth.
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("survives_to_next"))

    return df.sort("pick_score", descending=True).head(top_n)


def explain(row: dict, current_pick: int, next_pick: int | None) -> str:
    """One line of plain English for why a player is the recommendation."""
    name = row.get("name") or row.get("sleeper_name") or "player"
    pos = row.get("position", "?")
    vorp = row.get("vorp") or 0.0
    surv = row.get("survives_to_next")
    adp = row.get("adp")

    bits = [f"{name} ({pos}) - {vorp:.0f} pts over replacement"]
    if adp is not None:
        delta = current_pick - adp
        if delta > 6:
            bits.append(f"falling: ADP {adp:.0f}, on the board at {current_pick}")
        elif delta < -12:
            bits.append(f"a reach on ADP ({adp:.0f}) but the model likes him")
    if next_pick is not None and surv is not None:
        if surv < 0.25:
            bits.append(f"only {surv:.0%} to last until pick {next_pick} - take him now")
        elif surv > 0.7:
            bits.append(f"{surv:.0%} to last until {next_pick} - you can wait")
    return "; ".join(bits)
