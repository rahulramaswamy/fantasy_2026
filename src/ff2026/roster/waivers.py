"""The waiver wire.

An add/drop is a trade with the free-agent pool, so it is judged the same way a
trade is: by what it does to the starting lineup over the rest of the season,
not by comparing two raw point totals. That framing answers the two questions a
waiver claim actually raises:

  * **Who is worth adding?** Not the free agent with the most ROS points, but
    the one whose points land in a slot you would otherwise fill worse. A
    fourth good WR behind three better ones adds nothing on Sunday.
  * **Who can go?** The player whose removal costs the lineup the least -- and
    among bench players, whose removal costs the fewest rest-of-season points
    of insurance.

Every candidate move is scored on both: `lineup_gain` (starting-lineup ROS
points) and `depth_gain` (raw ROS points swapped onto the bench). Moves are
ranked on lineup gain first, because that is the part that scores; depth gain
breaks ties and is what you are buying when the lineup gain is zero.

Sleeper's trending-adds feed is shown alongside as a *market* read: a player
many managers are grabbing will not clear waivers uncontested, which matters
for FAAB bids and claim priority even though it says nothing about his value.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..config import LeagueConfig
from ..draft.value import PROJECTABLE
from ..trades.evaluate import apply_trade, optimal_lineup


@dataclass
class WaiverMove:
    add_id: str
    add_name: str
    add_pos: str
    drop_id: str | None
    drop_name: str | None
    drop_pos: str | None
    lineup_gain: float
    depth_gain: float
    add_ros: float
    drop_ros: float
    trending_adds: int = 0
    add_status: str | None = None

    @property
    def score(self) -> tuple[float, float]:
        return (round(self.lineup_gain, 1), round(self.depth_gain, 1))


@dataclass
class WaiverConfig:
    # Free agents considered per position, by rest-of-season points.
    per_position: int = 10
    # Only surface moves that gain at least this many ROS points somewhere.
    min_gain: float = 1.0
    top_n: int = 15
    value_col: str = "ros_points"
    # Never propose dropping these (your keepers, or players you just will not cut).
    protect: frozenset[str] = frozenset()


def free_agents(board: pl.DataFrame, rostered_ids: set[str]) -> pl.DataFrame:
    """Every projectable player on the board not on any roster in the league."""
    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"
    return board.filter(
        pl.col(id_col).is_not_null()
        & ~pl.col(id_col).is_in(list(rostered_ids))
        & pl.col("position").is_in(list(PROJECTABLE))
    )


def rostered_in_league(rosters: list[dict]) -> set[str]:
    """All Sleeper ids held by any team, including IR and taxi."""
    ids: set[str] = set()
    for roster in rosters:
        for key in ("players", "reserve", "taxi"):
            ids.update(str(p) for p in (roster.get(key) or []))
    return ids


def marginal_values(
    my_roster: pl.DataFrame, league: LeagueConfig, value_col: str = "ros_points"
) -> pl.DataFrame:
    """How much each player on my roster is worth to the starting lineup.

    `marginal` is the lineup's loss if the player vanished. Starters have a
    positive marginal (at least the gap to whoever would replace them); pure
    bench players have zero, and among those the lowest `value_col` is the
    natural drop.
    """
    id_col = "sleeper_id" if "sleeper_id" in my_roster.columns else "gsis_id"
    base, _ = optimal_lineup(my_roster, league, value_col)
    rows = []
    for pid in my_roster[id_col].to_list():
        without = my_roster.filter(pl.col(id_col) != pid)
        value, _ = optimal_lineup(without, league, value_col)
        rows.append({id_col: pid, "marginal": round(base - value, 1)})
    if not rows:
        return my_roster.with_columns(pl.lit(0.0).alias("marginal"))
    return my_roster.join(pl.DataFrame(rows), on=id_col, how="left").sort(
        ["marginal", value_col], descending=[False, False], nulls_last=True
    )


def find_moves(
    my_roster: pl.DataFrame,
    pool: pl.DataFrame,
    league: LeagueConfig,
    config: WaiverConfig | None = None,
    trending: dict[str, int] | None = None,
    roster_full: bool = True,
) -> list[WaiverMove]:
    """Rank add/drop pairs by what they do to my rest-of-season lineup.

    `pool` is the free-agent frame. When `roster_full` is False an add needs no
    corresponding drop, and the best pure adds are reported with `drop_id=None`.
    """
    config = config or WaiverConfig()
    value_col = config.value_col
    trending = trending or {}
    if pool.is_empty() or value_col not in pool.columns:
        return []

    id_col = "sleeper_id" if "sleeper_id" in pool.columns else "gsis_id"
    candidates = (
        pool.filter(pl.col(value_col).is_not_null())
        .sort(value_col, descending=True)
        .group_by("position", maintain_order=True)
        .head(config.per_position)
    )

    before, _ = optimal_lineup(my_roster, league, value_col)
    droppable = my_roster
    if config.protect:
        droppable = droppable.filter(~pl.col(id_col).is_in(list(config.protect)))
    drop_rows = droppable.to_dicts() if roster_full else []
    # The weakest few by value are the only realistic drops; evaluating every
    # starter as a drop for every add is wasted work and produces silly pairs.
    drop_rows.sort(key=lambda r: float(r.get(value_col) or 0.0))
    drop_rows = drop_rows[: max(6, len(drop_rows) // 2)]

    moves: list[WaiverMove] = []
    for add in candidates.to_dicts():
        add_id = str(add[id_col])
        add_ros = float(add.get(value_col) or 0.0)
        options: list[tuple[str | None, dict | None]] = (
            [(None, None)] if not roster_full else []
        )
        options.extend((str(r[id_col]), r) for r in drop_rows)

        for drop_id, drop in options:
            after_roster = apply_trade(
                my_roster, pool, [drop_id] if drop_id else [], [add_id]
            )
            after, _ = optimal_lineup(after_roster, league, value_col)
            drop_ros = float((drop or {}).get(value_col) or 0.0)
            lineup_gain = after - before
            depth_gain = add_ros - drop_ros
            if max(lineup_gain, depth_gain) < config.min_gain:
                continue
            moves.append(
                WaiverMove(
                    add_id=add_id,
                    add_name=str(add.get("name") or add_id),
                    add_pos=str(add.get("position") or "?"),
                    drop_id=drop_id,
                    drop_name=str(drop.get("name") or drop_id) if drop else None,
                    drop_pos=str(drop.get("position") or "?") if drop else None,
                    lineup_gain=lineup_gain,
                    depth_gain=depth_gain,
                    add_ros=add_ros,
                    drop_ros=drop_ros,
                    trending_adds=int(trending.get(add_id, 0)),
                    add_status=add.get("injury_status"),
                )
            )

    moves.sort(key=lambda m: m.score, reverse=True)
    # One row per free agent: the best drop for him. Five variants of the same
    # pickup with different cuts are noise.
    seen: set[str] = set()
    out: list[WaiverMove] = []
    for move in moves:
        if move.add_id in seen:
            continue
        seen.add(move.add_id)
        out.append(move)
        if len(out) >= config.top_n:
            break
    return out


def trending_counts(payload: list[dict]) -> dict[str, int]:
    """Sleeper's trending feed as {sleeper_id: adds in the lookback window}."""
    return {str(p.get("player_id")): int(p.get("count") or 0) for p in payload or []}
