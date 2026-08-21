"""`ff` -- command line interface.

Draft-night commands are the point of this tool, so they are the ones designed
for speed: `ff draft live` runs a polling loop that watches the Sleeper draft and
reprints a recommendation every time a pick is made, so there is nothing to type
while you are on the clock.
"""

from __future__ import annotations

import time
import warnings

import polars as pl
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import pipeline
from .config import LeagueConfig, get_settings
from .data.sleeper import SleeperClient
from .draft.agent import AgentConfig, explain, recommend
from .draft.board import DraftState, picks_frame, roster_counts
from .draft.value import tier_breaks
from .model.benchmark import benchmark
from .model.benchmark import summarize as bench_summarize
from .model.evaluate import backtest, coverage, summarize
from .model.projections import ProjectionConfig, fit_age_curve
from .power import power_table, read_the_table, team_strengths
from .roster.lineup import flags, set_lineup, weekly_values
from .roster.waivers import (
    WaiverConfig,
    find_moves,
    free_agents,
    marginal_values,
    rostered_in_league,
    trending_counts,
)
from .scoring import ScoringEngine
from .trades.evaluate import evaluate_trade
from .trades.finder import FinderConfig, find_trades

app = typer.Typer(
    help="Fantasy football model, draft agent and trade evaluator.", no_args_is_help=True
)
league_app = typer.Typer(help="League configuration.", no_args_is_help=True)
data_app = typer.Typer(help="Upstream data.", no_args_is_help=True)
board_app = typer.Typer(help="Build and inspect the draft board.", no_args_is_help=True)
model_app = typer.Typer(help="Model diagnostics and backtests.", no_args_is_help=True)
draft_app = typer.Typer(help="Live draft assistance.", no_args_is_help=True)
trade_app = typer.Typer(help="Trade evaluation.", no_args_is_help=True)
sleeper_app = typer.Typer(help="Raw Sleeper API helpers.", no_args_is_help=True)
roster_app = typer.Typer(help="Your roster and this week's lineup.", no_args_is_help=True)
waiver_app = typer.Typer(help="Waiver wire: who to add, who to drop.", no_args_is_help=True)

app.add_typer(league_app, name="league")
app.add_typer(data_app, name="data")
app.add_typer(board_app, name="board")
app.add_typer(model_app, name="model")
app.add_typer(draft_app, name="draft")
app.add_typer(trade_app, name="trade")
app.add_typer(sleeper_app, name="sleeper")
app.add_typer(roster_app, name="roster")
app.add_typer(waiver_app, name="waiver")

console = Console()

POS_COLORS = {"QB": "magenta", "RB": "green", "WR": "cyan", "TE": "yellow"}


def _fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _board_table(df: pl.DataFrame, title: str, limit: int = 25) -> Table:
    table = Table(title=title, header_style="bold")
    for col, justify in (
        ("#", "right"), ("Player", "left"), ("Pos", "left"), ("Tm", "left"),
        ("Proj", "right"), ("VORP", "right"), ("ADP", "right"), ("Floor", "right"),
        ("Ceil", "right"),
    ):
        table.add_column(col, justify=justify)

    for i, row in enumerate(df.head(limit).iter_rows(named=True), start=1):
        pos = row.get("position") or "?"
        table.add_row(
            str(i),
            str(row.get("name") or "?"),
            f"[{POS_COLORS.get(pos, 'white')}]{pos}[/]",
            str(row.get("team") or "-"),
            _fmt(row.get("proj_points"), 0),
            _fmt(row.get("vorp"), 0),
            _fmt(row.get("adp"), 1),
            _fmt(row.get("floor"), 0),
            _fmt(row.get("ceiling"), 0),
        )
    return table


def _load_league(path: str | None) -> LeagueConfig:
    try:
        return LeagueConfig.load(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc




def _resolve_league_id(league_id: str | None) -> str:
    settings = get_settings()
    lid = league_id or settings.sleeper_league_id
    if not lid:
        console.print("[red]Pass --league-id or set SLEEPER_LEAGUE_ID in .env[/red]")
        raise typer.Exit(1)
    return lid


def _current_week(default: int | None = None) -> int:
    """The regular-season week we are in, for rest-of-season maths.

    Sleeper's `week` counts preseason weeks too (`season_type: "pre"`, week 2
    in mid-August), and taking that literally would tell the model two
    regular-season games have been played and everyone missed them. Before the
    regular season the answer is week 0: nothing has happened yet.
    """
    if default is not None:
        return default
    with SleeperClient() as client:
        state = client.state()
    return regular_season_week(state)


def regular_season_week(state: dict) -> int:
    season_type = str(state.get("season_type") or "regular")
    week = int(state.get("week") or 1)
    if season_type == "pre":
        return 0
    if season_type == "post":
        return 18
    return week


def _ros_board(league_cfg, week: int | None, season: int | None):
    """Load the board with rest-of-season projections attached.

    Before week 1 there are no stats to fold in, which is normal rather than a
    problem, so the underlying warning is turned into a plain note.
    """
    wk = _current_week(week)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        board = pipeline.build_ros_board(league_cfg, wk, season=season)
    for warning in caught:
        message = str(warning.message)
        if "not yet started" in message:
            console.print(
                "[dim]Season hasn't started -- rest-of-season equals the "
                "preseason projection until week 1 games are played.[/dim]"
            )
        else:
            console.print(f"[yellow]{message}[/yellow]")
    return board, wk


def _resolve_me(client: SleeperClient, username: str | None) -> tuple[str, str]:
    """Work out which Sleeper user we are. Returns (user_id, label).

    Precedence is deliberate: an explicit --username always wins over anything
    stored in .env, because the flag is the more specific instruction. A stored
    SLEEPER_USER_ID that has gone stale should never silently override it.
    """
    settings = get_settings()

    if username:
        uid = client.resolve_user_id(username)
        if not uid:
            console.print(
                f"[red]Sleeper has no user named '{username}'.[/red]\n"
                "Usernames are not display names -- check the one you log in with."
            )
            raise typer.Exit(1)
        return uid, username

    if settings.sleeper_user_id:
        return settings.sleeper_user_id, f"user_id {settings.sleeper_user_id}"

    if settings.sleeper_username:
        uid = client.resolve_user_id(settings.sleeper_username)
        if not uid:
            console.print(
                f"[red]SLEEPER_USERNAME in .env ('{settings.sleeper_username}') "
                "is not a Sleeper user.[/red]"
            )
            raise typer.Exit(1)
        return uid, settings.sleeper_username

    console.print(
        "[red]I don't know who you are.[/red] Set SLEEPER_USERNAME in .env, "
        "or pass --username."
    )
    raise typer.Exit(1)


def _my_roster_ids(league_id: str, username: str | None) -> tuple[list[str], list, list]:
    """Find my roster in a league, with diagnostics when it isn't there.

    Failing to find a roster has three quite different causes -- wrong user,
    wrong league, or a league you are genuinely not in -- and they need
    different fixes, so the error says which one it is.
    """
    with SleeperClient() as client:
        uid, label = _resolve_me(client, username)
        league = client.league(league_id)
        rosters = client.league_rosters(league_id)
        users = client.league_users(league_id)

    if not league:
        console.print(
            f"[red]No Sleeper league with id {league_id}.[/red]\n"
            "Check SLEEPER_LEAGUE_ID in .env, or run "
            "`ff league sync --username <you>` to pick the right one."
        )
        raise typer.Exit(1)

    def _owns(roster: dict) -> bool:
        if str(roster.get("owner_id")) == str(uid):
            return True
        # Co-owned teams list additional managers separately.
        return str(uid) in {str(c) for c in (roster.get("co_owners") or [])}

    mine = next((r for r in rosters if _owns(r)), None)
    if mine:
        return [str(p) for p in (mine.get("players") or [])], rosters, users

    league_name = league.get("name", "?")
    season = league.get("season", "?")
    managers = sorted(
        (u.get("display_name") or u.get("user_id") or "?") for u in users
    )
    console.print(
        f"[red]{label} is not in '{league_name}' ({season}).[/red]\n"
        f"Resolved user_id: {uid}\n"
        f"Managers in that league: {', '.join(managers) or '(none)'}\n\n"
        "Most likely SLEEPER_LEAGUE_ID points at a different (or older) league. "
        "Run [bold]ff league sync --username <you>[/bold] to list your leagues "
        "and write the right id."
    )
    raise typer.Exit(1)


# --------------------------------------------------------------------- league


@league_app.command("sync")
def league_sync(
    league_id: str | None = typer.Option(None, help="Sleeper league id."),
    username: str | None = typer.Option(None, help="Sleeper username, to find your leagues."),
    season: int = typer.Option(2026, help="Season."),
    out: str = typer.Option("configs/league.yaml", help="Where to write the config."),
) -> None:
    """Pull your league's real scoring and roster rules from Sleeper."""
    settings = get_settings()
    league_id = league_id or settings.sleeper_league_id
    username = username or settings.sleeper_username

    with SleeperClient() as client:
        if not league_id:
            if not username:
                console.print("[red]Provide --league-id or --username.[/red]")
                raise typer.Exit(1)
            user_id = client.resolve_user_id(username)
            if not user_id:
                console.print(f"[red]No Sleeper user '{username}'.[/red]")
                raise typer.Exit(1)
            leagues = client.user_leagues(user_id, season)
            if not leagues:
                console.print(f"[red]No {season} leagues for {username}.[/red]")
                raise typer.Exit(1)
            table = Table(title=f"{username}'s {season} leagues")
            for col in ("#", "League", "ID", "Teams"):
                table.add_column(col)
            for i, lg in enumerate(leagues, 1):
                table.add_row(str(i), lg.get("name", "?"), lg.get("league_id", "?"),
                              str((lg.get("settings") or {}).get("num_teams", "?")))
            console.print(table)
            if len(leagues) == 1:
                league_id = leagues[0]["league_id"]
                console.print(f"[green]Using the only league: {league_id}[/green]")
            else:
                choice = typer.prompt("Which league number?", type=int)
                league_id = leagues[choice - 1]["league_id"]

        payload = client.league(league_id)

    if not payload:
        console.print(f"[red]League {league_id} not found.[/red]")
        raise typer.Exit(1)

    cfg = LeagueConfig.from_sleeper(payload)
    path = cfg.save(out)
    console.print(f"[green]Wrote {path}[/green]")
    _print_league(cfg)


@league_app.command("show")
def league_show(config: str | None = typer.Option(None, "--config")) -> None:
    """Show the active league configuration and how scoring is interpreted."""
    _print_league(_load_league(config))


def _print_league(cfg: LeagueConfig) -> None:
    engine = ScoringEngine(cfg)
    unsupported = engine.static_unsupported()
    lines = [
        f"[bold]{cfg.name}[/bold]  ({cfg.teams} teams, {cfg.season})",
        f"Starters : {', '.join(cfg.starting_slots)}",
        f"Bench    : {cfg.bench_size}   Roster size: {cfg.roster_size}",
        f"PPR      : {cfg.ppr}   Superflex: {cfg.superflex}",
        f"Scoring  : {len(cfg.scoring_settings)} rules",
    ]
    if unsupported:
        lines.append(
            f"[yellow]Unsupported scoring keys (need play-by-play): "
            f"{', '.join(unsupported)}[/yellow]"
        )
    else:
        lines.append("[green]All scoring rules are supported.[/green]")
    console.print(Panel("\n".join(lines), title="League"))


# ----------------------------------------------------------------------- data


@data_app.command("sync")
def data_sync(
    season: int = typer.Option(2026),
    lookback: int = typer.Option(4),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Warm every upstream cache so draft night needs no network."""
    cfg = _load_league(config)
    from .data import nflverse

    seasons = nflverse.seasons_back(season, lookback)
    steps = [
        ("weekly stats", lambda: nflverse.weekly_stats(seasons)),
        ("player master", nflverse.player_master),
        (f"{season} rosters", lambda: nflverse.rosters([season])),
        ("schedules", lambda: nflverse.schedules([season])),
        ("snap counts", lambda: nflverse.snap_counts(seasons)),
        ("injuries", lambda: nflverse.injuries(seasons)),
        ("expected points", lambda: nflverse.ff_opportunity(seasons)),
        ("id crosswalk", lambda: nflverse.ff_playerids()),
        ("draft picks", nflverse.draft_picks),
    ]
    for label, fn in steps:
        try:
            df = fn()
            console.print(f"[green]OK[/green]   {label}: {df.height:,} rows")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]FAIL[/red] {label}: {type(exc).__name__}: {exc}")

    try:
        cw = pipeline.build_crosswalk_table(force=True)
        console.print(f"[green]OK[/green]   sleeper crosswalk: {cw.height:,} players")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]FAIL[/red] sleeper crosswalk: {exc}")

    console.print(f"[dim]League: {cfg.name}[/dim]")


@data_app.command("crosswalk")
def data_crosswalk(force: bool = typer.Option(False, "--force")) -> None:
    """Check how well Sleeper ids resolve to nflverse ids."""
    from .ids import crosswalk_report

    cw = pipeline.build_crosswalk_table(force=force)
    report = crosswalk_report(cw)
    console.print(
        f"Matched [green]{report['matched']:,}[/green] / {report['total']:,} "
        f"({report['match_rate']:.1%})"
    )
    console.print(report["by_source"])
    if report["notable_unmatched"].height:
        console.print("[yellow]Most notable unmatched players:[/yellow]")
        console.print(report["notable_unmatched"])


# ---------------------------------------------------------------------- board


@board_app.command("build")
def board_build(
    season: int | None = typer.Option(None),
    lookback: int = typer.Option(4),
    no_market: bool = typer.Option(False, "--no-market", help="Skip ADP/trade-value feeds."),
    expert_weight: float = typer.Option(
        0.75, "--expert-weight",
        help="How much the board defers to expert consensus (0 = pure model).",
    ),
    force: bool = typer.Option(False, "--force", help="Bypass caches."),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Build projections and save the draft board."""
    cfg = _load_league(config)
    with console.status("Building projections..."):
        board = pipeline.build_board(
            cfg, season=season, lookback=lookback, with_market=not no_market,
            expert_weight=expert_weight, force=force,
        )
    path = pipeline.save_board(board)
    console.print(f"[green]Saved {board.height:,} players to {path}[/green]")
    console.print(_board_table(board, "Top of the board", limit=20))


@board_app.command("show")
def board_show(
    position: str | None = typer.Option(None, "--pos", help="Filter to one position."),
    limit: int = typer.Option(30),
    tiers: bool = typer.Option(False, "--tiers", help="Mark tier breaks."),
) -> None:
    """Print the saved board."""
    board = pipeline.load_board()
    if position:
        board = board.filter(pl.col("position") == position.upper())
    title = f"Draft board - {position.upper()}" if position else "Draft board"
    console.print(_board_table(board, title, limit))

    if tiers and position:
        breaks = tier_breaks(board, position.upper())
        console.print(f"[dim]Tier breaks after ranks: {breaks}[/dim]")


# ---------------------------------------------------------------------- model


@model_app.command("backtest")
def model_backtest(
    seasons: str = typer.Option("2022,2023,2024,2025", help="Comma-separated seasons."),
    lookback: int = typer.Option(4),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Score the model against the naive 'last season's PPG' baseline."""
    cfg = _load_league(config)
    season_list = [int(s) for s in seasons.split(",") if s.strip()]
    history_start = min(season_list) - lookback

    with console.status("Loading history..."):
        totals = pipeline.build_totals(
            cfg, max(season_list) + 1, lookback=lookback + len(season_list)
        )
        totals = totals.filter(pl.col("season") >= history_start)

    results = backtest(totals, season_list, ProjectionConfig(lookback=lookback))
    if results.is_empty():
        console.print("[red]Not enough data to backtest.[/red]")
        raise typer.Exit(1)

    console.print(summarize(results))
    console.print("\n[dim]Coverage (players the model can rank vs the baseline):[/dim]")
    for season in season_list:
        console.print(f"  {coverage(totals, season)}")


@model_app.command("benchmark")
def model_benchmark(
    seasons: str = typer.Option("2022,2023,2024,2025", help="Comma-separated seasons."),
    lookback: int = typer.Option(4),
    local_dir: str | None = typer.Option(
        None, "--local-dir", help="Path to a dynastyprocess/data checkout (offline use)."
    ),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Compare this model against expert consensus and a naive baseline.

    This is the honest test: the real alternative to the model is a free expert
    ranking, not a strawman. Rank metrics only -- expert consensus has no points.
    """
    cfg = _load_league(config)
    season_list = [int(s) for s in seasons.split(",") if s.strip()]

    with console.status("Loading history and expert rankings..."):
        totals = pipeline.build_totals(
            cfg, max(season_list) + 1, lookback=lookback + len(season_list)
        )
        results = benchmark(
            totals, season_list, ProjectionConfig(lookback=lookback), local_dir=local_dir
        )

    if results.is_empty():
        console.print(
            "[red]No benchmark results -- expert rankings unavailable for those seasons.[/red]"
        )
        raise typer.Exit(1)

    console.print(bench_summarize(results, by_position=True))
    console.print("\n[bold]Pooled across positions:[/bold]")
    console.print(bench_summarize(results, by_position=False))
    console.print(
        "\n[dim]spearman = rank accuracy across the field; "
        "top12 = share of the true top 12 identified.[/dim]"
    )


@model_app.command("age-curve")
def model_age_curve(
    lookback: int = typer.Option(8),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Show empirical year-over-year ageing, to sanity-check the model's curves."""
    cfg = _load_league(config)
    totals = pipeline.build_totals(cfg, 2026, lookback=lookback)
    curve = fit_age_curve(totals)
    if curve.is_empty():
        console.print("[yellow]Not enough paired seasons to fit an age curve.[/yellow]")
        raise typer.Exit(0)
    console.print(curve.filter(pl.col("n") >= 10))
    console.print(
        "[dim]Positive delta = players at that age improved the next season. "
        "Survivors bias this upward at the tail: players who collapse get cut "
        "rather than playing a bad season.[/dim]"
    )


# ---------------------------------------------------------------------- draft


def _resolve_draft(client: SleeperClient, draft_id: str | None,
                   username: str | None, season: int) -> tuple[str, str | None]:
    """Return (draft_id, user_id), prompting if necessary."""
    settings = get_settings()
    draft_id = draft_id or settings.sleeper_draft_id
    try:
        user_id, _ = _resolve_me(client, username)
    except typer.Exit:
        user_id = None

    if draft_id:
        return draft_id, user_id

    if not user_id:
        console.print("[red]Provide --draft-id, or --username so I can find your drafts.[/red]")
        raise typer.Exit(1)

    drafts = client.user_drafts(user_id, season)
    if not drafts:
        # A draft belongs to a league as well as to its members, and the
        # per-user feed can come back empty (for instance before a draft is
        # scheduled, or for a league joined after creation). The league's own
        # draft list is the more reliable route when we know the league.
        settings = get_settings()
        if settings.sleeper_league_id:
            drafts = client.league_drafts(settings.sleeper_league_id)
    if not drafts:
        console.print(
            f"[red]No {season} drafts found.[/red]\n"
            "Checked your user's drafts"
            + (
                f" and league {settings.sleeper_league_id}."
                if get_settings().sleeper_league_id
                else " (set SLEEPER_LEAGUE_ID in .env to also check your league)."
            )
            + "\nIf the draft isn't scheduled in Sleeper yet, there is nothing "
            "to connect to. Pass --draft-id once it exists."
        )
        raise typer.Exit(1)
    if len(drafts) == 1:
        return drafts[0]["draft_id"], user_id

    table = Table(title="Your drafts")
    for col in ("#", "Draft ID", "Status", "Type", "Teams"):
        table.add_column(col)
    for i, d in enumerate(drafts, 1):
        table.add_row(str(i), d.get("draft_id", "?"), d.get("status", "?"),
                      d.get("type", "?"), str((d.get("settings") or {}).get("teams", "?")))
    console.print(table)
    choice = typer.prompt("Which draft number?", type=int)
    return drafts[choice - 1]["draft_id"], user_id


def _render_draft(state: DraftState, board: pl.DataFrame, cfg: LeagueConfig,
                  agent_cfg: AgentConfig, top_n: int) -> None:
    """Print the full draft-night view: status, roster, recommendations."""
    drafted = state.drafted_player_ids()
    available = board
    if "sleeper_id" in board.columns:
        available = board.filter(
            pl.col("sleeper_id").is_null() | ~pl.col("sleeper_id").is_in(list(drafted))
        )

    my_ids = state.my_roster()
    counts = roster_counts(my_ids, board) if "sleeper_id" in board.columns else {}
    current, nxt = state.my_next_two_picks()
    until = state.picks_until_my_turn()

    if state.is_complete:
        header = "[bold green]Draft complete[/bold green]"
    elif state.is_my_turn():
        header = f"[bold green]YOU ARE ON THE CLOCK -- pick {state.next_pick_overall}[/bold green]"
    elif until is not None:
        header = (
            f"Pick [bold]{state.next_pick_overall}[/bold] (round {state.current_round}) "
            f"- [yellow]{until} pick{'s' if until != 1 else ''} until your turn "
            f"(#{current})[/yellow]"
        )
    else:
        header = f"Pick {state.next_pick_overall} - no picks left for you"

    roster_line = ", ".join(f"{p}{n}" for p, n in sorted(counts.items())) or "empty"
    console.print(
        Panel(
            f"{header}\n"
            f"Your slot: {state.my_slot or '?'}   Roster: {roster_line}   "
            f"Next picks: {current or '-'}, {nxt or '-'}",
            title=f"Draft {state.draft_id} ({state.draft_type}, {state.teams} teams)",
        )
    )

    if state.is_complete:
        return

    recs = recommend(
        available, cfg, counts, current or state.next_pick_overall, nxt,
        agent_cfg, top_n=top_n,
    )
    if recs.is_empty():
        console.print("[yellow]No players available to recommend.[/yellow]")
        return

    table = Table(title="Recommendations", header_style="bold")
    for col in ("#", "Player", "Pos", "Tm", "Score", "VORP", "ADP", "P(lasts)"):
        table.add_column(col, justify="right" if col not in ("Player", "Pos", "Tm") else "left")
    for i, row in enumerate(recs.iter_rows(named=True), 1):
        pos = row.get("position") or "?"
        surv = row.get("survives_to_next")
        table.add_row(
            str(i), str(row.get("name") or "?"),
            f"[{POS_COLORS.get(pos, 'white')}]{pos}[/]",
            str(row.get("team") or "-"),
            _fmt(row.get("pick_score"), 1),
            _fmt(row.get("vorp"), 0),
            _fmt(row.get("adp"), 1),
            f"{surv:.0%}" if surv is not None else "-",
        )
    console.print(table)

    top = recs.head(1).to_dicts()[0]
    console.print(f"[bold]->[/bold] {explain(top, current or state.next_pick_overall, nxt)}")


@draft_app.command("live")
def draft_live(
    draft_id: str | None = typer.Option(None, "--draft-id"),
    username: str | None = typer.Option(None, "--username"),
    season: int = typer.Option(2026),
    slot: int | None = typer.Option(None, "--slot", help="Your draft slot, if auto-detect fails."),
    interval: float = typer.Option(5.0, "--interval", help="Seconds between polls."),
    top_n: int = typer.Option(8, "--top"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Watch a live Sleeper draft and recommend a pick every time the board moves.

    Polls the draft picks endpoint and reprints only when something changed, so
    it is quiet while you wait and loud when it is your turn.
    """
    cfg = _load_league(config)
    board = pipeline.load_board()
    agent_cfg = AgentConfig()

    with SleeperClient() as client:
        did, user_id = _resolve_draft(client, draft_id, username, season)
        draft = client.draft(did)
        if not draft:
            console.print(f"[red]Draft {did} not found.[/red]")
            raise typer.Exit(1)

        last_count = -1
        console.print(f"[dim]Polling draft {did} every {interval}s. Ctrl-C to stop.[/dim]")
        try:
            while True:
                picks = client.draft_picks(did)
                state = DraftState.from_sleeper(draft, picks, my_user_id=user_id, my_slot=slot)

                if state.my_slot is None:
                    console.print(
                        "[yellow]Could not work out your draft slot. Pass --slot N.[/yellow]"
                    )

                if len(picks) != last_count:
                    last_count = len(picks)
                    console.clear()
                    _render_draft(state, board, cfg, agent_cfg, top_n)

                if state.is_complete:
                    console.print("[green]Draft complete.[/green]")
                    break

                time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/dim]")


@draft_app.command("recommend")
def draft_recommend(
    draft_id: str | None = typer.Option(None, "--draft-id"),
    username: str | None = typer.Option(None, "--username"),
    season: int = typer.Option(2026),
    slot: int | None = typer.Option(None, "--slot"),
    top_n: int = typer.Option(10, "--top"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """One-shot recommendation for the current state of a draft."""
    cfg = _load_league(config)
    board = pipeline.load_board()
    with SleeperClient() as client:
        did, user_id = _resolve_draft(client, draft_id, username, season)
        draft = client.draft(did)
        picks = client.draft_picks(did)
    state = DraftState.from_sleeper(draft or {}, picks, my_user_id=user_id, my_slot=slot)
    _render_draft(state, board, cfg, AgentConfig(), top_n)


@draft_app.command("picks")
def draft_picks_cmd(
    draft_id: str | None = typer.Option(None, "--draft-id"),
    username: str | None = typer.Option(None, "--username"),
    season: int = typer.Option(2026),
    limit: int = typer.Option(30),
) -> None:
    """Show the picks made so far."""
    with SleeperClient() as client:
        did, _ = _resolve_draft(client, draft_id, username, season)
        picks = client.draft_picks(did)
    board = pipeline.load_board() if picks else pl.DataFrame()
    console.print(picks_frame(picks, board).tail(limit))


# ---------------------------------------------------------------------- trade


@trade_app.command("eval")
def trade_eval(
    send: str = typer.Option(..., "--send", help="Comma-separated player names you give up."),
    receive: str = typer.Option(..., "--receive", help="Comma-separated player names you get."),
    roster: str | None = typer.Option(
        None, "--roster", help="Comma-separated names on your current roster."
    ),
    league_id: str | None = typer.Option(None, "--league-id"),
    username: str | None = typer.Option(None, "--username"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Evaluate a trade by its effect on your starting lineup."""
    cfg = _load_league(config)
    board = pipeline.load_board()

    def _resolve(names: str) -> list[str]:
        ids: list[str] = []
        for raw in names.split(","):
            name = raw.strip()
            if not name:
                continue
            hit = board.filter(pl.col("name").str.to_lowercase() == name.lower())
            if hit.is_empty():
                hit = board.filter(pl.col("name").str.contains(f"(?i){name}"))
            if hit.is_empty():
                console.print(f"[red]No player matching '{name}'.[/red]")
                raise typer.Exit(1)
            ids.append(str(hit["sleeper_id"][0] or hit["gsis_id"][0]))
        return ids

    send_ids = _resolve(send)
    recv_ids = _resolve(receive)

    if roster:
        my_ids = _resolve(roster)
    else:
        # Let _resolve_me handle the .env fallback so precedence is uniform.
        lid = _resolve_league_id(league_id)
        my_ids, _, _ = _my_roster_ids(lid, username)

    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"
    my_roster = board.filter(pl.col(id_col).is_in(my_ids))
    result = evaluate_trade(my_roster, board, cfg, send_ids, recv_ids)

    console.print(
        Panel(
            f"Starting lineup: [bold]{result['lineup_before']}[/bold] -> "
            f"[bold]{result['lineup_after']}[/bold]  "
            f"([{'green' if result['lineup_delta'] >= 0 else 'red'}]"
            f"{result['lineup_delta']:+}[/])\n"
            f"Market value out {result['market_out']} / in {result['market_in']} "
            f"({result['market_delta']:+})\n"
            f"Roster spots: {result['depth_delta']:+}\n\n"
            f"[bold]Verdict: {result['verdict']}[/bold]",
            title="Trade evaluation",
        )
    )


# -------------------------------------------------------------------- sleeper


@sleeper_app.command("selftest")
def sleeper_selftest(username: str | None = typer.Option(None, "--username")) -> None:
    """Hit every Sleeper endpoint this tool relies on and report status."""
    settings = get_settings()
    username = username or settings.sleeper_username
    with SleeperClient() as client:
        results = client.selftest(username)

    table = Table(title="Sleeper API selftest")
    for col in ("Endpoint", "Status", "Detail"):
        table.add_column(col)
    for endpoint, status, detail in results:
        color = {"OK": "green", "EMPTY": "yellow", "FAIL": "red"}.get(status, "white")
        table.add_row(endpoint, f"[{color}]{status}[/]", detail)
    console.print(table)


@sleeper_app.command("drafts")
def sleeper_drafts(
    username: str | None = typer.Option(None, "--username"),
    season: int = typer.Option(2026),
) -> None:
    """List your drafts for a season."""
    settings = get_settings()
    username = username or settings.sleeper_username
    if not username:
        console.print("[red]Pass --username or set SLEEPER_USERNAME.[/red]")
        raise typer.Exit(1)
    with SleeperClient() as client:
        uid = client.resolve_user_id(username)
        if not uid:
            console.print(f"[red]No such user: {username}[/red]")
            raise typer.Exit(1)
        drafts = client.user_drafts(uid, season)

    table = Table(title=f"{username}'s {season} drafts")
    for col in ("Draft ID", "Status", "Type", "Teams", "Rounds"):
        table.add_column(col)
    for d in drafts:
        s = d.get("settings") or {}
        table.add_row(d.get("draft_id", "?"), d.get("status", "?"), d.get("type", "?"),
                      str(s.get("teams", "?")), str(s.get("rounds", "?")))
    console.print(table)


@trade_app.command("find")
def trade_find(
    league_id: str | None = typer.Option(None, "--league-id"),
    username: str | None = typer.Option(None, "--username"),
    week: int | None = typer.Option(None, "--week", help="Defaults to the current NFL week."),
    season: int | None = typer.Option(None, "--season"),
    max_send: int = typer.Option(2, "--max-send"),
    max_receive: int = typer.Option(2, "--max-receive"),
    min_gain: float = typer.Option(5.0, "--min-gain", help="Minimum ROS points I must gain."),
    top_n: int = typer.Option(15, "--top"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Search the league for trades that help you AND your trade partner.

    A trade only happens if both managers think they won, so this only surfaces
    proposals that improve both starting lineups. Valued on rest-of-season points.
    """
    cfg = _load_league(config)
    lid = _resolve_league_id(league_id)

    with console.status("Projecting rest of season..."):
        board, wk = _ros_board(cfg, week, season)
    my_ids, rosters, users = _my_roster_ids(lid, username)

    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"
    my_roster = board.filter(pl.col(id_col).is_in(my_ids))

    by_user = {u.get("user_id"): u for u in users}
    opponents = {}
    for roster in rosters:
        ids = [str(p) for p in (roster.get("players") or [])]
        if set(ids) == set(my_ids) or not ids:
            continue
        user = by_user.get(roster.get("owner_id")) or {}
        name = user.get("display_name") or f"roster {roster.get('roster_id')}"
        opponents[name] = board.filter(pl.col(id_col).is_in(ids))

    with console.status(f"Searching {len(opponents)} opponents..."):
        ideas = find_trades(
            my_roster, opponents, board, cfg,
            FinderConfig(max_send=max_send, max_receive=max_receive,
                         min_my_gain=min_gain, top_n=top_n),
        )

    if not ideas:
        console.print(
            f"[yellow]No mutually beneficial trades found at week {wk} "
            f"(min gain {min_gain}). Try --min-gain 2 or --max-send 3.[/yellow]"
        )
        return

    table = Table(title=f"Trade ideas (week {wk}, rest-of-season points)", header_style="bold")
    for col in ("#", "Partner", "You send", "You get", "You gain", "They gain", "Spots"):
        numeric = col in ("#", "You gain", "They gain", "Spots")
        table.add_column(col, justify="right" if numeric else "left")
    for i, idea in enumerate(ideas, 1):
        table.add_row(
            str(i), idea.partner, ", ".join(idea.send_names), ", ".join(idea.receive_names),
            f"[green]+{idea.my_gain:.0f}[/]", f"+{idea.their_gain:.0f}",
            f"{idea.depth_delta:+d}" if idea.depth_delta else "0",
        )
    console.print(table)
    console.print(
        "[dim]Both columns are positive by construction -- these are trades the "
        "other manager has a reason to accept, not steals.[/dim]"
    )


@board_app.command("ros")
def board_ros(
    week: int | None = typer.Option(None, "--week"),
    season: int | None = typer.Option(None, "--season"),
    position: str | None = typer.Option(None, "--pos"),
    limit: int = typer.Option(30),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Rest-of-season projections -- the number to use once the season starts."""
    cfg = _load_league(config)
    with console.status("Projecting rest of season..."):
        board, wk = _ros_board(cfg, week, season)
    if position:
        board = board.filter(pl.col("position") == position.upper())

    table = Table(title=f"Rest of season, from week {wk}", header_style="bold")
    for col in ("#", "Player", "Pos", "Tm", "ROS pts", "ROS ppg", "Gm left", "GP", "Avail"):
        table.add_column(col, justify="left" if col in ("Player", "Pos", "Tm") else "right")
    for i, row in enumerate(board.head(limit).iter_rows(named=True), 1):
        pos = row.get("position") or "?"
        avail = row.get("availability")
        table.add_row(
            str(i), str(row.get("name") or "?"),
            f"[{POS_COLORS.get(pos, 'white')}]{pos}[/]", str(row.get("team") or "-"),
            _fmt(row.get("ros_points"), 0), _fmt(row.get("ros_ppg"), 1),
            _fmt(row.get("ros_games"), 1), _fmt(row.get("games_played"), 0),
            f"{avail:.0%}" if avail is not None else "-",
        )
    console.print(table)


@league_app.command("power")
def league_power(
    league_id: str | None = typer.Option(None, "--league-id"),
    week: int | None = typer.Option(None, "--week"),
    season: int | None = typer.Option(None, "--season"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Rank every team by preseason roster, rest-of-season roster, and results.

    The gaps matter more than the ranks: a strong roster with a bad record is
    unlucky and ready to sell; a weak roster with a good record is due to regress.
    """
    cfg = _load_league(config)
    lid = _resolve_league_id(league_id)

    with console.status("Projecting rest of season..."):
        board, wk = _ros_board(cfg, week, season)
    rosters, users = pipeline.league_rosters(lid)

    strengths = team_strengths(rosters, users, board, cfg)
    df = power_table(strengths)
    if df.is_empty():
        console.print("[red]No rosters found for that league.[/red]")
        raise typer.Exit(1)

    # An undrafted league has empty rosters, which would otherwise render as a
    # tidy table of zeroes -- a ranking that ranks nothing. Say so instead.
    rostered = sum(s.players for s in strengths)
    if rostered == 0:
        console.print(
            "[yellow]Every roster in this league is empty -- the draft "
            "hasn't happened yet.[/yellow]\n"
            "Power rankings compare drafted teams, so there is nothing to "
            "compare until picks are in.\n"
            "Until then, use [bold]ff board show[/bold] for player values and "
            "[bold]ff draft live[/bold] on draft day."
        )
        raise typer.Exit(0)
    if any(s.players == 0 for s in strengths):
        empty = [s.manager for s in strengths if s.players == 0]
        console.print(
            f"[yellow]Note: {len(empty)} roster(s) are empty and will rank "
            f"last: {', '.join(empty)}[/yellow]"
        )

    table = Table(title=f"Power rankings (week {wk})", header_style="bold")
    for col in ("ROS", "Manager", "Record", "Pre", "ROS pts", "Scored", "Luck", "Trend"):
        table.add_column(col, justify="left" if col == "Manager" else "right")
    for row in df.iter_rows(named=True):
        luck, trend = row["luck"], row["trend"]
        luck_s = f"[green]+{luck}[/]" if luck > 0 else (f"[red]{luck}[/]" if luck < 0 else "0")
        trend_s = f"[green]+{trend}[/]" if trend > 0 else (f"[red]{trend}[/]" if trend < 0 else "0")
        table.add_row(
            str(row["rank_ros"]), row["manager"], row["record"],
            f"#{row['rank_pre']}", _fmt(row["ros"], 0), f"#{row['rank_actual']}",
            luck_s, trend_s,
        )
    console.print(table)
    console.print(
        "[dim]Luck = record rank vs scoring rank. "
        "Trend = preseason rank vs ROS rank.[/dim]"
    )

    notes = read_the_table(df)
    if notes:
        console.print("\n[bold]What this means:[/bold]")
        for note in notes:
            console.print(f"  - {note}")


# --------------------------------------------------------------- in-season


def _status_str(row: dict) -> str:
    text = flags(row)
    if not text:
        return ""
    color = "red" if ("BYE" in text or any(
        s in text for s in ("IR", "Out", "PUP", "Sus", "Doubtful", "NA"))) else "yellow"
    return f"[{color}]{text}[/]"


def _season_context(cfg: LeagueConfig, league_id: str | None, username: str | None,
                    week: int | None, season: int | None):
    """Everything an in-season command needs: ROS board, week, my roster, league."""
    lid = _resolve_league_id(league_id)
    with console.status("Projecting rest of season..."):
        board, wk = _ros_board(cfg, week, season)
    my_ids, rosters, users = _my_roster_ids(lid, username)
    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"
    my_roster = board.filter(pl.col(id_col).is_in(my_ids))
    missing = [i for i in my_ids if i not in set(my_roster[id_col].to_list())]
    return board, wk, my_roster, missing, rosters, users, lid


def _weekly_board(cfg: LeagueConfig, board: pl.DataFrame, wk: int, season: int | None):
    """Board with `week_points` attached, using expert weekly numbers if fresh."""
    from .data import expert as expert_mod
    from .data import nflverse

    season = season or cfg.season
    schedule = nflverse.schedules([season])
    expert = None
    try:
        expert = expert_mod.weekly_ecr(ppr=cfg.ppr)
    except Exception as exc:  # noqa: BLE001 - optional signal
        console.print(f"[dim]Weekly expert feed unavailable ({type(exc).__name__}).[/dim]")
    weekly, used_expert = weekly_values(board, wk, schedule, season, expert)
    return weekly, used_expert


def _roster_table(df: pl.DataFrame, title: str, value_cols: list[tuple[str, str, int]],
                  slot_col: str | None = None) -> Table:
    table = Table(title=title, header_style="bold")
    cols = ([("Slot", "left")] if slot_col else []) + [
        ("Player", "left"), ("Pos", "left"), ("Tm", "left")
    ] + [(label, "right") for label, _, _ in value_cols] + [("Status", "left")]
    for label, justify in cols:
        table.add_column(label, justify=justify)
    for row in df.iter_rows(named=True):
        pos = row.get("position") or "?"
        cells = ([str(row.get(slot_col) or "")] if slot_col else []) + [
            str(row.get("name") or "?"),
            f"[{POS_COLORS.get(pos, 'white')}]{pos}[/]",
            str(row.get("team") or "-"),
        ] + [_fmt(row.get(col), digits) for _, col, digits in value_cols] + [
            _status_str(row)
        ]
        table.add_row(*cells)
    return table


@roster_app.command("show")
def roster_show(
    league_id: str | None = typer.Option(None, "--league-id"),
    username: str | None = typer.Option(None, "--username"),
    week: int | None = typer.Option(None, "--week"),
    season: int | None = typer.Option(None, "--season"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Your roster, with rest-of-season value, injury designations and drop order."""
    cfg = _load_league(config)
    board, wk, my_roster, missing, _, _, _ = _season_context(
        cfg, league_id, username, week, season
    )
    if my_roster.is_empty():
        console.print("[yellow]Your roster is empty (or nobody on it is projected).[/yellow]")
        raise typer.Exit(0)

    weekly, _ = _weekly_board(cfg, my_roster, wk, season)
    ranked = marginal_values(weekly, cfg, "ros_points").sort(
        ["marginal", "ros_points"], descending=[True, True]
    )
    console.print(_roster_table(
        ranked, f"Your roster (week {wk})",
        [("ROS pts", "ros_points", 0), ("ROS ppg", "ros_ppg", 1),
         ("Gm left", "ros_games", 1), ("Lineup +", "marginal", 0)],
    ))
    console.print(
        "[dim]Lineup + = rest-of-season points your starting lineup loses without "
        "him. Zero means bench; the lowest ROS pts among the zeroes is your "
        "natural drop.[/dim]"
    )
    if missing:
        console.print(
            f"[dim]{len(missing)} rostered player(s) are not projected "
            "(K/DEF, or unmatched ids).[/dim]"
        )


@roster_app.command("lineup")
def roster_lineup(
    league_id: str | None = typer.Option(None, "--league-id"),
    username: str | None = typer.Option(None, "--username"),
    week: int | None = typer.Option(None, "--week"),
    season: int | None = typer.Option(None, "--season"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Who to start this week, accounting for byes, injuries and expert weekly projections."""
    cfg = _load_league(config)
    _, wk, my_roster, missing, _, _, _ = _season_context(
        cfg, league_id, username, week, season
    )
    if my_roster.is_empty():
        console.print("[yellow]Your roster is empty (or nobody on it is projected).[/yellow]")
        raise typer.Exit(0)

    weekly, used_expert = _weekly_board(cfg, my_roster, wk, season)
    lineup = set_lineup(weekly, cfg, "week_points")

    value_cols = [("Week", "week_points", 1)]
    if used_expert:
        value_cols.append(("Expert", "week_pts", 1))
    value_cols.append(("ROS ppg", "ros_ppg", 1))
    if "week_opp" in weekly.columns:
        value_cols.insert(1, ("Opp", "week_opp", 0))

    console.print(_roster_table(
        lineup.starters, f"Start -- week {wk} ({lineup.points:.0f} projected)",
        value_cols, slot_col="slot",
    ))
    if not lineup.bench.is_empty():
        console.print(_roster_table(lineup.bench, "Sit", value_cols))

    if used_expert:
        console.print(
            "[dim]Week = 75% FantasyPros weekly consensus + 25% own rate, "
            "zeroed for byes and Out/IR.[/dim]"
        )
    else:
        console.print(
            "[dim]Week = own scoring rate, zeroed for byes and Out/IR; "
            "no fresh expert weekly projections yet.[/dim]"
        )
    if missing:
        console.print("[dim]K/DEF are not projected; set those by hand.[/dim]")

    notes = [r for r in lineup.starters.to_dicts() if r.get("week_note")] if \
        "week_note" in lineup.starters.columns else []
    for row in notes[:5]:
        console.print(f"  [dim]{row['name']}: {str(row['week_note'])[:140]}[/dim]")


@waiver_app.command("scan")
def waiver_scan(
    league_id: str | None = typer.Option(None, "--league-id"),
    username: str | None = typer.Option(None, "--username"),
    week: int | None = typer.Option(None, "--week"),
    season: int | None = typer.Option(None, "--season"),
    position: str | None = typer.Option(None, "--pos", help="Only look at one position."),
    protect: str | None = typer.Option(
        None, "--protect", help="Comma-separated names never to propose dropping."
    ),
    min_gain: float = typer.Option(1.0, "--min-gain", help="Minimum ROS points gained."),
    top_n: int = typer.Option(15, "--top"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Rank free agents by what adding them (and dropping someone) does to your lineup.

    Valued on rest-of-season points. Shows Sleeper's 24h add count so you know
    who will be contested on waivers.
    """
    cfg = _load_league(config)
    board, wk, my_roster, missing, rosters, _, _ = _season_context(
        cfg, league_id, username, week, season
    )
    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"

    pool = free_agents(board, rostered_in_league(rosters))
    if position:
        pool = pool.filter(pl.col("position") == position.upper())

    try:
        with SleeperClient() as client:
            trending = trending_counts(client.trending("add", limit=100))
    except Exception:  # noqa: BLE001 - cosmetic
        trending = {}

    protect_ids: set[str] = set()
    if protect:
        for raw in protect.split(","):
            name = raw.strip()
            hit = my_roster.filter(pl.col("name").str.contains(f"(?i){name}"))
            if hit.is_empty():
                console.print(f"[yellow]'{name}' is not on your roster; ignoring.[/yellow]")
            protect_ids.update(str(i) for i in hit[id_col].to_list())

    # The roster is full if Sleeper's count (including K/DEF we cannot value)
    # meets the league's roster size.
    roster_full = (my_roster.height + len(missing)) >= cfg.roster_size

    with console.status(f"Evaluating {pool.height} free agents..."):
        moves = find_moves(
            my_roster, pool, cfg,
            WaiverConfig(min_gain=min_gain, top_n=top_n, protect=frozenset(protect_ids)),
            trending=trending, roster_full=roster_full,
        )

    if not moves:
        console.print(
            f"[yellow]No add worth {min_gain:.0f}+ ROS points over what you have. "
            "Try --min-gain 0 to see marginal ones.[/yellow]"
        )
        return

    table = Table(title=f"Waiver targets (week {wk}, rest-of-season points)",
                  header_style="bold")
    for col, justify in (
        ("#", "right"), ("Add", "left"), ("Pos", "left"), ("ROS", "right"),
        ("Drop", "left"), ("ROS", "right"), ("Lineup +", "right"), ("Depth +", "right"),
        ("Adds 24h", "right"), ("Status", "left"),
    ):
        table.add_column(col, justify=justify)
    for i, m in enumerate(moves, 1):
        lg = f"[green]+{m.lineup_gain:.0f}[/]" if m.lineup_gain >= 0.5 else "0"
        table.add_row(
            str(i), m.add_name, f"[{POS_COLORS.get(m.add_pos, 'white')}]{m.add_pos}[/]",
            f"{m.add_ros:.0f}",
            m.drop_name or "[dim](open spot)[/]", f"{m.drop_ros:.0f}" if m.drop_name else "-",
            lg, f"{m.depth_gain:+.0f}",
            f"[magenta]{m.trending_adds}[/]" if m.trending_adds else "",
            f"[yellow]{m.add_status}[/]" if m.add_status else "",
        )
    console.print(table)
    console.print(
        "[dim]Lineup + = starting-lineup ROS points gained; Depth + = raw ROS points "
        "swapped onto the bench. A high Adds-24h count means you will need "
        "priority or FAAB.[/dim]"
    )
    if not roster_full:
        console.print("[dim]You have an open roster spot -- no drop needed.[/dim]")


@waiver_app.command("eval")
def waiver_eval(
    add: str = typer.Option(..., "--add", help="Comma-separated free agents to add."),
    drop: str | None = typer.Option(None, "--drop", help="Comma-separated players to drop."),
    league_id: str | None = typer.Option(None, "--league-id"),
    username: str | None = typer.Option(None, "--username"),
    week: int | None = typer.Option(None, "--week"),
    season: int | None = typer.Option(None, "--season"),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Evaluate a specific add/drop by its effect on your rest-of-season lineup."""
    cfg = _load_league(config)
    board, wk, my_roster, _, _, _, _ = _season_context(
        cfg, league_id, username, week, season
    )
    id_col = "sleeper_id" if "sleeper_id" in board.columns else "gsis_id"

    def _resolve(names: str, frame: pl.DataFrame, what: str) -> list[str]:
        ids: list[str] = []
        for raw in names.split(","):
            name = raw.strip()
            if not name:
                continue
            hit = frame.filter(pl.col("name").str.to_lowercase() == name.lower())
            if hit.is_empty():
                hit = frame.filter(pl.col("name").str.contains(f"(?i){name}"))
            if hit.is_empty():
                console.print(f"[red]No {what} matching '{name}'.[/red]")
                raise typer.Exit(1)
            ids.append(str(hit[id_col][0]))
        return ids

    add_ids = _resolve(add, board, "player")
    drop_ids = _resolve(drop, my_roster, "player on your roster") if drop else []

    result = evaluate_trade(my_roster, board, cfg, drop_ids, add_ids, value_col="ros_points")
    delta = result["lineup_delta"]
    names_in = ", ".join(result["receiving"]["name"].to_list())
    names_out = ", ".join(result["sending"]["name"].to_list()) or "(nobody)"
    console.print(Panel(
        f"Add [bold]{names_in}[/bold], drop [bold]{names_out}[/bold]\n"
        f"Starting lineup ROS: {result['lineup_before']} -> {result['lineup_after']}  "
        f"([{'green' if delta >= 0 else 'red'}]{delta:+}[/])\n"
        f"Roster spots: {result['depth_delta']:+}\n\n"
        f"[bold]Verdict: {result['verdict']}[/bold]",
        title=f"Waiver move (week {wk})",
    ))


@app.command("version")
def version() -> None:
    """Print the version."""
    from . import __version__

    console.print(f"ff2026 {__version__}")


if __name__ == "__main__":
    app()
