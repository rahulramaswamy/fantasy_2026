"""`ff` -- command line interface.

Draft-night commands are the point of this tool, so they are the ones designed
for speed: `ff draft live` runs a polling loop that watches the Sleeper draft and
reprints a recommendation every time a pick is made, so there is nothing to type
while you are on the clock.
"""

from __future__ import annotations

import time

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
from .model.evaluate import backtest, coverage, summarize
from .model.projections import ProjectionConfig, fit_age_curve
from .scoring import ScoringEngine
from .trades.evaluate import evaluate_trade

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

app.add_typer(league_app, name="league")
app.add_typer(data_app, name="data")
app.add_typer(board_app, name="board")
app.add_typer(model_app, name="model")
app.add_typer(draft_app, name="draft")
app.add_typer(trade_app, name="trade")
app.add_typer(sleeper_app, name="sleeper")

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
    force: bool = typer.Option(False, "--force", help="Bypass caches."),
    config: str | None = typer.Option(None, "--config"),
) -> None:
    """Build projections and save the draft board."""
    cfg = _load_league(config)
    with console.status("Building projections..."):
        board = pipeline.build_board(
            cfg, season=season, lookback=lookback, with_market=not no_market, force=force
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
    username = username or settings.sleeper_username
    user_id = settings.sleeper_user_id or (
        client.resolve_user_id(username) if username else None
    )

    if draft_id:
        return draft_id, user_id

    if not user_id:
        console.print("[red]Provide --draft-id, or --username so I can find your drafts.[/red]")
        raise typer.Exit(1)

    drafts = client.user_drafts(user_id, season)
    if not drafts:
        console.print(f"[red]No {season} drafts found for that user.[/red]")
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
        settings = get_settings()
        lid = league_id or settings.sleeper_league_id
        uname = username or settings.sleeper_username
        if not lid:
            console.print("[red]Pass --roster, or --league-id to pull your roster.[/red]")
            raise typer.Exit(1)
        with SleeperClient() as client:
            uid = settings.sleeper_user_id or (client.resolve_user_id(uname) if uname else None)
            rosters = client.league_rosters(lid)
        mine = next((r for r in rosters if str(r.get("owner_id")) == str(uid)), None)
        if not mine:
            console.print("[red]Could not find your roster in that league.[/red]")
            raise typer.Exit(1)
        my_ids = [str(p) for p in (mine.get("players") or [])]

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


@app.command("version")
def version() -> None:
    """Print the version."""
    from . import __version__

    console.print(f"ff2026 {__version__}")


if __name__ == "__main__":
    app()
