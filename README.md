# fantasy_2026

A fantasy football projection model, draft agent, and trade evaluator for
**Sleeper** leagues.

Built on free data. Scores history under *your* league's actual rules,
projects the season with calibrated uncertainty, and recommends picks live
during your draft based on opportunity cost — not just "best player available".

```
┌─ nflverse ──────┐   ┌─ Sleeper API ──┐   ┌─ ADP / trade values ─┐
│ box scores      │   │ league rules   │   │ what players cost    │
│ usage, snaps    │   │ rosters        │   │ survival probability │
│ expected points │   │ live draft     │   │ market read          │
└────────┬────────┘   └───────┬────────┘   └──────────┬───────────┘
         └────────────────────┼───────────────────────┘
                    projections → VORP → pick recommendation
```

## Quickstart

```bash
make install                              # or: pip install -e ".[dev]"
cp .env.example .env                      # add your Sleeper username

ff league sync --username <your-sleeper-name>   # pull your real scoring rules
ff data sync                                    # warm every cache
ff board build                                  # build the draft board
```

Then on draft night:

```bash
ff draft live --username <your-sleeper-name>
```

That polls your Sleeper draft and reprints a recommendation every time the board
moves — nothing to type while you are on the clock.

```
╭─────────────────────── Draft 1234567890 (snake, 12 teams) ───────────────────────╮
│ Pick 19 (round 2) - 3 picks until your turn (#22)                                │
│ Your slot: 3   Roster: RB1   Next picks: 22, 27                                  │
╰──────────────────────────────────────────────────────────────────────────────────╯
                              Recommendations
  #  Player               Pos  Tm    Score   VORP    ADP   P(lasts)
  1  Malik Nabers         WR   NYG    58.2     94   20.4        18%
  2  Brock Bowers         TE   LV     41.7     88   23.1        44%
  ...
-> Malik Nabers (WR) - 94 pts over replacement; only 18% to last until pick 27 - take him now
```

## What the numbers mean

| Column | Meaning |
|---|---|
| **VORP** | Projected points above the last startable player at that position, given *your* roster rules |
| **Score** | VORP minus what that position is likely to still offer at your next pick, scaled by roster need |
| **P(lasts)** | Probability the player survives to your next pick, from ADP and its standard deviation |
| **Floor / Ceiling** | 20th / 80th percentile, calibrated on a holdout season |

The **Score** column is the recommendation. It answers "which choice leaves me
best off two picks from now", which is the actual question at a draft pick.

## Commands

```bash
# League
ff league sync --username <you>      # pull scoring + roster rules from Sleeper
ff league show                       # confirm how your scoring is interpreted

# Data
ff data sync                         # warm all caches (do this before draft day)
ff data crosswalk                    # check Sleeper <-> nflverse ID match quality

# Board
ff board build                       # build and save projections
ff board show --pos RB --tiers       # positional board with tier breaks

# Draft
ff draft live --username <you>       # live draft assistant (the main event)
ff draft recommend                   # one-shot recommendation
ff draft picks                       # what has been taken

# Trades
ff trade eval --send "Player A" --receive "Player B"

# Model
ff model backtest                    # score the model vs the naive baseline
ff model age-curve                   # empirical ageing, to sanity-check the curves

# Sleeper
ff sleeper selftest --username <you> # verify every endpoint this tool needs
ff sleeper drafts --username <you>   # list your drafts
```

## Before draft day

Run this once, a day ahead:

```bash
ff sleeper selftest --username <you>   # confirm the API works from your machine
ff data crosswalk                      # confirm no notable player is unmatched
ff data sync                           # cache everything locally
ff board build                         # build the board
```

An unmatched player never appears on your board and you will not notice — which
is why `ff data crosswalk` ranks unmatched players by prominence.

## Does the model work?

Backtested 2022–2025 against the baseline every projection must beat — last
season's points per game — with both scored on the same players:

| Position | MAE (model / naive) | Spearman (model / naive) |
|---|---|---|
| QB | **71.8** / 84.5 | **0.549** / 0.472 |
| RB | **49.8** / 53.7 | **0.774** / 0.750 |
| WR | **43.9** / 52.2 | **0.773** / 0.739 |
| TE | **30.8** / 36.2 | **0.746** / 0.694 |

It wins on error and rank correlation at every position, and it can rank ~430
contributors per season where the baseline manages ~350. It still **loses to
naive on picking the exact top-12 RBs** — see
[docs/MODEL.md](docs/MODEL.md#known-limitations) for that and every other known
limitation.

## Documentation

- **[docs/DATASETS.md](docs/DATASETS.md)** — every data source, what it gives
  you, how to reach it, refresh cadence, and the gotchas (including the Sleeper
  endpoint quirks and the player-ID problem).
- **[docs/MODEL.md](docs/MODEL.md)** — how the model works, what it gets right,
  what it gets wrong, and the roadmap.

## Layout

```
src/ff2026/
├── config.py        League rules: scoring, roster slots, flex eligibility
├── scoring.py       Sleeper scoring settings -> points, per week
├── ids.py           Sleeper <-> nflverse player identity crosswalk
├── pipeline.py      End-to-end build: sources -> draft-ready board
├── cli.py           The `ff` command
├── data/
│   ├── sleeper.py   All 18 Sleeper endpoints, rate limited and cached
│   ├── nflverse.py  nflverse loaders
│   ├── market.py    ADP and trade values
│   └── cache.py     On-disk parquet cache with TTLs
├── model/
│   ├── features.py     Season totals, age, expected points
│   ├── projections.py  The projection model
│   └── evaluate.py     Backtesting vs the naive baseline
├── draft/
│   ├── value.py     Replacement level, VORP, tiers
│   ├── board.py     Live draft state and snake-order math
│   └── agent.py     Pick recommendation
└── trades/
    └── evaluate.py  Lineup-based trade evaluation
```

## Development

```bash
make test        # 65 tests, no network required
make lint
make typecheck
```

Tests run entirely on synthetic fixtures — no API calls, no data downloads.

## Caveats

- The **Sleeper API could not be reached from the sandbox this was built in**,
  so the client follows the documented endpoint set rather than live responses.
  `ff sleeper selftest` verifies all of it from your machine in one command.
- **K and DEF are not projected.** They are close to random and cost almost
  nothing in draft capital.
- Model output is an input to your judgement, not a replacement for it. The
  known limitations in [docs/MODEL.md](docs/MODEL.md#known-limitations) are worth
  reading before you trust a recommendation over your own read.

## Credits

Data from [nflverse](https://github.com/nflverse),
[Sleeper](https://docs.sleeper.com/),
[Fantasy Football Calculator](https://fantasyfootballcalculator.com/),
and [FantasyCalc](https://www.fantasycalc.com/).
