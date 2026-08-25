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

## Accuracy, honestly

Benchmarked against FantasyPros expert consensus (frozen pre-season, 2022-2025):

| Approach | Rank accuracy | Top-12 hit |
|---|---|---|
| Expert consensus | **0.714** | **0.594** |
| Blend (default) | 0.712 | 0.583 |
| This model alone | 0.654 | 0.510 |
| Naive baseline | 0.604 | 0.536 |

Expert rankings beat the model's own projections, because the model cannot see
offseason moves, depth charts or coaching changes. So the board **defers to
expert consensus for ordering** and uses the model for point magnitudes, which
is what value-over-replacement and opportunity cost actually need.

Reproduce with `ff model benchmark`. Details in [docs/MODEL.md](docs/MODEL.md).

## In-season

```bash
ff roster lineup                # who to start this week (byes, injuries, expert weekly numbers)
ff waiver scan                  # best add/drop moves, ranked by what they do to your lineup
ff waiver eval --add "Player" --drop "Player"   # check one specific move
ff roster show                  # your roster: ROS value, designations, natural drop order
ff board ros                    # rest-of-season projections (use these, not preseason)
ff trade find --league-id <id>  # trades that help you AND your partner
ff league power --league-id <id># who's actually good vs who's been lucky
```

A weekly routine that takes about a minute:

1. **Tuesday** — `ff waiver scan`. Every move is valued by what it does to
   your *starting lineup* over the rest of the season, not by raw points, so a
   fourth WR behind three better ones scores zero even if his number is big.
   The `Adds 24h` column is Sleeper's trending feed: a high count means the
   claim will be contested. `--protect "Name"` keeps someone off the drop list.
2. **Sunday morning** — `ff roster lineup`. Byes and Out/IR players are zeroed;
   Questionable players are haircut; during the season the number is 75%
   FantasyPros' weekly consensus projection (which is re-ranked after Friday
   injury reports) and 25% the player's own rate.

### How injuries are handled

- **Preseason**: through the expert-consensus blend. Experts see camp injuries,
  suspensions and holdouts; the model does not.
- **In-season**: every in-season command refreshes each player's current Sleeper
  designation (Out, Doubtful, Questionable, IR, PUP, Sus) rather than using
  the one saved when the board was built. Each designation is priced as
  *expected games missed* — one for Out, four for IR/PUP, three for a
  suspension — taken off the games remaining, while the scoring rate is left
  alone: a player on IR is not a worse player, he is a player with fewer games
  left. On top of that, a player who has already missed games is projected to
  keep missing some (`availability`), because absence is the best predictor of
  absence.
- **K and DEF** are not projected, so `ff roster lineup` leaves them to you.

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

# In-season
ff roster show                       # your roster, ROS value and drop order
ff roster lineup                     # start/sit for the current week
ff waiver scan [--pos RB] [--protect "Name"]   # ranked add/drop moves
ff waiver eval --add "Player" [--drop "Player"]

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
├── roster/
│   ├── lineup.py    This week's value: byes, injuries, expert weekly projections
│   └── waivers.py   Free agents, add/drop search, drop order
└── trades/
    ├── evaluate.py  Lineup-based trade evaluation
    └── finder.py    League-wide search for mutually beneficial trades
```

## Development

```bash
make test        # 111 tests, no network required
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
