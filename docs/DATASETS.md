# Datasets

Everything below is free and legally accessible. The paid options are listed at
the end for completeness, but nothing in this repo requires them.

Status notes reflect what was actually verified while building this repo
(August 2026). Where a source could not be reached from the build sandbox, that
is stated explicitly rather than assumed working.

---

## 1. nflverse — the backbone

**What it is:** a community-maintained, free set of NFL data releases covering
play-by-play back to 1999, weekly player stats, rosters, snap counts, depth
charts, injuries, Next Gen Stats, and PFR advanced stats. It is the single most
valuable free resource in football analytics and it is what this model is built
on.

**Access:** `nflreadpy` (Python, Polars-native).

```bash
pip install nflreadpy
```

> **Note:** `nfl_data_py` — the package most older tutorials use — is
> **deprecated**. nflverse has moved to `nflreadpy`, and no further
> `nfl_data_py` maintenance is planned. Use `nflreadpy`.

**Verified working**, with the shapes actually returned:

| Loader | What you get | Verified |
|---|---|---|
| `load_player_stats(seasons, summary_level="week")` | Per-player, per-week box scores. **150 columns** including passing/rushing/receiving splits, EPA, air yards, target share, WOPR, *and* kicker distance buckets (`fg_made_40_49`, etc.) | 19,422 rows for 2025; 140,750 for 2018–2025 |
| `load_players()` | Player master: `gsis_id`, `pfr_id`, `espn_id`, `pff_id`, birth date, draft capital | 25,046 rows |
| `load_rosters(seasons)` | Season rosters, incl. rookies | **2026: 2,930 rows** |
| `load_schedules(seasons)` | Opponent, bye weeks, **Vegas spread/total**, roof, surface, rest days | **2026: 272 games, weeks 1–18** |
| `load_snap_counts(seasons)` | Offensive/defensive/ST snap share — the cleanest role proxy | 26,612 rows for 2025 |
| `load_depth_charts(seasons)` | Weekly depth chart position and rank | 554,215 rows |
| `load_injuries(seasons)` | Official injury report: practice + game status | 6,068 rows for 2025 |
| `load_ff_opportunity(seasons)` | **Expected fantasy points from usage** (159 cols) | 6,054 rows for 2025 |
| `load_draft_picks()` | NFL draft history — the rookie prior | ✅ |
| `load_nextgen_stats(seasons, stat_type)` | Separation, cushion, time to throw, rush yards over expected | ✅ |
| `load_pfr_advstats(seasons, stat_type)` | Broken tackles, drops, YAC, pressure | ✅ |
| `load_ff_playerids()` | **Cross-platform ID map** (sleeper ↔ gsis ↔ mfl ↔ espn) | Blocked in sandbox¹ |
| `load_ff_rankings(type)` | FantasyPros expert consensus rankings (ECR) | Blocked in sandbox¹ |

¹ These two pull from the DynastyProcess GitHub repo, which this build sandbox's
network policy blocked. They are not broken — they will work on your machine.
The ID crosswalk has a fallback path that does not depend on them (see §5).

**Why `load_ff_opportunity` matters more than it looks:** it models *expected*
fantasy points from opportunity alone (air yards, carries, usage). Touchdown
rate regresses hard year over year; opportunity does not. The gap between a
player's actual and expected points is the best free regression signal there is.

**Refresh cadence:** a few times per week in season. **License:** open, community
maintained. Credit nflverse if you publish anything from it.

---

## 2. Sleeper API — your league

**What it is:** read-only, unauthenticated, no API key. This is where your
league's real rules, rosters, and — critically — the **live draft feed** come
from.

**Base URL:** `https://api.sleeper.app/v1` · **Docs:** <https://docs.sleeper.com/>

The complete endpoint set (all 18 are wrapped in `src/ff2026/data/sleeper.py`):

```
GET /user/<username_or_id>
GET /user/<user_id>/leagues/nfl/<season>
GET /user/<user_id>/drafts/nfl/<season>
GET /league/<league_id>
GET /league/<league_id>/rosters
GET /league/<league_id>/users
GET /league/<league_id>/matchups/<week>
GET /league/<league_id>/winners_bracket
GET /league/<league_id>/loses_bracket        <- Sleeper's spelling, not a typo
GET /league/<league_id>/transactions/<round>
GET /league/<league_id>/traded_picks
GET /league/<league_id>/drafts
GET /draft/<draft_id>
GET /draft/<draft_id>/picks                  <- the live draft feed
GET /draft/<draft_id>/traded_picks
GET /state/nfl
GET /players/nfl?position=<pos>&active=<bool>
GET /players/nfl/trending/<add|drop>?lookback_hours=<h>&limit=<n>
```

**Gotchas worth knowing before draft day:**

- **`loses_bracket`**, not `losers_bracket`. The latter 404s.
- **`/players/nfl` is ~5 MB.** Sleeper asks you to call it **at most once per
  day**. It is cached for 24h here. Use the `position`/`active` filters to
  shrink it.
- **Rate limit:** stay under ~1000 calls/minute. This client self-limits to 600.
- **404 means "no such thing"**, not an error — the client returns `None`.
- **There is no draft websocket.** Live draft tracking means polling
  `/draft/<id>/picks`. Every 5 seconds is plenty and nowhere near the limit.
- `/players/nfl` carries a `gsis_id` per player, which is the cheapest and most
  reliable way to join Sleeper to nflverse.

**Verify it yourself** before your draft — this hits every endpoint and reports
what works:

```bash
ff sleeper selftest --username <you>
```

> The Sleeper API was **not reachable from this build sandbox** (egress policy
> blocked `api.sleeper.app`), so the client was written against the documented
> endpoint set rather than live responses. That is exactly why `selftest`
> exists: one command confirms all of it from your machine.

---

## 3. Market data — what players actually cost

Projections tell you what a player is *worth*. Market data tells you what he
*costs*. Value minus price is the only edge that exists.

### Fantasy Football Calculator — ADP

Real ADP from live mock drafts. Free REST API, no key.

```
GET https://fantasyfootballcalculator.com/api/v1/adp/{format}?teams=12&year=2026&position=all
```

`format` ∈ `standard | ppr | half-ppr | 2qb | dynasty | rookie`.
Returns `adp`, `stdev`, `high`, `low`, `times_drafted`, `bye`.

The **`stdev` field is the important one** — it is what makes "will he last
until my next pick?" a real probability instead of a guess. Recent data covers
thousands of mock drafts per week during draft season.

*Terms:* free for personal and commercial use; they ask for attribution and
that you not hammer the API. Cached for 1 hour here.

### FantasyCalc — trade values

Trade values derived from **actual completed trades** across thousands of
leagues — not one analyst's opinion.

```
GET https://api.fantasycalc.com/values/current?isDynasty=false&numQbs=1&numTeams=12&ppr=1
```

Publishes **`sleeperId` directly**, so it joins to your roster exactly with no
name matching. Supports dynasty/redraft, superflex, and league size.

*Note:* FantasyCalc does not publish formal API docs; this endpoint is
widely used and stable but is not contractually guaranteed.

### Alternatives worth knowing

| Source | Use | Access |
|---|---|---|
| **FantasyPros ECR** | Expert consensus ranks | Free via `nflreadpy.load_ff_rankings()`; their own API is paid |
| **KeepTradeCut** | Crowdsourced dynasty values | Scraping only, no public API |
| **DynastyProcess** | Values + the ID crosswalk | Free CSVs on GitHub |
| **Sleeper trending** | Waiver-wire pulse (adds/drops) | Free, in the Sleeper API above |

---

## 4. Supplementary signals

Not wired into the v1 model, but the data is there and these are the highest-value
next additions:

- **Vegas lines** (already in `load_schedules`): implied team totals are the best
  single predictor of game script. A back on a 27-point favourite gets carries.
- **Weather** (`roof`, `surface`, `temp`, `wind` in schedules): wind above ~15mph
  measurably suppresses passing.
- **Snap share + depth charts**: the fastest way to catch a role change before
  it shows up in the box score.
- **Injury reports**: practice status (DNP/LP/FP) predicts game availability
  better than the game-status tag alone.
- **Next Gen Stats**: separation and target quality for receivers.

---

## 5. The ID problem (read this one)

Sleeper speaks its own numeric `player_id`. nflverse speaks `gsis_id`. Market
feeds speak names, and names are messy: suffixes (`Jr.`), accents, nicknames
(`Hollywood Brown` vs `Marquise Brown`), and genuinely duplicate names.

**An unmatched star is a silent, expensive bug on draft day** — he simply never
appears on your board and you never notice.

`src/ff2026/ids.py` resolves in three tiers, best evidence first:

1. **Sleeper's own `gsis_id`** from the player dump (covers most NFL players).
2. **DynastyProcess ID map** via `load_ff_playerids()`.
3. **Normalized name + position matching** — accent stripping, suffix removal,
   punctuation removal, plus an explicit alias table.

Match quality is **reported, not assumed**:

```bash
ff data crosswalk    # match rate by source + the most notable unmatched players
```

Unmatched players are ranked by Sleeper's `search_rank`, so anyone who actually
matters surfaces at the top of the list.

---

## 6. Paid options (not required)

| Source | Adds | Rough cost |
|---|---|---|
| **PFF** | Player grades, route participation, pressure | $$$ |
| **FantasyPros API** | Their projections and ECR programmatically | $$ |
| **SportsDataIO / Sportradar** | Real-time everything, commercial licence | $$$$ |
| **rotowire / 4for4** | Projections, snap projections | $ |

The free stack above is genuinely competitive for redraft. The main thing money
buys is **charting data** (route participation, pressure rates) that nflverse
only partially covers via FTN and PFR.

---

## Refresh cadence

| Data | Refresh | Why |
|---|---|---|
| nflverse weekly stats | 2–3× per week in season | Games finish, stats post |
| Sleeper `/players` | Daily (enforced) | 5 MB payload; Sleeper asks |
| ADP | Hourly in draft season | Moves fast near your draft |
| Trade values | Hourly in season | News moves them |
| Draft picks | Every 5s during a draft | It is a live feed |

```bash
ff data sync    # warms every cache so draft night needs no network
```
