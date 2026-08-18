# The model

## What it does

Projects season fantasy points for every relevant NFL player, **under your
league's actual scoring rules**, with calibrated uncertainty, then converts
those points into draft value and pick recommendations.

## Why it is built this way

Fantasy projection is a small-sample, high-variance problem. A season is 17
games, roughly half of a player's fantasy points come from touchdowns, and
touchdown rate barely persists year to year. That shapes every decision here:

1. **Score history under your rules, not generic PPR.** A TE-premium league and
   a standard league are genuinely different games. The model's target variable
   is the points *your* league pays out.

2. **Shrink toward a positional prior, in units of games.** A player with 4
   career games is mostly prior; a player with 50 is mostly himself. Expressing
   the shrinkage weight in games makes it an honest empirical-Bayes estimator
   instead of an arbitrary blend.

3. **Blend realized points with expected points from usage.** Opportunity
   (targets, carries, air yards) persists far better than efficiency. The
   `ffopportunity` dataset gives expected points for free.

4. **Age as a *level ratio*, not a flat penalty.** The player's history was
   produced at one age and we project at another, so the adjustment is the ratio
   of curve levels between those two points. This prices the *movement* along
   the curve rather than re-charging a player for being old every season.

5. **Project games separately from points per game.** Durability and
   productivity are different questions and deserve different estimators.

6. **Calibrate uncertainty on a holdout season.** Floor and ceiling come from
   residuals the model actually produced on a season it did not see, bucketed by
   position and projection tier — not from an assumed spread.

## Does it work?

Yes — with one honest exception. Backtested on 2022–2025, 12-team PPR, against
the naive baseline every projection must beat: *last season's points per game*.

Both models are scored **on exactly the same players**. This matters: the naive
baseline is structurally blind to rookies and anyone who missed the prior
season, so scoring it on its own survivorship-filtered subset would flatter it.

| Position | MAE (model / naive) | Spearman (model / naive) | Top-12 hit (model / naive) |
|---|---|---|---|
| QB | **71.8** / 84.5 | **0.549** / 0.472 | **0.562** / 0.521 |
| RB | **49.8** / 53.7 | **0.774** / 0.750 | 0.521 / **0.604** |
| WR | **43.9** / 52.2 | **0.773** / 0.739 | 0.479 / 0.479 |
| TE | **30.8** / 36.2 | **0.746** / 0.694 | 0.479 / **0.542** |

**The model wins MAE, RMSE and rank correlation at all four positions.**

**Where it loses:** naive still picks the exact top-12 better at RB and TE. The
shrinkage that helps everywhere else compresses the very top of those positions,
and the top of RB is precisely where drafts are won. With 48 slots across four
seasons a 0.08 gap is about four players — within noise, but it is consistent
enough that I would not call it noise. **This is the top open item.**

**Coverage** is the model's structural advantage — it can rank every contributor,
the baseline cannot:

| Season | Real contributors | Model ranks | Naive ranks |
|---|---|---|---|
| 2023 | 429 | **429** | 344 |
| 2024 | 421 | **421** | 350 |
| 2025 | 436 | **436** | 351 |

Reproduce any of this:

```bash
ff model backtest --seasons 2022,2023,2024,2025
```


## Benchmarked against expert consensus

Beating a naive baseline proves the machinery works. It does not prove the model
is worth using, because the real alternative is a free expert ranking. So the
model is also scored against **FantasyPros expert consensus (ECR)**, frozen at
the last August scrape before each season so no hindsight leaks in.

`ff model benchmark`, 2022-2025, identical player sets, rank metrics only (expert
consensus is a rank -- it has no points, so MAE/RMSE are undefined for it):

| Approach | Spearman | Top-12 hit |
|---|---|---|
| **Expert consensus (ECR)** | **0.714** | **0.594** |
| Blend, 75% expert | 0.712 | 0.583 |
| Blend, 50% expert | 0.702 | 0.578 |
| This model alone | 0.654 | 0.510 |
| Naive baseline | 0.604 | 0.536 |

**Expert consensus wins at every position.** This is not a surprise and it is not
a bug: the model is purely backward-looking. It sees a player's own statistics,
age and draft capital. It cannot see that he changed teams, that his team drafted
a replacement, that the coaching staff turned over, or that he is holding out. In
August that information is worth more than any amount of curve-fitting.

### What follows from that

The projection layer is **not** where this project adds value, and pretending
otherwise would be dishonest. The value is in the decision layer, which no
ranking list can provide:

1. **Your exact scoring**, applied to every historical week.
2. **Your exact roster rules**, which set replacement level.
3. **Your draft position, live** -- what survives to your next pick.

So the board defers to expert consensus for *ordering* and uses the model for
*magnitude*. `blend_rankings()` reassigns the model's positional point
distribution along the blended order: if the blend makes someone the 5th-best
receiver, he inherits the points the model gave the 5th-best receiver. VORP,
tier breaks and opportunity cost keep working, on a better ordering.

Default `--expert-weight 0.75`. Pure expert edges it on the pooled numbers, but
0.75 is at or above optimal for RB and TE, within noise elsewhere, and keeps the
model's coverage of the ~500 players the experts never rank. Set `0.0` for a
pure-model board.

## Two data quirks that will bite you

Both were found while validating the scoring engine against real 2025 data:

1. **nflverse's built-in `fantasy_points` column uses INT = −2** (the NFL.com
   standard). **Sleeper's default is −1.** If you score against nflverse's
   column you will be wrong for every quarterback. This engine always uses your
   league's own settings. Pinned by a test.

2. **`fumbles_lost_total` includes return-game fumbles** that
   `rushing + receiving + sack` fumbles miss (39 of 19,422 rows in 2025 — rare,
   but real, and concentrated on return men). Sleeper scores any lost fumble, so
   the total column is the correct one.

## Draft value

Projected points are the wrong unit for a draft. 300 points from a QB in a
1-QB league is worth much less than 300 from a RB, because the QB you would
otherwise start scores nearly as much and the RB does not.

**Replacement level** is derived from your league's actual rules: dedicated
starting slots × teams, plus flex slots allocated greedily to whichever eligible
players project best — which is what managers really do, and therefore where
replacement level really sits.

`VORP = projected points − replacement level at that position`

## The draft agent

The question at a pick is never "who is the best player available". It is
"which choice leaves me best off two picks from now". Taking a receiver costs
you the running back who will not survive until your next turn.

```
marginal value = VORP(player) − E[VORP of best player at his position
                                  still available at my next pick]
```

The expectation walks the pool best-first: player *i* is the best survivor if he
survives and everyone ahead of him does not. Survival comes from ADP and its
standard deviation, treating draft position as roughly normal. Crude, but
calibrated against thousands of real drafts, which beats a hand-wave.

That value is then scaled by **roster need** (starting slots → flex → bench
depth → surplus), and from round 8 onward tilted toward **ceiling**, because a
replacement-level bench player is worth nothing and variance is therefore free.

## Trade evaluation

A trade is not two piles of players. It is two **starting lineups over the
remaining weeks** — points scored on your bench are worth zero. So the evaluator
fills your best legal lineup before and after, and compares.

Market value is reported *alongside*, never instead. When your lineup value and
the market disagree, that disagreement is the actual information: a trade the
market hates and your lineup loves is exactly the one to make.


## Rest of season

Preseason projections answer the wrong question once games start: some of those
points are already banked and can't be traded for, and there is new evidence the
preseason number never saw. `ff board ros` fixes both.

**Rate** is re-estimated with the preseason projection as the prior and this
season's games as evidence — the same shrinkage form as the preseason model:

```
ros_ppg = (points_this_season + k x preseason_ppg) / (games_played + k)
```

`k = 2` (validated: 2 beat 0, 1, 3, 5 and 8 at every week tested).

**Availability** is modelled separately, and this mattered more than the rate:

```
availability = (games_played + 2 x prior_rate) / (team_games_played + 2)
ros_games    = team_games_remaining x availability
```

where `prior_rate` is the player's *own* projected durability
(`proj_games / 17`), not a league-wide constant. That choice makes the whole
thing degrade cleanly: with zero games played the availability term collapses to
the preseason durability and rest-of-season reduces exactly to the preseason
projection, so `ff board ros` is well-defined before week 1. It also measured
better than a flat prior at every week (wk 3: 0.729 vs 0.714).

Validation caught the bug that motivated it — the first version projected players
who had missed *every* game as though they'd play every remaining one. Missing
time is the strongest single predictor of missing more.

### Does it work? (2025, rank accuracy vs actual remaining points)

| From week | ROS model | Season-to-date only | Preseason only |
|---|---|---|---|
| 3 | **0.729** | 0.714 | 0.536 |
| 6 | **0.751** | 0.742 | 0.525 |
| 9 | **0.745** | 0.734 | 0.521 |
| 12 | **0.749** | 0.729 | 0.543 |

Preseason projections decay badly (0.536 → 0.543 while ROS climbs to 0.746),
which is precisely why in-season decisions must not use them. **Use `ros_points`
for every trade, waiver and start/sit call once the season is under way.**

## Finding trades

`ff trade find` searches the league for proposals rather than judging ones you
already thought of. It rests on one idea: **a trade only happens if both managers
think they won.**

So a proposal is surfaced only when it improves your starting lineup *and* your
partner's. Those exist because rosters are unbalanced — you have three startable
running backs and one receiver, someone has the mirror image, and the surplus is
worth more to the other team than to its owner.

Two design points worth knowing:

- **Only bench surplus is tradeable.** A player already in your starting lineup
  can't be given away for free. A roster where everyone starts has no trades in
  it, and the finder correctly returns nothing.
- **Lopsided trades are filtered out** (`min_their_gain_ratio`, default 0.25).
  Gaining 125 while your partner gains 5 is technically mutual and will still be
  declined. Set the ratio to 0 to see them anyway.

## Power rankings

`ff league power` ranks every team three ways — preseason roster, rest-of-season
roster, and actual results — because standings measure winning, not quality. The
gaps are the point:

- **Strong roster, bad record** → unlucky, may be ready to sell. Your best trade
  partner.
- **Weak roster, good record** → riding luck, due to regress. A good team to sell
  to, because they feel like winners.

## Known limitations

Stated plainly, because a projection you trust blindly is worse than none:

- **The projection layer loses to free expert rankings** (see benchmark above).
  Mitigated by blending, not solved. Closing that gap means feeding the model
  offseason context -- depth charts, team changes, coaching -- which it is
  currently blind to.
- **RB/TE top-12 identification** trails the naive baseline (above). Open item.
- **ROS ignores strength of schedule.** Remaining opponents are counted but
  not weighted by difficulty; Vegas lines are loaded and still unused.
- **The trade finder assumes opponents value players as I do.** It uses my
  projections for their lineup too, so it finds trades that are good if they
  agree with my model. Market value is not yet used as a second opinion.
- **No strength of schedule.** Vegas lines are loaded but unused.
- **K and DEF are not projected.** They are close to random and worth roughly
  nothing in draft capital; projecting them properly needs team-level modelling.
- **Rookies lean on draft capital alone.** No college production, athletic
  testing, or landing-spot adjustment.
- **Age curves are parametric constants**, sanity-checked against empirical
  year-over-year deltas (`ff model age-curve`) but not fitted from them —
  survivorship makes those deltas unreliable at the tail, since players who
  collapse get cut rather than playing a bad season.
- **No injury-risk modelling** beyond historical games played.
- **Survival probability assumes independence** across players, which slightly
  understates the tail.

## Roadmap, in the order I would do it

0. **Strength of schedule in ROS.** Vegas lines are already loaded; weighting
   remaining games by opponent difficulty is the cheapest remaining accuracy win.
1. **Close the gap to expert consensus.** The model is blind to offseason
   change. Depth-chart position, team switches and target-competition features
   would attack the largest measured weakness directly.

1. **Fix top-of-position compression at RB/TE** — likely a tier-aware shrinkage
   that trusts large samples more at the top of the distribution.
2. **Rest-of-season projections** with weekly updating — turns this from a draft
   tool into a season-long one.
3. **Strength of schedule** from Vegas implied team totals (data already loaded).
4. **Monte Carlo draft simulation** — simulate the room's picks from ADP to get
   distributions over draft outcomes rather than point estimates.
5. **Full-season lineup simulation** for trades, instead of a static ROS lineup.
6. **Waiver-wire recommendations** using Sleeper trending + opportunity changes.
