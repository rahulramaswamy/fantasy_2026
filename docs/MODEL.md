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

## Known limitations

Stated plainly, because a projection you trust blindly is worse than none:

- **RB/TE top-12 identification** trails the naive baseline (above). Open item.
- **No in-season update.** The model projects full seasons. Rest-of-season
  reprojection with weekly Bayesian updating is the biggest missing feature.
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

1. **Fix top-of-position compression at RB/TE** — likely a tier-aware shrinkage
   that trusts large samples more at the top of the distribution.
2. **Rest-of-season projections** with weekly updating — turns this from a draft
   tool into a season-long one.
3. **Strength of schedule** from Vegas implied team totals (data already loaded).
4. **Monte Carlo draft simulation** — simulate the room's picks from ADP to get
   distributions over draft outcomes rather than point estimates.
5. **Full-season lineup simulation** for trades, instead of a static ROS lineup.
6. **Waiver-wire recommendations** using Sleeper trending + opportunity changes.
