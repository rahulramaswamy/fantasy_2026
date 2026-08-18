"""League scoring engine.

Translates a Sleeper `scoring_settings` map into points against nflverse weekly
player stats. Sleeper's keys are the source of truth for a league's rules, so we
key the whole engine off them rather than off a hand-maintained scoring enum.

Anything we cannot compute from nflverse weekly aggregates (mostly TD-length
bonuses, which need play-by-play) is reported in `unsupported_keys` instead of
being silently dropped -- a league that uses them will otherwise get quietly
wrong projections.
"""

from __future__ import annotations

import polars as pl

from .config import LeagueConfig

# Sleeper scoring key -> nflverse weekly stat column, for straight multiplications.
DIRECT_STAT_MAP: dict[str, str] = {
    # Passing
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "passing_interceptions",
    "pass_2pt": "passing_2pt_conversions",
    "pass_cmp": "completions",
    "pass_att": "attempts",
    "pass_fd": "passing_first_downs",
    "pass_sack": "sacks_suffered",
    # Rushing
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rush_2pt": "rushing_2pt_conversions",
    "rush_att": "carries",
    "rush_fd": "rushing_first_downs",
    # Receiving
    "rec": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "rec_2pt": "receiving_2pt_conversions",
    "rec_tgt": "targets",
    "rec_fd": "receiving_first_downs",
    # Misc
    "fum": "fumbles_total",
    "fum_lost": "fumbles_lost_total",
    "st_td": "special_teams_tds",
    "fum_rec_td": "fumble_recovery_tds",
    # Kicking
    "xpm": "pat_made",
    "xpmiss": "pat_missed",
    "fgm_0_19": "fg_made_0_19",
    "fgm_20_29": "fg_made_20_29",
    "fgm_30_39": "fg_made_30_39",
    "fgm_40_49": "fg_made_40_49",
    "fgm_50_59": "fg_made_50_59",
    "fgm_60p": "fg_made_60_",
    "fgmiss": "fg_missed",
    "fgm": "fg_made",
}

# Per-game threshold bonuses: key -> (stat column, threshold).
THRESHOLD_BONUS_MAP: dict[str, tuple[str, float]] = {
    "bonus_pass_yd_300": ("passing_yards", 300),
    "bonus_pass_yd_400": ("passing_yards", 400),
    "bonus_rush_yd_100": ("rushing_yards", 100),
    "bonus_rush_yd_200": ("rushing_yards", 200),
    "bonus_rec_yd_100": ("receiving_yards", 100),
    "bonus_rec_yd_200": ("receiving_yards", 200),
    "bonus_pass_cmp_25": ("completions", 25),
    "bonus_rush_att_20": ("carries", 20),
}

# Position-conditional reception bonuses (TE premium and friends).
POSITION_RECEPTION_BONUS: dict[str, str] = {
    "bonus_rec_te": "TE",
    "bonus_rec_rb": "RB",
    "bonus_rec_wr": "WR",
}

# Keys we knowingly cannot derive from weekly aggregates. Requires play-by-play.
KNOWN_UNSUPPORTED = frozenset(
    {
        "pass_td_40p", "pass_td_50p", "rush_td_40p", "rush_td_50p",
        "rec_td_40p", "rec_td_50p", "bonus_fd_qb", "bonus_fd_rb",
        "bonus_fd_wr", "bonus_fd_te",
    }
)

# Sleeper's fgm_50p means "50+ yards" in leagues that don't split 50-59/60+.
FGM_50_PLUS = "fgm_50p"

# Team-defense scoring keys. Scored from team stats, not player stats.
DEF_KEYS = frozenset(
    {
        "def_td", "sack", "int", "fum_rec", "safe", "blk_kick", "def_st_td",
        "def_st_ff", "def_st_fr", "st_ff", "st_fr", "ff", "def_2pt", "def_pr_td",
        "def_kr_td", "pts_allow_0", "pts_allow_1_6", "pts_allow_7_13",
        "pts_allow_14_20", "pts_allow_21_27", "pts_allow_28_34", "pts_allow_35p",
        "yds_allow_0_100", "yds_allow_100_199", "yds_allow_200_299",
        "yds_allow_300_349", "yds_allow_350_399", "yds_allow_400_449",
        "yds_allow_450_499", "yds_allow_500_549", "yds_allow_550p",
    }
)


class ScoringEngine:
    """Applies a league's scoring settings to weekly nflverse stat lines."""

    def __init__(self, league: LeagueConfig) -> None:
        self.league = league
        self.scoring = {k: v for k, v in league.scoring_settings.items() if v}
        self.unsupported_keys: list[str] = []

    def _build_terms(self, columns: set[str]) -> list[pl.Expr]:
        terms: list[pl.Expr] = []
        self.unsupported_keys = []

        for key, points in self.scoring.items():
            if key in DEF_KEYS:
                # Scored separately in score_defense(); not a player-stat term.
                continue

            if key in DIRECT_STAT_MAP:
                col = DIRECT_STAT_MAP[key]
                if col in columns:
                    terms.append(pl.col(col).fill_null(0).cast(pl.Float64) * points)
                else:
                    self.unsupported_keys.append(key)
            elif key == FGM_50_PLUS:
                cols = [c for c in ("fg_made_50_59", "fg_made_60_") if c in columns]
                if cols:
                    expr = pl.sum_horizontal([pl.col(c).fill_null(0) for c in cols])
                    terms.append(expr.cast(pl.Float64) * points)
                else:
                    self.unsupported_keys.append(key)
            elif key in THRESHOLD_BONUS_MAP:
                col, threshold = THRESHOLD_BONUS_MAP[key]
                if col in columns:
                    terms.append(
                        (pl.col(col).fill_null(0) >= threshold).cast(pl.Float64) * points
                    )
                else:
                    self.unsupported_keys.append(key)
            elif key in POSITION_RECEPTION_BONUS:
                pos = POSITION_RECEPTION_BONUS[key]
                if "receptions" in columns and "position" in columns:
                    terms.append(
                        pl.when(pl.col("position") == pos)
                        .then(pl.col("receptions").fill_null(0).cast(pl.Float64) * points)
                        .otherwise(0.0)
                    )
                else:
                    self.unsupported_keys.append(key)
            else:
                self.unsupported_keys.append(key)

        return terms

    def score_weekly(self, df: pl.DataFrame, alias: str = "fantasy_points") -> pl.DataFrame:
        """Add a per-week fantasy points column computed under this league's rules."""
        terms = self._build_terms(set(df.columns))
        if not terms:
            return df.with_columns(pl.lit(0.0).alias(alias))
        return df.with_columns(pl.sum_horizontal(terms).alias(alias))

    def score_stat_line(self, stats: dict[str, float], position: str = "WR") -> float:
        """Score a single projected stat line (a dict of nflverse column -> value)."""
        row = {**stats, "position": position}
        df = pl.DataFrame([row])
        # Threshold bonuses are per-game and meaningless on a season total, so a
        # projected season line should be scored per-game and multiplied instead.
        return float(self.score_weekly(df)["fantasy_points"][0])

    def static_unsupported(self) -> list[str]:
        """Scoring keys this engine cannot compute, independent of any dataframe.

        `unsupported_keys` is populated during scoring and therefore reflects
        which columns a *particular* frame happened to have. For reporting a
        league's configuration we want the structural answer instead.
        """
        known = (
            set(DIRECT_STAT_MAP)
            | set(THRESHOLD_BONUS_MAP)
            | set(POSITION_RECEPTION_BONUS)
            | {FGM_50_PLUS}
            | DEF_KEYS
        )
        return sorted(k for k in self.scoring if k not in known)

    def describe(self) -> str:
        supported = len(self.scoring) - len(self.unsupported_keys)
        missing = ", ".join(self.unsupported_keys) or "none"
        return (
            f"{self.league.name}: {supported} scoring rules applied, "
            f"{len(self.unsupported_keys)} unsupported ({missing})"
        )


def default_ppr_settings(ppr: float = 1.0) -> dict[str, float]:
    """A standard Sleeper-shaped scoring map, for testing and for leagues not yet synced."""
    return {
        "pass_yd": 0.04,
        "pass_td": 4.0,
        "pass_int": -1.0,
        "pass_2pt": 2.0,
        "rush_yd": 0.1,
        "rush_td": 6.0,
        "rush_2pt": 2.0,
        "rec": ppr,
        "rec_yd": 0.1,
        "rec_td": 6.0,
        "rec_2pt": 2.0,
        "fum_lost": -2.0,
        "st_td": 6.0,
    }
