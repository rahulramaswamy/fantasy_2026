import polars as pl
import pytest

from ff2026.config import LeagueConfig
from ff2026.scoring import default_ppr_settings


@pytest.fixture
def ppr_league() -> LeagueConfig:
    return LeagueConfig(
        name="Test PPR",
        teams=12,
        roster_positions=[
            "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
            "BN", "BN", "BN", "BN", "BN", "BN",
        ],
        scoring_settings=default_ppr_settings(1.0),
        ppr=1.0,
    )


@pytest.fixture
def superflex_league() -> LeagueConfig:
    return LeagueConfig(
        name="Test SF",
        teams=10,
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN"],
        scoring_settings=default_ppr_settings(0.5),
        ppr=0.5,
        superflex=True,
    )


@pytest.fixture
def board() -> pl.DataFrame:
    """A small synthetic board: enough players to have a real replacement level."""
    rows = []
    specs = {"QB": 40, "RB": 70, "WR": 90, "TE": 40}
    for pos, count in specs.items():
        base = {"QB": 320.0, "RB": 300.0, "WR": 290.0, "TE": 220.0}[pos]
        for i in range(count):
            points = base - i * (base / (count + 5))
            rows.append(
                {
                    "gsis_id": f"{pos}-{i}",
                    "sleeper_id": f"s{pos}{i}",
                    "name": f"{pos} Player {i}",
                    "position": pos,
                    "team": "FA",
                    "proj_points": round(points, 1),
                    "proj_ppg": round(points / 16, 2),
                    "proj_games": 16.0,
                    "floor": round(points * 0.7, 1),
                    "ceiling": round(points * 1.3, 1),
                    "adp": float(i * 4 + {"QB": 30, "RB": 1, "WR": 2, "TE": 25}[pos]),
                    "adp_stdev": 10.0,
                }
            )
    return pl.DataFrame(rows)
