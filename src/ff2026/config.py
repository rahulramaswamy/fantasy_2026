"""Configuration: environment settings and league (scoring + roster) definition."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Positions we actually project. Everything else is ignored by the model.
OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")
ALL_POSITIONS = (*OFFENSE_POSITIONS, "K", "DEF")

# Sleeper roster slot -> the set of positions eligible to fill it.
SLOT_ELIGIBILITY: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "K": ("K",),
    "DEF": ("DEF",),
    "FLEX": ("RB", "WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
    "REC_FLEX": ("WR", "TE"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
    "IDP_FLEX": (),
}
# Slots that do not start anyone.
BENCH_SLOTS = frozenset({"BN", "IR", "TAXI"})


class Settings(BaseSettings):
    """Environment-driven settings. Read from the process env and a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sleeper_username: str | None = None
    sleeper_user_id: str | None = None
    sleeper_league_id: str | None = None
    sleeper_draft_id: str | None = None
    ff_data_dir: Path = Path("./data")
    # Optional local checkout of dynastyprocess/data, for offline expert rankings.
    ff_dp_local_dir: str | None = None

    @property
    def cache_dir(self) -> Path:
        d = self.ff_data_dir / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def artifacts_dir(self) -> Path:
        d = self.ff_data_dir / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d


class LeagueConfig(BaseModel):
    """A league's rules. Mirrors the shape Sleeper returns so it can be synced directly."""

    name: str = "My League"
    league_id: str | None = None
    season: int = 2026
    teams: int = 12
    # Sleeper's roster_positions array,
    # e.g. ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF","BN",...]
    roster_positions: list[str] = Field(
        default_factory=lambda: [
            "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
            "BN", "BN", "BN", "BN", "BN", "BN",
        ]
    )
    # Sleeper's scoring_settings map, e.g. {"pass_yd": 0.04, "rec": 1.0, ...}
    scoring_settings: dict[str, float] = Field(default_factory=dict)
    # PPR-ish label used only for choosing market/ADP feeds.
    ppr: float = 1.0
    superflex: bool = False

    @property
    def starting_slots(self) -> list[str]:
        """Roster slots that start a player each week, in order."""
        return [s for s in self.roster_positions if s not in BENCH_SLOTS]

    @property
    def bench_size(self) -> int:
        return sum(1 for s in self.roster_positions if s == "BN")

    @property
    def roster_size(self) -> int:
        return len(self.roster_positions)

    def slot_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for slot in self.starting_slots:
            counts[slot] = counts.get(slot, 0) + 1
        return counts

    def starters_at(self, position: str) -> int:
        """Dedicated (non-flex) starting slots for a position."""
        return sum(1 for s in self.starting_slots if s == position)

    def flex_slots(self) -> dict[str, int]:
        """Flex-type slots and their counts."""
        return {
            slot: n
            for slot, n in self.slot_counts().items()
            if slot in SLOT_ELIGIBILITY and len(SLOT_ELIGIBILITY[slot]) > 1
        }

    @classmethod
    def from_sleeper(cls, league: dict[str, Any]) -> LeagueConfig:
        """Build from the payload of GET /v1/league/<league_id>."""
        scoring = {k: float(v) for k, v in (league.get("scoring_settings") or {}).items()}
        roster_positions = list(league.get("roster_positions") or [])
        settings = league.get("settings") or {}
        return cls(
            name=league.get("name") or "My League",
            league_id=league.get("league_id"),
            season=int(league.get("season") or 2026),
            teams=int(settings.get("num_teams") or len(roster_positions) and 12),
            roster_positions=roster_positions,
            scoring_settings=scoring,
            ppr=float(scoring.get("rec", 0.0)),
            superflex="SUPER_FLEX" in roster_positions or roster_positions.count("QB") > 1,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> LeagueConfig:
        """Load from YAML, falling back to configs/league.yaml then the bundled example."""
        candidates = [Path(path)] if path else [
            Path("configs/league.yaml"),
            Path("configs/league.example.yaml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                data = yaml.safe_load(candidate.read_text()) or {}
                return cls.model_validate(data)
        raise FileNotFoundError(
            f"No league config found (looked at: {', '.join(str(c) for c in candidates)}). "
            "Run `ff league sync --league-id <id>` to generate one from Sleeper."
        )

    def save(self, path: str | Path = "configs/league.yaml") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
        return p


def get_settings() -> Settings:
    return Settings()


def data_dir() -> Path:
    return Path(os.environ.get("FF_DATA_DIR", "./data"))
