"""Sleeper API client (read-only, unauthenticated).

Every endpoint the app touches lives here, so if Sleeper changes a path there is
exactly one place to fix. Sleeper asks callers to stay under ~1000 requests per
minute and to pull the full player dump at most once a day; both are enforced
here rather than left to the caller.

Endpoint reference: https://docs.sleeper.com/
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import httpx

from .cache import TTL_DAY, TTL_HOUR, cached_json

BASE_URL = "https://api.sleeper.app/v1"
AVATAR_URL = "https://sleepercdn.com/avatars"

# Sleeper's documented ceiling is ~1000 calls/minute; stay well under it.
MAX_CALLS_PER_MINUTE = 600


class SleeperError(RuntimeError):
    """Raised when Sleeper returns an error or an unusable payload."""


class _RateLimiter:
    """Simple sliding-window limiter shared by all clients in the process."""

    def __init__(self, max_calls: int = MAX_CALLS_PER_MINUTE, window: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window = window
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.window:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                sleep_for = self.window - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self._calls.append(time.monotonic())


_limiter = _RateLimiter()


class SleeperClient:
    """Thin, typed-ish wrapper over the Sleeper v1 REST API."""

    def __init__(self, timeout: float = 15.0, retries: int = 3) -> None:
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"User-Agent": "ff2026/0.1 (personal fantasy tooling)"},
            follow_redirects=True,
        )
        self.retries = retries

    def __enter__(self) -> SleeperClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ---------------------------------------------------------------- plumbing

    def _get(self, path: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            _limiter.acquire()
            try:
                resp = self._client.get(path)
            except httpx.HTTPError as exc:  # network/timeout
                last_exc = exc
                time.sleep(2**attempt)
                continue

            if resp.status_code == 404:
                # Sleeper uses 404 for "no such league/draft"; surface as None.
                return None
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code >= 500:
                last_exc = SleeperError(f"{resp.status_code} from {path}")
                time.sleep(2**attempt)
                continue
            if resp.status_code >= 400:
                raise SleeperError(f"{resp.status_code} from {path}: {resp.text[:200]}")

            try:
                return resp.json()
            except ValueError as exc:
                raise SleeperError(f"Non-JSON response from {path}") from exc

        raise SleeperError(f"Failed to GET {path} after {self.retries} attempts: {last_exc}")

    # ------------------------------------------------------------------- state

    def state(self, sport: str = "nfl") -> dict[str, Any]:
        """Current season/week. GET /state/<sport>"""
        return self._get(f"/state/{sport}") or {}

    # ------------------------------------------------------------------- users

    def user(self, username_or_id: str) -> dict[str, Any] | None:
        """GET /user/<username_or_user_id>"""
        return self._get(f"/user/{username_or_id}")

    def user_leagues(self, user_id: str, season: int, sport: str = "nfl") -> list[dict[str, Any]]:
        """GET /user/<user_id>/leagues/<sport>/<season>"""
        return self._get(f"/user/{user_id}/leagues/{sport}/{season}") or []

    def user_drafts(self, user_id: str, season: int, sport: str = "nfl") -> list[dict[str, Any]]:
        """GET /user/<user_id>/drafts/<sport>/<season>"""
        return self._get(f"/user/{user_id}/drafts/{sport}/{season}") or []

    # ----------------------------------------------------------------- leagues

    def league(self, league_id: str) -> dict[str, Any] | None:
        """GET /league/<league_id>"""
        return self._get(f"/league/{league_id}")

    def league_rosters(self, league_id: str) -> list[dict[str, Any]]:
        """GET /league/<league_id>/rosters"""
        return self._get(f"/league/{league_id}/rosters") or []

    def league_users(self, league_id: str) -> list[dict[str, Any]]:
        """GET /league/<league_id>/users"""
        return self._get(f"/league/{league_id}/users") or []

    def matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        """GET /league/<league_id>/matchups/<week>"""
        return self._get(f"/league/{league_id}/matchups/{week}") or []

    def transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        """GET /league/<league_id>/transactions/<week>"""
        return self._get(f"/league/{league_id}/transactions/{week}") or []

    def traded_picks(self, league_id: str) -> list[dict[str, Any]]:
        """GET /league/<league_id>/traded_picks"""
        return self._get(f"/league/{league_id}/traded_picks") or []

    def winners_bracket(self, league_id: str) -> list[dict[str, Any]]:
        """GET /league/<league_id>/winners_bracket"""
        return self._get(f"/league/{league_id}/winners_bracket") or []

    def losers_bracket(self, league_id: str) -> list[dict[str, Any]]:
        """GET /league/<league_id>/loses_bracket

        Note the path really is `loses_bracket` -- that spelling is Sleeper's,
        not a typo here, and `losers_bracket` 404s.
        """
        return self._get(f"/league/{league_id}/loses_bracket") or []

    def league_drafts(self, league_id: str) -> list[dict[str, Any]]:
        """GET /league/<league_id>/drafts"""
        return self._get(f"/league/{league_id}/drafts") or []

    # ------------------------------------------------------------------ drafts

    def draft(self, draft_id: str) -> dict[str, Any] | None:
        """GET /draft/<draft_id>"""
        return self._get(f"/draft/{draft_id}")

    def draft_picks(self, draft_id: str) -> list[dict[str, Any]]:
        """GET /draft/<draft_id>/picks -- the live feed during a draft."""
        return self._get(f"/draft/{draft_id}/picks") or []

    def draft_traded_picks(self, draft_id: str) -> list[dict[str, Any]]:
        """GET /draft/<draft_id>/traded_picks"""
        return self._get(f"/draft/{draft_id}/traded_picks") or []

    # ----------------------------------------------------------------- players

    def players(
        self,
        sport: str = "nfl",
        position: str | None = None,
        active: bool | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """GET /players/<sport>?position=<pos>&active=<bool>

        The unfiltered dump is ~5MB and Sleeper asks that it run at most once a
        day, so it is cached for 24h. Filtering by position/active shrinks the
        payload a lot when you only need, say, active WRs.
        """
        params: list[str] = []
        if position:
            params.append(f"position={position}")
        if active is not None:
            params.append(f"active={str(active).lower()}")
        query = f"?{'&'.join(params)}" if params else ""
        key = f"sleeper_players_{sport}{'_' + '_'.join(params) if params else ''}"
        return cached_json(
            key,
            lambda: self._get(f"/players/{sport}{query}") or {},
            ttl=TTL_DAY,
            force=force,
        )

    def trending(
        self,
        kind: str = "add",
        sport: str = "nfl",
        lookback_hours: int = 24,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """GET /players/<sport>/trending/<add|drop> -- waiver-wire pulse."""
        if kind not in ("add", "drop"):
            raise ValueError("kind must be 'add' or 'drop'")
        return cached_json(
            f"sleeper_trending_{kind}_{lookback_hours}_{limit}",
            lambda: self._get(
                f"/players/{sport}/trending/{kind}"
                f"?lookback_hours={lookback_hours}&limit={limit}"
            )
            or [],
            ttl=TTL_HOUR,
        )

    # --------------------------------------------------------------- utilities

    @staticmethod
    def avatar_url(avatar_id: str, thumb: bool = False) -> str:
        return f"{AVATAR_URL}/{'thumbs/' if thumb else ''}{avatar_id}"

    def resolve_user_id(self, username_or_id: str) -> str | None:
        """Accept either a username or a raw user_id and return the user_id."""
        if username_or_id.isdigit():
            return username_or_id
        user = self.user(username_or_id)
        return user.get("user_id") if user else None

    def selftest(self, username: str | None = None) -> list[tuple[str, str, str]]:
        """Hit every endpoint we depend on and report status.

        Returns (endpoint, status, detail) rows. Run this from a machine with
        outbound network access before draft day -- it is the fastest way to
        confirm nothing upstream has moved.
        """
        results: list[tuple[str, str, str]] = []

        def check(label: str, fn: Any) -> Any:
            try:
                out = fn()
            except Exception as exc:  # noqa: BLE001 - selftest reports, never raises
                results.append((label, "FAIL", f"{type(exc).__name__}: {exc}"))
                return None
            if out is None or (hasattr(out, "__len__") and len(out) == 0):
                results.append((label, "EMPTY", "reachable but returned nothing"))
                return out
            size = len(out) if hasattr(out, "__len__") else 1
            results.append((label, "OK", f"{size} items"))
            return out

        state = check("GET /state/nfl", self.state)
        season = int((state or {}).get("season") or 2026)

        user = None
        if username:
            user = check(f"GET /user/{username}", lambda: self.user(username))
        if user and user.get("user_id"):
            uid = user["user_id"]
            leagues = check(
                f"GET /user/{uid}/leagues/nfl/{season}",
                lambda: self.user_leagues(uid, season),
            )
            check(
                f"GET /user/{uid}/drafts/nfl/{season}",
                lambda: self.user_drafts(uid, season),
            )
            if leagues:
                lid = leagues[0]["league_id"]
                check(f"GET /league/{lid}", lambda: self.league(lid))
                check(f"GET /league/{lid}/rosters", lambda: self.league_rosters(lid))
                check(f"GET /league/{lid}/users", lambda: self.league_users(lid))
                drafts = check(f"GET /league/{lid}/drafts", lambda: self.league_drafts(lid))
                if drafts:
                    did = drafts[0]["draft_id"]
                    check(f"GET /draft/{did}", lambda: self.draft(did))
                    check(f"GET /draft/{did}/picks", lambda: self.draft_picks(did))

        check("GET /players/nfl", lambda: self.players())
        check("GET /players/nfl/trending/add", lambda: self.trending("add"))
        return results
