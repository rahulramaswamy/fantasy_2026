"""Tests for Sleeper identity resolution.

Getting this wrong is invisible and expensive: a username sitting in the
user_id field compares against numeric roster owner ids, never matches, and
every league command reports that you are not in your own league.
"""

import pytest
import typer

from ff2026 import cli
from ff2026.config import Settings


class StubClient:
    """Mimics SleeperClient.resolve_user_id: digits pass through, names look up."""

    KNOWN = {"rahulr2000": "868566942836654080"}

    def __init__(self):
        self.lookups = []

    def resolve_user_id(self, username_or_id: str):
        self.lookups.append(username_or_id)
        if username_or_id.isdigit():
            return username_or_id
        return self.KNOWN.get(username_or_id)


@pytest.fixture
def settings(monkeypatch):
    def _set(**kwargs):
        monkeypatch.setattr(cli, "get_settings", lambda: Settings(**kwargs))
    return _set


def test_username_in_the_user_id_field_still_resolves(settings):
    """The exact mistake that broke a real league lookup."""
    settings(sleeper_user_id="rahulr2000")
    uid, label = cli._resolve_me(StubClient(), None)
    assert uid == "868566942836654080"
    assert label == "rahulr2000"


def test_numeric_user_id_passes_through(settings):
    settings(sleeper_user_id="868566942836654080")
    uid, _ = cli._resolve_me(StubClient(), None)
    assert uid == "868566942836654080"


def test_explicit_username_beats_stored_id(settings):
    settings(sleeper_user_id="999999", sleeper_username="someone_else")
    client = StubClient()
    uid, label = cli._resolve_me(client, "rahulr2000")
    assert uid == "868566942836654080"
    assert label == "rahulr2000"
    assert client.lookups == ["rahulr2000"]


def test_falls_back_to_username_from_env(settings):
    settings(sleeper_username="rahulr2000")
    uid, _ = cli._resolve_me(StubClient(), None)
    assert uid == "868566942836654080"


def test_unknown_user_exits_with_a_clear_error(settings):
    settings(sleeper_username="not_a_real_user")
    with pytest.raises(typer.Exit):
        cli._resolve_me(StubClient(), None)


def test_no_identity_configured_exits(settings):
    settings()
    with pytest.raises(typer.Exit):
        cli._resolve_me(StubClient(), None)


def test_whitespace_is_tolerated(settings):
    settings(sleeper_username="  rahulr2000  ")
    uid, _ = cli._resolve_me(StubClient(), None)
    assert uid == "868566942836654080"
