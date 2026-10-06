import pytest


@pytest.fixture(autouse=True)
def _no_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default path must stay local even if a developer key is exported."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
