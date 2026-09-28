import pytest

from app import config


def test_missing_keys_raise_clear_error(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        config.validate_env()
    assert "GROQ_API_KEY" in str(exc.value)
    assert "TAVILY_API_KEY" in str(exc.value)


def test_present_keys_pass(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "y")
    config.validate_env()
