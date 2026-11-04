"""Проверки источников настроек, ограничений значений и защиты секретов."""

from pathlib import Path

import pytest

from app.config import ConfigurationError, load_settings


@pytest.mark.usefixtures("configured_environment")
def test_default_settings() -> None:
    """Настройки соответствуют согласованным значениям архитектуры."""
    settings = load_settings()

    assert settings.deepseek_api_key.get_secret_value() == "example-test-key-not-real"
    assert str(settings.deepseek_base_url) == "https://api.deepseek.com/"
    assert settings.deepseek_model == "deepseek-flash"
    assert settings.deepseek_timeout_seconds == 30
    assert settings.deepseek_max_attempts == 2
    assert settings.sqlite_path == Path("data/shop-assistant.sqlite3")
    assert settings.conversation_ttl_hours == 24
    assert settings.classifier_context_messages == 5
    assert "example-test-key-not-real" not in repr(settings)


def test_missing_key_is_reported_without_validation_details() -> None:
    """Отсутствующий обязательный ключ сопровождается понятной ошибкой."""
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY") as caught:
        load_settings()

    assert "input_value" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.usefixtures("configured_environment")
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DEEPSEEK_API_KEY", ""),
        ("DEEPSEEK_API_KEY", "   "),
        ("DEEPSEEK_MODEL", ""),
        ("DEEPSEEK_MODEL", "   "),
        ("DEEPSEEK_BASE_URL", "not-a-url"),
        ("DEEPSEEK_BASE_URL", "http://api.deepseek.com"),
        ("DEEPSEEK_BASE_URL", "https://example-user:example-password@api.deepseek.com"),
        ("DEEPSEEK_TIMEOUT_SECONDS", "0"),
        ("DEEPSEEK_TIMEOUT_SECONDS", "31"),
        ("DEEPSEEK_TIMEOUT_SECONDS", "nan"),
        ("DEEPSEEK_TIMEOUT_SECONDS", "inf"),
        ("DEEPSEEK_MAX_ATTEMPTS", "0"),
        ("DEEPSEEK_MAX_ATTEMPTS", "3"),
        ("CONVERSATION_TTL_HOURS", "0"),
        ("CONVERSATION_TTL_HOURS", "invalid-sensitive-value"),
        ("CLASSIFIER_CONTEXT_MESSAGES", "-1"),
        ("CLASSIFIER_CONTEXT_MESSAGES", "1.5"),
        ("SQLITE_PATH", ""),
        ("SQLITE_PATH", "   "),
    ],
)
def test_invalid_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Некорректные настройки отклоняются без раскрытия переданных значений.

    Args:
        monkeypatch: Средство временного изменения окружения pytest.
        name: Имя проверяемой переменной.
        value: Некорректное демонстрационное значение.
    """
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=name) as caught:
        load_settings()

    assert "example-test-key-not-real" not in str(caught.value)
    if value.strip():
        assert value not in str(caught.value)


def test_dotenv_loading_and_environment_priority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Файл .env загружается, а переменные окружения имеют приоритет.

    Args:
        monkeypatch: Средство временного изменения окружения pytest.
        tmp_path: Изолированный рабочий каталог.
    """
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=example-dotenv-key-not-real\n"
        "DEEPSEEK_TIMEOUT_SECONDS=15\n"
        "DEEPSEEK_MAX_ATTEMPTS=1\n"
        "CONVERSATION_TTL_HOURS=48\n"
        "SQLITE_PATH=local/test.sqlite3\n"
        "DEEPSEEK_MODEL= example-model \n",
        encoding="utf-8",
    )
    from_file = load_settings()
    assert from_file.deepseek_api_key.get_secret_value() == "example-dotenv-key-not-real"
    assert from_file.deepseek_timeout_seconds == 15
    assert from_file.deepseek_max_attempts == 1
    assert from_file.conversation_ttl_hours == 48
    assert from_file.sqlite_path == Path("local/test.sqlite3")
    assert from_file.deepseek_model == "example-model"

    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "20")
    assert load_settings().deepseek_timeout_seconds == 20


@pytest.mark.usefixtures("configured_environment")
@pytest.mark.parametrize("limit", [0, 1, 8])
def test_classifier_context_limit_from_environment(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    """Настройка истории принимает неотрицательные целые числа, включая отключение.

    Args:
        monkeypatch: Средство временного изменения окружения.
        limit: Число предыдущих сообщений для классификатора.
    """
    monkeypatch.setenv("CLASSIFIER_CONTEXT_MESSAGES", str(limit))
    assert load_settings().classifier_context_messages == limit
