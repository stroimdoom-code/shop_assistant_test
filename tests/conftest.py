"""Изолированные настройки тестов без локальных ключей и внешних сервисов."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Изолировать рабочий каталог и переменные настроек для каждого теста.

    Args:
        monkeypatch: Средство временного изменения окружения pytest.
        tmp_path: Отдельный каталог теста, не содержащий пользовательский .env.
    """
    monkeypatch.chdir(tmp_path)
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_ATTEMPTS",
        "SQLITE_PATH",
        "CONVERSATION_TTL_HOURS",
        "CLASSIFIER_CONTEXT_MESSAGES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Установить явно фиктивный ключ для проверок каркаса без вызовов API.

    Args:
        monkeypatch: Средство временного изменения окружения pytest.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "example-test-key-not-real")
