"""Проверки HTTP-контракта и проверки конфигурации при запуске FastAPI."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigurationError, Settings
from app.database import (
    ConversationRepository,
    DatabaseError,
    KnowledgeRepository,
    MessageRepository,
    initialize_database,
    open_database,
)
from app.main import create_app


@pytest.mark.usefixtures("configured_environment")
def test_health_endpoint() -> None:
    """API отдаёт стабильный JSON после успешной проверки конфигурации."""
    application = create_app()
    with TestClient(application) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["content-type"] == "application/json"
        assert isinstance(application.state.settings, Settings)
        assert "example-test-key-not-real" not in response.text


def test_application_refuses_start_without_key() -> None:
    """Ошибочная конфигурация останавливает приложение до обслуживания запросов."""
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"), TestClient(create_app()):
        pytest.fail("Приложение не должно запускаться без обязательных настроек")


@pytest.mark.usefixtures("configured_environment")
def test_startup_initializes_database(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FastAPI автоматически создаёт SQLite v1 до обслуживания HTTP-запросов.

    Args:
        monkeypatch: Средство временного изменения настроек.
        tmp_path: Изолированный каталог теста.
    """
    path = tmp_path / "api" / "shop.sqlite3"
    monkeypatch.setenv("SQLITE_PATH", str(path))
    assert not path.exists()
    with TestClient(create_app()) as client:
        assert path.exists()
        assert client.get("/health").status_code == 200
        with open_database(path) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            assert KnowledgeRepository(connection).read().products[0].price_rubles == 2490


@pytest.mark.usefixtures("configured_environment")
@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE products SET price_rubles = 2500",
        "UPDATE store_settings SET discounts_allowed = 1",
    ],
)
def test_startup_rejects_changed_regulation_without_changing_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, statement: str
) -> None:
    """Изменение исходного факта блокирует запуск без удаления истёкшей истории.

    Args:
        monkeypatch: Средство временного изменения настроек.
        tmp_path: Изолированный каталог теста.
        statement: SQL-допустимое изменение исходного бизнес-факта.
    """
    path = tmp_path / "shop.sqlite3"
    monkeypatch.setenv("SQLITE_PATH", str(path))
    old_time = datetime.now(UTC) - timedelta(days=2)
    initialize_database(path, now=old_time)
    with open_database(path) as connection:
        conversation, _ = ConversationRepository(connection).create(now=old_time)
        MessageRepository(connection).append_user_message(
            conversation.session_token_hash, "Сохранить при ошибке", now=old_time
        )
        connection.execute(statement)
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="Исходный регламент"), TestClient(create_app()):
        pytest.fail("Приложение не должно запускаться при изменённом регламенте")
    assert path.read_bytes() == before


@pytest.mark.usefixtures("configured_environment")
def test_startup_cleans_only_expired_conversations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Запуск API применяет TTL из настроек и каскадно удаляет старую историю.

    Args:
        monkeypatch: Средство временного изменения настроек.
        tmp_path: Изолированный каталог теста.
    """
    path = tmp_path / "shop.sqlite3"
    monkeypatch.setenv("SQLITE_PATH", str(path))
    monkeypatch.setenv("CONVERSATION_TTL_HOURS", "1")
    current = datetime.now(UTC)
    initialize_database(path, now=current)
    with open_database(path) as connection:
        conversations = ConversationRepository(connection)
        expired, _ = conversations.create(now=current - timedelta(hours=2))
        active, _ = conversations.create(now=current)
        MessageRepository(connection).append_user_message(
            expired.session_token_hash, "Старое", now=current - timedelta(hours=2)
        )
        MessageRepository(connection).append_user_message(
            active.session_token_hash, "Новое", now=current
        )
    with TestClient(create_app()), open_database(path) as connection:
        assert connection.execute("SELECT id FROM conversations").fetchone()[0] == active.id
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
