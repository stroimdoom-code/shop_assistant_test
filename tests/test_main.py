"""Проверки HTTP API, сессий, безопасных ошибок и раздачи сборки React через FastAPI."""

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.assistant import AssistantTimeoutError, AssistantUnavailableError
from app.config import ConfigurationError, Settings
from app.database import (
    ConversationRepository,
    DatabaseError,
    KnowledgeRepository,
    Message,
    MessageRepository,
    hash_session_token,
    initialize_database,
    open_database,
)
from app.main import create_app, get_assistant_service


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


class _AssistantStub:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    async def process(self, session_token_hash: str, message: str) -> Message:
        """Сохранить пару сообщений или имитировать сбой после сохранения вопроса.

        Args:
            session_token_hash: Серверный хеш токена.
            message: Неизменённый пользовательский текст.

        Returns:
            Сохранённый ответ с временной отметкой из SQLite.

        Raises:
            Exception: Явно заданная тестом ошибка сервиса.
        """
        self.calls.append((session_token_hash, message))
        with open_database(self.settings.sqlite_path) as connection:
            messages = MessageRepository(connection, self.settings.conversation_ttl_hours)
            saved = messages.append_user_message(session_token_hash, message)
            if self.error is not None:
                raise self.error
            return messages.append_assistant_message(saved.conversation_id, "Проверенный ответ")


@pytest.fixture
def chat_client(configured_environment: None) -> Iterator[tuple[TestClient, _AssistantStub]]:
    """Создать HTTP-клиент с подменённым сервисом и настоящей временной SQLite.

    Args:
        configured_environment: Изолированная конфигурация с фиктивным ключом.

    Yields:
        HTTP-клиент и записывающая подмена сервиса без вызовов DeepSeek.
    """
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        stub = _AssistantStub(application.state.settings)
        application.dependency_overrides[get_assistant_service] = lambda: stub
        yield client, stub


def test_chat_creates_and_continues_session_with_complete_history(
    chat_client: tuple[TestClient, _AssistantStub],
) -> None:
    """POST создаёт одну сессию, сохраняет исходный текст и возвращает полную историю.

    Args:
        chat_client: Клиент с подменённым сервисом.
    """
    client, stub = chat_client
    assert client.get("/api/chat/history").json() == {"messages": []}
    assert not stub.calls
    timestamps = []
    original = "  Сколько стоит X1?\nНе сокращайте текст.  "
    for index in range(6):
        text = original if index == 0 else f"Уточнение {index}"
        response = client.post("/api/chat/messages", json={"message": text})
        assert response.status_code == 200
        assert set(response.json()) == {"answer", "created_at"}
        assert response.json()["answer"] == "Проверенный ответ"
        assert datetime.fromisoformat(response.json()["created_at"]).utcoffset() == timedelta(0)
        timestamps.append(response.json()["created_at"])
        assert response.headers["cache-control"] == "no-store"
    token = client.cookies.get("shop_assistant_session")
    assert token is not None and len(token) == 43
    assert stub.calls[0] == (hash_session_token(token), original)
    assert len({item[0] for item in stub.calls}) == 1
    response = client.get("/api/chat/history")
    assert response.status_code == 200 and "set-cookie" not in response.headers
    history = response.json()["messages"]
    assert len(history) == 12
    assert [item["role"] for item in history] == ["user", "assistant"] * 6
    assert history[0]["content"] == original
    assert [item["created_at"] for item in history[1::2]] == timestamps
    assert all(set(item) == {"role", "content", "created_at"} for item in history)
    dates = [datetime.fromisoformat(item["created_at"]) for item in history]
    assert dates == sorted(dates)
    assert all(item.utcoffset() == timedelta(0) for item in dates)
    assert token not in response.text and hash_session_token(token) not in response.text
    with open_database(stub.settings.sqlite_path) as connection:
        rows = connection.execute("SELECT * FROM conversations").fetchall()
        assert len(rows) == 1 and rows[0]["session_token_hash"] == hash_session_token(token)
        assert token not in str([tuple(row) for row in rows])


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.usefixtures("configured_environment")
def test_cookie_security_and_configured_lifetime(
    monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    """Cookie ограничена чат-маршрутами, временем TTL и подходящим транспортом.

    Args:
        monkeypatch: Средство подмены настройки TTL.
        scheme: Протокол тестового HTTP-клиента.
    """
    monkeypatch.setenv("CONVERSATION_TTL_HOURS", "2")
    application = create_app()
    with TestClient(application, base_url=f"{scheme}://testserver") as client:
        stub = _AssistantStub(application.state.settings)
        application.dependency_overrides[get_assistant_service] = lambda: stub
        response = client.post("/api/chat/messages", json={"message": "Привет"})
        cookie = response.headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=lax" in cookie
        assert "Path=/api/chat" in cookie and "Max-Age=7200" in cookie
        assert "Domain=" not in cookie
        assert ("Secure" in cookie) == (scheme == "https")
        assert client.get("/api/chat/history").json()["messages"]
        assert "shop_assistant_session" not in client.get("/health").request.headers.get(
            "cookie", ""
        )


@pytest.mark.parametrize("state", ["absent", "malformed", "empty", "unknown", "expired"])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_invalid_or_absent_cookie_is_cleared_or_replaced(
    chat_client: tuple[TestClient, _AssistantStub], state: str, method: str
) -> None:
    """GET не создаёт сессию, а POST заменяет недействительную cookie новой.

    Args:
        chat_client: HTTP-клиент и подмена сервиса.
        state: Отсутствующая или недействительная cookie.
        method: Чтение истории либо отправка сообщения.
    """
    client, stub = chat_client
    token = {"malformed": "invalid-token", "empty": "", "unknown": "A" * 43}.get(state)
    if state == "expired":
        old = datetime.now(UTC) - timedelta(hours=25)
        with open_database(stub.settings.sqlite_path) as connection:
            conversation, token = ConversationRepository(connection).create(now=old)
            MessageRepository(connection).append_user_message(
                conversation.session_token_hash, "Просроченная история", now=old
            )
    if token is not None:
        client.cookies.set(
            "shop_assistant_session", token, domain="testserver.local", path="/api/chat"
        )
    response = (
        client.get("/api/chat/history")
        if method == "GET"
        else client.post("/api/chat/messages", json={"message": "Новое сообщение"})
    )
    assert response.status_code == 200
    with open_database(stub.settings.sqlite_path) as connection:
        count = connection.execute("SELECT count(*) FROM conversations").fetchone()[0]
        messages = connection.execute("SELECT count(*) FROM messages").fetchone()[0]
    if method == "GET":
        assert response.json() == {"messages": []}
        assert count == 0 and messages == 0 and not stub.calls
        assert client.cookies.get("shop_assistant_session") is None
        if state == "absent":
            assert "set-cookie" not in response.headers
        else:
            cookie = response.headers["set-cookie"]
            assert "Max-Age=0" in cookie and "Path=/api/chat" in cookie
            assert "HttpOnly" in cookie and "SameSite=lax" in cookie
    else:
        replacement = client.cookies.get("shop_assistant_session")
        assert replacement and replacement != token
        assert count == 1 and messages == 2 and len(stub.calls) == 1


def test_get_history_neither_renews_cookie_nor_activity(
    chat_client: tuple[TestClient, _AssistantStub],
) -> None:
    """Чтение истории не обновляет серверный TTL и время жизни cookie.

    Args:
        chat_client: Клиент с изолированной сессией.
    """
    client, stub = chat_client
    client.post("/api/chat/messages", json={"message": "Начало"})
    token = client.cookies.get("shop_assistant_session")
    assert token is not None
    with open_database(stub.settings.sqlite_path) as connection:
        before = ConversationRepository(connection).find_active(hash_session_token(token))
    for _ in range(2):
        response = client.get("/api/chat/history")
        assert response.status_code == 200 and "set-cookie" not in response.headers
    with open_database(stub.settings.sqlite_path) as connection:
        after = ConversationRepository(connection).find_active(hash_session_token(token))
    assert before == after


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": " \n\t"},
        {"message": "я" * 2001},
        {"message": 123},
        {"message": True},
        {"message": None},
        {"message": ["текст"]},
        {"message": "Вопрос", "role": "assistant"},
        {"message": "Вопрос", "history": []},
        {"message": "Вопрос", "messages": []},
        {"message": "Вопрос", "conversation_id": "чужой"},
        {"message": "Вопрос", "session_token": "чужой"},
        ["Вопрос"],
    ],
)
def test_invalid_body_is_safe_and_has_no_side_effects(
    chat_client: tuple[TestClient, _AssistantStub], payload: object
) -> None:
    """Некорректный ввод и попытки назначить историю или роли отклоняются до сервиса.

    Args:
        chat_client: Клиент и записывающая подмена.
        payload: Некорректный JSON-документ.
    """
    client, stub = chat_client
    response = client.post("/api/chat/messages", json=payload)
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert response.json()["error"]["code"] == "invalid_request"
    assert set(response.json()["error"]) == {"code", "message"}
    assert "set-cookie" not in response.headers and not stub.calls
    with open_database(stub.settings.sqlite_path) as connection:
        assert connection.execute("SELECT count(*) FROM conversations").fetchone()[0] == 0


def test_maximum_length_and_raw_json_validation(
    chat_client: tuple[TestClient, _AssistantStub],
) -> None:
    """Допускаются 2000 символов, повреждённый JSON не раскрывает исходный ввод.

    Args:
        chat_client: Клиент с подменой сервиса.
    """
    client, stub = chat_client
    message = " я" * 1000
    assert client.post("/api/chat/messages", json={"message": message}).status_code == 200
    assert stub.calls[0][1] == message
    response = client.post(
        "/api/chat/messages",
        content='{"message": secret-value',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422 and "secret-value" not in response.text
    assert len(stub.calls) == 1


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (AssistantUnavailableError("private-key SQL secret"), 503, "service_unavailable"),
        (AssistantTimeoutError("private-key SQL secret"), 504, "request_timeout"),
        (TimeoutError("private-key SQL secret"), 504, "request_timeout"),
        (RuntimeError("private-key SQL secret"), 500, "internal_error"),
    ],
)
def test_service_errors_keep_question_and_session_without_saving_error_answer(
    chat_client: tuple[TestClient, _AssistantStub],
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    status: int,
    code: str,
) -> None:
    """HTTP-ошибка сохраняет cookie новой сессии и уже записанный вопрос.

    Args:
        chat_client: Клиент с подменённым сервисом.
        caplog: Перехваченные журнальные сообщения.
        error: Ошибка после сохранения пользовательского сообщения.
        status: Ожидаемый HTTP-статус.
        code: Машинный код ответа.
    """
    client, stub = chat_client
    stub.error = error
    response = client.post("/api/chat/messages", json={"message": "Сохраните мой вопрос"})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert set(response.json()["error"]) == {"code", "message"}
    assert "private-key" not in response.text and "SQL" not in response.text
    assert "private-key" not in caplog.text
    assert "HttpOnly" in response.headers["set-cookie"]
    history = client.get("/api/chat/history").json()["messages"]
    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "Сохраните мой вопрос")
    ]
    stub.error = None
    assert client.post("/api/chat/messages", json={"message": "Повтор"}).status_code == 200
    assert stub.calls[0][0] == stub.calls[1][0]


@pytest.mark.parametrize("route", ["/api/chat/history", "/api/chat/messages"])
def test_sqlite_session_error_is_safe(
    chat_client: tuple[TestClient, _AssistantStub], monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """Ошибка получения сессии даёт 503 без SQL, ключей и вызова ассистента.

    Args:
        chat_client: Изолированный клиент.
        monkeypatch: Подмена чтения сессии.
        route: Маршрут истории или отправки.
    """
    client, stub = chat_client

    def fail_session(settings: Settings, token: str | None, *, create: bool) -> tuple[None, None]:
        raise sqlite3.OperationalError("private-key SELECT secret")

    monkeypatch.setattr(main_module, "_resolve_session", fail_session)
    response = (
        client.get(route)
        if route.endswith("history")
        else client.post(route, json={"message": "Вопрос"})
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert "private-key" not in response.text and "SELECT" not in response.text
    assert not stub.calls


@pytest.mark.parametrize("route", ["/api/chat/history", "/api/chat/messages"])
def test_session_resolution_failure_preserves_valid_cookie_and_history(
    chat_client: tuple[TestClient, _AssistantStub], monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """Временный сбой SQLite не удаляет действующую cookie и доступ к истории.

    Args:
        chat_client: HTTP-клиент с подменённым сервисом и временной SQLite.
        monkeypatch: Средство временной подмены разрешения сессии.
        route: Маршрут чтения истории либо отправки сообщения.
    """
    client, stub = chat_client
    assert client.post("/api/chat/messages", json={"message": "Первый вопрос"}).status_code == 200
    token = client.cookies.get("shop_assistant_session")
    assert token is not None
    before = client.get("/api/chat/history").json()

    def fail_session(settings: Settings, token: str | None, *, create: bool) -> tuple[None, None]:
        raise sqlite3.OperationalError("Временный сбой SQLite")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(main_module, "_resolve_session", fail_session)
        response = (
            client.get(route)
            if route.endswith("history")
            else client.post(route, json={"message": "Не сохранять при сбое"})
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"
    assert "set-cookie" not in response.headers
    assert client.cookies.get("shop_assistant_session") == token
    assert len(stub.calls) == 1
    assert client.get("/api/chat/history").json() == before
    assert client.post("/api/chat/messages", json={"message": "Повтор"}).status_code == 200
    assert client.cookies.get("shop_assistant_session") == token
    assert stub.calls[0][0] == stub.calls[1][0] == hash_session_token(token)
    assert len(client.get("/api/chat/history").json()["messages"]) == 4


def test_history_isolated_by_cookie_and_openapi_contract(
    chat_client: tuple[TestClient, _AssistantStub],
) -> None:
    """Другой токен не открывает историю; OpenAPI описывает ответы и безопасные ошибки.

    Args:
        chat_client: Клиент и подмена сервиса.
    """
    client, stub = chat_client
    client.post("/api/chat/messages", json={"message": "Приватный вопрос"})
    first_token = client.cookies.get("shop_assistant_session")
    client.cookies.clear()
    assert client.get("/api/chat/history").json() == {"messages": []}
    client.post("/api/chat/messages", json={"message": "Другой диалог"})
    history = client.get("/api/chat/history").json()["messages"]
    assert len(history) == 2 and history[0]["content"] == "Другой диалог"
    assert first_token != client.cookies.get("shop_assistant_session")
    assert stub.calls[0][0] != stub.calls[1][0]
    schema = client.get("/openapi.json").json()
    post = schema["paths"]["/api/chat/messages"]["post"]
    assert {"200", "422", "503", "504", "500"} <= set(post["responses"])
    assert schema["components"]["schemas"]["ChatRequest"]["additionalProperties"] is False
    assert client.get("/does-not-exist").json()["error"]["code"] == "http_error"
    preflight = client.options("/api/chat/messages", headers={"origin": "https://other.example"})
    assert "access-control-allow-origin" not in preflight.headers
    assert preflight.status_code == 405


@pytest.mark.usefixtures("configured_environment")
def test_production_chat_and_assets_do_not_intercept_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Сборка раздаётся только на корне и assets, сохраняя маршруты и ошибки API.

    Args:
        monkeypatch: Подмена расположения production-сборки.
        tmp_path: Временный каталог с имитацией результата Vite.
    """
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<html lang="ru"><div id="root"></div></html>', encoding="utf-8"
    )
    (assets / "chat.js").write_text('const title = "Помощник магазина";', encoding="utf-8")
    (assets / "chat.css").write_text("body { margin: 0; }", encoding="utf-8")
    monkeypatch.setattr(main_module, "_WEB_DIST", dist)
    application = create_app()
    with TestClient(application) as client:
        stub = _AssistantStub(application.state.settings)
        application.dependency_overrides[get_assistant_service] = lambda: stub
        index = client.get("/")
        assert index.status_code == 200 and '<div id="root">' in index.text
        assert index.headers["content-type"].startswith("text/html")
        assert index.headers["cache-control"] == "no-cache"
        assert client.get("/assets/chat.js").status_code == 200
        assert client.get("/assets/chat.css").headers["content-type"].startswith("text/css")
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/chat/history").json() == {"messages": []}
        assert client.post("/api/chat/messages", json={"message": "Вопрос"}).status_code == 200
        assert len(client.get("/api/chat/history").json()["messages"]) == 2
        assert client.get("/api/missing").status_code == 404
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/another-page").status_code == 404
        assert client.get("/assets/%2e%2e/index.html").status_code == 404
        assert client.get("/openapi.json").status_code == 200
        assert "/" not in client.get("/openapi.json").json()["paths"]


@pytest.mark.usefixtures("configured_environment")
def test_missing_frontend_build_does_not_prevent_api_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """API работает без Node и сборки; корень безопасно сообщает недоступность чата.

    Args:
        monkeypatch: Подмена пути на отсутствующую сборку.
        tmp_path: Изолированный каталог теста.
    """
    monkeypatch.setattr(main_module, "_WEB_DIST", tmp_path / "missing-dist")
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/chat/history").json() == {"messages": []}
        response = client.get("/")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"
        assert "missing-dist" not in response.text
