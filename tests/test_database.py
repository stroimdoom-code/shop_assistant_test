"""Проверки SQLite v1, исходных фактов, транзакций и изоляции диалогов."""

import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app import database
from app.database import (
    ConversationRepository,
    DatabaseError,
    KnowledgeRepository,
    MessageRepository,
    SessionUnavailableError,
    hash_session_token,
    initialize_database,
    open_database,
)

NOW = datetime(2026, 11, 4, 12, 0, 0, 123456, tzinfo=UTC)
SqlValue = str | int | float | None


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    """Создать отдельную SQLite v1 с фиксированными часами.

    Args:
        tmp_path: Изолированный каталог теста.

    Returns:
        Путь к инициализированной БД.
    """
    path = tmp_path / "nested" / "shop.sqlite3"
    initialize_database(path, now=NOW)
    return path


def _snapshot(path: Path) -> tuple[str, ...]:
    with open_database(path) as connection:
        return tuple(connection.iterdump())


def test_schema_seed_and_six_facts(database_path: Path) -> None:
    """Шесть таблиц и два индекса представляют ровно исходный регламент.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        assert tables == {
            "products",
            "delivery_policy",
            "return_policy",
            "store_settings",
            "conversations",
            "messages",
        }
        assert indexes == {"idx_conversations_last_activity", "idx_messages_conversation_created"}
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM conversations").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        facts = KnowledgeRepository(connection).read()
        assert len(facts.products) == 1
        assert facts.products[0].name == "Наушники X1"
        assert facts.products[0].price_rubles == 2490
        assert facts.products[0].is_in_stock
        assert facts.delivery == database.DeliveryPolicy("курьер", 1, "18:00", "21:00")
        assert facts.returns.period_days == 14
        assert facts.returns.refund_method == "same_card"
        assert facts.returns.refund_working_days == 3
        assert facts.returns.opened_without_defect_allowed
        assert facts.returns.defect_accepted_for_expertise
        assert not facts.returns.immediate_refund_promised
        assert not facts.discounts_allowed


def test_reinitialization_preserves_data_and_extra_products(database_path: Path) -> None:
    """Повторный запуск сохраняет активную историю и дополнительные товары.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        connection.execute(
            "INSERT INTO products (id, name, price_rubles, is_in_stock) VALUES (?, ?, ?, ?)",
            (9, "Дополнительный тестовый товар", 500, 0),
        )
        # Идентификатор начального товара технический, имя и факты остаются исходными.
        connection.execute("UPDATE products SET id = ? WHERE id = ?", (7, 1))
        conversation, token = ConversationRepository(connection).create(now=NOW)
        MessageRepository(connection).append_user_message(
            hash_session_token(token), "Вопрос", now=NOW
        )
        assert not connection.in_transaction
    before = _snapshot(database_path)
    initialize_database(database_path, now=NOW + timedelta(hours=1))
    assert _snapshot(database_path) == before
    with open_database(database_path) as connection:
        assert len(KnowledgeRepository(connection).read().products) == 2
        assert (
            MessageRepository(connection)
            .get_history(conversation.session_token_hash, now=NOW)[0]
            .content
            == "Вопрос"
        )


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM products",
        "DELETE FROM delivery_policy",
        "DELETE FROM return_policy",
        "DELETE FROM store_settings",
        "UPDATE products SET price_rubles = 2500 WHERE id = 1",
        "UPDATE products SET is_in_stock = 0 WHERE id = 1",
        "UPDATE products SET name = 'Другой товар' WHERE id = 1",
        "UPDATE delivery_policy SET delivery_days_offset = 2",
        "UPDATE delivery_policy SET slot_start = '17:00'",
        "UPDATE delivery_policy SET slot_end = '22:00'",
        "UPDATE return_policy SET return_period_days = 15",
        "UPDATE return_policy SET refund_processing_working_days = 4",
        "UPDATE return_policy SET opened_item_without_defect_allowed = 0",
        "UPDATE return_policy SET defect_accepted_for_expertise = 0",
        "UPDATE return_policy SET immediate_refund_promised = 1",
        "UPDATE store_settings SET discounts_allowed = 1",
        "INSERT INTO products VALUES (2, 'Наушники X1', 2500, 1)",
    ],
)
def test_changed_or_missing_facts_refuse_start_without_repair(
    database_path: Path, statement: str
) -> None:
    """Python отклоняет даже SQL-допустимые изменения, сохраняя файл и историю.

    Args:
        database_path: Изолированная SQLite v1.
        statement: Изменение или удаление обязательного факта.
    """
    with open_database(database_path) as connection:
        conversation, _ = ConversationRepository(connection).create(now=NOW)
        MessageRepository(connection).append_user_message(
            conversation.session_token_hash, "Сохранить историю", now=NOW
        )
        connection.execute(statement)
    before_bytes = database_path.read_bytes()
    before_data = _snapshot(database_path)
    # Истёкший диалог также должен сохраниться: проверка регламента предшествует очистке.
    with pytest.raises(DatabaseError):
        initialize_database(database_path, now=NOW + timedelta(hours=48))
    assert database_path.read_bytes() == before_bytes
    assert _snapshot(database_path) == before_data
    with open_database(database_path) as connection, pytest.raises(DatabaseError):
        KnowledgeRepository(connection).read()


@pytest.mark.parametrize(
    ("statement", "parameters"),
    [
        ("INSERT INTO products VALUES (?, ?, ?, ?)", (2, "", 100, 1)),
        ("INSERT INTO products VALUES (?, ?, ?, ?)", (2, "Товар", 0, 1)),
        ("INSERT INTO products VALUES (?, ?, ?, ?)", (2, "Товар", 1.5, 1)),
        ("INSERT INTO products VALUES (?, ?, ?, ?)", (2, "Товар", "не число", 1)),
        ("INSERT INTO products VALUES (?, ?, ?, ?)", (2, "Товар", 100, 2)),
        ("INSERT INTO products VALUES (?, ?, ?, ?)", (2, None, 100, 1)),
        ("UPDATE delivery_policy SET id = ?", (2,)),
        ("UPDATE delivery_policy SET delivery_method = ?", ("courier",)),
        ("UPDATE delivery_policy SET delivery_days_offset = ?", (1.5,)),
        ("UPDATE delivery_policy SET slot_start = ?", ("24:00",)),
        ("UPDATE delivery_policy SET slot_end = ?", ("21:60",)),
        ("UPDATE delivery_policy SET slot_start = ?", ("22:00",)),
        ("UPDATE return_policy SET id = ?", (2,)),
        ("UPDATE return_policy SET return_period_days = ?", (0,)),
        ("UPDATE return_policy SET return_period_days = ?", (2.5,)),
        ("UPDATE return_policy SET refund_method = ?", ("cash",)),
        ("UPDATE return_policy SET refund_processing_working_days = ?", (-1,)),
        ("UPDATE return_policy SET opened_item_without_defect_allowed = ?", (2,)),
        ("UPDATE return_policy SET defect_accepted_for_expertise = ?", (2,)),
        ("UPDATE return_policy SET immediate_refund_promised = ?", (2,)),
        ("UPDATE store_settings SET id = ?", (2,)),
        ("UPDATE store_settings SET discounts_allowed = ?", (2,)),
    ],
)
def test_sql_constraints(
    database_path: Path, statement: str, parameters: tuple[SqlValue, ...]
) -> None:
    """SQL отклоняет некорректную форму данных без изменения регламента.

    Args:
        database_path: Изолированная SQLite v1.
        statement: Параметризованный запрос с некорректными данными.
        parameters: Значения параметров.
    """
    before = _snapshot(database_path)
    with open_database(database_path) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(statement, parameters)
    assert _snapshot(database_path) == before


@pytest.mark.parametrize("phase", ["schema", "seed", "validation"])
def test_initialization_rolls_back_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Ошибка откатывает DDL, индексы, записи и user_version одной транзакцией.

    Args:
        tmp_path: Изолированный каталог.
        monkeypatch: Средство внедрения сбоя.
        phase: Этап, на котором возникает ошибка.
    """
    path = tmp_path / "rollback.sqlite3"
    if phase == "schema":
        monkeypatch.setattr(
            database,
            "_SCHEMA",
            {
                **database._SCHEMA,
                ("index", "broken"): "CREATE INDEX broken ON missing_table (id)",
            },
        )
    elif phase == "seed":
        monkeypatch.setattr(
            database,
            "_SEED",
            (
                *database._SEED,
                ("INSERT INTO missing_table (id) VALUES (?)", (1,)),
            ),
        )
    else:

        def fail_validation(connection: sqlite3.Connection) -> None:
            """Внедрить сбой после записи версии и начальных данных.

            Args:
                connection: Соединение внутри ещё не зафиксированной инициализации.
            """
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            assert connection.execute("SELECT count(*) FROM products").fetchone()[0] == 1
            raise DatabaseError("Внедрённый сбой после создания схемы и данных")

        monkeypatch.setattr(database, "_validate_database", fail_validation)
    with pytest.raises(DatabaseError):
        initialize_database(path, now=NOW)
    with open_database(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_schema").fetchall() == []


@pytest.mark.parametrize(
    "statement",
    [
        "PRAGMA user_version = 2",
        "DROP INDEX idx_messages_conversation_created",
        "ALTER TABLE products ADD COLUMN invented TEXT",
    ],
)
def test_unsupported_or_damaged_schema_is_not_changed(database_path: Path, statement: str) -> None:
    """Неизвестная версия и повреждённая схема не исправляются автоматически.

    Args:
        database_path: Изолированная SQLite v1.
        statement: Изменение версии или структуры.
    """
    with open_database(database_path) as connection:
        connection.execute(statement)
    before = database_path.read_bytes()
    with pytest.raises(DatabaseError):
        initialize_database(database_path, now=NOW)
    assert database_path.read_bytes() == before


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE unrelated (id INTEGER)",
        "CREATE TABLE sqlitex_other (id INTEGER)",
    ],
)
def test_nonempty_version_zero_is_rejected(tmp_path: Path, statement: str) -> None:
    """Непустая схема без версии не дополняется таблицами приложения.

    Args:
        tmp_path: Изолированный каталог.
        statement: Посторонняя схема с версией 0.
    """
    path = tmp_path / "foreign.sqlite3"
    with open_database(path) as connection:
        connection.execute(statement)
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="версией 0"):
        initialize_database(path, now=NOW)
    assert path.read_bytes() == before


def test_corrupted_file_is_preserved(tmp_path: Path) -> None:
    """Некорректный файл SQLite остаётся без изменений после понятной ошибки.

    Args:
        tmp_path: Изолированный каталог.
    """
    path = tmp_path / "broken.sqlite3"
    path.write_bytes(b"not a sqlite database")
    with pytest.raises(DatabaseError, match="Не удалось"):
        initialize_database(path, now=NOW)
    assert path.read_bytes() == b"not a sqlite database"


def test_connections_are_separate_configured_and_closed(database_path: Path) -> None:
    """Каждое соединение включает PRAGMA и закрывается владельцем контекста.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as first, open_database(database_path) as second:
        assert first is not second
        for connection in (first, second):
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        ConversationRepository(first).create(now=NOW)
        assert second.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        first.execute("SELECT 1")


def test_only_hash_is_stored_and_hash_is_unique(database_path: Path) -> None:
    """БД хранит уникальный SHA-256 без исходного токена.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        repository = ConversationRepository(connection)
        conversation, token = repository.create(now=NOW)
        assert len(token) >= 43
        assert len(conversation.session_token_hash) == 64
        assert UUID(conversation.id).version == 4
        assert repository.find_active(hash_session_token(token), now=NOW) == conversation
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                (
                    str(uuid4()),
                    conversation.session_token_hash,
                    "2026-11-04T12:00:00.123456Z",
                    "2026-11-04T12:00:00.123456Z",
                ),
            )
        for invalid_hash in ("a" * 63, "A" * 64, "g" * 64):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?, ?)",
                    (
                        str(uuid4()),
                        invalid_hash,
                        "2026-11-04T12:00:00.123456Z",
                        "2026-11-04T12:00:00.123456Z",
                    ),
                )
    assert token not in "\n".join(_snapshot(database_path))
    assert token.encode() not in database_path.read_bytes()


def test_history_is_isolated_ordered_and_uses_server_roles(database_path: Path) -> None:
    """История изолирована по сессии и упорядочена по времени, затем id.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        conversations = ConversationRepository(connection)
        messages = MessageRepository(connection)
        first, _ = conversations.create(now=NOW)
        second, _ = conversations.create(now=NOW)
        content = "'); DROP TABLE products; --"
        user = messages.append_user_message(
            first.session_token_hash, content, now=NOW + timedelta(seconds=2)
        )
        assert not connection.in_transaction
        assistant = messages.append_assistant_message(
            first.id, "Обычный ответ", now=NOW + timedelta(seconds=2)
        )
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (first.id, "user", "Более раннее время", "2026-11-04T12:00:01.123456Z"),
        )
        messages.append_user_message(second.session_token_hash, "Другой диалог", now=NOW)
        history = messages.get_history(first.session_token_hash, now=NOW + timedelta(seconds=3))
        assert [item.content for item in history] == [
            "Более раннее время",
            content,
            "Обычный ответ",
        ]
        assert [item.role for item in history] == ["user", "user", "assistant"]
        assert history[1:] == (user, assistant)
        assert all(item.conversation_id == first.id for item in history)
        assert (
            messages.get_history(second.session_token_hash, now=NOW)[0].content == "Другой диалог"
        )
        assert connection.execute("SELECT count(*) FROM products").fetchone()[0] == 1
        found = conversations.find_active(first.session_token_hash, now=NOW)
        assert found is not None
        assert found.last_activity_at == NOW + timedelta(seconds=2)
    # Сообщение пользователя уже зафиксировано до будущего сетевого вызова LLM.
    with open_database(database_path) as reopened:
        assert len(MessageRepository(reopened).get_history(first.session_token_hash, now=NOW)) == 3


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_ttl_boundary_blocks_history_and_writes(database_path: Path, delta: int) -> None:
    """На точной границе TTL диалог уже нельзя читать, продлевать или продолжать.

    Args:
        database_path: Изолированная SQLite v1.
        delta: Микросекунды относительно границы TTL в один час.
    """
    with open_database(database_path) as connection:
        conversations = ConversationRepository(connection, ttl_hours=1)
        messages = MessageRepository(connection, ttl_hours=1)
        conversation, _ = conversations.create(now=NOW)
        messages.append_user_message(conversation.session_token_hash, "Вопрос", now=NOW)
        current = NOW + timedelta(hours=1, microseconds=delta)
        if delta < 0:
            assert conversations.find_active(conversation.session_token_hash, now=current)
            assert len(messages.get_history(conversation.session_token_hash, now=current)) == 1
        else:
            assert conversations.find_active(conversation.session_token_hash, now=current) is None
            assert messages.get_history(conversation.session_token_hash, now=current) == ()
            assert not conversations.touch(conversation.id, now=current)
            with pytest.raises(SessionUnavailableError):
                messages.append_user_message(conversation.session_token_hash, "Поздно", now=current)
            with pytest.raises(SessionUnavailableError):
                messages.append_assistant_message(conversation.id, "Поздно", now=current)
            # Физическое удаление ещё не выполнялось, но доступ уже запрещён.
            assert connection.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1


def test_activity_extension_and_cascading_cleanup(database_path: Path) -> None:
    """Очистка учитывает принятую активность и не затрагивает другие диалоги.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        conversations = ConversationRepository(connection, ttl_hours=1)
        messages = MessageRepository(connection, ttl_hours=1)
        expired, _ = conversations.create(now=NOW)
        active, _ = conversations.create(now=NOW)
        messages.append_user_message(expired.session_token_hash, "Старое", now=NOW)
        messages.append_user_message(
            active.session_token_hash, "Новое", now=NOW + timedelta(minutes=30)
        )
        messages.append_assistant_message(active.id, "Ответ", now=NOW + timedelta(minutes=40))
        assert conversations.cleanup_expired(now=NOW + timedelta(hours=1)) == 1
        assert conversations.cleanup_expired(now=NOW + timedelta(hours=1)) == 0
        assert (
            len(messages.get_history(active.session_token_hash, now=NOW + timedelta(hours=1))) == 2
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM messages WHERE conversation_id = ?", (expired.id,)
            ).fetchone()[0]
            == 0
        )
        assert conversations.delete(active.id)
        assert not conversations.delete(active.id)
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_startup_cleanup_uses_configured_ttl(database_path: Path) -> None:
    """Общая инициализация удаляет только диалоги на границе выбранного TTL.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        conversations = ConversationRepository(connection)
        old, _ = conversations.create(now=NOW)
        current, _ = conversations.create(now=NOW + timedelta(minutes=30))
        MessageRepository(connection).append_user_message(old.session_token_hash, "Старое", now=NOW)
    initialize_database(database_path, ttl_hours=1, now=NOW + timedelta(hours=1))
    with open_database(database_path) as connection:
        assert connection.execute("SELECT id FROM conversations").fetchone()[0] == current.id
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_user_message_and_activity_rollback_together(database_path: Path) -> None:
    """Ошибка обновления активности откатывает уже вставленное сообщение.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        conversation, _ = ConversationRepository(connection).create(now=NOW)
        connection.execute(
            """CREATE TEMP TRIGGER fail_touch BEFORE UPDATE ON conversations
               BEGIN SELECT RAISE(ABORT, 'injected failure'); END"""
        )
        with pytest.raises(sqlite3.IntegrityError):
            MessageRepository(connection).append_user_message(
                conversation.session_token_hash, "Не сохранять", now=NOW + timedelta(minutes=1)
            )
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        assert not connection.in_transaction
        assert (
            ConversationRepository(connection).find_active(conversation.session_token_hash, now=NOW)
            == conversation
        )


@pytest.mark.parametrize("content", ["", " ", "\t\n", "\u2003"])
def test_python_rejects_blank_messages(database_path: Path, content: str) -> None:
    """Python отклоняет все пробельные строки, включая пропускаемые SQLite trim.

    Args:
        database_path: Изолированная SQLite v1.
        content: Пустой либо пробельный текст.
    """
    with open_database(database_path) as connection:
        conversation, _ = ConversationRepository(connection).create(now=NOW)
        messages = MessageRepository(connection)
        with pytest.raises(ValueError):
            messages.append_user_message(conversation.session_token_hash, content, now=NOW)
        with pytest.raises(ValueError):
            messages.append_assistant_message(conversation.id, content, now=NOW)
        assert messages.get_history(conversation.session_token_hash, now=NOW) == ()


def test_foreign_keys_roles_and_message_constraints(database_path: Path) -> None:
    """SQLite отклоняет сообщения без диалога, недопустимые роли и пустой текст.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        conversation, _ = ConversationRepository(connection).create(now=NOW)
        for identifier, role, content in (
            (str(uuid4()), "user", "Нет диалога"),
            (conversation.id, "system", "Недопустимая роль"),
            (conversation.id, "user", "   "),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO messages (conversation_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (identifier, role, content, "2026-11-04T12:00:00.123456Z"),
                )


@pytest.mark.parametrize("ttl", [0, -1, True, 10**30])
def test_invalid_ttl_is_rejected(database_path: Path, ttl: int) -> None:
    """Срок неактивности проверяется до записи данных.

    Args:
        database_path: Изолированная SQLite v1.
        ttl: Некорректное число часов.
    """
    with pytest.raises(ValueError):
        initialize_database(database_path, ttl_hours=ttl, now=NOW)
    with open_database(database_path) as connection, pytest.raises(ValueError):
        ConversationRepository(connection, ttl_hours=ttl)


def test_invalid_identifiers_time_and_missing_sessions(database_path: Path) -> None:
    """Некорректные входы и отсутствующие сессии не создают сообщений.

    Args:
        database_path: Изолированная SQLite v1.
    """
    with open_database(database_path) as connection:
        conversations = ConversationRepository(connection)
        messages = MessageRepository(connection)
        with pytest.raises(ValueError):
            conversations.create(now=NOW.replace(tzinfo=None))
        with pytest.raises(ValueError):
            conversations.find_active("not-a-hash", now=NOW)
        with pytest.raises(ValueError):
            conversations.delete("not-a-uuid")
        with pytest.raises(ValueError):
            hash_session_token("   ")
        unknown_hash = hash_session_token("example-missing-token-not-real")
        assert conversations.find_active(unknown_hash, now=NOW) is None
        assert messages.get_history(unknown_hash, now=NOW) == ()
        with pytest.raises(SessionUnavailableError):
            messages.append_user_message(unknown_hash, "Нет сессии", now=NOW)
        with pytest.raises(SessionUnavailableError):
            messages.append_assistant_message(str(uuid4()), "Нет сессии", now=NOW)
        conversation, _ = conversations.create(now=NOW)
        with pytest.raises(SessionUnavailableError):
            messages.append_user_message(
                conversation.session_token_hash, "Время назад", now=NOW - timedelta(seconds=1)
            )
        with pytest.raises(ValueError):
            messages.append_assistant_message(
                conversation.id, "Время назад", now=NOW - timedelta(seconds=1)
            )
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-time",
        "2026-11-04T12:00:00",
        "2026-11-04T12:00:00Z",
        "2026-11-04T17:00:00.123456+05:00",
    ],
)
def test_corrupted_timestamp_is_rejected(database_path: Path, timestamp: str) -> None:
    """Время в БД должно иметь единственный канонический формат UTC.

    Args:
        database_path: Изолированная SQLite v1.
        timestamp: Некорректное или неканоническое время.
    """
    with open_database(database_path) as connection:
        ConversationRepository(connection).create(now=NOW)
        connection.execute("UPDATE conversations SET created_at = ?", (timestamp,))
    before = database_path.read_bytes()
    with pytest.raises(DatabaseError):
        initialize_database(database_path, now=NOW)
    assert database_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["ok", "regulation", "configuration", "ttl"])
def test_cli_uses_shared_initialization(tmp_path: Path, mode: str) -> None:
    """CLI инициализирует БД либо явно завершает работу без изменения плохой БД.

    Args:
        tmp_path: Изолированный каталог запуска CLI.
        mode: Успех, некорректный регламент, отсутствие настроек или некорректный TTL.
    """
    path = tmp_path / "cli" / "shop.sqlite3"
    if mode == "regulation":
        initialize_database(path, now=NOW)
        with open_database(path) as connection:
            connection.execute("UPDATE products SET price_rubles = 2500")
    before = path.read_bytes() if path.exists() else None
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment["PYTHONUTF8"] = "1"
    environment["SQLITE_PATH"] = str(path)
    if mode == "ttl":
        environment["CONVERSATION_TTL_HOURS"] = str(10**30)
    if mode != "configuration":
        environment["DEEPSEEK_API_KEY"] = "example-cli-key-not-real"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.init_db"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert "example-cli-key-not-real" not in result.stdout + result.stderr
    if mode == "ok":
        assert result.returncode == 0
        assert "исходный регламент проверен" in result.stdout
        assert path.exists()
    else:
        assert result.returncode == 1
        assert "остановлена" in result.stderr
        if before is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == before


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE products SET price_rubles = -1",
        "INSERT INTO products VALUES (2, char(9), 100, 1)",
        "INSERT INTO messages (conversation_id, role, content, created_at) "
        "VALUES ('missing', 'user', 'orphan', '2026-11-04T12:00:00.123456Z')",
    ],
)
def test_invalid_data_inserted_with_disabled_checks_is_not_repaired(
    database_path: Path, statement: str
) -> None:
    """Python и проверка целостности отклоняют обход ограничений SQLite.

    Args:
        database_path: Изолированная SQLite v1.
        statement: Нарушение регламента или связей при отключённых ограничениях.
    """
    with open_database(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(statement)
    before = database_path.read_bytes()
    with pytest.raises(DatabaseError):
        initialize_database(database_path, now=NOW)
    assert database_path.read_bytes() == before


def test_version_zero_with_internal_schema_is_not_empty(tmp_path: Path) -> None:
    """Даже оставшаяся внутренняя таблица означает непустую схему с версией 0.

    Args:
        tmp_path: Изолированный каталог.
    """
    path = tmp_path / "prior.sqlite3"
    with open_database(path) as connection:
        connection.execute("CREATE TABLE prior (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        connection.execute("DROP TABLE prior")
    before = path.read_bytes()
    with pytest.raises(DatabaseError, match="версией 0"):
        initialize_database(path, now=NOW)
    assert path.read_bytes() == before


def test_equivalent_sql_formatting_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Проверка структуры не считает иной регистр и пробелы SQL повреждением БД.

    Args:
        tmp_path: Изолированный каталог.
        monkeypatch: Средство временной замены форматирования утверждённого DDL.
    """
    path = tmp_path / "equivalent.sqlite3"
    alternate_schema = {
        key: statement.replace("CREATE TABLE", "create table").replace(" (", "(")
        for key, statement in database._SCHEMA.items()
    }
    with monkeypatch.context() as temporary:
        temporary.setattr(database, "_SCHEMA", alternate_schema)
        initialize_database(path, now=NOW)
    before = _snapshot(path)
    initialize_database(path, now=NOW)
    assert _snapshot(path) == before
