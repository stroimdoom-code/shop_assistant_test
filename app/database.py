"""SQLite v1: исходный регламент, атомарная инициализация и история диалогов."""

import hashlib
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

_SCHEMA: dict[tuple[str, str], str] = {
    ("table", "products"): """CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    price_rubles INTEGER NOT NULL
        CHECK (typeof(price_rubles) = 'integer' AND price_rubles > 0),
    is_in_stock INTEGER NOT NULL CHECK (is_in_stock IN (0, 1))
)""",
    ("table", "delivery_policy"): """CREATE TABLE delivery_policy (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    delivery_method TEXT NOT NULL CHECK (delivery_method = 'курьер'),
    delivery_days_offset INTEGER NOT NULL
        CHECK (typeof(delivery_days_offset) = 'integer' AND delivery_days_offset > 0),
    slot_start TEXT NOT NULL CHECK (
        slot_start GLOB '[0-1][0-9]:[0-5][0-9]'
        OR slot_start GLOB '2[0-3]:[0-5][0-9]'
    ),
    slot_end TEXT NOT NULL CHECK (
        slot_end GLOB '[0-1][0-9]:[0-5][0-9]'
        OR slot_end GLOB '2[0-3]:[0-5][0-9]'
    ),
    CHECK (slot_start < slot_end)
)""",
    ("table", "return_policy"): """CREATE TABLE return_policy (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    return_period_days INTEGER NOT NULL
        CHECK (typeof(return_period_days) = 'integer' AND return_period_days > 0),
    refund_method TEXT NOT NULL CHECK (refund_method = 'same_card'),
    refund_processing_working_days INTEGER NOT NULL CHECK (
        typeof(refund_processing_working_days) = 'integer'
        AND refund_processing_working_days > 0
    ),
    opened_item_without_defect_allowed INTEGER NOT NULL
        CHECK (opened_item_without_defect_allowed IN (0, 1)),
    defect_accepted_for_expertise INTEGER NOT NULL
        CHECK (defect_accepted_for_expertise IN (0, 1)),
    immediate_refund_promised INTEGER NOT NULL
        CHECK (immediate_refund_promised IN (0, 1))
)""",
    ("table", "store_settings"): """CREATE TABLE store_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    discounts_allowed INTEGER NOT NULL CHECK (discounts_allowed IN (0, 1))
)""",
    ("table", "conversations"): """CREATE TABLE conversations (
    id TEXT PRIMARY KEY NOT NULL,
    session_token_hash TEXT NOT NULL UNIQUE CHECK (
        length(session_token_hash) = 64
        AND session_token_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL
)""",
    ("table", "messages"): """CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    created_at TEXT NOT NULL
)""",
    ("index", "idx_conversations_last_activity"): """CREATE INDEX idx_conversations_last_activity
    ON conversations (last_activity_at)""",
    (
        "index",
        "idx_messages_conversation_created",
    ): """CREATE INDEX idx_messages_conversation_created
    ON messages (conversation_id, created_at, id)""",
}

_SEED: tuple[tuple[str, tuple[str | int, ...]], ...] = (
    (
        "INSERT INTO products (id, name, price_rubles, is_in_stock) VALUES (?, ?, ?, ?)",
        (1, "Наушники X1", 2490, 1),
    ),
    (
        """INSERT INTO delivery_policy
        (id, delivery_method, delivery_days_offset, slot_start, slot_end)
        VALUES (?, ?, ?, ?, ?)""",
        (1, "курьер", 1, "18:00", "21:00"),
    ),
    (
        """INSERT INTO return_policy
        (id, return_period_days, refund_method, refund_processing_working_days,
         opened_item_without_defect_allowed, defect_accepted_for_expertise,
         immediate_refund_promised) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (1, 14, "same_card", 3, 1, 1, 0),
    ),
    ("INSERT INTO store_settings (id, discounts_allowed) VALUES (?, ?)", (1, 0)),
)


class DatabaseError(RuntimeError):
    """Ошибка структуры или содержимого SQLite с безопасным сообщением."""


class SessionUnavailableError(RuntimeError):
    """Диалог отсутствует, просрочен или не допускает обновления активности."""


@dataclass(frozen=True, slots=True)
class Product:
    """Товар из регламента.

    Attributes:
        id: Технический идентификатор.
        name: Название товара.
        price_rubles: Положительная цена в целых рублях.
        is_in_stock: Наличие без сведений о количестве.
    """

    id: int
    name: str
    price_rubles: int
    is_in_stock: bool


@dataclass(frozen=True, slots=True)
class DeliveryPolicy:
    """Общие условия доставки.

    Attributes:
        method: Способ доставки.
        days_offset: Доставка на следующий день без дополнительных условий.
        slot_start: Начало слота.
        slot_end: Конец слота.
    """

    method: str
    days_offset: int
    slot_start: str
    slot_end: str


@dataclass(frozen=True, slots=True)
class ReturnPolicy:
    """Раздельные факты об обычном возврате и экспертизе брака.

    Attributes:
        period_days: Срок обычного возврата без домысливания типа дней.
        refund_method: Возврат на ту же карту.
        refund_working_days: Срок поступления денег при обычном возврате.
        opened_without_defect_allowed: Разрешён возврат вскрытого товара без брака.
        defect_accepted_for_expertise: Брак принимается на экспертизу.
        immediate_refund_promised: Обещана ли немедленная выплата при браке.
    """

    period_days: int
    refund_method: str
    refund_working_days: int
    opened_without_defect_allowed: bool
    defect_accepted_for_expertise: bool
    immediate_refund_promised: bool


@dataclass(frozen=True, slots=True)
class Regulation:
    """Проверенный снимок четырёх таблиц регламента без кеширования.

    Attributes:
        products: Товары, включая обязательные наушники X1.
        delivery: Условия доставки.
        returns: Условия обычного возврата и экспертизы.
        discounts_allowed: Разрешены ли скидки.
    """

    products: tuple[Product, ...]
    delivery: DeliveryPolicy
    returns: ReturnPolicy
    discounts_allowed: bool


@dataclass(frozen=True, slots=True)
class Conversation:
    """Диалог, в котором хранится только хеш анонимного токена.

    Attributes:
        id: Серверный UUID.
        session_token_hash: SHA-256 токена в строчном шестнадцатеричном виде.
        created_at: Время создания в UTC.
        last_activity_at: Время последнего принятого сообщения пользователя.
    """

    id: str
    session_token_hash: str
    created_at: datetime
    last_activity_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    """Сообщение из серверной истории.

    Attributes:
        id: Идентификатор, уточняющий порядок при совпадении времени.
        conversation_id: UUID диалога.
        role: Роль, назначенная серверным методом записи.
        content: Непустой текст сообщения.
        created_at: Время создания в UTC.
    """

    id: int
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


def _now(value: datetime | None) -> datetime:
    value = datetime.now(UTC) if value is None else value
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Время должно содержать часовой пояс")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _now(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise DatabaseError("В БД обнаружено некорректное время") from None
    if parsed.tzinfo is None or _timestamp(parsed) != value:
        raise DatabaseError("В БД требуется каноническое время UTC с шестью знаками")
    return parsed


def _ttl(value: int) -> timedelta:
    if type(value) is not int or value <= 0:
        raise ValueError("Срок хранения должен быть положительным целым числом часов")
    try:
        return timedelta(hours=value)
    except OverflowError:
        raise ValueError("Срок хранения слишком велик") from None


def _cutoff(now: datetime, ttl: timedelta) -> str:
    try:
        return _timestamp(now - ttl)
    except OverflowError:
        raise ValueError("Срок хранения выходит за допустимый диапазон времени") from None


def _require_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Текст должен быть непустой строкой")
    return value


def _require_hash(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("Требуется SHA-256-хеш в строчном шестнадцатеричном виде")
    return value


def _require_id(value: str) -> str:
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError("Требуется канонический UUID диалога") from None
    return value


def hash_session_token(token: str) -> str:
    """Вычислить хеш токена без сохранения исходного значения.

    Args:
        token: Непустой токен из доверенной обработки cookie.

    Returns:
        SHA-256 в виде 64 строчных шестнадцатеричных символов.

    Raises:
        ValueError: Токен пуст или не является строкой.
    """
    return hashlib.sha256(_require_text(token).encode("utf-8")).hexdigest()


@contextmanager
def open_database(path: Path) -> Iterator[sqlite3.Connection]:
    """Открыть отдельное соединение и закрыть его после работы.

    Args:
        path: Путь SQLite; родительский каталог должен существовать.

    Yields:
        Соединение с foreign_keys=ON, busy_timeout=5000 и явными транзакциями.

    Raises:
        sqlite3.Error: Не удалось открыть или настроить БД.
    """
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        yield connection
    finally:
        connection.close()


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Выполнить короткую транзакцию с откатом при незавершённом выходе.

    Args:
        connection: Открытое соединение без активной транзакции.

    Yields:
        Управление вызывающему коду внутри BEGIN IMMEDIATE.

    Raises:
        sqlite3.Error: Начало транзакции или фиксация не удались.
    """
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.commit()
    finally:
        # Откатываем также ошибку фиксации или прерывание выполнения.
        if connection.in_transaction:
            connection.rollback()


def _integer(row: sqlite3.Row, name: str) -> int:
    value = row[name]
    if type(value) is not int:
        raise DatabaseError("Некорректный тип числового поля регламента")
    return value


def _positive(row: sqlite3.Row, name: str) -> int:
    value = _integer(row, name)
    if value <= 0:
        raise DatabaseError("Числовое поле регламента должно быть положительным")
    return value


def _text(row: sqlite3.Row, name: str) -> str:
    value = row[name]
    if not isinstance(value, str) or not value.strip():
        raise DatabaseError("В БД отсутствует обязательное непустое текстовое поле")
    return value


def _flag(row: sqlite3.Row, name: str) -> bool:
    value = _integer(row, name)
    if value not in (0, 1):
        raise DatabaseError("Логическое поле регламента должно быть 0 или 1")
    return bool(value)


def _single(rows: list[sqlite3.Row]) -> sqlite3.Row:
    if len(rows) != 1 or rows[0]["id"] != 1:
        raise DatabaseError("Отсутствует обязательная единственная запись регламента")
    return rows[0]


class KnowledgeRepository:
    """Чтение регламента с проверкой соответствия шести исходным фактам."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Сохранить соединение, принадлежащее вызывающему коду.

        Args:
            connection: Настроенное соединение SQLite v1.
        """
        self.connection = connection

    def read(self) -> Regulation:
        """Прочитать четыре таблицы без подстановок отсутствующих данных.

        Returns:
            Проверенные исходные факты и дополнительные корректные товары.

        Raises:
            DatabaseError: Факт отсутствует, изменён или имеет некорректный тип.
            sqlite3.Error: Ошибка чтения SQLite.
        """
        products = tuple(
            Product(
                _integer(row, "id"),
                _text(row, "name"),
                _positive(row, "price_rubles"),
                _flag(row, "is_in_stock"),
            )
            for row in self.connection.execute(
                "SELECT id, name, price_rubles, is_in_stock FROM products ORDER BY id"
            )
        )
        x1 = [product for product in products if product.name == "Наушники X1"]
        if len(x1) != 1 or x1[0].price_rubles != 2490 or not x1[0].is_in_stock:
            raise DatabaseError("Исходный регламент нарушен: наушники X1, цена 2490 ₽, в наличии")
        row = _single(self.connection.execute("SELECT * FROM delivery_policy").fetchall())
        delivery = DeliveryPolicy(
            _text(row, "delivery_method"),
            _positive(row, "delivery_days_offset"),
            _text(row, "slot_start"),
            _text(row, "slot_end"),
        )
        if delivery != DeliveryPolicy("курьер", 1, "18:00", "21:00"):
            raise DatabaseError("Исходный регламент нарушен: условия доставки")
        row = _single(self.connection.execute("SELECT * FROM return_policy").fetchall())
        returns = ReturnPolicy(
            _positive(row, "return_period_days"),
            _text(row, "refund_method"),
            _positive(row, "refund_processing_working_days"),
            _flag(row, "opened_item_without_defect_allowed"),
            _flag(row, "defect_accepted_for_expertise"),
            _flag(row, "immediate_refund_promised"),
        )
        if returns != ReturnPolicy(14, "same_card", 3, True, True, False):
            raise DatabaseError("Исходный регламент нарушен: условия возврата и экспертизы")
        row = _single(self.connection.execute("SELECT * FROM store_settings").fetchall())
        discounts_allowed = _flag(row, "discounts_allowed")
        if discounts_allowed:
            raise DatabaseError("Исходный регламент нарушен: discounts_allowed должен быть 0")
        return Regulation(products, delivery, returns, discounts_allowed)


def _conversation(row: sqlite3.Row) -> Conversation:
    identifier = _text(row, "id")
    token_hash = _text(row, "session_token_hash")
    try:
        _require_id(identifier)
        _require_hash(token_hash)
    except ValueError:
        raise DatabaseError("В БД обнаружен некорректный идентификатор диалога или хеш") from None
    created = _parse_timestamp(_text(row, "created_at"))
    active = _parse_timestamp(_text(row, "last_activity_at"))
    if active < created:
        raise DatabaseError("Активность диалога не может предшествовать его созданию")
    return Conversation(identifier, token_hash, created, active)


def _message(row: sqlite3.Row) -> Message:
    role = _text(row, "role")
    if role not in ("user", "assistant"):
        raise DatabaseError("В истории обнаружена недопустимая роль")
    validated_role: Literal["user", "assistant"] = "user" if role == "user" else "assistant"
    return Message(
        _integer(row, "id"),
        _text(row, "conversation_id"),
        validated_role,
        _text(row, "content"),
        _parse_timestamp(_text(row, "created_at")),
    )


class ConversationRepository:
    """Создание и поиск диалогов, обновление активности и физическая очистка."""

    def __init__(self, connection: sqlite3.Connection, ttl_hours: int = 24) -> None:
        """Сохранить соединение и проверить срок неактивности.

        Args:
            connection: Соединение текущего запроса.
            ttl_hours: Положительное число часов неактивности.

        Raises:
            ValueError: Срок неактивности некорректен.
        """
        self.connection = connection
        self.ttl = _ttl(ttl_hours)

    def create(self, *, now: datetime | None = None) -> tuple[Conversation, str]:
        """Создать серверные UUID и токен, сохранив только хеш токена.

        Args:
            now: Время с часовым поясом; по умолчанию UTC сейчас.

        Returns:
            Созданный диалог и исходный токен для будущей защищённой cookie.

        Raises:
            ValueError: Передано время без часового пояса.
            sqlite3.Error: Запись диалога не удалась.
        """
        current = _now(now)
        token = secrets.token_urlsafe(32)
        conversation = Conversation(str(uuid4()), hash_session_token(token), current, current)
        self.connection.execute(
            """INSERT INTO conversations (id, session_token_hash, created_at, last_activity_at)
               VALUES (?, ?, ?, ?)""",
            (
                conversation.id,
                conversation.session_token_hash,
                _timestamp(current),
                _timestamp(current),
            ),
        )
        return conversation, token

    def find_active(
        self, session_token_hash: str, *, now: datetime | None = None
    ) -> Conversation | None:
        """Найти диалог по хешу только до истечения TTL.

        Args:
            session_token_hash: SHA-256 токена, не исходный токен.
            now: Текущее время с часовым поясом.

        Returns:
            Активный диалог либо None, включая точную границу TTL.

        Raises:
            ValueError: Хеш или время некорректны.
            DatabaseError: Запись диалога повреждена.
        """
        row = self.connection.execute(
            "SELECT * FROM conversations WHERE session_token_hash = ? AND last_activity_at > ?",
            (_require_hash(session_token_hash), _cutoff(_now(now), self.ttl)),
        ).fetchone()
        return None if row is None else _conversation(row)

    def touch(self, conversation_id: str, *, now: datetime | None = None) -> bool:
        """Обновить активность существующего непросроченного диалога.

        Более ранняя отметка ожидавшего запроса не уменьшает уже записанную активность.

        Args:
            conversation_id: Канонический UUID диалога.
            now: Время принятия сообщения пользователя.

        Returns:
            True для активного диалога; False при истечении или времени до его создания.

        Raises:
            ValueError: UUID или время некорректны.
        """
        current = _now(now)
        result = self.connection.execute(
            """UPDATE conversations SET last_activity_at = max(last_activity_at, ?)
               WHERE id = ? AND last_activity_at > ? AND created_at <= ?""",
            (
                _timestamp(current),
                _require_id(conversation_id),
                _cutoff(current, self.ttl),
                _timestamp(current),
            ),
        )
        return result.rowcount == 1

    def delete(self, conversation_id: str) -> bool:
        """Удалить диалог и каскадно его сообщения.

        Args:
            conversation_id: Канонический UUID диалога.

        Returns:
            True, если диалог существовал.

        Raises:
            ValueError: UUID некорректен.
        """
        return (
            self.connection.execute(
                "DELETE FROM conversations WHERE id = ?", (_require_id(conversation_id),)
            ).rowcount
            == 1
        )

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Удалить просроченные диалоги вместе с их сообщениями.

        Args:
            now: Текущее время с часовым поясом.

        Returns:
            Число удалённых диалогов, не сообщений.

        Raises:
            ValueError: Время некорректно.
        """
        return self.connection.execute(
            "DELETE FROM conversations WHERE last_activity_at <= ?",
            (_cutoff(_now(now), self.ttl),),
        ).rowcount


class MessageRepository:
    """Запись сообщений и чтение серверной истории с проверкой TTL."""

    def __init__(self, connection: sqlite3.Connection, ttl_hours: int = 24) -> None:
        """Сохранить соединение и правила проверки сессий.

        Args:
            connection: Соединение текущего запроса.
            ttl_hours: Положительное число часов неактивности.

        Raises:
            ValueError: Срок неактивности некорректен.
        """
        self.connection = connection
        self.conversations = ConversationRepository(connection, ttl_hours)

    def _insert(
        self, conversation_id: str, role: Literal["user", "assistant"], content: str, now: datetime
    ) -> Message:
        cursor = self.connection.execute(
            """INSERT INTO messages (conversation_id, role, content, created_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, role, _require_text(content), _timestamp(now)),
        )
        identifier = cursor.lastrowid
        if identifier is None:
            raise DatabaseError("Не удалось получить идентификатор сообщения")
        return Message(identifier, conversation_id, role, content, now)

    def append_user_message(
        self, session_token_hash: str, content: str, *, now: datetime | None = None
    ) -> Message:
        """Атомарно проверить сессию, сохранить сообщение и обновить активность.

        Args:
            session_token_hash: Хеш токена текущей сессии.
            content: Непустой текст пользователя.
            now: Время принятия сообщения с часовым поясом.

        Returns:
            Сохранённое сообщение с серверной ролью user.

        Raises:
            SessionUnavailableError: Диалог отсутствует, просрочен или время до его создания.
            ValueError: Хеш, текст или время некорректны.
            sqlite3.Error: Запись не удалась; транзакция откатывается.
        """
        _require_text(content)
        with transaction(self.connection):
            # Часы читаем после получения блокировки, чтобы учесть истечение TTL в очереди.
            current = _now(now)
            conversation = self.conversations.find_active(session_token_hash, now=current)
            if conversation is None:
                raise SessionUnavailableError("Сессия отсутствует или срок её действия истёк")
            message = self._insert(conversation.id, "user", content, current)
            if not self.conversations.touch(conversation.id, now=current):
                raise SessionUnavailableError("Не удалось обновить активность диалога")
        return message

    def append_assistant_message(
        self, conversation_id: str, content: str, *, now: datetime | None = None
    ) -> Message:
        """Сохранить ответ отдельной транзакцией без продления активности.

        Args:
            conversation_id: UUID диалога, полученный сервером.
            content: Непустой обычный ответ; техническую ошибку сюда не передают.
            now: Время ответа с часовым поясом.

        Returns:
            Сохранённое сообщение с ролью assistant.

        Raises:
            SessionUnavailableError: Диалог удалён или просрочен.
            ValueError: UUID, текст или время некорректны.
            sqlite3.Error: Запись не удалась.
        """
        _require_text(content)
        with transaction(self.connection):
            current = _now(now)
            row = self.connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND last_activity_at > ?",
                (_require_id(conversation_id), _cutoff(current, self.conversations.ttl)),
            ).fetchone()
            if row is None:
                raise SessionUnavailableError("Диалог отсутствует или срок его действия истёк")
            conversation = _conversation(row)
            if current < conversation.created_at:
                raise ValueError("Ответ не может предшествовать созданию диалога")
            message = self._insert(conversation.id, "assistant", content, current)
        return message

    def get_history(
        self, session_token_hash: str, *, now: datetime | None = None
    ) -> tuple[Message, ...]:
        """Прочитать историю активного диалога в порядке времени и id.

        Args:
            session_token_hash: Хеш токена текущей сессии.
            now: Текущее время с часовым поясом.

        Returns:
            Сообщения либо пустой кортеж при отсутствии или истечении сессии.

        Raises:
            ValueError: Хеш или время некорректны.
            DatabaseError: В истории обнаружены некорректные значения.
        """
        rows = self.connection.execute(
            """SELECT m.* FROM messages AS m JOIN conversations AS c ON c.id = m.conversation_id
               WHERE c.session_token_hash = ? AND c.last_activity_at > ?
               ORDER BY m.created_at, m.id""",
            (_require_hash(session_token_hash), _cutoff(_now(now), self.conversations.ttl)),
        )
        return tuple(_message(row) for row in rows)


def _normalized_sql(value: str) -> str:
    # Пробелы и регистр SQL не меняют структуру; значения строковых литералов сохраняем.
    tokens = re.findall(
        r"'(?:''|[^'])*'|[a-zA-Z_][a-zA-Z_0-9]*|[0-9]+|[^\s]", value.strip().rstrip(";")
    )
    return " ".join(token if token.startswith("'") else token.lower() for token in tokens)


def _validate_database(connection: sqlite3.Connection) -> None:
    objects = connection.execute(
        "SELECT type, name, sql FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'"
    ).fetchall()
    actual = {(row["type"], row["name"]): _normalized_sql(row["sql"]) for row in objects}
    expected = {key: _normalized_sql(value) for key, value in _SCHEMA.items()}
    if actual != expected:
        raise DatabaseError("Структура SQLite не соответствует утверждённой схеме v1")
    if [row[0] for row in connection.execute("PRAGMA quick_check")] != ["ok"]:
        raise DatabaseError("Проверка целостности или ограничений SQLite не пройдена")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise DatabaseError("В БД обнаружены нарушенные связи сообщений и диалогов")
    KnowledgeRepository(connection).read()
    for row in connection.execute("SELECT * FROM conversations"):
        _conversation(row)
    for row in connection.execute("SELECT * FROM messages"):
        _message(row)


def initialize_database(path: Path, ttl_hours: int = 24, *, now: datetime | None = None) -> None:
    """Создать SQLite v1 или проверить БД и очистить истёкшие диалоги.

    Args:
        path: Путь из настроек; родительский каталог создаётся при необходимости.
        ttl_hours: Положительный срок неактивности в часах.
        now: Текущее время с часовым поясом для проверки TTL.

    Raises:
        DatabaseError: Версия, структура, исходные факты или файл БД некорректны.
        ValueError: Срок неактивности или время некорректны.
    """
    _ttl(ttl_hours)
    current = _now(now)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open_database(path) as connection, transaction(connection):
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if connection.execute("SELECT 1 FROM sqlite_schema").fetchone() is not None:
                    raise DatabaseError("Непустая схема с версией 0: автоматический сброс запрещён")
                # executescript может завершить начатую транзакцию, поэтому выполняем по одному.
                for statement in _SCHEMA.values():
                    connection.execute(statement)
                for statement, parameters in _SEED:
                    connection.execute(statement, parameters)
                connection.execute("PRAGMA user_version = 1")
            elif version != 1:
                raise DatabaseError("Неподдерживаемая версия SQLite: требуется версия 1")
            _validate_database(connection)
            ConversationRepository(connection, ttl_hours).cleanup_expired(now=current)
    except (sqlite3.Error, OSError):
        raise DatabaseError(
            "Не удалось инициализировать SQLite: проверьте файл, права доступа и целостность БД"
        ) from None
