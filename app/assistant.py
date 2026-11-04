"""Сервис диалога: последовательная обработка, история и ответы только по регламенту."""

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Protocol

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.database import (
    ConversationRepository,
    DatabaseError,
    KnowledgeRepository,
    Message,
    MessageRepository,
    Product,
    Regulation,
    SessionUnavailableError,
    open_database,
)
from app.llm import ClassificationError, ClassificationTimeoutError, DeepSeekClassifier
from app.schemas import (
    ClassificationResult,
    ContextMessage,
    DefectTopic,
    DeliveryTopic,
    DiscountTopic,
    OrderStatusTopic,
    OutOfScopeTopic,
    ProductTopic,
    ReturnTopic,
    Topic,
)

TECHNICAL_ERROR_MESSAGE = "Сейчас не могу обработать запрос. Попробуйте ещё раз чуть позже"
_LOGGER = logging.getLogger(__name__)


class AssistantUnavailableError(RuntimeError):
    """Обработка недоступна из-за провайдера, хранилища или состояния сессии."""


class AssistantTimeoutError(AssistantUnavailableError):
    """Обработка не завершилась за разрешённое время ожидания провайдера."""


class RequestClassifier(Protocol):
    """Контракт классификатора без зависимости сценария от HTTP-транспорта."""

    async def classify(
        self, message: str, *, context: Sequence[ContextMessage] = ()
    ) -> ClassificationResult:
        """Определить темы и детали сообщения, не формируя покупательский ответ.

        Args:
            message: Текст текущего сообщения.
            context: Предыдущие сообщения, выбранные сервером.

        Returns:
            Проверенная классификация всех частей запроса.

        Raises:
            ClassificationError: Провайдер не смог классифицировать запрос.
        """
        ...


def _days(count: int, *, working: bool = False) -> str:
    if 11 <= count % 100 <= 14:
        index = 2
    elif count % 10 == 1:
        index = 0
    elif 2 <= count % 10 <= 4:
        index = 1
    else:
        index = 2
    noun = ("день", "дня", "дней")[index]
    adjective = ("рабочий ", "рабочих ", "рабочих ")[index] if working else ""
    return f"{count} {adjective}{noun}"


def _normalized_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def _find_products(name: str, regulation: Regulation) -> list[Product]:
    normalized = _normalized_name(name)
    matches: list[Product] = []
    for product in regulation.products:
        full_name = _normalized_name(product.name)
        model_code = full_name.split()[-1]
        # Короткое обозначение берём только из каталога и не угадываем по части названия.
        if normalized == full_name or (
            normalized == model_code and any(character.isdigit() for character in model_code)
        ):
            matches.append(product)
    return matches


def _product_reply(topic: ProductTopic, regulation: Regulation) -> tuple[list[str], str | None]:
    if topic.product_name is None:
        return [], "Какая модель вас интересует?"
    products = _find_products(topic.product_name, regulation)
    if not products:
        return [f"У меня нет сведений о модели «{topic.product_name}»."], None
    if len(products) != 1:
        return [], "Уточните, пожалуйста, полное название модели."
    product = products[0]
    parts: list[str] = []
    for information in topic.requested_info:
        if information.kind == "price":
            price = format(product.price_rubles, ",").replace(",", " ")
            parts.append(f"Цена модели «{product.name}» — {price} ₽.")
        elif information.kind == "availability":
            availability = "есть в наличии" if product.is_in_stock else "нет в наличии"
            parts.append(f"Модель «{product.name}» {availability}.")
        else:
            parts.append(
                f"В регламенте нет сведений о характеристике «{information.name}» "
                f"модели «{product.name}»."
            )
    return parts, None


def _delivery_reply(regulation: Regulation) -> str:
    delivery = regulation.delivery
    if delivery.method != "курьер":
        raise DatabaseError("Неизвестный способ доставки в регламенте")
    period = (
        "на следующий день" if delivery.days_offset == 1 else f"через {_days(delivery.days_offset)}"
    )
    return (
        f"Общие условия доставки: курьером {period}, "
        f"с {delivery.slot_start} до {delivery.slot_end}."
    )


def _defect_reply(regulation: Regulation) -> str:
    policy = regulation.returns
    if not policy.defect_accepted_for_expertise or policy.immediate_refund_promised:
        raise DatabaseError("Неподдерживаемые условия экспертизы")
    return (
        "Товар с браком принимается на экспертизу. Немедленный возврат денег не гарантирован. "
        "О сроке и результате экспертизы сведений нет."
    )


def _return_reply(topic: ReturnTopic, regulation: Regulation) -> tuple[list[str], str | None]:
    if topic.defective is True:
        return [_defect_reply(regulation)], None
    policy = regulation.returns
    if topic.return_day is not None and topic.return_day > policy.period_days:
        text = f"Срок возврата товара без брака — {_days(policy.period_days)}. Он уже истёк."
        question = "Есть ли у товара неисправность?" if topic.defective is None else None
        return [text], question
    if policy.refund_method != "same_card":
        raise DatabaseError("Неизвестный способ возврата денег в регламенте")
    parts = [f"Срок возврата товара без брака — {_days(policy.period_days)}."]
    if policy.opened_without_defect_allowed:
        parts.append("Вскрытый товар без брака тоже можно вернуть в этот срок.")
    parts.append(
        "Деньги возвращаются на ту же карту. "
        f"Срок поступления — {_days(policy.refund_working_days, working=True)}."
    )
    return [" ".join(parts)], None


def _topic_reply(topic: Topic, regulation: Regulation) -> tuple[list[str], str | None]:
    if isinstance(topic, ProductTopic):
        return _product_reply(topic, regulation)
    if isinstance(topic, DeliveryTopic):
        return [_delivery_reply(regulation)], None
    if isinstance(topic, OrderStatusTopic):
        order = (
            f"заказа №{topic.order_number}"
            if topic.order_number is not None
            else "конкретного заказа"
        )
        return [
            f"Я не могу проверить статус {order}: у меня нет доступа к данным о заказах.",
            _delivery_reply(regulation),
        ], None
    if isinstance(topic, ReturnTopic):
        return _return_reply(topic, regulation)
    if isinstance(topic, DefectTopic):
        if topic.defective is None:
            return [], "Есть ли у товара неисправность?"
        if topic.defective:
            return [_defect_reply(regulation)], None
        return _return_reply(
            ReturnTopic.model_validate({**topic.model_dump(), "type": "return"}), regulation
        )
    if isinstance(topic, DiscountTopic):
        # Процент из классификации не участвует ни в расчётах, ни в выборе исключений.
        if regulation.discounts_allowed:
            raise DatabaseError("Исходный запрет скидок нарушен")
        return ["Скидок нет"], None
    return ["По этой части вопроса у меня нет сведений."], None


def _compose_reply(classification: ClassificationResult, regulation: Regulation) -> str:
    parts: list[str] = []
    questions: list[str] = []
    for topic in classification.topics:
        fragments, question = _topic_reply(topic, regulation)
        parts.extend(fragments)
        if question is not None:
            questions.append(question)
    if classification.topics and all(
        isinstance(topic, OutOfScopeTopic) for topic in classification.topics
    ):
        parts.append("Могу помочь с товаром, доставкой и возвратами.")
    if classification.needs_clarification and not questions:
        # Свободный текст причины от LLM не используем как готовый ответ покупателю.
        questions.append("Уточните, пожалуйста, что вы хотите узнать?")
    if questions:
        parts.append(questions[0])
    return "\n\n".join(dict.fromkeys(parts))


class ShopAssistant:
    """Полный сценарий обработки сообщения без HTTP-маршрута и браузерного интерфейса."""

    def __init__(self, settings: Settings, classifier: RequestClassifier | None = None) -> None:
        """Сохранить настройки и необязательную подмену классификатора.

        Args:
            settings: Проверенные настройки SQLite, TTL и лимита контекста.
            classifier: Классификатор для тестов; по умолчанию создаётся DeepSeekClassifier.
        """
        self._settings = settings
        self._classifier = classifier

    def _prepare(
        self, session_token_hash: str, message: str
    ) -> tuple[str, tuple[ContextMessage, ...]]:
        with open_database(self._settings.sqlite_path) as connection:
            conversations = ConversationRepository(
                connection, self._settings.conversation_ttl_hours
            )
            messages = MessageRepository(connection, self._settings.conversation_ttl_hours)
            conversation = conversations.find_active(session_token_hash)
            conversations.cleanup_expired()
            if conversation is None:
                raise SessionUnavailableError("Сессия отсутствует или срок её действия истёк")
            limit = self._settings.classifier_context_messages
            previous = messages.get_history(session_token_hash) if limit else ()
            context = (
                tuple(
                    ContextMessage(role=item.role, content=item.content)
                    for item in previous[-limit:]
                )
                if limit
                else ()
            )
            saved = messages.append_user_message(session_token_hash, message)
        return saved.conversation_id, context

    def _finish(self, conversation_id: str, classification: ClassificationResult) -> Message:
        with open_database(self._settings.sqlite_path) as connection:
            regulation = KnowledgeRepository(connection).read()
            reply = _compose_reply(classification, regulation)
            saved = MessageRepository(
                connection, self._settings.conversation_ttl_hours
            ).append_assistant_message(conversation_id, reply)
        return saved

    async def respond(self, session_token_hash: str, message: str) -> str:
        """Обработать сообщение активной сессии и сохранить обычный ответ.

        Args:
            session_token_hash: SHA-256 токена текущей серверной сессии.
            message: Непустой текст нового сообщения пользователя.

        Returns:
            Ответ по проверенному регламенту либо безопасная техническая фраза.
            Техническая фраза не сохраняется в историю.

        Raises:
            ValueError: Текст или формат хеша некорректны; классификатор не вызывается.
        """
        try:
            return (await self.process(session_token_hash, message)).content
        except AssistantUnavailableError:
            return TECHNICAL_ERROR_MESSAGE

    async def process(self, session_token_hash: str, message: str) -> Message:
        """Обработать запрос, возвращая сохранённый ответ или типизированную ошибку.

        Args:
            session_token_hash: SHA-256 токена активной сессии.
            message: Непустой исходный текст пользователя.

        Returns:
            Сохранённый ответ с фактической временной отметкой UTC из SQLite.

        Raises:
            ValueError: Текст или формат хеша некорректны.
            AssistantTimeoutError: Исчерпано ожидание классификации.
            AssistantUnavailableError: Не удалось прочитать данные или сохранить ответ.
        """
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Сообщение должно быть непустой строкой")
        stage = "подготовка истории"
        try:
            conversation_id, context = await run_in_threadpool(
                self._prepare, session_token_hash, message
            )
            stage = "классификация"
            classifier = (
                self._classifier
                if self._classifier is not None
                else DeepSeekClassifier(self._settings)
            )
            classification = ClassificationResult.model_validate(
                await classifier.classify(message, context=context)
            )
            stage = "формирование и сохранение ответа"
            return await run_in_threadpool(self._finish, conversation_id, classification)
        except (ClassificationTimeoutError, TimeoutError):
            _LOGGER.warning("Обработка сообщения не завершена; этап: %s; ошибка: timeout", stage)
            raise AssistantTimeoutError(TECHNICAL_ERROR_MESSAGE) from None
        except (
            ClassificationError,
            DatabaseError,
            sqlite3.Error,
            OSError,
            SessionUnavailableError,
            ValidationError,
        ) as error:
            _LOGGER.warning(
                "Обработка сообщения не завершена; этап: %s; ошибка: %s",
                stage,
                type(error).__name__,
            )
            raise AssistantUnavailableError(TECHNICAL_ERROR_MESSAGE) from None


class AssistantService:
    """Сериализация запросов одного диалога в пределах одного worker приложения."""

    def __init__(self, settings: Settings, classifier: RequestClassifier | None = None) -> None:
        """Создать сценарий и реестр блокировок активных обработок.

        Args:
            settings: Проверенные настройки приложения.
            classifier: Необязательная подмена провайдера для изолированных тестов.
        """
        self._assistant = ShopAssistant(settings, classifier)
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def _session_lock(self, session_token_hash: str) -> AsyncIterator[None]:
        lock, users = self._locks.get(session_token_hash, (asyncio.Lock(), 0))
        self._locks[session_token_hash] = (lock, users + 1)
        try:
            async with lock:
                yield
        finally:
            # Учитываем ожидающие запросы, включая отменённые во время ожидания.
            _, remaining = self._locks[session_token_hash]
            if remaining == 1:
                del self._locks[session_token_hash]
            else:
                self._locks[session_token_hash] = (lock, remaining - 1)

    async def process(self, session_token_hash: str, message: str) -> Message:
        """Выполнить весь сценарий последовательно для данного диалога.

        Args:
            session_token_hash: SHA-256 токена серверной сессии.
            message: Исходный текст нового сообщения пользователя.

        Returns:
            Ответ, записанный в SQLite, с его временной отметкой.

        Raises:
            ValueError: Вход сценария некорректен.
            AssistantTimeoutError: Провайдер не ответил вовремя.
            AssistantUnavailableError: Зависимость или сессия недоступна.
        """
        async with self._session_lock(session_token_hash):
            return await self._assistant.process(session_token_hash, message)
