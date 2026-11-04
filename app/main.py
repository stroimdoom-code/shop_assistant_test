"""HTTP API чата: валидация, анонимная cookie, история и безопасные ошибки."""

import logging
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.assistant import AssistantService, AssistantTimeoutError, AssistantUnavailableError
from app.config import Settings, load_settings
from app.database import (
    ConversationRepository,
    DatabaseError,
    MessageRepository,
    hash_session_token,
    initialize_database,
    open_database,
    transaction,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
)

_LOGGER = logging.getLogger(__name__)
_COOKIE_NAME = "shop_assistant_session"
_COOKIE_PATH = "/api/chat"
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
_ERRORS = {
    422: (
        "invalid_request",
        "Некорректный запрос. Проверьте сообщение и его длину до 2000 символов",
    ),
    503: ("service_unavailable", "Сейчас не могу обработать запрос. Попробуйте ещё раз чуть позже"),
    504: ("request_timeout", "Сервис не ответил вовремя. Попробуйте ещё раз чуть позже"),
    500: ("internal_error", "Не удалось обработать запрос. Попробуйте ещё раз чуть позже"),
}


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Проверить настройки, инициализировать SQLite и очистить истёкшие диалоги."""
    settings = load_settings()
    await run_in_threadpool(
        initialize_database, settings.sqlite_path, settings.conversation_ttl_hours
    )
    application.state.settings = settings
    application.state.assistant_service = AssistantService(settings)
    yield


def get_assistant_service(request: Request) -> AssistantService:
    """Получить общий сервис приложения с блокировками диалогов.

    Args:
        request: Текущий запрос FastAPI.

    Returns:
        Сервис, созданный при старте; зависимость заменяется в HTTP-тестах.
    """
    return cast(AssistantService, request.app.state.assistant_service)


def _apply_cookie(request: Request, response: Response) -> None:
    if not hasattr(request.state, "chat_cookie"):
        return
    token: str | None = request.state.chat_cookie
    secure = request.url.scheme == "https"
    if token is None:
        response.delete_cookie(
            _COOKIE_NAME, path=_COOKIE_PATH, secure=secure, httponly=True, samesite="lax"
        )
    else:
        settings = cast(Settings, request.app.state.settings)
        response.set_cookie(
            _COOKIE_NAME,
            token,
            max_age=settings.conversation_ttl_hours * 3600,
            path=_COOKIE_PATH,
            secure=secure,
            httponly=True,
            samesite="lax",
        )


def _error_response(request: Request, status: int) -> JSONResponse:
    code, message = _ERRORS.get(status, ("http_error", "Запрос не может быть выполнен"))
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message))
    response = JSONResponse(status_code=status, content=payload.model_dump())
    response.headers["Cache-Control"] = "no-store"
    _apply_cookie(request, response)
    return response


def _resolve_session(
    settings: Settings, token: str | None, *, create: bool
) -> tuple[str | None, str | None]:
    token_hash = (
        hash_session_token(token) if token is not None and _TOKEN_PATTERN.fullmatch(token) else None
    )
    with open_database(settings.sqlite_path) as connection, transaction(connection):
        conversations = ConversationRepository(connection, settings.conversation_ttl_hours)
        conversations.cleanup_expired()
        conversation = conversations.find_active(token_hash) if token_hash is not None else None
        if conversation is not None:
            return conversation.session_token_hash, token
        if create:
            conversation, new_token = conversations.create()
            return conversation.session_token_hash, new_token
    return None, None


def _read_history(settings: Settings, token_hash: str) -> HistoryResponse:
    with open_database(settings.sqlite_path) as connection:
        messages = MessageRepository(connection, settings.conversation_ttl_hours).get_history(
            token_hash
        )
    return HistoryResponse(
        messages=[
            HistoryMessage(role=item.role, content=item.content, created_at=item.created_at)
            for item in messages
        ]
    )


def create_app() -> FastAPI:
    """Создать независимый экземпляр HTTP-приложения.

    Returns:
        Приложение с проверкой SQLite при старте, health и двумя маршрутами чата.
    """
    application = FastAPI(title="Shop Assistant", version="0.1.0", lifespan=_lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        """Скрыть входные значения и детали ошибок валидации.

        Args:
            request: Текущий HTTP-запрос.
            error: Ошибка валидации, содержимое которой не публикуется.

        Returns:
            Единая безопасная ошибка HTTP 422.
        """
        return _error_response(request, 422)

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        """Привести ошибки HTTP к единому безопасному формату.

        Args:
            request: Текущий HTTP-запрос.
            error: HTTP-ошибка без публикации её произвольного текста.

        Returns:
            Безопасный JSON с исходным HTTP-статусом.
        """
        return _error_response(request, error.status_code)

    @application.exception_handler(AssistantUnavailableError)
    @application.exception_handler(sqlite3.Error)
    @application.exception_handler(DatabaseError)
    @application.exception_handler(OSError)
    @application.exception_handler(TimeoutError)
    async def unavailable(request: Request, error: Exception) -> JSONResponse:
        """Преобразовать ошибки сервиса и SQLite без раскрытия внутренних деталей.

        Args:
            request: Текущий HTTP-запрос.
            error: Ошибка зависимости или таймаута.

        Returns:
            HTTP 504 при таймауте, иначе HTTP 503.
        """
        _LOGGER.warning("HTTP-обработка недоступна; ошибка: %s", type(error).__name__)
        return _error_response(
            request, 504 if isinstance(error, (AssistantTimeoutError, TimeoutError)) else 503
        )

    @application.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        """Обработать непредвиденный сбой на внешней границе приложения.

        Args:
            request: Текущий HTTP-запрос.
            error: Внутренняя ошибка, текст и трассировка которой не публикуются.

        Returns:
            Безопасная ошибка HTTP 500.
        """
        _LOGGER.error("Внутренняя ошибка HTTP-обработки: %s", type(error).__name__)
        return _error_response(request, 500)

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Сообщить о доступности HTTP API без обращения к внешним сервисам.

        Returns:
            Статус доступности процесса API; не проверяет SQLite и DeepSeek.
        """
        return HealthResponse()

    @application.post(
        "/api/chat/messages",
        response_model=ChatResponse,
        responses={status: {"model": ErrorResponse} for status in _ERRORS},
        tags=["chat"],
    )
    async def send_message(
        payload: ChatRequest,
        request: Request,
        response: Response,
        service: Annotated[AssistantService, Depends(get_assistant_service)],
    ) -> ChatResponse:
        """Принять сообщение, создать или продолжить сессию и вернуть готовый ответ.

        Args:
            payload: Проверенный исходный текст сообщения.
            request: Запрос с непрозрачной cookie.
            response: Ответ для установки защищённой cookie.
            service: Сервис, владеющий бизнес-логикой и последовательностью обработки.

        Returns:
            Сохранённый ответ и его фактическое время UTC.
        """
        settings = cast(Settings, request.app.state.settings)
        token = request.cookies.get(_COOKIE_NAME)
        if token is not None:
            request.state.chat_cookie = None
        token_hash, new_token = await run_in_threadpool(
            _resolve_session, settings, token, create=True
        )
        if token_hash is None or new_token is None:
            raise AssistantUnavailableError("Не удалось создать сессию")
        request.state.chat_cookie = new_token
        saved = await service.process(token_hash, payload.message)
        response.headers["Cache-Control"] = "no-store"
        _apply_cookie(request, response)
        return ChatResponse(answer=saved.content, created_at=saved.created_at)

    @application.get(
        "/api/chat/history",
        response_model=HistoryResponse,
        responses={status: {"model": ErrorResponse} for status in (503, 500)},
        tags=["chat"],
    )
    async def history(request: Request, response: Response) -> HistoryResponse:
        """Вернуть полную историю без создания сессии и продления её активности.

        Args:
            request: Запрос с необязательной cookie.
            response: Ответ для удаления недействительной cookie.

        Returns:
            История в хронологическом порядке либо пустой список.
        """
        settings = cast(Settings, request.app.state.settings)
        token = request.cookies.get(_COOKIE_NAME)
        token_hash, _ = await run_in_threadpool(_resolve_session, settings, token, create=False)
        if token is not None and token_hash is None:
            request.state.chat_cookie = None
        result = (
            await run_in_threadpool(_read_history, settings, token_hash)
            if token_hash is not None
            else HistoryResponse(messages=[])
        )
        response.headers["Cache-Control"] = "no-store"
        _apply_cookie(request, response)
        return result

    return application


app = create_app()
