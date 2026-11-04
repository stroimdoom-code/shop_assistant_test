"""Создание FastAPI-приложения и минимальная проверка его работоспособности."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.config import load_settings
from app.database import initialize_database
from app.schemas import HealthResponse


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Проверить настройки, инициализировать SQLite и очистить истёкшие диалоги."""
    settings = load_settings()
    await run_in_threadpool(
        initialize_database, settings.sqlite_path, settings.conversation_ttl_hours
    )
    application.state.settings = settings
    yield


def create_app() -> FastAPI:
    """Создать независимый экземпляр HTTP-приложения.

    Returns:
        Приложение с проверкой конфигурации и SQLite при старте, маршрутом /health.
    """
    application = FastAPI(title="Shop Assistant", version="0.1.0", lifespan=_lifespan)

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Сообщить о доступности HTTP API без обращения к внешним сервисам.

        Returns:
            Статус доступности процесса API; не проверяет SQLite и DeepSeek.
        """
        return HealthResponse()

    return application


app = create_app()
