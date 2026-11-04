"""Создание FastAPI-приложения и минимальная проверка его работоспособности."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import load_settings
from app.schemas import HealthResponse


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Проверить настройки до начала обслуживания HTTP-запросов."""
    application.state.settings = load_settings()
    yield


def create_app() -> FastAPI:
    """Создать независимый экземпляр HTTP-приложения.

    Returns:
        FastAPI-приложение с проверкой конфигурации при старте и маршрутом /health.
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
