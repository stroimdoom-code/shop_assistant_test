"""Модель ответа минимального проверочного HTTP endpoint."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Ответ о работоспособности HTTP API.

    Attributes:
        status: Значение «ok», подтверждающее доступность процесса API.
    """

    status: Literal["ok"] = "ok"
