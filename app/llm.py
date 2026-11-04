"""Асинхронная классификация через DeepSeek с ограниченными повторами и проверкой JSON."""

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import httpx2 as httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.config import Settings
from app.schemas import ClassificationResult, ContextMessage, NonBlankText

_LOGGER = logging.getLogger(__name__)
_MESSAGE_TEXT = TypeAdapter(NonBlankText)


class ClassificationError(RuntimeError):
    """Явная ошибка классификации без ключей, ответа провайдера или пользовательского текста."""


class AuthenticationError(ClassificationError):
    """Провайдер отклонил авторизацию; повтор запроса не выполняется."""


class _CompletionMessage(BaseModel):
    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    role: Literal["assistant"]
    content: str


class _CompletionChoice(BaseModel):
    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    finish_reason: Literal["stop"]
    message: _CompletionMessage


class _Completion(BaseModel):
    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    choices: list[_CompletionChoice] = Field(min_length=1, max_length=1)


class DeepSeekClassifier:
    """Адаптер понимания запросов, не принимающий бизнес-решений и не читающий SQLite."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        """Загрузить инструкцию и сохранить настройки сетевого вызова.

        Args:
            settings: Проверенные настройки из окружения.
            transport: Подмена HTTP-транспорта для изолированных тестов.
            prompt_path: Путь к инструкции; по умолчанию prompts/classifier.md в проекте.

        Raises:
            ClassificationError: Инструкция недоступна, пуста или ключ имеет неверный формат.
        """
        path = prompt_path or Path(__file__).resolve().parents[1] / "prompts" / "classifier.md"
        try:
            prompt = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise ClassificationError("Не удалось загрузить инструкцию классификатора") from None
        if not prompt.strip():
            raise ClassificationError("Инструкция классификатора не должна быть пустой")
        key = settings.deepseek_api_key.get_secret_value()
        if not key.isascii() or any(
            ord(character) <= 32 or ord(character) == 127 for character in key
        ):
            raise ClassificationError("Некорректный формат ключа DeepSeek")
        self._settings = settings
        self._transport = transport
        self._instruction = (
            prompt
            + "\n\nJSON Schema результата:\n"
            + json.dumps(ClassificationResult.model_json_schema(), ensure_ascii=False)
        )

    async def classify(
        self, message: str, *, context: Sequence[ContextMessage] = ()
    ) -> ClassificationResult:
        """Классифицировать сообщение с переданным недавним контекстом.

        Args:
            message: Непустой текст текущего сообщения.
            context: Сообщения user/assistant, выбранные вызывающим серверным кодом.

        Returns:
            Все распознанные темы с деталями и признаком необходимости уточнения.

        Raises:
            ValueError: Текст или контекст некорректны; сетевой вызов не выполняется.
            AuthenticationError: Ошибка авторизации, без повторной попытки.
            ClassificationError: Постоянная ошибка провайдера или исчерпание попыток.
        """
        try:
            text = _MESSAGE_TEXT.validate_python(message, strict=True)
            history = [ContextMessage.model_validate(item) for item in context]
        except ValidationError:
            raise ValueError("Некорректный текст сообщения или контекст классификации") from None
        messages = [
            {"role": "system", "content": self._instruction},
            *[item.model_dump() for item in history],
            {"role": "user", "content": text},
        ]
        payload = {
            "model": self._settings.deepseek_model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 2048,
            "stream": False,
        }
        endpoint = str(self._settings.deepseek_base_url).rstrip("/") + "/chat/completions"
        headers = {"Authorization": "Bearer " + self._settings.deepseek_api_key.get_secret_value()}
        timeout = self._settings.deepseek_timeout_seconds
        attempts = self._settings.deepseek_max_attempts
        async with httpx.AsyncClient(
            transport=self._transport, timeout=timeout, follow_redirects=False
        ) as client:
            for attempt in range(1, attempts + 1):
                try:
                    # Ограничиваем весь вызов, а не только ожидание очередного сетевого пакета.
                    async with asyncio.timeout(timeout):
                        response = await client.post(endpoint, json=payload, headers=headers)
                except (
                    TimeoutError,
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.RemoteProtocolError,
                    httpx.ProxyError,
                    httpx.DecodingError,
                ):
                    _LOGGER.warning(
                        "Классификация: временная сетевая ошибка, попытка %s из %s",
                        attempt,
                        attempts,
                    )
                    continue
                except (httpx.HTTPError, httpx.InvalidURL):
                    raise ClassificationError("Ошибка HTTP-запроса классификации") from None
                status = response.status_code
                if status in (401, 403):
                    raise AuthenticationError("DeepSeek отклонил авторизацию")
                if status in (408, 425, 429) or 500 <= status <= 599:
                    _LOGGER.warning(
                        "Классификация: временный HTTP-статус %s, попытка %s из %s",
                        status,
                        attempt,
                        attempts,
                    )
                    continue
                if not 200 <= status <= 299:
                    raise ClassificationError("DeepSeek отклонил запрос классификации")
                try:
                    completion = _Completion.model_validate_json(response.content)
                    return ClassificationResult.model_validate_json(
                        completion.choices[0].message.content
                    )
                except ValidationError:
                    # Ответ и детали ValidationError не выводим: они содержат исходные данные.
                    _LOGGER.warning(
                        "Классификация: невалидный результат, попытка %s из %s", attempt, attempts
                    )
        raise ClassificationError("Не удалось классифицировать запрос: попытки исчерпаны")
