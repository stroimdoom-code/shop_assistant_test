"""Загрузка и проверка настроек окружения без раскрытия секретных значений."""

from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Ошибка настроек с безопасным для вывода пользователю сообщением."""


class Settings(BaseSettings):
    """Проверенные настройки приложения из окружения и локального файла .env.

    Attributes:
        deepseek_api_key: Обязательный непустой секретный ключ провайдера.
        deepseek_base_url: HTTPS-адрес API без встроенных учётных данных.
        deepseek_model: Идентификатор модели из согласованной архитектуры.
        deepseek_timeout_seconds: Таймаут попытки, не более 30 секунд.
        deepseek_max_attempts: Число попыток на сообщение, включая первую, от 1 до 2.
        sqlite_path: Путь к SQLite относительно рабочего каталога.
        conversation_ttl_hours: Срок хранения и неактивности диалога в часах.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
        frozen=True,
    )

    deepseek_api_key: SecretStr
    deepseek_base_url: HttpUrl = HttpUrl("https://api.deepseek.com")
    deepseek_model: str = Field(default="deepseek-flash", min_length=1)
    deepseek_timeout_seconds: float = Field(default=30, gt=0, le=30, allow_inf_nan=False)
    deepseek_max_attempts: int = Field(default=2, ge=1, le=2)
    sqlite_path: Path = Path("data/shop-assistant.sqlite3")
    conversation_ttl_hours: int = Field(default=24, gt=0)

    @field_validator("deepseek_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        """Проверить, что обязательный ключ не пуст и не состоит из пробелов.

        Args:
            value: Секретное значение, полученное из настроек.

        Returns:
            Исходный ключ без раскрытия его содержимого.

        Raises:
            ValueError: Ключ не задан или состоит из пробелов.
        """
        if not value.get_secret_value().strip():
            raise ValueError("Задайте непустой DEEPSEEK_API_KEY")
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl) -> HttpUrl:
        """Проверить безопасный транспорт и отсутствие учётных данных в URL.

        Args:
            value: Разобранный адрес API.

        Returns:
            Проверенный адрес API.

        Raises:
            ValueError: Адрес использует HTTP или содержит логин/пароль.
        """
        if value.scheme != "https" or value.username is not None or value.password is not None:
            raise ValueError("Укажите HTTPS-адрес без учётных данных")
        return value

    @field_validator("deepseek_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Убрать внешние пробелы и проверить идентификатор модели.

        Args:
            value: Настроенный идентификатор модели.

        Returns:
            Непустой идентификатор без внешних пробелов.

        Raises:
            ValueError: Идентификатор пуст.
        """
        if not value.strip():
            raise ValueError("Укажите DEEPSEEK_MODEL")
        return value.strip()

    @field_validator("sqlite_path", mode="before")
    @classmethod
    def validate_sqlite_path(cls, value: object) -> object:
        """Отклонить пустую строку до её преобразования в текущий каталог.

        Args:
            value: Значение пути до валидации Pydantic.

        Returns:
            Исходный непустой путь для дальнейшей проверки типа.

        Raises:
            ValueError: Вместо пути передана пустая строка.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError("Укажите SQLITE_PATH")
        return value


def load_settings() -> Settings:
    """Прочитать настройки, скрывая значения при ошибке валидации.

    Returns:
        Валидированные настройки из окружения и .env в рабочем каталоге.

    Raises:
        ConfigurationError: Обязательные настройки отсутствуют или некорректны.
    """
    try:
        return Settings()
    except ValidationError as error:
        # Показываем только имена полей: исходные значения могут содержать секреты.
        fields = sorted({str(item["loc"][0]).upper() for item in error.errors()})
        raise ConfigurationError(
            "Проверьте обязательные настройки и допустимые значения: " + ", ".join(fields)
        ) from None
