"""CLI инициализации SQLite и проверки исходного регламента по настройкам окружения."""

import sys

from app.config import ConfigurationError, load_settings
from app.database import DatabaseError, initialize_database


def main() -> int:
    """Инициализировать SQLite той же функцией, что используется при старте API.

    Returns:
        Код 0 при успехе или 1 при ошибке настроек либо проверки БД.
    """
    try:
        settings = load_settings()
        initialize_database(settings.sqlite_path, settings.conversation_ttl_hours)
    except (ConfigurationError, DatabaseError, ValueError) as error:
        print(f"Инициализация SQLite остановлена: {error}", file=sys.stderr)
        return 1
    print("SQLite v1 готова; исходный регламент проверен.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
