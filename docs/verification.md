# Результаты реализации карточки 06

**Итог: каркас приведён к полному утверждённому дереву.** Пользователь явно
разрешил файлы будущих компонентов с описанием назначения без реализации
будущей логики. Все 21 путь из дерева карточки присутствуют; сверка структуры
и проверки качества повторно выполнены после её исправления.

Проверено 04.11.2026 в Windows / PowerShell, Python **3.12.3**.
Рабочая директория: `C:\Users\goood\Desktop\project\shop-assistant-test`.
До работы директория была пустой и не являлась Git-репозиторием.
Карточки [01](https://trello.com/c/aB4obw2V),
[02](https://trello.com/c/o6Y8gEiv), [03](https://trello.com/c/McJ4Vt4A)
и [06](https://trello.com/c/eGWxxccv) прочитаны перед реализацией.

## Файлы и изменения

Все перечисленные файлы созданы; существующего кода не было.

```text
shop-assistant-test/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── assistant.py
│   ├── llm.py
│   ├── database.py
│   ├── schemas.py
│   └── README.md
├── prompts/
│   ├── system.md
│   └── classifier.md
├── web/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_main.py
│   ├── test_assistant.py
│   ├── test_database.py
│   └── test_security.py
├── docs/
│   ├── architecture.md
│   └── verification.md
├── scripts/
│   └── init_db.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

- `app/main.py`: FastAPI, фабрика приложения, проверка настроек в lifespan,
  минимальный `GET /health`.
- `app/config.py`: настройки из окружения/`.env`, обязательный секретный ключ,
  проверка URL, таймаута, числа попыток, пути SQLite, срока хранения;
  безопасное сообщение об ошибке с именами полей.
- `app/schemas.py`: единственная используемая модель `HealthResponse`.
- `tests/`: 22 изолированных тестовых случая API и конфигурации.
- `app/assistant.py`, `app/llm.py`, `app/database.py`, `scripts/init_db.py`
  и три файла будущих тестов: описания назначения в модульных docstrings,
  без функций, классов и исполняемой логики.
- `prompts/`: описания назначения системного промпта и инструкции
  классификатора; рабочие промпты ещё не реализованы.
- `web/`: комментарии с назначением HTML, CSS и JavaScript;
  интерфейс, HTTP-запросы и подключение к FastAPI не реализованы.
- `pyproject.toml`: зависимости приложения и `dev`, Ruff, mypy strict,
  Bandit, pytest с ошибками на предупреждениях, coverage с ветвями и порогом 90%.
- `.env.example`/`.gitignore`: перечень настроек без настоящих секретов,
  исключение локальных настроек, БД, окружения и кешей.
- Markdown: установка, конфигурация, запуск, границы SQLite, документация
  пакета и архитектура с явным разделением текущей и будущей реализации.

Кроме исходников локально появились игнорируемые `.venv/`,
`shop_assistant.egg-info/`, кеши, `.coverage` и `coverage.xml`.
Реальный `.env`, БД и Git-репозиторий в корне не создавались.

## Установка

Создано новое изолированное окружение без `--system-site-packages`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --timeout 15 --retries 0 -e ".[dev]"
```

Результат: `Successfully built shop-assistant` и `Successfully installed ...`;
код завершения 0.

Первоначальные попытки в песочнице прерваны после зависания `pip` при записи
кеша в пользовательский каталог вне рабочей директории. Диагностический стек
показывал `pip._internal.network.cache._write → adjacent_tmp_file →
tempfile.NamedTemporaryFile`. Прямой запрос к PyPI возвращал HTTP 200.
После разрешённого запуска установки с расширенным доступом зависимости
успешно установились в проектную `.venv`.

Первый запуск тестов остановился при сборе тестов: Starlette 1.10.0 выдаёт
предупреждение об устаревшем HTTPX. При настройке `filterwarnings = ["error"]`
оно правильно привело к ошибке. Зависимость тестового клиента заменена на
HTTPX2, нижняя граница FastAPI обновлена до проверенной 0.162.0.
Предупреждение не подавлялось. Основание — установленный код TestClient и
[официальная документация Starlette](https://raw.githubusercontent.com/Kludex/starlette/main/docs/testclient.md).

Фактически выполненная корректировка окружения:

```powershell
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --timeout 15 --retries 0 --upgrade pip -e ".[dev]"
.\.venv\Scripts\python.exe -m pip uninstall -y httpx httpcore
```

Результат: установлен `pip 26.3`, `httpx2 2.16.0`, `httpcore2 2.16.0`,
`truststore 0.10.5`; устаревшие тестовые зависимости HTTPX/HTTPCore удалены
только из `.venv`. Обе команды завершились с кодом 0.

Основные проверенные версии:

| Компонент | Версия |
| --- | --- |
| FastAPI / Starlette | 0.162.0 / 1.10.0 |
| Pydantic / pydantic-settings | 2.14.2 / 2.16.2 |
| Uvicorn | 0.54.0 |
| HTTPX2 | 2.16.0 |
| Ruff | 0.17.3 |
| mypy | 1.20.2 |
| Bandit | 1.9.4 |
| pytest / pytest-cov / coverage | 9.2.0 / 7.1.0 / 7.17.1 |
| pip-audit / pip | 2.10.1 / 26.3 |

Точные версии указаны для отчёта, не являются lock-файлом. Проект устанавливает
совместимые диапазоны из `pyproject.toml`; будущие версии нужно проверять заново.

## Итоговые проверки

Ниже `python` означает `.\.venv\Scripts\python.exe` из корня проекта.
Вывод приведён с сохранением результатов; длинные локальные пути и служебные
строки сокращены. Все итоговые команды этого раздела завершились с кодом 0.

### Ruff

```text
python -m ruff check .
All checks passed!

python -m ruff format --check .
20 files already formatted
```

### mypy

```text
python -m mypy
Success: no issues found in 14 source files
```

### Bandit

```text
python -m bandit -c pyproject.toml -r app scripts
Test results:
    No issues identified.
Code scanned:
    Total lines of code: 169
    Total lines skipped (#nosec): 0
    Total potential issues skipped due to specifically being disabled: 0
Total issues (by severity):
    Undefined: 0
    Low: 0
    Medium: 0
    High: 0
Files skipped (0)
```

Bandit анализирует приложение и скрипты; pytest-утверждения в тестах не являются
предметом этой проверки. Правила Bandit в приложении не отключены.

### pytest и coverage

```text
python -m pytest --cov --cov-report=term-missing --cov-report=xml
collected 22 items
tests\test_config.py .................... [ 90%]
tests\test_main.py ..                     [100%]

Name               Stmts   Miss Branch BrPart  Cover   Missing
--------------------------------------------------------------
app\__init__.py        0      0      0      0   100%
app\assistant.py       0      0      0      0   100%
app\config.py         43      0      8      0   100%
app\database.py        0      0      0      0   100%
app\llm.py             0      0      0      0   100%
app\main.py           16      0      0      0   100%
app\schemas.py         4      0      0      0   100%
--------------------------------------------------------------
TOTAL                 63      0      8      0   100%
Coverage XML written to file coverage.xml
Required test coverage of 90.0% reached. Total coverage: 100.00%
22 passed in 1.09s
```

Проверены: значения настроек по умолчанию; обязательность ключа; 17 случаев
некорректных настроек; загрузка `.env` и приоритет окружения; сокрытие секретов;
HTTP-контракт `/health`; отказ запуска без ключа. Пользовательские ключи и
локальный `.env` не используются, каждый тест меняет рабочий каталог на
отдельный временный. Внешних LLM-вызовов нет.

Файлы `test_assistant.py`, `test_database.py` и `test_security.py` содержат
только описания, не собирают тестовых случаев и не увеличивают число
выполненных тестов. Показатель 100% для будущих модулей с `Stmts = 0`
не является проверкой их ещё не реализованной функциональности.

### pip-audit

```text
python -m pip_audit --skip-editable --timeout 30 --cache-dir .venv/pip-audit-cache
No known vulnerabilities found
Name           Skip Reason
-------------- -------------------------------
shop-assistant distribution marked as editable
```

Проверено установленное окружение приложения и инструментов. Пропущен только
локальный editable-пакет, зависимости не исключались. Кеш находится внутри
проекта. Это результат базы известных уязвимостей на дату запуска, а не
доказательство отсутствия любых уязвимостей.

### Совместимость зависимостей

```text
python -m pip check
No broken requirements found.
```

## Реальный запуск HTTP-сервера

Этот запуск проверен при первоначальном создании HTTP-каркаса. При исправлении
структуры код `main.py`, `config.py` и `schemas.py` не менялся; API и старт
приложения дополнительно покрыты повторным запуском автоматических тестов выше.

В отдельном процессе временно задано явно фиктивное значение ключа
`example-local-key-not-real`, затем выполнено:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Фактический вывод:

```text
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765
```

HTTP-запросы выполнялись стандартным `urllib.request` с таймаутом 5 секунд:

```text
GET /health: 200 application/json {"status":"ok"}
GET /openapi.json: 200 health documented
```

После проверки тестовый сервер остановлен. Настоящий API-ключ не использован;
работа DeepSeek по этому запуску не утверждается.

Отдельный запуск той же команды без `DEEPSEEK_API_KEY` завершился с кодом **3**:

```text
app.config.ConfigurationError: Проверьте обязательные настройки и допустимые значения: DEEPSEEK_API_KEY
ERROR:    Application startup failed. Exiting.
Exit code: 3
```

Это ожидаемая проверка ошибочной конфигурации, а не непрошедшая итоговая
проверка каркаса. Исходные значения Pydantic в сообщении не раскрываются.

## Git-ignore и документация

Поскольку корень не является Git-репозиторием, правила `.gitignore` проверены
в отдельном временном Git-репозитории: скопирован фактический `.gitignore`,
выполнен `git check-ignore --no-index` для `.env`, `.env.local`,
`.env.example`, `.venv/example`, `data/local.sqlite3`.

```text
.env
.env.local
.venv/example
data/local.sqlite3
OK: secrets, venv and SQLite ignored; .env.example remains visible; project Git untouched
```

Код проверки 0; `.env.example` не игнорируется. Проверять существующие
отслеживаемые секреты нечего: `.git` и реальный `.env` отсутствуют. Коммиты
не создавались.

Отдельная AST-проверка всех созданных Python-файлов подтвердила наличие
русскоязычных модульных и публичных docstrings:

```text
OK: 14 module docstrings, 18 public class/function docstrings
```

Список путей из утверждённого дерева проверен по фактическим файлам.
AST-проверка будущих Python-компонентов подтвердила отсутствие исполняемых
инструкций: каждый из семи файлов содержит только модульный docstring.

```text
OK: all 21 approved file paths exist
OK: 7 future Python files contain only module docstrings
```

Google Style проверяется Ruff. Каждый текущий самостоятельный модуль описан
в `app/README.md`, главном README и `docs/architecture.md`.

## Границы результата

Создан работающий каркас и все файлы утверждённого дерева. Согласно ответу
пользователя будущие компоненты оформлены файлами с описанием назначения
без реализации их логики. Схема SQLite зависит от утверждения карточки 03.
Скрипт `scripts/init_db.py` присутствует, но не содержит исполняемого кода:
его запуск не создаёт БД и не подтверждает готовность регламента. Рабочая
команда инициализации ещё отсутствует, это явно отражено в README.
Ни обработка регламента, ни три краш-теста готового ассистента не реализованы
и не объявляются проверенными. Файлы промптов и веб-чата также содержат
только описания назначения и не используются работающим API.

Первоначальное сокращение дерева исправлено после уточнения пользователя;
дополнительные файлы тестов каркаса и документации сохранены с их фактическим
назначением. Проверки качества повторно выполнены для расширенной структуры.
Карточки Trello при выполнении локальной задачи не изменялись.
