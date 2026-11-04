"""Проверки HTTP-контракта и проверки конфигурации при запуске FastAPI."""

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigurationError, Settings
from app.main import create_app


@pytest.mark.usefixtures("configured_environment")
def test_health_endpoint() -> None:
    """API отдаёт стабильный JSON после успешной проверки конфигурации."""
    application = create_app()
    with TestClient(application) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["content-type"] == "application/json"
        assert isinstance(application.state.settings, Settings)
        assert "example-test-key-not-real" not in response.text


def test_application_refuses_start_without_key() -> None:
    """Ошибочная конфигурация останавливает приложение до обслуживания запросов."""
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"), TestClient(create_app()):
        pytest.fail("Приложение не должно запускаться без обязательных настроек")
