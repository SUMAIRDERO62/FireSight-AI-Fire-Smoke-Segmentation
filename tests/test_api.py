from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO

from app.backend.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint_lists_both_supported_models():
    response = TestClient(app).get("/models")
    assert response.status_code == 200
    assert set(response.json()["models"]) == {"unet", "yolo11-seg"}


def test_segment_returns_service_unavailable_without_checkpoint():
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(buffer, format="PNG")
    response = TestClient(app).post(
        "/segment?model=unet",
        files={"file": ("image.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 503

