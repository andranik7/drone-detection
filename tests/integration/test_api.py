"""Integration tests for the FastAPI serving endpoint."""

import io
from PIL import Image
from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_detections():
    # Create a dummy image
    img = Image.new("RGB", (640, 640), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "count" in data
