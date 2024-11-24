import pytest
from fastapi.testclient import TestClient
from app.api import app
from PIL import Image
import io


@pytest.fixture
def client():
    return TestClient(app)


def create_dummy_image(format="JPEG"):
    image = Image.new("RGB", (1024, 1024), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def test_retouch_valid_image(client):
    image = create_dummy_image()
    response = client.post("/retouch", files={"file": ("dummy.jpg", image, "image/jpeg")})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.headers["content-type"] == "image/jpeg", "Response is not a JPEG image."


def test_retouch_invalid_file_type(client):
    response = client.post("/retouch", files={"file": ("dummy.txt", b"This is not an image", "text/plain")})
    assert response.status_code == 500, f"Expected 500, got {response.status_code}"
    assert "Invalid file type" in response.json()["detail"], "Incorrect error message for invalid file type."


def test_retouch_no_file(client):
    response = client.post("/retouch")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    assert "No file provided." in response.json()["detail"], "Incorrect error message for missing file."


def test_retouch_corrupted_image(client):
    corrupted_image = io.BytesIO(b"This is not a valid image file")
    response = client.post("/retouch", files={"file": ("corrupted.jpg", corrupted_image, "image/jpeg")})
    assert response.status_code == 500, f"Expected 500, got {response.status_code}"
    assert "not a valid image" in response.json()["detail"], "Incorrect error message for corrupted image."


def test_retouch_inference_failure(client, monkeypatch):
    def mock_retouch_image(*args, **kwargs):
        return None

    monkeypatch.setattr("model.inference.FaceRetoucher.retouch_image", mock_retouch_image)

    image = create_dummy_image()
    response = client.post("/retouch", files={"file": ("dummy.jpg", image, "image/jpeg")})
    assert response.status_code == 500, f"Expected 500, got {response.status_code}"
    assert "Model inference failed" in response.json()["detail"], "Incorrect error message for inference failure."
