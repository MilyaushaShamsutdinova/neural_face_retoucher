import pytest
from fastapi.testclient import TestClient
from app.api import app
from io import BytesIO
from PIL import Image

client = TestClient(app)

@pytest.mark.asyncio
async def test_retouch_image():
    image = Image.new("RGB", (100, 100), color="black")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post(
        "/retouch",
        files={"file": ("static\sample.png", image_bytes, "image/png")}
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    retouched_image = Image.open(BytesIO(response.content))
    assert retouched_image.size == (100, 100)  # Ensure that the image size is correct
    assert retouched_image.mode == "RGB"  # Ensure that the image mode is RGB

@pytest.mark.asyncio
async def test_invalid_file_type():
    content = b"Hello, this is a text file."
    response = client.post(
        "/retouch",
        files={"file": ("test_text.txt", content, "text/plain")}
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.text
