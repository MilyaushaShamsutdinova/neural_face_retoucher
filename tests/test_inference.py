import pytest
from PIL import Image
import torch
from model.inference import FaceRetoucher


@pytest.fixture
def face_retoucher():
    return FaceRetoucher("model/weights/generator.pth")


def test_preprocess_image(face_retoucher):
    image = Image.new("RGB", (1024, 1024))
    processed = face_retoucher.preprocess_image(image)
    assert processed.shape == (1, 3, 512, 512), "Preprocessed image dimensions are incorrect."
    assert isinstance(processed, torch.Tensor), "Preprocessed output is not a tensor."


def test_postprocess_image(face_retoucher):
    dummy_tensor = torch.randn(1, 3, 512, 512)
    postprocessed = face_retoucher.postprocess_image(dummy_tensor)
    assert isinstance(postprocessed, Image.Image), "Postprocessed output is not a PIL.Image object."


def test_retouch_image(face_retoucher):
    image = Image.new("RGB", (1024, 1024))
    retouched_image = face_retoucher.retouch_image(image)
    assert isinstance(retouched_image, Image.Image), "Retouched output is not a PIL.Image object."


def test_retouch(face_retoucher, tmp_path):
    dummy_image_path = tmp_path / "dummy.jpg"
    image = Image.new("RGB", (1024, 1024))
    image.save(dummy_image_path)

    retouched_image = face_retoucher.retouch(dummy_image_path)
    assert isinstance(retouched_image, Image.Image), "Retouched output is not a PIL.Image object."
