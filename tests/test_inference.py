import unittest
from PIL import Image
from model.inference import FaceRetoucher


class TestFaceRetoucher(unittest.TestCase):
    def setUp(self):
        self.model_path = "model/weights/generator.pth"
        self.sample_image_path = "static/sample.png"
        self.retoucher = FaceRetoucher(generator_weights_path=self.model_path)

    def test_preprocess_image(self):
        image = Image.open(self.sample_image_path)
        preprocessed = self.retoucher.preprocess_image(image)
        self.assertEqual(preprocessed.shape, (1, 3, 512, 512))

    def test_postprocess_image(self):
        import torch
        tensor_image = torch.randn(1, 3, 512, 512)
        postprocessed_image = self.retoucher.postprocess_image(tensor_image)
        self.assertIsInstance(postprocessed_image, Image.Image)

    def test_retouch(self):
        retouched_image = self.retoucher.retouch(self.sample_image_path)
        self.assertIsInstance(retouched_image, Image.Image)
        retouched_image.save("static/retouched_test_output.jpg")
