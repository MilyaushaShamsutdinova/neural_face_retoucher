import torch
from torchvision.transforms import functional as F
from PIL import Image
from model.model import Generator


class FaceRetoucher:
    def __init__(self, generator_weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(n_residual=15).to(self.device)
        self.generator.load_state_dict(torch.load(generator_weights_path, map_location=self.device, weights_only=True))
        self.generator.eval()

    def preprocess_image(self, image: Image.Image, image_size=512):
        """Preprocess input image."""
        image = image.convert("RGB")
        image = F.resize(image, (image_size, image_size))
        image = F.to_tensor(image).unsqueeze(0).to(self.device)
        image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return image

    def postprocess_image(self, tensor_image):
        """Convert output tensor to PIL image."""
        tensor_image = tensor_image.squeeze(0).cpu().detach()
        tensor_image = tensor_image * 0.5 + 0.5
        return F.to_pil_image(tensor_image)

    def retouch(self, image_path):
        """Generate a retouched image."""
        input_image = Image.open(image_path)
        preprocessed_image = self.preprocess_image(input_image)
        with torch.no_grad():
            retouched_tensor = self.generator(preprocessed_image)
        retouched_image = self.postprocess_image(retouched_tensor)
        return retouched_image
