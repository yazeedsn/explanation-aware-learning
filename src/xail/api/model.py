# app/model.py

import torch
from torchvision import transforms
from PIL import Image

# from .your_model import YourModel


class ModelService:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # self.model = YourModel(...)
        # checkpoint = torch.load(
        #     checkpoint_path,
        #     map_location=self.device,
        # )

        # self.model.load_state_dict(checkpoint)
        # self.model.to(self.device)
        # self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        x = self.transform(image).unsqueeze(0)
        x = x.to(self.device)

        logits = torch.tensor([0.7, 0.1]) # self.model(x)

        probabilities = torch.sigmoid(logits)

        return probabilities.cpu()
