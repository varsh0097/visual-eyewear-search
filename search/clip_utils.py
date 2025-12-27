import torch
from transformers import CLIPModel, CLIPProcessor

DEVICE = torch.device("cpu")

class CLIPTextImageScorer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def text_image_similarity(self, image, text):
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # cosine similarity in CLIP space
        sim = outputs.logits_per_image.softmax(dim=1)[0][0].item()
        return sim
