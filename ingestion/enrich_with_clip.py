import os
import pickle
import random
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

INPUT_FILE = os.path.join("data", "embeddings", "embeddings_resnet.pkl")
OUTPUT_FILE = os.path.join("data", "embeddings", "enriched_embeddings.pkl")
DEVICE = torch.device("cpu")

BRANDS = ["Ray-Ban", "Lenskart Air", "Vincent Chase", "John Jacobs", "Oakley"]
STYLES = [
    "Aviator Sunglasses", "Wayfarer Glasses", "Round Frame Glasses",
    "Cat Eye Glasses", "Rimless Glasses", "Rectangular Glasses"
]

def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    for item in tqdm(data):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(text=STYLES, images=img, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                out = model(**inputs)
            label = STYLES[out.logits_per_image.softmax(dim=1).argmax()]
            item["predicted_style"] = label.split()[0]
            item["brand"] = random.choice(BRANDS)
            item["price"] = random.randint(1000, 8000)
        except:
            pass

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
