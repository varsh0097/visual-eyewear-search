import os
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

ROOT_DIR = os.path.join("data", "raw_images")
OUTPUT_FILE = os.path.join("data", "embeddings", "embeddings_resnet.pkl")
DEVICE = torch.device("cpu")

def get_model_and_transforms():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)
    return model, preprocess

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    model, preprocess = get_model_and_transforms()

    data = []
    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, f)
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        vec = model(tensor).flatten().cpu().numpy()
                    data.append({
                        "product_id": os.path.basename(root),
                        "image_path": path,
                        "vector": vec
                    })
                except Exception as e:
                    print(f"Skip {path}: {e}")

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
