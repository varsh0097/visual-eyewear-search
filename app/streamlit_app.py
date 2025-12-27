# ================= PATH FIX =================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ===========================================

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import csv
from datetime import datetime

from search.faiss_search import FaissSearchEngine

# ================= CONFIG =================
DEVICE = torch.device("cpu")
FEEDBACK_FILE = os.path.join("data", "feedback", "user_feedback.csv")

# ================= MODEL LOADING =================
@st.cache_resource
def load_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, preprocess

# ================= IMAGE CACHE =================
@st.cache_data
def load_display_image(path):
    return Image.open(path)

# ================= FEEDBACK LOGIC =================
def get_product_boosts():
    boosts = {}
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["product_id"]
                boosts[pid] = boosts.get(pid, 0) + 0.5
    return boosts

def save_feedback(product_id, query_style):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "product_id", "searched_style", "action"])
        writer.writerow([datetime.now(), product_id, query_style, "like"])

    st.success(f"Feedback recorded for Product ID: {product_id}")

# ================= MAIN APP =================
def main():
    st.set_page_config(page_title="LensAI", layout="wide")
    st.title("LensAI: Multimodal Visual Search for Eyewear")

    st.caption(
        "Upload an eyewear image and optionally refine results using a text description "
        "(for example: tortoise shell, gold frame)."
    )

    model, preprocess = load_model()
    search_engine = FaissSearchEngine()

    # ---------------- SIDEBAR FILTERS ----------------
    st.sidebar.header("Filters")

    all_brands = sorted(list(set([item["brand"] for item in search_engine.metadata])))
    all_brands.insert(0, "All")
    selected_brand = st.sidebar.selectbox("Brand", all_brands)

    selected_style = st.sidebar.selectbox(
        "Frame Style",
        ["All", "Aviator", "Wayfarer", "Round", "Cat", "Rimless", "Rectangular"]
    )

    max_price = st.sidebar.slider("Maximum Price", 500, 10000, 5000)

    # ---------------- MULTIMODAL TEXT INPUT ----------------
    common_modifiers = [
        "",
        "black frame",
        "gold frame",
        "silver frame",
        "tortoise shell",
        "transparent frame",
        "thin metal frame",
        "thick acetate frame"
    ]

    selected_modifier = st.sidebar.selectbox(
        "Quick text modifiers",
        common_modifiers
    )

    text_modifier = st.sidebar.text_input(
        "Custom text modifier (optional)",
        value=selected_modifier,
        placeholder="Example: tortoise shell, gold frame"
    )

    # ---------------- IMAGE UPLOAD ----------------
    uploaded = st.file_uploader(
        "Upload an eyewear image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Query Image", width=280)

        if st.button("Search"):
            img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                query_vector = model(img_tensor).flatten().cpu().numpy()

            filters = {
                "brand": selected_brand,
                "style": selected_style,
                "max_price": max_price
            }

            boosts = get_product_boosts()

            with st.spinner("Searching similar eyewear..."):
                results = search_engine.search(
                    query_vector=query_vector,
                    top_k=5,
                    filters=filters,
                    boosts=boosts,
                    text_query=text_modifier
                )

            # ---------------- RESULTS ----------------
            if results:
                st.subheader("Search Results")
                cols = st.columns(len(results))

                for i, item in enumerate(results):
                    with cols[i]:
                        st.image(
                            load_display_image(item["image_path"]),
                            use_container_width=True
                        )
                        st.caption(f"{item['brand']} | {item['predicted_style']}")
                        st.text(f"Price: ₹{item['price']}")

                        if st.button("Like", key=f"like_{i}"):
                            save_feedback(item["product_id"], selected_style)
            else:
                st.warning("No similar products found.")

if __name__ == "__main__":
    main()
