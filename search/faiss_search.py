from search.clip_utils import CLIPTextImageScorer
from PIL import Image
import os
import pickle
import faiss
import numpy as np


INDEX_FILE = os.path.join("data", "embeddings", "faiss.index")
META_FILE = os.path.join("data", "embeddings", "faiss_metadata.pkl")

class FaissSearchEngine:
    def __init__(self):
        self.index = faiss.read_index(INDEX_FILE)
        self._text_cache = {}
        with open(META_FILE, "rb") as f:
            self.metadata = pickle.load(f)
            
        # CLIP scorer for multimodal fusion
        self.clip_scorer = CLIPTextImageScorer()

    def search(
        self,
        query_vector,
        top_k=5,
        filters=None,
        boosts=None,
        text_query=None,
        alpha=0.7,   # image similarity weight
        beta=0.3     # text similarity weight
    ):
        """
        query_vector : numpy array from ResNet
        text_query   : optional natural language modifier
        """

        query_vector = np.array([query_vector]).astype("float32")

        # Retrieve more candidates, then re-rank
        distances, indices = self.index.search(query_vector, top_k * 2)

        results = []

        for dist, idx in zip(distances[0], indices[0]):
            item = self.metadata[idx]

            # --------- STRUCTURED FILTERS ----------
            if filters:
                if filters.get("brand") and filters["brand"] != "All":
                    if item["brand"] != filters["brand"]:
                        continue

                if filters.get("style") and filters["style"] != "All":
                    if filters["style"].lower() not in item["predicted_style"].lower():
                        continue

                if item["price"] > filters.get("max_price", float("inf")):
                    continue

            # --------- FEEDBACK BOOST ----------
            boost = boosts.get(item["product_id"], 0) if boosts else 0

            # --------- TEXT–IMAGE SIMILARITY ----------
            text_penalty = 0.0
            if text_query and text_query.strip():
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    text_sim = self.clip_scorer.text_image_similarity(img, text_query)
                    text_penalty = 1.0 - text_sim  # lower is better
                except Exception:
                    text_penalty = 0.5  # safe fallback

            # --------- FINAL FUSED SCORE ----------
            final_score = (alpha * dist) + (beta * text_penalty) - boost

            results.append((final_score, item))

        # Rank by final fused score
        results.sort(key=lambda x: x[0])

        return [r[1] for r in results[:top_k]]
