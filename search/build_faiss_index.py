import os
import pickle
import faiss
import numpy as np

INPUT_FILE = os.path.join("data", "embeddings", "enriched_embeddings.pkl")
OUTPUT_INDEX = os.path.join("data", "embeddings", "faiss.index")
OUTPUT_META = os.path.join("data", "embeddings", "faiss_metadata.pkl")

def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    vectors = np.array([item["vector"] for item in data]).astype("float32")

    # L2 distance (Euclidean)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, OUTPUT_INDEX)

    # Save metadata separately (FAISS stores only vectors)
    with open(OUTPUT_META, "wb") as f:
        pickle.dump(data, f)

    print(f"FAISS index built with {index.ntotal} vectors")

if __name__ == "__main__":
    main()
