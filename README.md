# Visual Similarity Search for Eyewear

## Overview
This project implements an AI-powered visual search system for eyewear.  
Users can upload an image of glasses and retrieve visually similar products based on
frame shape, color, and style. The system also supports multimodal search by allowing
users to refine results using natural language text modifiers.

## Key Features
- Image-based visual similarity search using deep learning embeddings
- Multimodal search (image + text) using CLIP
- Attribute-aware filtering (brand, price, frame style)
- Feedback-driven ranking improvement
- Scalable vector search using FAISS
- Smart cropping using face detection to automatically isolate eyewear region in real-world images


## System Architecture
1. Image ingestion and preprocessing
2. Feature extraction using ResNet50
3. Vector indexing using FAISS
4. Multimodal re-ranking using CLIP
5. Interactive search interface using Streamlit

(Refer to architecture.png)

## Models Used
- ResNet50 (CNN) for image embeddings
- CLIP (ViT-B/32) for text–image similarity

## Vector Search
- Vector database: FAISS
- Distance metric: Euclidean (L2)

## Multimodal Search
Users can optionally provide a text modifier (e.g., "tortoise shell", "gold frame").
The system re-ranks visually similar results using CLIP-based text–image similarity.

## Smart Cropping (Bonus)
To handle real-world images such as selfies or photos with complex backgrounds, the system includes a smart cropping step.
If a face is detected in the uploaded image, the eyewear region is automatically cropped before feature extraction.
If no face is detected, the system safely falls back to using the full image.

This improves robustness and visual relevance without requiring perfect segmentation.

## Feedback Loop
User interactions (likes) are logged and used to boost frequently selected products
for similar visual queries.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
