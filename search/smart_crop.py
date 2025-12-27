import cv2
import numpy as np
from PIL import Image

import mediapipe as mp

# Explicitly load FaceDetection module (Windows-safe)
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

def smart_crop_eyewear(pil_image):
    """
    Detect face and crop eyewear region.
    Falls back to full image if no face is detected.
    """

    image = np.array(pil_image.convert("RGB"))
    h, w, _ = image.shape

    results = mp_face_detection.process(image)

    if results is None or not results.detections:
        return pil_image

    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box

    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    # Heuristic crop for eyewear region
    crop_y1 = int(y + 0.25 * bh)
    crop_y2 = int(y + 0.65 * bh)
    crop_x1 = int(x + 0.05 * bw)
    crop_x2 = int(x + 0.95 * bw)


    # Clamp to image boundaries
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if cropped.size == 0:
        return pil_image

    return Image.fromarray(cropped)
