"""
Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
"""

import sys
import os
import numpy as np
import cv2
import joblib
import pandas as pd
from PIL import Image

from feature_engineering.features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_train", "best_random_forest.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    "Blur Score", "Edge Density", "Brightness", "Entropy",
    "Mean Red", "Mean Green", "Mean Blue",
    "Std Red", "Std Green", "Std Blue",
    "Sobel X", "Sobel Y",
    "FFT High", "FFT Low",
    "LBP Mean", "LBP Variance",
    "RMS Contrast", "Saturation Mean",
    "Tenengrad", "Noise Estimate", "Pixel Grid Score"
]


def predict(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (256, 256))

    features = extract_features(img)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    probabilities = model.predict_proba(features_df)[0]
    return float(f"{probabilities[1]:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found — {image_path}")
        sys.exit(1)

    print(predict(image_path))