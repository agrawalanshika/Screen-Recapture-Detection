import os
import cv2
import pandas as pd

from features import extract_features


REAL_PATH = "dataset/Real"
SCREEN_PATH = "dataset/Screen"      

IMAGE_SIZE = (256, 256)


feature_names = [

    "Blur Score",
    "Edge Density",
    "Brightness",
    "Entropy",

    "Mean Red",
    "Mean Green",
    "Mean Blue",

    "Std Red",
    "Std Green",
    "Std Blue",

    "Sobel X",
    "Sobel Y",

    "FFT High",
    "FFT Low",

    "LBP Mean",
    "LBP Variance",

    "RMS Contrast",
    "Saturation Mean",

    "Tenengrad",
    "Noise Estimate",
    "Pixel Grid Score"
]

rows = []


def process_folder(folder_path, label):

    images = sorted(os.listdir(folder_path))

    for image_name in images:

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read: {image_name}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)

        features = extract_features(img)

    

        row = [image_name]

        row.extend(features.tolist())

        row.append(label)

        rows.append(row)


print("Creating Feature Dataset\n")

print("Processing REAL Images...")
process_folder(REAL_PATH, 0)

print("Processing SCREEN Images...")
process_folder(SCREEN_PATH, 1)


columns = ["Filename"] + feature_names + ["Label"]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("features.csv", index=False)

print("\nDataset Created Successfully!")


print("\nDataset Shape :", df.shape)

print("\nFirst 5 Rows\n")
print(df.head())

print("\nCSV Saved As : features.csv")