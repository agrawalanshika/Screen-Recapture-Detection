import os
import time
import cv2
import joblib
import numpy as np

from features import extract_features

MODEL_PATH = "best_random_forest.pkl"

REAL_PATH = "dataset/Real"
SCREEN_PATH = "dataset/Screen"

IMAGE_SIZE = (256, 256)

model = joblib.load(MODEL_PATH)

times = []

fastest_image = ""
slowest_image = ""

fastest_time = float("inf")
slowest_time = 0


def benchmark_folder(folder):

    global fastest_time
    global slowest_time
    global fastest_image
    global slowest_image

    images = sorted(os.listdir(folder))

    for image_name in images:

        image_path = os.path.join(folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)

        start = time.perf_counter()

        features = extract_features(img)
        features = features.reshape(1, -1)

        prediction = model.predict(features)

        end = time.perf_counter()

        elapsed = (end - start) * 1000  # milliseconds

        times.append(elapsed)

        if elapsed < fastest_time:
            fastest_time = elapsed
            fastest_image = image_name

        if elapsed > slowest_time:
            slowest_time = elapsed
            slowest_image = image_name


print("RUNNING LATENCY BENCHMARK\n")

benchmark_folder(REAL_PATH)
benchmark_folder(SCREEN_PATH)

times = np.array(times)

average_latency = np.mean(times)
minimum_latency = np.min(times)
maximum_latency = np.max(times)
std_latency = np.std(times)

images_per_second = 1000 / average_latency


print("\nLATENCY RESULTS")

print(f"Total Images        : {len(times)}")
print(f"Average Latency     : {average_latency:.2f} ms")
print(f"Minimum Latency     : {minimum_latency:.2f} ms")
print(f"Maximum Latency     : {maximum_latency:.2f} ms")
print(f"Std Deviation       : {std_latency:.2f} ms")

print(f"\nImages / Second     : {images_per_second:.2f}")


print("\nAnalysis Completed Successfully!")
