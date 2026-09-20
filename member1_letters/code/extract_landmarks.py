import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = BASE_DIR / "landmarks.csv"

# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -----------------------------
# Store extracted data
# -----------------------------
data = []

# -----------------------------
# Read each letter folder
# -----------------------------
for letter_folder in sorted(DATASET_DIR.iterdir()):

    if not letter_folder.is_dir():
        continue

    label = letter_folder.name.upper()

    print(f"Processing letter: {label}")

    for image_file in letter_folder.iterdir():

        if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = cv2.imread(str(image_file))

        if image is None:
            continue

        # Convert BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hand
        results = hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            continue

        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Wrist coordinates
        wrist = hand_landmarks.landmark[0]

        features = []

        # Extract 21 landmarks
        for landmark in hand_landmarks.landmark:

            # Make coordinates relative to wrist
            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z

            features.extend([x, y, z])

        # 63 features + label
        data.append(features + [label])

# -----------------------------
# Create DataFrame
# -----------------------------
columns = []

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

columns.append("label")

df = pd.DataFrame(data, columns=columns)

# -----------------------------
# Save CSV
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False)

hands.close()

print("\n--------------------------------")
print("Landmark extraction completed!")
print(f"Total samples: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")
print("--------------------------------")