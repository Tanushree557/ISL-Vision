import cv2
import mediapipe as mp
import csv
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Output CSV
OUTPUT_FILE = BASE_DIR / "landmarks.csv"

# MediaPipe
mp_hands = mp.solutions.hands

# Store rows
rows = []

# Create MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    # A to Z
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

        folder = DATASET_DIR / label

        if not folder.exists():
            print(f"Folder not found: {label}")
            continue

        print(f"Processing {label}...")

        # Read images
        for image_path in folder.iterdir():

            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect hand
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                # Use wrist as reference point
                wrist_x = hand.landmark[0].x
                wrist_y = hand.landmark[0].y
                wrist_z = hand.landmark[0].z

                features = []

                # 21 landmarks × 3 coordinates
                for landmark in hand.landmark:
                    features.extend([
                        landmark.x - wrist_x,
                        landmark.y - wrist_y,
                        landmark.z - wrist_z
                    ])

                rows.append([label] + features)

print("Writing landmark data...")

# CSV header
header = ["label"]

for i in range(21):
    header += [
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ]

# Save CSV
with open(OUTPUT_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)

print("DONE!")
print(f"Total samples extracted: {len(rows)}")
print(f"Saved to: {OUTPUT_FILE}")