import cv2
import mediapipe as mp
import csv
from pathlib import Path
import numpy as np

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = BASE_DIR / "word_landmarks.csv"

mp_hands = mp.solutions.hands

rows = []

print("Starting word landmark extraction...")

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    # Each folder represents one word
    for word_folder in sorted(DATASET_DIR.iterdir()):

        if not word_folder.is_dir():
            continue

        word = word_folder.name
        print(f"Processing: {word}")

        for video_path in word_folder.glob("*.mp4"):

            cap = cv2.VideoCapture(str(video_path))

            frame_features = []

            while True:
                success, frame = cap.read()

                if not success:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb)

                if results.multi_hand_landmarks:

                    hand = results.multi_hand_landmarks[0]

                    wrist_x = hand.landmark[0].x
                    wrist_y = hand.landmark[0].y
                    wrist_z = hand.landmark[0].z

                    features = []

                    for landmark in hand.landmark:
                        features.extend([
                            landmark.x - wrist_x,
                            landmark.y - wrist_y,
                            landmark.z - wrist_z
                        ])

                    frame_features.append(features)

            cap.release()

            # Skip video if no hand was detected
            if len(frame_features) == 0:
                continue

            frame_features = np.array(frame_features)

            # Calculate mean and standard deviation
            mean_features = np.mean(
                frame_features,
                axis=0
            )

            std_features = np.std(
                frame_features,
                axis=0
            )

            final_features = np.concatenate([
                mean_features,
                std_features
            ])

            rows.append(
                [word] + final_features.tolist()
            )

print("\nWriting word landmark data...")

# Create CSV header
header = ["label"]

for i in range(21):
    header += [
        f"mean_x{i}",
        f"mean_y{i}",
        f"mean_z{i}"
    ]

for i in range(21):
    header += [
        f"std_x{i}",
        f"std_y{i}",
        f"std_z{i}"
    ]

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(header)
    writer.writerows(rows)

print("\nWORD LANDMARK EXTRACTION COMPLETE!")
print(f"Total videos processed: {len(rows)}")
print(f"Saved to: {OUTPUT_FILE}")