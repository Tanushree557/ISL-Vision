import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "letter_model.pkl"

model = joblib.load(MODEL_FILE)


# -----------------------------
# Feature names
# -----------------------------
FEATURE_NAMES = []

for i in range(21):
    FEATURE_NAMES.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    predicted_letter = "-"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Wrist
        wrist = hand.landmark[0]

        features = []

        # 21 landmarks × 3 = 63 features
        for landmark in hand.landmark:

            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z

            features.extend([x, y, z])

        # Create DataFrame
        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )

        # Prediction
        predicted_letter = model.predict(
            input_data
        )[0]

        # Probabilities
        probabilities = model.predict_proba(
            input_data
        )[0]

        classes = model.classes_

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        # Show top 3 predictions
        y_position = 150

        for index in top_indices:

            letter = classes[index]
            confidence = probabilities[index] * 100

            text = f"{letter}: {confidence:.1f}%"

            cv2.putText(
                frame,
                text,
                (30, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y_position += 35


    # -----------------------------
    # Big prediction
    # -----------------------------

    cv2.putText(
        frame,
        f"DETECTED: {predicted_letter}",
        (30, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (30, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ISL LETTER TEST",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break


cap.release()
hands.close()
cv2.destroyAllWindows()