import cv2
import mediapipe as mp
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "letter_model.pkl"

model = joblib.load(MODEL_FILE)

print("ISL Letter Recognition Model Loaded")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    hands.close()
    exit()

print("Camera started")
print("Press Q to exit")

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    predicted_letter = ""

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

        input_data = np.array(features).reshape(1, -1)

        predicted_letter = model.predict(input_data)[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 78),
        (45, 45, 75),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.90,
        frame,
        0.10,
        0
    )

    cv2.putText(
        frame,
        "ISL",
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LETTER RECOGNITION",
        (85, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )

    card_x1 = width - 270
    card_y1 = 105
    card_x2 = width - 25
    card_y2 = 245

    card_overlay = frame.copy()

    cv2.rectangle(
        card_overlay,
        (card_x1, card_y1),
        (card_x2, card_y2),
        (255, 255, 255),
        -1
    )

    frame = cv2.addWeighted(
        card_overlay,
        0.94,
        frame,
        0.06,
        0
    )

    cv2.putText(
        frame,
        "DETECTED LETTER",
        (card_x1 + 18, card_y1 + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 100, 110),
        1
    )

    if predicted_letter:

        cv2.putText(
            frame,
            predicted_letter,
            (card_x1 + 88, card_y1 + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (80, 70, 210),
            4
        )

        cv2.putText(
            frame,
            "ISL alphabet",
            (card_x1 + 72, card_y1 + 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 120, 130),
            1
        )

    else:

        cv2.putText(
            frame,
            "--",
            (card_x1 + 92, card_y1 + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (140, 140, 150),
            3
        )

        cv2.putText(
            frame,
            "Show your hand",
            (card_x1 + 60, card_y1 + 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 120, 130),
            1
        )

    if predicted_letter:
        status = "HAND DETECTED"
    else:
        status = "WAITING FOR HAND"

    cv2.circle(
        frame,
        (card_x1 + 25, card_y2 - 23),
        6,
        (90, 190, 110) if predicted_letter else (170, 170, 170),
        -1
    )

    cv2.putText(
        frame,
        status,
        (card_x1 + 40, card_y2 - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (90, 90, 100),
        1
    )

    bottom_overlay = frame.copy()

    cv2.rectangle(
        bottom_overlay,
        (0, height - 55),
        (width, height),
        (45, 45, 75),
        -1
    )

    frame = cv2.addWeighted(
        bottom_overlay,
        0.92,
        frame,
        0.08,
        0
    )

    cv2.putText(
        frame,
        "Show an ISL alphabet sign",
        (25, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (width - 150, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 230),
        1
    )

    cv2.imshow(
        "ISL Vision - Letter Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Letter Recognition Closed")
