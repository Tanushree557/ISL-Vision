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
print("Show an ISL letter A-Z")
print("Press Q to exit")
print("Press B or click BACK TO MAIN MENU to return")

window_name = "ISL Vision - Letter Recognition"

back_clicked = False


def mouse_callback(event, x, y, flags, param):

    global back_clicked

    if event == cv2.EVENT_LBUTTONDOWN:

        height, width = param

        button_x1 = 20
        button_y1 = height - 48
        button_x2 = 220
        button_y2 = height - 10

        if (
            button_x1 <= x <= button_x2
            and button_y1 <= y <= button_y2
        ):
            back_clicked = True


cv2.namedWindow(window_name)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        (height, width)
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

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

        input_data = np.array(
            features
        ).reshape(1, -1)

        predicted_letter = model.predict(
            input_data
        )[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    # -------------------------------
    # TOP HEADER
    # -------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 75),
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
        (25, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LETTER RECOGNITION",
        (82, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )

    # -------------------------------
    # DETECTED LETTER CARD
    # -------------------------------

    card_x1 = width - 300
    card_y1 = 100
    card_x2 = width - 25
    card_y2 = 280

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
        (card_x1 + 20, card_y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 100, 110),
        1
    )

    if predicted_letter:

        cv2.putText(
            frame,
            predicted_letter.upper(),
            (card_x1 + 95, card_y1 + 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (80, 70, 210),
            5
        )

    else:

        cv2.putText(
            frame,
            "--",
            (card_x1 + 105, card_y1 + 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (140, 140, 150),
            4
        )

        cv2.putText(
            frame,
            "Show your hand",
            (card_x1 + 65, card_y1 + 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 120, 130),
            1
        )

    # -------------------------------
    # STATUS
    # -------------------------------

    if results.multi_hand_landmarks:

        status = "HAND DETECTED"
        status_color = (90, 190, 110)

    else:

        status = "WAITING FOR HAND"
        status_color = (170, 170, 170)

    cv2.circle(
        frame,
        (card_x1 + 25, card_y2 - 25),
        6,
        status_color,
        -1
    )

    cv2.putText(
        frame,
        status,
        (card_x1 + 40, card_y2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (90, 90, 100),
        1
    )

    # -------------------------------
    # BOTTOM BAR
    # -------------------------------

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

    # Back button

    cv2.rectangle(
        frame,
        (20, height - 48),
        (220, height - 10),
        (80, 70, 150),
        -1
    )

    cv2.putText(
        frame,
        "< BACK TO MAIN MENU",
        (32, height - 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Show an ISL letter A-Z",
        (245, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Q = Exit",
        (width - 90, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 230),
        1
    )

    cv2.imshow(
        window_name,
        frame
    )

    # -------------------------------
    # KEYBOARD CONTROLS
    # -------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("b"):
        break

    if back_clicked:
        break


cap.release()

hands.close()

cv2.destroyAllWindows()

print("Letter Recognition Closed")