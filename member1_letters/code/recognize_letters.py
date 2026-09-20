import cv2
import mediapipe as mp
import joblib
import pandas as pd
import subprocess
import time
import threading
import queue

from pathlib import Path
from collections import deque, Counter


# =========================================================
# MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "letter_model.pkl"

model = joblib.load(MODEL_FILE)


# =========================================================
# WINDOWS VOICE SYSTEM
# =========================================================

voice_queue = queue.Queue()
voice_running = True


def voice_worker():

    while voice_running:

        try:
            letter = voice_queue.get(timeout=0.1)

        except queue.Empty:
            continue

        try:

            print(f"Speaking: {letter}")

            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object "
                "System.Speech.Synthesis.SpeechSynthesizer; "
                f"$speaker.Speak('{letter}'); "
                "$speaker.Dispose();"
            )

            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    command
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        except Exception as error:

            print("Voice error:", error)

        finally:

            voice_queue.task_done()


# Start ONE voice worker
voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)

voice_thread.start()


def speak_letter(letter):

    if letter != "-":

        voice_queue.put(letter)


# =========================================================
# FEATURE NAMES - 63 FEATURES
# =========================================================

FEATURE_NAMES = []

for i in range(21):

    FEATURE_NAMES.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("Camera could not be opened.")

    voice_running = False

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# =========================================================
# PREDICTION
# =========================================================

# General prediction smoothing
prediction_history = deque(maxlen=5)

current_letter = "-"


# =========================================================
# STABLE LETTER DETECTION
# =========================================================

# This stores recent predictions for voice confirmation
stable_history = deque(maxlen=12)

# Letter that was last spoken
last_confirmed_letter = "-"

# Number of identical predictions needed before speaking
STABLE_FRAMES = 12


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # =====================================================
    # MIRROR CAMERA
    # =====================================================

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]


    # =====================================================
    # RGB CONVERSION
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = hands.process(
        rgb_frame
    )


    # =====================================================
    # HAND DETECTED
    # =====================================================

    if results.multi_hand_landmarks:

        # -------------------------------------------------
        # DRAW LANDMARKS ON ALL HANDS
        # -------------------------------------------------

        for detected_hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                detected_hand,
                mp_hands.HAND_CONNECTIONS
            )


        # -------------------------------------------------
        # FIRST HAND → LETTER RECOGNITION
        # -------------------------------------------------

        hand = results.multi_hand_landmarks[0]

        wrist = hand.landmark[0]

        features = []


        for landmark in hand.landmark:

            x = landmark.x - wrist.x
            y = landmark.y - wrist.y
            z = landmark.z - wrist.z

            features.extend([
                x,
                y,
                z
            ])


        # -------------------------------------------------
        # CREATE MODEL INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        try:

            prediction = model.predict(
                input_data
            )[0]


            # Add prediction to normal smoothing
            prediction_history.append(
                prediction
            )


            # Current displayed letter
            current_letter = Counter(
                prediction_history
            ).most_common(1)[0][0]


            # =================================================
            # STABLE PREDICTION FOR VOICE
            # =================================================

            stable_history.append(
                prediction
            )


            # Check whether all recent predictions
            # are the same
            if len(stable_history) == STABLE_FRAMES:

                stable_letter = Counter(
                    stable_history
                ).most_common(1)[0][0]


                # Count how many times this letter occurred
                stable_count = stable_history.count(
                    stable_letter
                )


                # -------------------------------------------------
                # ONLY SPEAK IF LETTER IS REALLY STABLE
                # -------------------------------------------------

                if (

                    stable_count == STABLE_FRAMES

                    and stable_letter != last_confirmed_letter

                ):

                    print(
                        f"Confirmed letter: {stable_letter}"
                    )

                    speak_letter(
                        stable_letter
                    )

                    last_confirmed_letter = stable_letter

                    # Clear history so a new letter
                    # must become stable again
                    stable_history.clear()


        except Exception as error:

            print(
                "Prediction error:",
                error
            )

            current_letter = "-"

            stable_history.clear()


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        cv2.putText(
            frame,
            "HAND DETECTED",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 100),
            2
        )


    # =====================================================
    # NO HAND
    # =====================================================

    else:

        current_letter = "-"

        prediction_history.clear()

        stable_history.clear()

        # Allow the same letter to be spoken
        # when the hand is shown again
        last_confirmed_letter = "-"


        cv2.putText(
            frame,
            "SHOW YOUR HAND",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 255),
            2
        )


    # =====================================================
    # HEADER
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 110),
        (45, 30, 75),
        -1
    )


    cv2.putText(
        frame,
        "ISL LETTER RECOGNITION",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Indian Sign Language - A to Z",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1
    )


    # =====================================================
    # DETECTED LETTER BOX
    # =====================================================

    box_left = width - 330
    box_top = 130
    box_right = width - 30
    box_bottom = 390


    cv2.rectangle(
        frame,
        (box_left, box_top),
        (box_right, box_bottom),
        (55, 45, 85),
        -1
    )


    cv2.rectangle(
        frame,
        (box_left, box_top),
        (box_right, box_bottom),
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "DETECTED LETTER",
        (box_left + 35, box_top + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2
    )


    # =====================================================
    # ACTUAL LETTER
    # =====================================================

    cv2.putText(
        frame,
        str(current_letter),
        (box_left + 105, box_top + 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        5.0,
        (255, 255, 255),
        9
    )


    # =====================================================
    # NUMBER OF HANDS
    # =====================================================

    if results.multi_hand_landmarks:

        count = len(
            results.multi_hand_landmarks
        )


        cv2.putText(
            frame,
            f"Hands detected: {count}",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # =====================================================
    # VOICE STATUS
    # =====================================================

    cv2.putText(
        frame,
        "Voice: ON",
        (30, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # =====================================================
    # VOICE CONFIRMATION STATUS
    # =====================================================

    stable_count_display = len(stable_history)

    cv2.putText(
        frame,
        f"Voice confirmation: {stable_count_display}/{STABLE_FRAMES}",
        (30, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


    # =====================================================
    # INSTRUCTION
    # =====================================================

    cv2.putText(
        frame,
        "Press Q to exit",
        (width - 220, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "ISL LETTER RECOGNITION",
        frame
    )


    # =====================================================
    # EXIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        break


# =========================================================
# CLEANUP
# =========================================================

voice_running = False

cap.release()

hands.close()

cv2.destroyAllWindows()