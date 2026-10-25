import base64
import io
import os
import pickle
import threading
import uuid

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, render_template_string, request


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

LETTER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "letter_model.pkl"
)

WORD_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "word_model_23.keras"
)

WORD_LABELS_PATH = os.path.join(
    MODEL_DIR,
    "word_labels_23.pkl"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD LETTER MODEL
# ============================================================

letter_model = None
letter_model_error = None

try:
    # IMPORTANT:
    # Your letter_model.pkl must be loaded using joblib.
    letter_model = joblib.load(LETTER_MODEL_PATH)

    print("======================================")
    print("LETTER MODEL LOADED")
    print(type(letter_model))
    print("======================================")

except Exception as e:
    letter_model_error = str(e)

    print("======================================")
    print("LETTER MODEL ERROR")
    print(e)
    print("======================================")


# ============================================================
# LOAD WORD MODEL
# ============================================================

word_model = None
word_model_error = None

try:
    word_model = tf.keras.models.load_model(
        WORD_MODEL_PATH
    )

    print("======================================")
    print("WORD MODEL LOADED")
    print(type(word_model))
    print("======================================")

except Exception as e:
    word_model_error = str(e)

    print("======================================")
    print("WORD MODEL ERROR")
    print(e)
    print("======================================")


# ============================================================
# LOAD WORD LABELS
# ============================================================

word_labels = None

try:
    with open(WORD_LABELS_PATH, "rb") as f:
        word_labels = pickle.load(f)

    print("======================================")
    print("WORD LABELS LOADED")
    print(type(word_labels))
    print("======================================")

except Exception as e:

    if word_model_error:
        word_model_error += "\n"

    word_model_error = (
        word_model_error or ""
    ) + "LABEL ERROR: " + str(e)

    print("======================================")
    print("WORD LABEL ERROR")
    print(e)
    print("======================================")


# ============================================================
# MEDIAPIPE
# ============================================================

from mediapipe.python.solutions import hands as mp_hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# LETTER RECOGNITION
# ============================================================

LETTER_STABLE_REQUIRED = 4

letter_state = {
    "last_letter": None,
    "stable_count": 0
}


def decode_image(data_url):
    """
    Convert browser camera image into OpenCV BGR image.
    """

    if not data_url:
        return None

    try:

        if "," in data_url:
            data_url = data_url.split(",", 1)[1]

        image_bytes = base64.b64decode(data_url)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        return frame

    except Exception:
        return None


def extract_letter_features(frame):
    """
    Letter model preprocessing.

    21 hand landmarks
    X, Y, Z
    Wrist-relative coordinates

    21 × 3 = 63 features
    """

    if frame is None:
        return None, False

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands_detector.process(rgb)

    if not result.multi_hand_landmarks:
        return None, False

    hand = result.multi_hand_landmarks[0]

    wrist = hand.landmark[0]

    features = []

    for landmark in hand.landmark:

        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])

    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features, True


def predict_letter(frame):

    if letter_model is None:

        return {
            "success": False,
            "hand_detected": False,
            "error":
                letter_model_error
                or "Letter model is not loaded."
        }

    features, hand_detected = extract_letter_features(
        frame
    )

    # --------------------------------------------------------
    # NO HAND
    # --------------------------------------------------------

    if not hand_detected:

        letter_state["last_letter"] = None
        letter_state["stable_count"] = 0

        return {
            "success": True,
            "hand_detected": False,
            "letter": "",
            "confidence": 0.0,
            "stable": False
        }

    try:

        x = features.reshape(
            1,
            -1
        )

        prediction = letter_model.predict(x)[0]

        confidence = 0.0

        if hasattr(
            letter_model,
            "predict_proba"
        ):

            probabilities = (
                letter_model.predict_proba(x)[0]
            )

            confidence = float(
                np.max(probabilities)
            )

        if isinstance(
            prediction,
            np.generic
        ):
            prediction = prediction.item()

        letter = str(
            prediction
        ).upper().strip()

        # ----------------------------------------------------
        # STABILITY
        # ----------------------------------------------------

        if confidence >= 0.50:

            if (
                letter
                ==
                letter_state["last_letter"]
            ):

                letter_state[
                    "stable_count"
                ] += 1

            else:

                letter_state[
                    "last_letter"
                ] = letter

                letter_state[
                    "stable_count"
                ] = 1

        else:

            letter_state[
                "stable_count"
            ] = 0

        stable = (
            letter_state["stable_count"]
            >= LETTER_STABLE_REQUIRED
        )

        return {

            "success": True,

            "hand_detected": True,

            "letter": letter,

            "confidence": confidence,

            "stable": stable
        }

    except Exception as e:

        return {

            "success": False,

            "hand_detected": True,

            "error": str(e)
        }


# ============================================================
# WORD RECOGNITION
# ============================================================

MAX_FRAMES = 16

FEATURES_PER_FRAME = 162

CONFIDENCE_THRESHOLD = 0.55

STABLE_COUNT_REQUIRED = 5


# ============================================================
# EXACT ANGLE TRIPLETS USED BY LAPTOP MODEL
# ============================================================

ANGLE_TRIPLETS = [

    (1, 2, 3),
    (2, 3, 4),

    (5, 6, 7),
    (6, 7, 8),

    (9, 10, 11),
    (10, 11, 12),

    (13, 14, 15),
    (14, 15, 16),

    (17, 18, 19),
    (18, 19, 20),

    (0, 5, 9),
    (0, 9, 13),
    (0, 13, 17),

    (5, 9, 13),
    (9, 13, 17)
]


def calculate_angle(a, b, c):

    ba = np.array(
        [
            a.x - b.x,
            a.y - b.y,
            a.z - b.z
        ],
        dtype=np.float32
    )

    bc = np.array(
        [
            c.x - b.x,
            c.y - b.y,
            c.z - b.z
        ],
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = (
        np.dot(ba, bc)
        /
        denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = np.arccos(
        cosine
    ) / np.pi

    return float(angle)


def extract_hand_features(
    hand_landmarks
):
    """
    Exact 79 features.

    63 normalized coordinates
    +
    15 angles
    +
    1 hand size

    Total = 79
    """

    landmarks = (
        hand_landmarks.landmark
    )

    wrist = landmarks[0]

    # --------------------------------------------------------
    # 63 COORDINATE FEATURES
    # --------------------------------------------------------

    coordinates = []

    for landmark in landmarks:

        coordinates.extend(
            [
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ]
        )

    # --------------------------------------------------------
    # HAND SIZE
    #
    # IMPORTANT:
    # Wrist landmark 0
    # Middle MCP landmark 9
    # --------------------------------------------------------

    dx = (
        landmarks[9].x
        -
        landmarks[0].x
    )

    dy = (
        landmarks[9].y
        -
        landmarks[0].y
    )

    dz = (
        landmarks[9].z
        -
        landmarks[0].z
    )

    hand_size = float(
        np.sqrt(
            dx * dx
            +
            dy * dy
            +
            dz * dz
        )
    )

    if hand_size < 1e-8:
        hand_size = 1e-8

    coordinates = [
        value / hand_size
        for value in coordinates
    ]

    # --------------------------------------------------------
    # 15 ANGLES
    # --------------------------------------------------------

    angles = []

    for a, b, c in ANGLE_TRIPLETS:

        angle = calculate_angle(
            landmarks[a],
            landmarks[b],
            landmarks[c]
        )

        angles.append(angle)

    # --------------------------------------------------------
    # FINAL 79 FEATURES
    # --------------------------------------------------------

    features = (
        coordinates
        +
        angles
        +
        [hand_size]
    )

    return np.asarray(
        features,
        dtype=np.float32
    )


def extract_word_features(frame):

    if frame is None:
        return None, False

    # IMPORTANT:
    # Do NOT flip the image before MediaPipe.
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands_detector.process(
        rgb
    )

    if not result.multi_hand_landmarks:
        return None, False

    left_hand = None
    right_hand = None

    # --------------------------------------------------------
    # USE MEDIAPIPE HANDEDNESS
    # --------------------------------------------------------

    if result.multi_handedness:

        for i, handedness in enumerate(
            result.multi_handedness
        ):

            label = (
                handedness
                .classification[0]
                .label
                .lower()
            )

            if label == "left":

                left_hand = (
                    result.multi_hand_landmarks[i]
                )

            elif label == "right":

                right_hand = (
                    result.multi_hand_landmarks[i]
                )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if (
        left_hand is None
        and
        right_hand is None
    ):

        if len(
            result.multi_hand_landmarks
        ) == 1:

            left_hand = (
                result.multi_hand_landmarks[0]
            )

        else:

            left_hand = (
                result.multi_hand_landmarks[0]
            )

            right_hand = (
                result.multi_hand_landmarks[1]
            )

    # --------------------------------------------------------
    # 79 FEATURES PER HAND
    # --------------------------------------------------------

    zero_features = np.zeros(
        79,
        dtype=np.float32
    )

    if left_hand is not None:

        left_features = (
            extract_hand_features(
                left_hand
            )
        )

    else:

        left_features = zero_features

    if right_hand is not None:

        right_features = (
            extract_hand_features(
                right_hand
            )
        )

    else:

        right_features = zero_features

    # --------------------------------------------------------
    # 4 TWO-HAND RELATIONSHIP FEATURES
    # --------------------------------------------------------

    relationship_features = np.zeros(
        4,
        dtype=np.float32
    )

    if (
        left_hand is not None
        and
        right_hand is not None
    ):

        left_wrist = (
            left_hand.landmark[0]
        )

        right_wrist = (
            right_hand.landmark[0]
        )

        dx = (
            right_wrist.x
            -
            left_wrist.x
        )

        dy = (
            right_wrist.y
            -
            left_wrist.y
        )

        dz = (
            right_wrist.z
            -
            left_wrist.z
        )

        distance = float(
            np.sqrt(
                dx * dx
                +
                dy * dy
                +
                dz * dz
            )
        )

        relationship_features = np.asarray(
            [
                dx,
                dy,
                dz,
                distance
            ],
            dtype=np.float32
        )

    # --------------------------------------------------------
    # TOTAL
    #
    # Left = 79
    # Right = 79
    # Relationship = 4
    #
    # TOTAL = 162
    # --------------------------------------------------------

    features = np.concatenate(
        [
            left_features,
            right_features,
            relationship_features
        ]
    )

    return (
        features.astype(np.float32),
        True
    )


# ============================================================
# WORD SESSION
# ============================================================

word_sessions = {}

word_sessions_lock = threading.Lock()


def get_word_session(
    session_id
):

    with word_sessions_lock:

        if session_id not in word_sessions:

            word_sessions[
                session_id
            ] = {

                "sequence": [],

                "last_prediction": None,

                "stable_count": 0,

                "current_word": ""
            }

        return word_sessions[
            session_id
        ]


def reset_word_session(
    session_id
):

    with word_sessions_lock:

        word_sessions[
            session_id
        ] = {

            "sequence": [],

            "last_prediction": None,

            "stable_count": 0,

            "current_word": ""
        }


def decode_word_label(
    predicted_index
):

    try:

        # Your labels are a LabelEncoder.
        if hasattr(
            word_labels,
            "inverse_transform"
        ):

            return str(
                word_labels.inverse_transform(
                    [predicted_index]
                )[0]
            )

        if isinstance(
            word_labels,
            dict
        ):

            return str(
                word_labels[
                    predicted_index
                ]
            )

        return str(
            word_labels[
                predicted_index
            ]
        )

    except Exception:

        return str(
            predicted_index
        )


def predict_word(
    frame,
    session_id
):

    if (
        word_model is None
        or
        word_labels is None
    ):

        return {

            "success": False,

            "hand_detected": False,

            "error":
                word_model_error
                or
                "Word model is not loaded.",

            "session_id": session_id
        }

    features, hand_detected = (
        extract_word_features(
            frame
        )
    )

    state = get_word_session(
        session_id
    )

    # --------------------------------------------------------
    # NO HAND
    #
    # Clear sequence/history/current word.
    # This matches the working laptop behavior.
    # --------------------------------------------------------

    if not hand_detected:

        state["sequence"] = []

        state["last_prediction"] = None

        state["stable_count"] = 0

        state["current_word"] = ""

        return {

            "success": True,

            "session_id": session_id,

            "hand_detected": False,

            "word": "",

            "confidence": 0.0,

            "frames": 0,

            "stable": False
        }

    # --------------------------------------------------------
    # ADD CURRENT FRAME
    # --------------------------------------------------------

    state["sequence"].append(
        features
    )

    if len(
        state["sequence"]
    ) > MAX_FRAMES:

        state["sequence"].pop(0)

    frames_count = len(
        state["sequence"]
    )

    # --------------------------------------------------------
    # WAIT UNTIL 16 FRAMES
    # --------------------------------------------------------

    if frames_count < MAX_FRAMES:

        return {

            "success": True,

            "session_id": session_id,

            "hand_detected": True,

            "word":
                state["current_word"],

            "confidence": 0.0,

            "frames": frames_count,

            "stable": False
        }

    try:

        sequence = np.asarray(
            state["sequence"],
            dtype=np.float32
        )

        sequence = sequence.reshape(
            1,
            MAX_FRAMES,
            FEATURES_PER_FRAME
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = word_model.predict(
            sequence,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[
                predicted_index
            ]
        )

        word = decode_word_label(
            predicted_index
        ).lower()

        # ----------------------------------------------------
        # CONFIDENCE + STABILITY
        # ----------------------------------------------------

        if (
            confidence
            >=
            CONFIDENCE_THRESHOLD
        ):

            if (
                word
                ==
                state["last_prediction"]
            ):

                state[
                    "stable_count"
                ] += 1

            else:

                state[
                    "last_prediction"
                ] = word

                state[
                    "stable_count"
                ] = 1

            state[
                "current_word"
            ] = word

        else:

            state[
                "stable_count"
            ] = 0

        stable = (
            state["stable_count"]
            >=
            STABLE_COUNT_REQUIRED
        )

        return {

            "success": True,

            "session_id": session_id,

            "hand_detected": True,

            "word":
                state["current_word"],

            "confidence":
                confidence,

            "frames":
                frames_count,

            "stable":
                stable
        }

    except Exception as e:

        return {

            "success": False,

            "session_id": session_id,

            "hand_detected": True,

            "error": str(e)
        }


# ============================================================
# FULL SENTENCE OPTIONS
#
# Same 23 words from your Sentence Recognition program.
# ============================================================

SENTENCE_OPTIONS = {

    "hello": [
        "Hello.",
        "Hello, how are you?",
        "Hello my friend.",
        "Hello everyone.",
        "Hello, nice to meet you."
    ],

    "thank_you": [
        "Thank you.",
        "Thank you very much.",
        "Thank you for your help.",
        "Thank you for coming.",
        "Thank you so much."
    ],

    "go": [
        "I want to go.",
        "Please go there.",
        "Let's go.",
        "I will go.",
        "We should go."
    ],

    "help": [
        "Please help me.",
        "I need help.",
        "Can you help me?",
        "Please help.",
        "I need your help."
    ],

    "yes": [
        "Yes.",
        "Yes, I understand.",
        "Yes, please.",
        "Yes, I will.",
        "Yes, that's correct."
    ],

    "no": [
        "No.",
        "No, thank you.",
        "No, I don't want.",
        "No, that's not correct.",
        "No problem."
    ],

    "please": [
        "Please help me.",
        "Please sit down.",
        "Please go there.",
        "Please give me water.",
        "Please stop."
    ],

    "friend": [
        "He is my friend.",
        "She is my friend.",
        "My friend is here.",
        "I want to meet my friend.",
        "My friend will come."
    ],

    "market": [
        "I want to go to the market.",
        "I am going to the market.",
        "Let's go to the market.",
        "We are at the market.",
        "I need to buy something from the market."
    ],

    "mother": [
        "My mother is at home.",
        "I want to see my mother.",
        "My mother is going to the market.",
        "Please call my mother.",
        "My mother needs help."
    ],

    "okay": [
        "Okay.",
        "Okay, thank you.",
        "Okay, I will go.",
        "Okay, I understand.",
        "Okay, no problem."
    ],

    "drink": [
        "I want to drink water.",
        "Please give me something to drink.",
        "I need a drink.",
        "I want to drink.",
        "Let's drink water."
    ],

    "father": [
        "My father is at home.",
        "I want to see my father.",
        "My father is going to the market.",
        "Please call my father.",
        "My father needs help."
    ],

    "you": [
        "How are you?",
        "What are you doing?",
        "You are my friend.",
        "Where are you going?",
        "Can you help me?"
    ],

    "today": [
        "Today I am going to school.",
        "Today I will go to the market.",
        "Today I need help.",
        "Today I want to drink water.",
        "Today I am happy."
    ],

    "food": [
        "I want food.",
        "Please give me food.",
        "I am hungry and want food.",
        "The food is ready.",
        "I want to eat food."
    ],

    "teacher": [
        "My teacher is here.",
        "I want to talk to my teacher.",
        "Please call my teacher.",
        "My teacher will help me.",
        "I am going to my teacher."
    ],

    "sorry": [
        "Sorry.",
        "I am sorry.",
        "Sorry, please help me.",
        "I am very sorry.",
        "Sorry for my mistake."
    ],

    "sit": [
        "Please sit down.",
        "I want to sit.",
        "You can sit here.",
        "Please sit here.",
        "Let's sit down."
    ],

    "stop": [
        "Please stop.",
        "Stop there.",
        "Please stop doing that.",
        "Stop the vehicle.",
        "You should stop."
    ],

    "hospital": [
        "I want to go to the hospital.",
        "Please take me to the hospital.",
        "We need to go to the hospital.",
        "The hospital is nearby.",
        "I am going to the hospital."
    ],

    "school": [
        "I am going to school.",
        "I want to go to school.",
        "Today I have school.",
        "My friend is at school.",
        "I am coming from school."
    ],

    "what": [
        "What is this?",
        "What are you doing?",
        "What happened?",
        "What do you want?",
        "What is your name?"
    ]
}


# ============================================================
# TWO-WORD + THREE-WORD CONTEXT OPTIONS
# ============================================================

CONTEXT_OPTIONS = {

    # ----------------------------
    # TWO WORDS
    # ----------------------------

    ("please", "help"): [
        "Please help me.",
        "Please help me with this.",
        "Please help me now.",
        "Please help my friend.",
        "Please help my mother."
    ],

    ("please", "sit"): [
        "Please sit down.",
        "Please sit here.",
        "Please sit near me.",
        "Please sit in the chair.",
        "Please sit and wait."
    ],

    ("please", "go"): [
        "Please go there.",
        "Please go to school.",
        "Please go to the market.",
        "Please go home.",
        "Please go with my friend."
    ],

    ("please", "drink"): [
        "Please drink water.",
        "Please drink some water.",
        "Please drink this.",
        "Please drink slowly.",
        "Please drink and rest."
    ],

    ("please", "stop"): [
        "Please stop.",
        "Please stop there.",
        "Please stop doing that.",
        "Please stop the vehicle.",
        "Please stop and listen."
    ],

    ("hello", "friend"): [
        "Hello my friend.",
        "Hello, my friend.",
        "Hello friend, how are you?",
        "Hello friend, nice to see you.",
        "Hello my good friend."
    ],

    ("hello", "you"): [
        "Hello, how are you?",
        "Hello, what are you doing?",
        "Hello, where are you going?",
        "Hello, nice to meet you.",
        "Hello, can you help me?"
    ],

    ("hello", "mother"): [
        "Hello mother.",
        "Hello, how are you mother?",
        "Hello my mother.",
        "Hello mother, I am here.",
        "Hello mother, please help me."
    ],

    ("hello", "father"): [
        "Hello father.",
        "Hello, how are you father?",
        "Hello my father.",
        "Hello father, I am here.",
        "Hello father, please help me."
    ],

    ("hello", "teacher"): [
        "Hello teacher.",
        "Hello, how are you teacher?",
        "Hello teacher, I am here.",
        "Hello teacher, please help me.",
        "Hello my teacher."
    ],

    ("hello", "today"): [
        "Hello, how are you today?",
        "Hello, I am happy today.",
        "Hello, what are you doing today?",
        "Hello, where are you going today?",
        "Hello, have a nice day today."
    ],

    ("you", "market"): [
        "Are you going to the market?",
        "You are going to the market.",
        "Did you go to the market?",
        "Will you go to the market?",
        "Can you go to the market?"
    ],

    ("you", "school"): [
        "Are you going to school?",
        "You are going to school.",
        "Did you go to school?",
        "Will you go to school?",
        "Can you go to school?"
    ],

    ("you", "hospital"): [
        "Are you going to the hospital?",
        "You are going to the hospital.",
        "Did you go to the hospital?",
        "Will you go to the hospital?",
        "Can you go to the hospital?"
    ],

    ("you", "food"): [
        "Do you want food?",
        "You can have food.",
        "Did you eat food?",
        "Will you eat food?",
        "Can you give me food?"
    ],

    ("you", "drink"): [
        "Do you want a drink?",
        "You should drink water.",
        "Did you drink water?",
        "Will you drink water?",
        "Can you give me a drink?"
    ],

    ("you", "friend"): [
        "You are my friend.",
        "Is he your friend?",
        "Is she your friend?",
        "Your friend is here.",
        "Where is your friend?"
    ],

    ("you", "teacher"): [
        "You are my teacher.",
        "Is your teacher here?",
        "Where is your teacher?",
        "Talk to your teacher.",
        "Your teacher will help you."
    ],

    ("today", "school"): [
        "Today I am going to school.",
        "Today we have school.",
        "Today I will go to school.",
        "Today my friend is at school.",
        "Today I have class at school."
    ],

    ("today", "market"): [
        "Today I am going to the market.",
        "Today we will go to the market.",
        "Today I need to go to the market.",
        "Today I want to visit the market.",
        "Today my mother is going to the market."
    ],

    ("today", "hospital"): [
        "Today I am going to the hospital.",
        "Today we need to go to the hospital.",
        "Today I will visit the hospital.",
        "Today my father is at the hospital.",
        "Today my mother is at the hospital."
    ],

    ("today", "food"): [
        "Today I want food.",
        "Today I will eat food.",
        "Today the food is ready.",
        "Today I want good food.",
        "Today we will have food."
    ],

    ("today", "drink"): [
        "Today I want to drink water.",
        "Today I will drink water.",
        "Today I need a drink.",
        "Today please give me water.",
        "Today we will drink water."
    ],

    ("today", "friend"): [
        "Today I will meet my friend.",
        "Today my friend is coming.",
        "Today I am with my friend.",
        "Today I want to see my friend.",
        "Today my friend will help me."
    ],

    ("today", "teacher"): [
        "Today I will meet my teacher.",
        "Today my teacher is here.",
        "Today I will talk to my teacher.",
        "Today my teacher will help me.",
        "Today I have class with my teacher."
    ],

    ("father", "school"): [
        "My father is going to school.",
        "My father is at school.",
        "My father will come to school.",
        "My father is near the school.",
        "My father wants to go to school."
    ],

    ("father", "market"): [
        "My father is going to the market.",
        "My father is at the market.",
        "My father will go to the market.",
        "My father wants to go to the market.",
        "My father is coming from the market."
    ],

    ("father", "hospital"): [
        "My father is at the hospital.",
        "I need to take my father to the hospital.",
        "My father is going to the hospital.",
        "Please help my father at the hospital.",
        "My father needs to go to the hospital."
    ],

    ("father", "food"): [
        "My father wants food.",
        "My father is eating food.",
        "Please give my father food.",
        "My father needs food.",
        "I will give food to my father."
    ],

    ("father", "drink"): [
        "My father wants a drink.",
        "Please give my father water.",
        "My father wants to drink water.",
        "My father needs a drink.",
        "I will give water to my father."
    ],

    ("mother", "school"): [
        "My mother is going to school.",
        "My mother is at school.",
        "My mother will come to school.",
        "My mother is near the school.",
        "My mother wants to go to school."
    ],

    ("mother", "market"): [
        "My mother is going to the market.",
        "My mother is at the market.",
        "My mother will go to the market.",
        "My mother wants to go to the market.",
        "My mother is coming from the market."
    ],

    ("mother", "hospital"): [
        "My mother is at the hospital.",
        "I need to take my mother to the hospital.",
        "My mother is going to the hospital.",
        "Please help my mother at the hospital.",
        "My mother needs to go to the hospital."
    ],

    ("mother", "food"): [
        "My mother wants food.",
        "My mother is preparing food.",
        "Please give my mother food.",
        "My mother needs food.",
        "I will give food to my mother."
    ],

    ("mother", "drink"): [
        "My mother wants a drink.",
        "Please give my mother water.",
        "My mother wants to drink water.",
        "My mother needs a drink.",
        "I will give water to my mother."
    ],

    ("friend", "school"): [
        "My friend is at school.",
        "My friend is going to school.",
        "I will meet my friend at school.",
        "My friend wants to go to school.",
        "I am going to school with my friend."
    ],

    ("friend", "market"): [
        "My friend is at the market.",
        "My friend is going to the market.",
        "I will go to the market with my friend.",
        "My friend wants to go to the market.",
        "I will meet my friend at the market."
    ],

    ("friend", "hospital"): [
        "My friend is at the hospital.",
        "My friend is going to the hospital.",
        "I will go to the hospital with my friend.",
        "My friend needs help at the hospital.",
        "I will visit my friend at the hospital."
    ],

    ("friend", "food"): [
        "My friend wants food.",
        "I will eat food with my friend.",
        "Please give food to my friend.",
        "My friend is eating food.",
        "I will give food to my friend."
    ],

    ("friend", "drink"): [
        "My friend wants a drink.",
        "I will drink water with my friend.",
        "Please give my friend water.",
        "My friend wants to drink water.",
        "I will give a drink to my friend."
    ],

    ("go", "school"): [
        "I want to go to school.",
        "Let's go to school.",
        "I am going to school.",
        "We should go to school.",
        "Please go to school."
    ],

    ("go", "market"): [
        "I want to go to the market.",
        "Let's go to the market.",
        "I am going to the market.",
        "We should go to the market.",
        "Please go to the market."
    ],

    ("go", "hospital"): [
        "I want to go to the hospital.",
        "Let's go to the hospital.",
        "We should go to the hospital.",
        "Please take me to the hospital.",
        "I am going to the hospital."
    ],

    ("go", "teacher"): [
        "Go to the teacher.",
        "I will go to my teacher.",
        "Let's go to the teacher.",
        "Please go to the teacher.",
        "We should go to the teacher."
    ],

    ("go", "friend"): [
        "I will go to my friend.",
        "Let's go to my friend.",
        "I want to go with my friend.",
        "Please go with my friend.",
        "We should go to our friend."
    ],

    ("help", "you"): [
        "I will help you.",
        "Can I help you?",
        "I want to help you.",
        "Please let me help you.",
        "How can I help you?"
    ],

    ("help", "today"): [
        "I need help today.",
        "Please help me today.",
        "Can you help me today?",
        "I will help you today.",
        "We need help today."
    ],

    ("help", "school"): [
        "Please help me at school.",
        "I need help at school.",
        "Can you help me at school?",
        "My teacher will help me at school.",
        "I need your help at school."
    ],

    ("help", "hospital"): [
        "Please help me at the hospital.",
        "I need help at the hospital.",
        "Can you help me go to the hospital?",
        "Please help my friend at the hospital.",
        "Please help my mother at the hospital."
    ],

    ("go", "today"): [
        "I will go today.",
        "Let's go today.",
        "I want to go today.",
        "We should go today.",
        "Please go today."
    ],

    ("food", "today"): [
        "I want food today.",
        "I will eat food today.",
        "Please give me food today.",
        "We will have food today.",
        "Today I want good food."
    ],

    ("food", "school"): [
        "I want food at school.",
        "I will eat food at school.",
        "Please give me food at school.",
        "My friend has food at school.",
        "We will eat food at school."
    ],

    ("food", "market"): [
        "I want to buy food at the market.",
        "Let's buy food at the market.",
        "I am going to the market for food.",
        "Please buy food from the market.",
        "We need food from the market."
    ],

    ("drink", "today"): [
        "I want to drink today.",
        "I will drink water today.",
        "Please give me a drink today.",
        "We will drink water today.",
        "Today I need a drink."
    ],

    ("drink", "school"): [
        "I want to drink water at school.",
        "Please give me water at school.",
        "I will drink water at school.",
        "We need water at school.",
        "Please give my friend a drink at school."
    ],

    ("drink", "hospital"): [
        "I want to drink water at the hospital.",
        "Please give me water at the hospital.",
        "I need a drink at the hospital.",
        "Please give my mother water at the hospital.",
        "Please give my father water at the hospital."
    ],

    ("teacher", "school"): [
        "My teacher is at school.",
        "I will meet my teacher at school.",
        "My teacher teaches at school.",
        "I am going to school to meet my teacher.",
        "Please talk to my teacher at school."
    ],

    ("teacher", "help"): [
        "My teacher will help me.",
        "Please ask the teacher for help.",
        "I need help from my teacher.",
        "My teacher can help you.",
        "Please help my teacher."
    ],

    ("sorry", "please"): [
        "Sorry, please help me.",
        "Sorry, please forgive me.",
        "I am sorry, please help.",
        "Sorry, please don't be angry.",
        "I am very sorry, please."
    ],

    ("sorry", "thank_you"): [
        "Sorry, and thank you.",
        "I am sorry, thank you.",
        "Sorry for the mistake, thank you.",
        "I am sorry and thank you for understanding.",
        "Sorry, thank you for helping me."
    ],

    ("sit", "school"): [
        "Please sit at school.",
        "I will sit at school.",
        "Please sit in the school.",
        "We can sit at school.",
        "You can sit at school."
    ],

    ("sit", "friend"): [
        "I will sit with my friend.",
        "Please sit with my friend.",
        "Let's sit with our friend.",
        "My friend and I will sit.",
        "You can sit with my friend."
    ],

    ("stop", "go"): [
        "Stop and then go.",
        "Please stop before you go.",
        "Stop, don't go.",
        "You should stop and go later.",
        "Please stop and wait before going."
    ],

    ("stop", "today"): [
        "Please stop today.",
        "I want you to stop today.",
        "Please stop doing that today.",
        "You should stop today.",
        "Let's stop for today."
    ],

    ("okay", "thank_you"): [
        "Okay, thank you.",
        "Okay, thank you very much.",
        "Okay, thank you for your help.",
        "Okay, thank you so much.",
        "Okay, thank you for coming."
    ],

    ("okay", "help"): [
        "Okay, please help me.",
        "Okay, I need help.",
        "Okay, can you help me?",
        "Okay, please help.",
        "Okay, I need your help."
    ],

    ("okay", "go"): [
        "Okay, let's go.",
        "Okay, I will go.",
        "Okay, we can go.",
        "Okay, please go.",
        "Okay, let's go there."
    ],

    ("no", "thank_you"): [
        "No, thank you.",
        "No, thank you very much.",
        "No, thank you for asking.",
        "No, but thank you.",
        "No, thank you for your help."
    ],

    ("yes", "thank_you"): [
        "Yes, thank you.",
        "Yes, thank you very much.",
        "Yes, thank you for helping me.",
        "Yes, thank you so much.",
        "Yes, thank you for asking."
    ],

    # ----------------------------
    # THREE WORDS
    # ----------------------------

    ("go", "school", "today"): [
        "I am going to school today.",
        "Let's go to school today.",
        "I will go to school today.",
        "We should go to school today.",
        "Please go to school today."
    ],

    ("go", "market", "today"): [
        "I am going to the market today.",
        "Let's go to the market today.",
        "I will go to the market today.",
        "We should go to the market today.",
        "Please go to the market today."
    ],

    ("go", "hospital", "today"): [
        "I am going to the hospital today.",
        "Let's go to the hospital today.",
        "I will go to the hospital today.",
        "We should go to the hospital today.",
        "Please take me to the hospital today."
    ],

    ("go", "friend", "today"): [
        "I will go to my friend today.",
        "I will meet my friend today.",
        "Let's go to my friend today.",
        "I want to see my friend today.",
        "I am going with my friend today."
    ],

    ("hello", "friend", "today"): [
        "Hello my friend, how are you today?",
        "Hello friend, I am happy to see you today.",
        "Hello my friend, have a nice day today.",
        "Hello friend, what are you doing today?",
        "Hello my good friend today."
    ],

    ("hello", "you", "today"): [
        "Hello, how are you today?",
        "Hello, what are you doing today?",
        "Hello, where are you going today?",
        "Hello, are you okay today?",
        "Hello, can you help me today?"
    ],

    ("please", "help", "today"): [
        "Please help me today.",
        "Please help me with this today.",
        "Please help my friend today.",
        "Please help my mother today.",
        "Please help me at school today."
    ],

    ("please", "sit", "today"): [
        "Please sit here today.",
        "Please sit down today.",
        "Please sit with me today.",
        "Please sit and wait today.",
        "Please sit at school today."
    ],

    ("please", "go", "school"): [
        "Please go to school.",
        "Please go to school now.",
        "Please go to school with me.",
        "Please go to school today.",
        "Please go to the school."
    ],

    ("please", "go", "market"): [
        "Please go to the market.",
        "Please go to the market now.",
        "Please go to the market with me.",
        "Please go to the market today.",
        "Please go to the market and buy food."
    ],

    ("please", "go", "hospital"): [
        "Please go to the hospital.",
        "Please take me to the hospital.",
        "Please go to the hospital now.",
        "Please take my mother to the hospital.",
        "Please take my father to the hospital."
    ],

    ("you", "school", "today"): [
        "Are you going to school today?",
        "You are going to school today.",
        "Will you go to school today?",
        "Did you go to school today?",
        "Can you go to school today?"
    ],

    ("you", "market", "today"): [
        "Are you going to the market today?",
        "You are going to the market today.",
        "Will you go to the market today?",
        "Did you go to the market today?",
        "Can you go to the market today?"
    ],

    ("you", "hospital", "today"): [
        "Are you going to the hospital today?",
        "You are going to the hospital today.",
        "Will you go to the hospital today?",
        "Did you go to the hospital today?",
        "Can you go to the hospital today?"
    ],

    ("you", "food", "today"): [
        "Do you want food today?",
        "You will eat food today.",
        "Did you eat food today?",
        "Will you eat food today?",
        "Can I give you food today?"
    ],

    ("you", "drink", "today"): [
        "Do you want to drink today?",
        "You should drink water today.",
        "Did you drink water today?",
        "Will you drink water today?",
        "Can I give you a drink today?"
    ],

    ("father", "school", "today"): [
        "My father is going to school today.",
        "My father is at school today.",
        "My father will go to school today.",
        "I will go to school with my father today.",
        "My father wants to go to school today."
    ],

    ("father", "market", "today"): [
        "My father is going to the market today.",
        "My father is at the market today.",
        "My father will go to the market today.",
        "I will go to the market with my father today.",
        "My father wants to go to the market today."
    ],

    ("father", "hospital", "today"): [
        "My father is at the hospital today.",
        "My father is going to the hospital today.",
        "I will take my father to the hospital today.",
        "My father needs help at the hospital today.",
        "We need to go to the hospital for my father today."
    ],

    ("mother", "school", "today"): [
        "My mother is going to school today.",
        "My mother is at school today.",
        "My mother will go to school today.",
        "I will go to school with my mother today.",
        "My mother wants to go to school today."
    ],

    ("mother", "market", "today"): [
        "My mother is going to the market today.",
        "My mother is at the market today.",
        "My mother will go to the market today.",
        "I will go to the market with my mother today.",
        "My mother wants to go to the market today."
    ],

    ("mother", "hospital", "today"): [
        "My mother is at the hospital today.",
        "My mother is going to the hospital today.",
        "I will take my mother to the hospital today.",
        "My mother needs help at the hospital today.",
        "We need to go to the hospital for my mother today."
    ],

    ("friend", "school", "today"): [
        "My friend is at school today.",
        "My friend is going to school today.",
        "I will meet my friend at school today.",
        "I am going to school with my friend today.",
        "My friend wants to go to school today."
    ],

    ("friend", "market", "today"): [
        "My friend is at the market today.",
        "My friend is going to the market today.",
        "I will go to the market with my friend today.",
        "I will meet my friend at the market today.",
        "My friend wants to go to the market today."
    ],

    ("friend", "hospital", "today"): [
        "My friend is at the hospital today.",
        "My friend is going to the hospital today.",
        "I will visit my friend at the hospital today.",
        "I will go to the hospital with my friend today.",
        "My friend needs help at the hospital today."
    ],

    ("friend", "food", "today"): [
        "My friend wants food today.",
        "I will eat food with my friend today.",
        "Please give food to my friend today.",
        "My friend is eating food today.",
        "I will give food to my friend today."
    ],

    ("friend", "drink", "today"): [
        "My friend wants a drink today.",
        "I will drink water with my friend today.",
        "Please give my friend water today.",
        "My friend wants to drink water today.",
        "I will give a drink to my friend today."
    ],

    ("teacher", "school", "today"): [
        "My teacher is at school today.",
        "I will meet my teacher at school today.",
        "My teacher will help me at school today.",
        "I am going to school to meet my teacher today.",
        "I will talk to my teacher at school today."
    ],

    ("help", "school", "today"): [
        "I need help at school today.",
        "Please help me at school today.",
        "My teacher will help me at school today.",
        "Can you help me at school today?",
        "I need your help at school today."
    ],

    ("help", "hospital", "today"): [
        "I need help at the hospital today.",
        "Please help me at the hospital today.",
        "Can you help me at the hospital today?",
        "Please help my mother at the hospital today.",
        "Please help my father at the hospital today."
    ],

    ("food", "market", "today"): [
        "I will buy food at the market today.",
        "I am going to the market for food today.",
        "Let's buy food at the market today.",
        "Please buy food from the market today.",
        "We need food from the market today."
    ],

    ("drink", "school", "today"): [
        "I want to drink water at school today.",
        "Please give me water at school today.",
        "I will drink water at school today.",
        "We need water at school today.",
        "Please give my friend a drink at school today."
    ],

    ("okay", "thank_you", "you"): [
        "Okay, thank you, you are very kind.",
        "Okay, thank you for helping me.",
        "Okay, thank you, I appreciate you.",
        "Okay, thank you for your help.",
        "Okay, thank you, I am grateful to you."
    ],

    ("yes", "thank_you", "you"): [
        "Yes, thank you, you are very kind.",
        "Yes, thank you for helping me.",
        "Yes, thank you, I appreciate you.",
        "Yes, thank you for your help.",
        "Yes, thank you, I am grateful to you."
    ],

    ("no", "thank_you", "you"): [
        "No, thank you, you are very kind.",
        "No, thank you for helping me.",
        "No, thank you, I appreciate you.",
        "No, thank you for your help.",
        "No, thank you, I am grateful to you."
    ],

    ("sorry", "thank_you", "you"): [
        "I am sorry, and thank you for helping me.",
        "Sorry, thank you for your help.",
        "I am sorry, thank you for understanding.",
        "Sorry, and thank you for being kind.",
        "I am sorry, thank you very much."
    ]
}


# ============================================================
# SENTENCE OPTION LOGIC
# ============================================================

def get_sentence_options(words):

    # Clean words
    words = [
        str(word).lower().strip()
        for word in words
        if str(word).strip()
    ]

    # Same maximum as desktop program
    words = words[-6:]

    # --------------------------------------------------------
    # PRIORITY 1: LAST THREE WORDS
    # --------------------------------------------------------

    if len(words) >= 3:

        last_three = tuple(
            words[-3:]
        )

        if last_three in CONTEXT_OPTIONS:

            return (
                CONTEXT_OPTIONS[
                    last_three
                ],
                list(last_three)
            )

    # --------------------------------------------------------
    # PRIORITY 2: LAST TWO WORDS
    # --------------------------------------------------------

    if len(words) >= 2:

        last_two = tuple(
            words[-2:]
        )

        if last_two in CONTEXT_OPTIONS:

            return (
                CONTEXT_OPTIONS[
                    last_two
                ],
                list(last_two)
            )

    # --------------------------------------------------------
    # PRIORITY 3: LAST SINGLE WORD
    # --------------------------------------------------------

    if len(words) >= 1:

        last_word = words[-1]

        if last_word in SENTENCE_OPTIONS:

            return (
                SENTENCE_OPTIONS[
                    last_word
                ],
                [last_word]
            )

        return (
            [
                f"I recognize the word {last_word}."
            ],
            [last_word]
        )

    return [], []


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/letter-recognition")
def letter_recognition():

    return render_template(
        "letter_recognition.html"
    )


@app.route("/letter-learning")
def letter_learning():

    return render_template(
        "letter_learning.html"
    )


@app.route("/word-learning")
def word_learning():

    return render_template(
        "word_learning.html"
    )


@app.route("/word-recognition")
def word_recognition():

    return render_template(
        "word_recognition.html"
    )


@app.route("/sentence-recognition")
def sentence_recognition():

    return render_template(
        "sentence_recognition.html"
    )

@app.route("/speech-to-text")
def speech_to_text():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ISL Vision - Speech to Text</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                text-align: center;
                padding: 25px;
            }
            .card {
                max-width: 650px;
                margin: 30px auto;
                padding: 25px;
                background: white;
                border-radius: 15px;
            }
            textarea {
                width: 95%;
                height: 180px;
                padding: 10px;
                font-size: 17px;
                box-sizing: border-box;
                margin: 15px 0;
            }
            button {
                padding: 12px 18px;
                margin: 5px;
                cursor: pointer;
                border: none;
                border-radius: 7px;
                background: #2457d6;
                color: white;
            }
            #status { color: #444; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🎤 Speech-to-Text</h1>
            <p>Speak naturally. Your speech will appear as text.</p>

            <button id="startBtn">Start Speaking</button>
            <button id="stopBtn" disabled>Stop</button>

            <p id="status">Status: Ready</p>

            <textarea id="result"
                placeholder="Your speech will appear here..."></textarea>
            <br>

            <button id="copyBtn">Copy Text</button>
            <button id="clearBtn">Clear</button>

            <p><a href="/">Back to ISL Vision Home</a></p>
        </div>

        <script>
            const SpeechRecognition =
                window.SpeechRecognition ||
                window.webkitSpeechRecognition;

            let recognition = null;
            let finalText = "";
            let isListening = false;

            const resultBox = document.getElementById("result");
            const status = document.getElementById("status");
            const startBtn = document.getElementById("startBtn");
            const stopBtn = document.getElementById("stopBtn");

            startBtn.onclick = function () {
                if (!SpeechRecognition) {
                    status.textContent =
                        "Speech recognition is not supported. Try Chrome.";
                    return;
                }

                if (isListening) return;

                recognition = new SpeechRecognition();
                recognition.lang = "en-IN";
                recognition.continuous = true;
                recognition.interimResults = true;

                recognition.onstart = function () {
                    isListening = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    status.textContent = "Listening... Speak now.";
                };

                recognition.onresult = function (event) {
                    let interimText = "";

                    for (let i = event.resultIndex;
                         i < event.results.length; i++) {
                        const transcript =
                            event.results[i][0].transcript;

                        if (event.results[i].isFinal) {
                            finalText += transcript + " ";
                        } else {
                            interimText += transcript;
                        }
                    }

                    resultBox.value = finalText + interimText;
                };

                recognition.onerror = function (event) {
                    status.textContent = "Recognition error: " + event.error;
                };

                recognition.onend = function () {
                    isListening = false;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    status.textContent = "Status: Stopped";
                };

                try {
                    recognition.start();
                } catch (error) {
                    status.textContent =
                        "Could not start. Please try again.";
                }
            };

            stopBtn.onclick = function () {
                if (recognition && isListening) {
                    recognition.stop();
                }
            };

            document.getElementById("clearBtn").onclick = function () {
                finalText = "";
                resultBox.value = "";
                status.textContent = "Status: Text cleared";
            };

            document.getElementById("copyBtn").onclick = async function () {
                if (!resultBox.value.trim()) {
                    status.textContent = "There is no text to copy.";
                    return;
                }

                try {
                    await navigator.clipboard.writeText(resultBox.value);
                    status.textContent = "Text copied!";
                } catch (error) {
                    status.textContent =
                        "Copy unavailable. Select the text and copy it manually.";
                }
            };
        </script>
    </body>
    </html>
    """)


# ============================================================
# LETTER API
# ============================================================

@app.route(
    "/api/letter_predict",
    methods=["POST"]
)
def api_letter_predict():

    data = request.get_json(
        silent=True
    ) or {}

    image_data = data.get(
        "image"
    )

    frame = decode_image(
        image_data
    )

    if frame is None:

        return jsonify({

            "success": False,

            "error":
                "Invalid camera image."
        })

    result = predict_letter(
        frame
    )

    return jsonify(
        result
    )


# ============================================================
# WORD API
# ============================================================

@app.route(
    "/api/word_predict",
    methods=["POST"]
)
def api_word_predict():

    data = request.get_json(
        silent=True
    ) or {}

    image_data = data.get(
        "image"
    )

    session_id = (
        data.get("session_id")
        or
        str(uuid.uuid4())
    )

    frame = decode_image(
        image_data
    )

    if frame is None:

        return jsonify({

            "success": False,

            "session_id":
                session_id,

            "error":
                "Invalid camera image."
        })

    result = predict_word(
        frame,
        session_id
    )

    return jsonify(
        result
    )


# ============================================================
# WORD RESET API
# ============================================================


@app.route(
    "/api/word_reset",
    methods=["POST"]
)
def api_word_reset():

    data = request.get_json(silent=True) or {}

    # Always create a valid session ID
    session_id = (
        data.get("session_id")
        or str(uuid.uuid4())
    )

    # Reset the sequence for this session
    reset_word_session(session_id)

    return jsonify({
        "success": True,
        "session_id": session_id
    })


# ============================================================
# SENTENCE OPTIONS API
# ============================================================

@app.route(
    "/api/sentence_options",
    methods=["POST"]
)
def api_sentence_options():

    data = request.get_json(
        silent=True
    ) or {}

    words = data.get(
        "words",
        []
    )

    options, matched_words = (
        get_sentence_options(
            words
        )
    )

    cleaned_words = [
        str(word).lower().strip()
        for word in words
        if str(word).strip()
    ]

    cleaned_words = cleaned_words[-6:]

    return jsonify({

        "success": True,

        "words":
            cleaned_words,

        "matched_words":
            matched_words,

        "options":
            options
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "ok",

        "letter_model_loaded":
            letter_model is not None,

        "word_model_loaded":
            word_model is not None,

        "word_labels_loaded":
            word_labels is not None,

        "letter_model_error":
            letter_model_error,

        "word_model_error":
            word_model_error
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("==========================================")
    print("       ISL VISION MOBILE/PUBLIC APP")
    print("==========================================")

    print(
        "Letter model:",
        "LOADED"
        if letter_model is not None
        else "ERROR"
    )

    print(
        "Word model:",
        "LOADED"
        if word_model is not None
        else "ERROR"
    )

    print(
        "Word labels:",
        "LOADED"
        if word_labels is not None
        else "ERROR"
    )

    print("==========================================")
    print("Server starting...")
    print("==========================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )
