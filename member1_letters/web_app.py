from flask import Flask, render_template_string, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import base64
import os

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_FILE = BASE_DIR / "letter_model.pkl"

# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)

# ============================================================
# MEDIAPIPE
# Two real hands can be detected.
# Each hand is processed separately by the existing
# one-hand Random Forest model.
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

# ============================================================
# CSS
# ============================================================

STYLE = """
<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f7fb;
    color: #111827;
}

.header {
    background: #111827;
    color: white;
    padding: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.container {
    max-width: 1000px;
    margin: 25px auto;
    padding: 20px;
    text-align: center;
}

h1 {
    font-size: 34px;
    margin-bottom: 10px;
}

.subtitle {
    color: #6b7280;
    margin-bottom: 30px;
}

.button {
    border: none;
    padding: 13px 25px;
    margin: 8px;
    border-radius: 10px;
    background: #111827;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

.button:hover {
    background: #374151;
}

.back-button {
    background: #4b5563;
}

.exit-button {
    background: #991b1b;
}

/* ==========================================================
   CAMERA
   ========================================================== */

.camera-area {
    position: relative;
    width: 760px;
    max-width: 95vw;
    margin: 20px auto;
    background: black;
    border-radius: 12px;
    overflow: hidden;
}

#video {
    width: 100%;
    display: block;
    transform: scaleX(-1);
}

#landmarkCanvas {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    transform: scaleX(-1);
}

.prediction {
    font-size: 60px;
    font-weight: bold;
    margin: 12px;
    min-height: 70px;
}

.detected {
    font-size: 17px;
    color: #4b5563;
    margin-bottom: 10px;
}

/* ==========================================================
   LEARNING
   ========================================================== */

.learning-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    max-width: 850px;
    margin: 25px auto;
}

.letter-button {
    padding: 17px 10px;
    font-size: 24px;
    font-weight: bold;
    border: 2px solid #d1d5db;
    border-radius: 12px;
    background: white;
    cursor: pointer;
}

.letter-button:hover {
    background: #e5e7eb;
}

.learning-image {
    margin: 25px auto;
    background: white;
    padding: 15px;
    border-radius: 12px;
    max-width: 450px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
}

.learning-image img {
    width: 100%;
    height: 350px;
    object-fit: contain;
    border-radius: 8px;
}

.learning-title {
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 15px;
}

.message {
    background: white;
    padding: 25px;
    margin: 25px auto;
    max-width: 700px;
    border-radius: 12px;
}

@media (max-width: 700px) {
    .learning-grid {
        grid-template-columns: repeat(4, 1fr);
    }

    .camera-area {
        width: 95vw;
    }

    .prediction {
        font-size: 48px;
    }
}
</style>
"""

# ============================================================
# HOME PAGE
# ============================================================

HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ISL Vision</title>
    {{ style|safe }}
</head>
<body>

<div class="header">ISL VISION</div>

<div class="container">

    <h1>Indian Sign Language Recognition</h1>

    <div class="subtitle">
        AI-Powered Indian Sign Language A–Z Recognition
    </div>

    <button class="button" onclick="location.href='/recognize'">
        A–Z RECOGNITION
    </button>

    <button class="button" onclick="location.href='/learning'">
        LEARN A–Z
    </button>

    <button class="button exit-button" onclick="location.href='/exit'">
        EXIT
    </button>

</div>

</body>
</html>
"""

# ============================================================
# RECOGNITION PAGE
# ============================================================

RECOGNITION_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>A–Z Recognition</title>
    {{ style|safe }}
</head>

<body>

<div class="header">ISL VISION</div>

<div class="container">

    <h1>A–Z Recognition</h1>

    <div class="camera-area">
        <video id="video" autoplay playsinline></video>
        <canvas id="landmarkCanvas"></canvas>
    </div>

    <!-- ONLY ONE FINAL LETTER IS SHOWN -->
    <div class="prediction" id="prediction">-</div>

    <div class="detected" id="handCount">
        No hands detected
    </div>

    <button class="button" id="startButton" onclick="startCamera()">
        START CAMERA
    </button>

    <button class="button back-button" onclick="goBack()">
        ← BACK
    </button>

</div>

<script>
const video = document.getElementById("video");
const canvas = document.getElementById("landmarkCanvas");
const ctx = canvas.getContext("2d");

const predictionText = document.getElementById("prediction");
const handCountText = document.getElementById("handCount");
const startButton = document.getElementById("startButton");

let stream = null;
let timer = null;

let lastFinalLetter = "";
let stableCount = 0;
let lastSpoken = "";

/*
   Landmark smoothing.
   This keeps the red dots and green lines from jumping.
*/
let previousHands = [];


/* ==========================================================
   START CAMERA
   ========================================================== */

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        resizeCanvas();

        startButton.innerText = "CAMERA RUNNING";

        if (timer !== null) {
            clearInterval(timer);
        }

        timer = setInterval(recognizeFrame, 250);
    } catch (error) {
        console.error(error);
        alert("Please allow camera permission.");
    }
}


/* ==========================================================
   CANVAS SIZE
   ========================================================== */

function resizeCanvas() {
    if (video.videoWidth === 0) {
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}


/* ==========================================================
   SMOOTH LANDMARKS
   ========================================================== */

function smoothHands(currentHands) {
    const alpha = 0.65;

    if (previousHands.length !== currentHands.length) {
        previousHands = currentHands.map(hand => ({
            points: hand.points.map(p => ({
                x: p.x,
                y: p.y,
                z: p.z
            })),
            connections: hand.connections
        }));

        return currentHands;
    }

    const smoothed = [];

    for (let h = 0; h < currentHands.length; h++) {
        const current = currentHands[h];
        const previous = previousHands[h];

        const points = [];

        for (let i = 0; i < current.points.length; i++) {
            points.push({
                x: alpha * current.points[i].x +
                   (1 - alpha) * previous.points[i].x,

                y: alpha * current.points[i].y +
                   (1 - alpha) * previous.points[i].y,

                z: alpha * current.points[i].z +
                   (1 - alpha) * previous.points[i].z
            });
        }

        smoothed.push({
            points: points,
            connections: current.connections
        });
    }

    previousHands = smoothed;

    return smoothed;
}


/* ==========================================================
   RECOGNIZE FRAME
   ========================================================== */

async function recognizeFrame() {
    if (!stream || video.readyState < 2) {
        return;
    }

    resizeCanvas();

    /*
       IMPORTANT:
       The image sent to Python is NOT flipped.
       This keeps the model input consistent with the
       original working recognition program.
    */

    const capture = document.createElement("canvas");

    capture.width = video.videoWidth;
    capture.height = video.videoHeight;

    const captureCtx = capture.getContext("2d");

    captureCtx.drawImage(
        video,
        0,
        0,
        capture.width,
        capture.height
    );

    const imageData = capture.toDataURL("image/png");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        const result = await response.json();

        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        /* ==================================================
           LANDMARKS FOR 0, 1 OR 2 REAL HANDS
           ================================================== */

        if (result.hands && result.hands.length > 0) {

            const visibleHands =
                smoothHands(result.hands);

            handCountText.innerText =
                visibleHands.length +
                (visibleHands.length === 1
                    ? " hand detected"
                    : " hands detected");

            for (const hand of visibleHands) {

                ctx.strokeStyle = "#39ff14";
                ctx.lineWidth = 3;

                for (const connection of hand.connections) {

                    const p1 =
                        hand.points[connection[0]];

                    const p2 =
                        hand.points[connection[1]];

                    /*
                       The canvas is mirrored together with
                       the video using CSS, so we use the
                       original MediaPipe X coordinate here.
                    */

                    const x1 =
                        p1.x * canvas.width;

                    const y1 =
                        p1.y * canvas.height;

                    const x2 =
                        p2.x * canvas.width;

                    const y2 =
                        p2.y * canvas.height;

                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }

                for (const point of hand.points) {

                    const x =
                        point.x * canvas.width;

                    const y =
                        point.y * canvas.height;

                    ctx.beginPath();

                    ctx.arc(
                        x,
                        y,
                        5,
                        0,
                        Math.PI * 2
                    );

                    ctx.fillStyle = "#ff3030";
                    ctx.fill();
                }
            }

        } else {

            handCountText.innerText =
                "No hands detected";

            previousHands = [];
        }


        /* ==================================================
           FINAL LETTER

           IMPORTANT:
           We do NOT display:
              LEFT: A
              RIGHT: B

           We display ONLY:
              A

           If both hands produce the same letter, that
           letter is used.

           If they differ, the first stable hand prediction
           is used. The user sees only ONE final letter.
           ================================================== */

        let finalLetter = null;

        if (
            result.predictions &&
            result.predictions.length > 0
        ) {

            const predictions =
                result.predictions.filter(
                    letter =>
                        letter !== null &&
                        letter !== undefined &&
                        letter !== ""
                );

            if (predictions.length === 1) {

                finalLetter = predictions[0];

            } else if (predictions.length > 1) {

                /*
                   If both hands give the same letter,
                   use that letter.
                */

                if (
                    predictions.every(
                        letter =>
                            letter === predictions[0]
                    )
                ) {

                    finalLetter =
                        predictions[0];

                } else {

                    /*
                       Different letters on two hands:
                       keep ONE final letter instead of
                       showing left/right labels.
                    */

                    finalLetter =
                        predictions[0];
                }
            }
        }


        /* ==================================================
           STABLE FINAL LETTER
           ================================================== */

        if (finalLetter) {

            predictionText.innerText =
                finalLetter;

            if (
                finalLetter === lastFinalLetter
            ) {

                stableCount++;

            } else {

                lastFinalLetter =
                    finalLetter;

                stableCount = 1;
            }

            /*
               4 × 250 ms ≈ 1 second.
            */

            if (
                stableCount >= 4 &&
                lastSpoken !== finalLetter
            ) {

                speakLetter(finalLetter);

                lastSpoken =
                    finalLetter;
            }

        } else {

            predictionText.innerText = "-";

            lastFinalLetter = "";
            stableCount = 0;
        }

    } catch (error) {
        console.error(
            "Recognition error:",
            error
        );
    }
}


/* ==========================================================
   SPEAK ONLY THE FINAL LETTER
   ========================================================== */

function speakLetter(letter) {

    if (!window.speechSynthesis) {
        return;
    }

    window.speechSynthesis.cancel();

    const speech =
        new SpeechSynthesisUtterance(letter);

    speech.rate = 0.9;
    speech.pitch = 1.0;
    speech.volume = 1.0;

    window.speechSynthesis.speak(speech);
}


/* ==========================================================
   STOP CAMERA
   ========================================================== */

function stopCamera() {

    if (timer !== null) {
        clearInterval(timer);
        timer = null;
    }

    if (stream) {
        stream.getTracks().forEach(
            track => track.stop()
        );

        stream = null;
    }

    video.srcObject = null;

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    previousHands = [];
}


/* ==========================================================
   BACK
   ========================================================== */

function goBack() {
    stopCamera();
    window.location.href = "/";
}

window.addEventListener(
    "beforeunload",
    stopCamera
);
</script>

</body>
</html>
"""

# ============================================================
# LEARNING PAGE
# ============================================================

LEARNING_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Learn A–Z</title>
    {{ style|safe }}
</head>

<body>

<div class="header">ISL VISION</div>

<div class="container">

    <h1>Learn A–Z</h1>

    <div class="subtitle">
        Select a letter to view its ISL sign.
    </div>

    <div class="learning-grid">

        {% for letter in letters %}

        <button
            class="letter-button"
            onclick="showLetter('{{ letter }}')">
            {{ letter }}
        </button>

        {% endfor %}

    </div>

    <div id="imageArea">

        <div class="message">
            <h2>Select a letter</h2>
            <p>Choose A–Z to see the sign.</p>
        </div>

    </div>

    <button
        class="button back-button"
        onclick="location.href='/'">
        ← BACK
    </button>

</div>

<script>
async function showLetter(letter) {

    const area =
        document.getElementById("imageArea");

    area.innerHTML =
        "<div class='message'>" +
        "<h2>Loading " +
        letter +
        "...</h2>" +
        "</div>";

    try {

        const response =
            await fetch(
                "/learning_image/" + letter
            );

        const data =
            await response.json();

        if (!data.image) {

            area.innerHTML =
                "<div class='message'>" +
                "<h2>No image found</h2>" +
                "</div>";

            return;
        }

        area.innerHTML =
            "<div class='learning-title'>" +
            "ISL Sign – " +
            letter +
            "</div>" +

            "<div class='learning-image'>" +

            "<img src='" +
            data.image +
            "' alt='ISL " +
            letter +
            "'>" +

            "</div>";

    } catch (error) {

        console.error(error);

        area.innerHTML =
            "<div class='message'>" +
            "<h2>Error loading image</h2>" +
            "</div>";
    }
}
</script>

</body>
</html>
"""

# ============================================================
# EXIT PAGE
# ============================================================

EXIT_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Exit</title>
    {{ style|safe }}
</head>

<body>

<div class="header">ISL VISION</div>

<div class="container">

    <div class="message">

        <h1>Thank You!</h1>

        <p>You can now close this browser tab.</p>

        <button
            class="button"
            onclick="window.close()">
            CLOSE
        </button>

    </div>

</div>

</body>
</html>
"""

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(
        HOME_PAGE,
        style=STYLE
    )


@app.route("/recognize")
def recognize():
    return render_template_string(
        RECOGNITION_PAGE,
        style=STYLE
    )


@app.route("/learning")
def learning():

    letters = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    return render_template_string(
        LEARNING_PAGE,
        style=STYLE,
        letters=letters
    )


@app.route("/exit")
def exit_page():
    return render_template_string(
        EXIT_PAGE,
        style=STYLE
    )


# ============================================================
# DATASET FOLDER
# ============================================================

def find_letter_folder(letter):

    if not DATASET_DIR.exists():
        return None

    for folder in DATASET_DIR.iterdir():

        if (
            folder.is_dir()
            and folder.name.lower()
            == letter.lower()
        ):
            return folder

    return None


# ============================================================
# ONE LEARNING IMAGE
# ============================================================

@app.route("/learning_image/<letter>")
def learning_image(letter):

    folder = find_letter_folder(letter)

    if folder is None:
        return jsonify({"image": None})

    image_files = []

    for file in folder.iterdir():

        if (
            file.is_file()
            and file.suffix.lower()
            in (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):
            image_files.append(file)

    image_files.sort(
        key=lambda x: x.name.lower()
    )

    if not image_files:
        return jsonify({"image": None})

    image_file = image_files[0]

    return jsonify({
        "image":
            "/dataset_image/"
            + folder.name
            + "/"
            + image_file.name
    })


# ============================================================
# SERVE DATASET IMAGE
# ============================================================

@app.route("/dataset_image/<letter>/<filename>")
def dataset_image(letter, filename):

    folder = find_letter_folder(letter)

    if folder is None:
        return "Folder not found", 404

    file_path = folder / filename

    if not file_path.exists():
        return "Image not found", 404

    return send_from_directory(
        str(folder),
        filename
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if (
            not data
            or "image" not in data
        ):
            return jsonify({
                "predictions": [],
                "hands": []
            })


        image_data = data["image"]

        if "," in image_data:

            image_data = image_data.split(",", 1)[1]


        image_bytes = base64.b64decode(image_data)


        image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )


        image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )


        if image is None:

            return jsonify({
                "predictions": [],
                "hands": []
            })


        # ----------------------------------------------------
        # BGR TO RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        results = hands.process(rgb)


        if not results.multi_hand_landmarks:

            return jsonify({
                "predictions": [],
                "hands": []
            })


        all_hands = []
        predictions = []


        # ----------------------------------------------------
        # PROCESS EACH REAL HAND
        # ----------------------------------------------------

        for hand_landmarks in \
            results.multi_hand_landmarks:


            # ------------------------------------------------
            # LANDMARK POINTS
            # ------------------------------------------------

            points = []

            for landmark in \
                hand_landmarks.landmark:

                points.append({

                    "x":
                        float(landmark.x),

                    "y":
                        float(landmark.y),

                    "z":
                        float(landmark.z)

                })


            # ------------------------------------------------
            # CONNECTIONS
            # ------------------------------------------------

            connections = []

            for connection in \
                mp_hands.HAND_CONNECTIONS:

                connections.append([

                    int(connection[0]),

                    int(connection[1])

                ])


            all_hands.append({

                "points":
                    points,

                "connections":
                    connections

            })


            # ------------------------------------------------
            # SAME 63 FEATURES USED DURING TRAINING
            # ------------------------------------------------

            wrist = hand_landmarks.landmark[0]


            features = []

            for landmark in \
                hand_landmarks.landmark:

                x = landmark.x - wrist.x

                y = landmark.y - wrist.y

                z = landmark.z - wrist.z


                features.extend([
                    x,
                    y,
                    z
                ])


            feature_array = np.array(
                    features,
                    dtype=np.float32
                ).reshape(
                    1,
                    -1
                )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                    feature_array
                )[0]


            predictions.append(
                str(prediction)
            )


        return jsonify({

            "predictions":
                predictions,

            "hands":
                all_hands
        })


    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return jsonify({

            "predictions": [],

            "hands": [],

            "error":
                str(error)
        })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
            os.environ.get(
                "PORT",
                5000
            )
        )

    print()
    print(
        "=========================================="
    )
    print(
        "              ISL VISION"
    )
    print(
        "=========================================="
    )
    print(
        "A-Z recognition"
    )
    print(
        "1 or 2 hand detection"
    )
    print(
        f"Model: {MODEL_FILE}"
    )
    print(
        f"Dataset: {DATASET_DIR}"
    )
    print(
        f"Port: {port}"
    )
    print(
        "=========================================="
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
