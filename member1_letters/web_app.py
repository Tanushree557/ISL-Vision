from flask import Flask, render_template_string, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import base64

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "letter_model.pkl"
LEARNING_DIR = BASE_DIR / "learning_images"

model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AI-Based Indian Sign Language Recognition</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f7fb;
    color: #172033;
}

.header {
    background: #172033;
    color: white;
    padding: 18px;
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 24px;
}

.header p {
    margin: 7px 0 0;
    font-size: 14px;
}

.container {
    max-width: 950px;
    margin: 24px auto;
    padding: 0 15px;
}

.tabs {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}

button {
    border: 0;
    border-radius: 10px;
    padding: 11px 17px;
    font-size: 15px;
    cursor: pointer;
}

.tab-btn {
    background: #dfe6f2;
    color: #172033;
}

.tab-btn.active {
    background: #172033;
    color: white;
}

.panel {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
}

#learningPanel {
    display: none;
}

.camera-box {
    position: relative;
    width: 100%;
    max-width: 720px;
    margin: auto;
    background: black;
    border-radius: 14px;
    overflow: hidden;
}

#video {
    width: 100%;
    display: block;
    transform: scaleX(-1);
}

#overlay {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    transform: scaleX(-1);
}

.controls {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin: 18px 0;
}

.start {
    background: #172033;
    color: white;
}

.stop {
    background: #e5e9f0;
    color: #172033;
}

.result-box {
    text-align: center;
    margin-top: 15px;
}

.result-label {
    font-size: 15px;
    color: #687386;
}

#letter {
    font-size: 82px;
    font-weight: bold;
    min-height: 100px;
    margin-top: 5px;
}

#status {
    color: #687386;
    min-height: 24px;
}

.small {
    font-size: 13px;
    color: #687386;
    text-align: center;
    margin-top: 14px;
}

.learning-title {
    text-align: center;
}

.letter-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(55px, 1fr));
    gap: 9px;
    margin-top: 20px;
}

.letter-btn {
    background: #e9eef7;
    color: #172033;
    font-weight: bold;
}

.learning-card {
    max-width: 520px;
    margin: 25px auto;
    text-align: center;
}

#learningImage {
    max-width: 100%;
    max-height: 450px;
    border-radius: 12px;
    display: none;
    margin: 15px auto;
}

#learningMessage {
    color: #687386;
    min-height: 25px;
}

</style>
</head>

<body>

<div class="header">

    <h1>AI-Based Indian Sign Language Recognition</h1>

    <p>A–Z Letter Recognition and Learning</p>

</div>


<div class="container">


<div class="tabs">

    <button
        class="tab-btn active"
        id="recognitionTab"
        onclick="showRecognition()">

        Letter Recognition

    </button>


    <button
        class="tab-btn"
        id="learningTab"
        onclick="showLearning()">

        Learning Mode

    </button>

</div>


<div class="panel" id="cameraPanel">

    <div class="camera-box">

        <video
            id="video"
            autoplay
            playsinline
            muted>
        </video>

        <canvas id="overlay"></canvas>

    </div>


    <div class="controls">

        <button
            class="start"
            onclick="startCamera()">

            Start Camera

        </button>


        <button
            class="stop"
            onclick="stopCamera()">

            Stop Camera

        </button>

    </div>


    <div class="result-box">

        <div class="result-label">
            Detected Letter
        </div>

        <div id="letter">
            —
        </div>

        <div id="status">
            Click Start Camera
        </div>

    </div>


    <div class="small">
        Hold one clear ISL hand sign in front of the camera.
    </div>

</div>


<div class="panel" id="learningPanel">

    <h2 class="learning-title">
        ISL A–Z Learning Mode
    </h2>

    <p class="small">
        Select a letter to view its ISL sign.
    </p>


    <div
        class="letter-buttons"
        id="letterButtons">
    </div>


    <div class="learning-card">

        <h2 id="selectedLetter">
            Select a letter
        </h2>


        <img
            id="learningImage"
            alt="ISL sign">


        <div id="learningMessage">
            Choose A–Z above.
        </div>

    </div>

</div>


</div>


<script>

const video = document.getElementById("video");

const overlay = document.getElementById("overlay");

const ctx = overlay.getContext("2d");

let stream = null;

let running = false;

let timer = null;

let processing = false;

let stableLetter = "";

let stableCount = 0;

let lastSpokenLetter = "";

let noHandCount = 0;

const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");


function showRecognition() {

    document.getElementById("cameraPanel").style.display = "block";

    document.getElementById("learningPanel").style.display = "none";

    document.getElementById("recognitionTab").classList.add("active");

    document.getElementById("learningTab").classList.remove("active");

}


function showLearning() {

    document.getElementById("cameraPanel").style.display = "none";

    document.getElementById("learningPanel").style.display = "block";

    document.getElementById("recognitionTab").classList.remove("active");

    document.getElementById("learningTab").classList.add("active");

    stopCamera();

}


function buildLetterButtons() {

    const box = document.getElementById("letterButtons");

    letters.forEach(function(letter) {

        const button = document.createElement("button");

        button.className = "letter-btn";

        button.textContent = letter;

        button.onclick = function() {

            loadLearningLetter(letter);

        };

        box.appendChild(button);

    });

}


async function loadLearningLetter(letter) {

    const title = document.getElementById("selectedLetter");

    const img = document.getElementById("learningImage");

    const message = document.getElementById("learningMessage");

    title.textContent = letter;

    img.style.display = "none";

    message.textContent = "Loading...";

    try {

        const response = await fetch("/learning/" + letter);

        if (!response.ok) {

            message.textContent = "Image not found.";

            return;

        }

        img.src = "/learning/" + letter + "?t=" + Date.now();

        img.onload = function() {

            img.style.display = "block";

            message.textContent = "ISL sign for " + letter;

        };

        img.onerror = function() {

            img.style.display = "none";

            message.textContent = "Image not found.";

        };

    } catch (error) {

        message.textContent = "Image could not be loaded.";

    }

}


function resizeCanvas() {

    if (video.videoWidth > 0 && video.videoHeight > 0) {

        overlay.width = video.videoWidth;

        overlay.height = video.videoHeight;

    }

}


async function startCamera() {

    if (running) {
        return;
    }

    if (
        !navigator.mediaDevices ||
        !navigator.mediaDevices.getUserMedia
    ) {

        document.getElementById("status").textContent =
            "Camera is not supported by this browser.";

        return;

    }

    try {

        document.getElementById("status").textContent =
            "Requesting camera...";


        stream = await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "user"
            },

            audio: false

        });


        video.srcObject = stream;


        await new Promise(function(resolve) {

            if (
                video.readyState >= 2 &&
                video.videoWidth > 0
            ) {

                resolve();

            } else {

                video.onloadedmetadata = function() {
                    resolve();
                };

            }

        });


        await video.play();


        while (
            video.videoWidth === 0 ||
            video.videoHeight === 0
        ) {

            await new Promise(function(resolve) {

                setTimeout(resolve, 100);

            });

        }


        resizeCanvas();


        running = true;

        processing = false;

        stableLetter = "";

        stableCount = 0;

        lastSpokenLetter = "";

        noHandCount = 0;


        document.getElementById("status").textContent =
            "Camera running...";


        timer = setInterval(
            processFrame,
            350
        );


    } catch (error) {

        console.error(error);

        document.getElementById("status").textContent =
            "Camera could not be started. Allow camera permission.";

    }

}


function stopCamera() {

    running = false;

    processing = false;


    if (timer) {

        clearInterval(timer);

        timer = null;

    }


    if (stream) {

        stream.getTracks().forEach(function(track) {

            track.stop();

        });

        stream = null;

    }


    video.srcObject = null;


    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    document.getElementById("status").textContent =
        "Camera stopped.";

}


async function processFrame() {

    if (!running) {
        return;
    }

    if (processing) {
        return;
    }

    if (
        video.videoWidth === 0 ||
        video.videoHeight === 0
    ) {
        return;
    }


    processing = true;


    try {

        const canvas = document.createElement("canvas");

        canvas.width = video.videoWidth;

        canvas.height = video.videoHeight;


        const c = canvas.getContext("2d");


        c.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );


        const image =
            canvas.toDataURL(
                "image/jpeg",
                0.75
            );


        const response = await fetch(
            "/predict",
            {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    image: image
                })

            }
        );


        if (!response.ok) {

            document.getElementById("status").textContent =
                "Prediction server error.";

            return;

        }


        const data = await response.json();


        drawLandmarks(data.hands || []);


        if (
            !data.hands ||
            data.hands.length === 0
        ) {

            noHandCount++;


            if (noHandCount >= 4) {

                stableLetter = "";

                stableCount = 0;

                lastSpokenLetter = "";

                document.getElementById("letter").textContent =
                    "—";

            }


            document.getElementById("status").textContent =
                "No hand detected";


            return;

        }


        noHandCount = 0;


        let best = data.hands[0];


        for (const hand of data.hands) {

            if (
                (hand.confidence || 0) >
                (best.confidence || 0)
            ) {

                best = hand;

            }

        }


        const predictedLetter = best.letter;


        if (!predictedLetter) {
            return;
        }


        if (
            predictedLetter === stableLetter
        ) {

            stableCount++;

        } else {

            stableLetter = predictedLetter;

            stableCount = 1;

        }


        document.getElementById("status").textContent =
            "Detecting: " + predictedLetter;


        if (stableCount >= 4) {

            document.getElementById("letter").textContent =
                predictedLetter;


            if (
                lastSpokenLetter !== predictedLetter
            ) {

                speakLetter(predictedLetter);

                lastSpokenLetter = predictedLetter;

            }

        }


    } catch (error) {

        console.error(error);

        document.getElementById("status").textContent =
            "Connection problem. Trying again...";

    } finally {

        processing = false;

    }

}


function drawLandmarks(allHands) {

    ctx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    if (
        !allHands ||
        allHands.length === 0
    ) {

        return;

    }


    ctx.lineWidth = 3;

    ctx.strokeStyle = "#00ff66";

    ctx.fillStyle = "#ff3333";


    for (const hand of allHands) {

        const landmarks = hand.landmarks || [];


        for (const connection of hand.connections || []) {

            const a = landmarks[connection[0]];

            const b = landmarks[connection[1]];


            if (!a || !b) {
                continue;
            }


            ctx.beginPath();

            ctx.moveTo(
                a.x * overlay.width,
                a.y * overlay.height
            );

            ctx.lineTo(
                b.x * overlay.width,
                b.y * overlay.height
            );

            ctx.stroke();

        }


        for (const point of landmarks) {

            ctx.beginPath();

            ctx.arc(
                point.x * overlay.width,
                point.y * overlay.height,
                5,
                0,
                Math.PI * 2
            );

            ctx.fill();

        }

    }

}


function speakLetter(letter) {

    if (!("speechSynthesis" in window)) {
        return;
    }


    window.speechSynthesis.cancel();


    const utterance =
        new SpeechSynthesisUtterance(letter);


    utterance.rate = 0.8;

    utterance.pitch = 1.0;

    utterance.volume = 1.0;


    window.speechSynthesis.speak(utterance);

}


buildLetterButtons();

</script>

</body>
</html>
"""


@app.route("/")
def home():

    return render_template_string(HTML)


@app.route("/learning/<letter>")
def learning(letter):

    letter = letter.upper()


    if (
        len(letter) != 1 or
        letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):

        return "Invalid letter", 404


    image_file = f"{letter}.jpg"

    full_path = LEARNING_DIR / image_file


    if not full_path.exists():

        return "Image not found", 404


    return send_from_directory(
        LEARNING_DIR,
        image_file
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        payload = request.get_json(
            silent=True
        )


        if (
            not payload or
            "image" not in payload
        ):

            return jsonify({
                "hands": [],
                "error": "No image received"
            }), 400


        image_data = payload["image"]


        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]


        image_bytes = base64.b64decode(
            image_data
        )


        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )


        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return jsonify({
                "hands": [],
                "error": "Could not decode image"
            }), 400


        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        results = hands.process(
            rgb_image
        )


        if not results.multi_hand_landmarks:

            return jsonify({
                "hands": []
            })


        response_hands = []


        for hand_landmarks in results.multi_hand_landmarks:

            wrist = hand_landmarks.landmark[0]

            features = []


            for landmark in hand_landmarks.landmark:

                x = landmark.x - wrist.x

                y = landmark.y - wrist.y

                z = landmark.z - wrist.z

                features.extend([
                    x,
                    y,
                    z
                ])


            features_array = np.array(
                features,
                dtype=np.float32
            ).reshape(
                1,
                -1
            )


            prediction = model.predict(
                features_array
            )[0]


            confidence = 0.0


            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    features_array
                )[0]

                confidence = float(
                    np.max(probabilities)
                )


            landmark_list = []


            for point in hand_landmarks.landmark:

                landmark_list.append({
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z)
                })


            connections = []


            for connection in mp_hands.HAND_CONNECTIONS:

                connections.append([
                    int(connection[0]),
                    int(connection[1])
                ])


            response_hands.append({

                "letter": str(prediction),

                "confidence": confidence,

                "landmarks": landmark_list,

                "connections": connections

            })


        return jsonify({
            "hands": response_hands
        })


    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )


        return jsonify({

            "hands": [],

            "error": str(error)

        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )