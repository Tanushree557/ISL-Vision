"""
Mobile word recognition helper.

The main Flask application performs browser-camera recognition directly.
This file is kept as a standalone helper for the mobile project and
uses the same 23-word model.
"""

import os
import pickle
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "word_model_23.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "word_labels_23.pkl"
)


def load_model_and_labels():

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABEL_PATH, "rb") as f:
        labels = pickle.load(f)

    if hasattr(labels, "classes_"):
        classes = list(labels.classes_)

    elif isinstance(labels, dict):

        if all(isinstance(k, int) for k in labels.keys()):
            classes = [
                labels[k]
                for k in sorted(labels.keys())
            ]
        else:
            classes = list(labels.values())

    else:
        classes = list(labels)

    return model, classes


def predict_sequence(model, classes, sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (16, 162):
        raise ValueError(
            f"Expected sequence shape (16, 162), "
            f"received {sequence.shape}"
        )

    prediction = model.predict(
        sequence.reshape(1, 16, 162),
        verbose=0
    )[0]

    index = int(np.argmax(prediction))

    return (
        classes[index],
        float(prediction[index])
    )


if __name__ == "__main__":

    model, classes = load_model_and_labels()

    print("Word model loaded successfully.")
    print("Number of classes:", len(classes))
    print("Classes:")

    for i, name in enumerate(classes):
        print(i, name)