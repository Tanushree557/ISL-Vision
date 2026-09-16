import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "word_landmarks.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "word_model.pkl"

MODEL_DIR.mkdir(exist_ok=True)


print("Loading word landmark data...")

data = pd.read_csv(DATA_FILE)

X = data.drop("label", axis=1)
y = data["label"]


print(f"Total samples: {len(data)}")
print(f"Total features: {X.shape[1]}")
print("Words:", sorted(y.unique()))


# Create training and testing data
train_indices = []
test_indices = []

rng = np.random.RandomState(42)

for word in sorted(y.unique()):

    indices = np.where(y.values == word)[0]

    rng.shuffle(indices)

    if len(indices) == 1:
        # Keep single-sample words in training
        train_indices.extend(indices)

    else:
        test_count = max(1, int(len(indices) * 0.20))

        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])


X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


print("\nTraining Word Recognition Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


print("\nTesting model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


joblib.dump(
    model,
    MODEL_FILE
)


print("\nWORD MODEL TRAINING COMPLETE!")

print(
    f"Model saved to: {MODEL_FILE}"
)