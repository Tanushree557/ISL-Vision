import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = BASE_DIR / "landmarks.csv"
MODEL_FILE = BASE_DIR / "letter_model.pkl"


# -----------------------------
# Load landmark data
# -----------------------------
df = pd.read_csv(CSV_FILE)

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}")


# -----------------------------
# Separate features and labels
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]


print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {y.nunique()}")


# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Create Random Forest model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# -----------------------------
# Train model
# -----------------------------
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)


# -----------------------------
# Test model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# -----------------------------
# Save trained model
# -----------------------------
joblib.dump(model, MODEL_FILE)

print("\n--------------------------------")
print("Training completed successfully!")
print(f"Model saved to: {MODEL_FILE}")
print("--------------------------------")