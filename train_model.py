import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Project paths
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "landmarks.csv"
MODEL_FILE = BASE_DIR / "letter_model.pkl"

# Load landmark data
print("Loading landmark data...")
data = pd.read_csv(DATA_FILE)

# Separate input features and labels
X = data.drop("label", axis=1)
y = data["label"]

print(f"Total samples: {len(data)}")
print(f"Total features: {X.shape[1]}")
print("Letters:", sorted(y.unique()))

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create AI model
print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Test
print("Testing model...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
joblib.dump(model, MODEL_FILE)

print("\nMODEL TRAINING COMPLETE!")
print(f"Model saved to: {MODEL_FILE}")