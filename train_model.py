import os
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import load_all_labels

# Load labels and file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "DeepShip")
df = load_all_labels(data_dir)

# Extract features (MFCCs)
def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Apply feature extraction
print("Extracting features...")
df["features"] = df["filepath"].apply(extract_features)
df = df[df["features"].notnull()]  # Drop rows where features couldn't be extracted

# Prepare features and labels
X = np.vstack(df["features"].values)
y = df["ship_type"].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the trained model (joblib only)
model_path = os.path.join(script_dir, "trained_model.joblib")
joblib.dump(clf, model_path)
print(f"\n✅ Model saved to {model_path}")
