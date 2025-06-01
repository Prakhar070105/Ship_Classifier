import os
import librosa
import numpy as np
import joblib

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "trained_model.pkl")

# Load trained model
clf = joblib.load(model_path)

# Function to extract features
def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Test file (replace with your actual file path)
test_file = os.path.join(script_dir, "data", "DeepShip", "Cargo", "27.wav")
  # <-- Replace 'test.wav' with your file

# Run prediction
features = extract_features(test_file)
if features is not None:
    prediction = clf.predict([features])
    print(f"\nPredicted ship type: {prediction[0]}")
else:
    print("Failed to extract features.")
