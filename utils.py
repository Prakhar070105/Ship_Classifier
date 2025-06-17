import numpy as np
import pandas as pd
import os
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier

FEATURES_FILE = "new_data.csv"
MODEL_FILE = "trained_model.joblib"

def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

def append_training_data(features, label):
    row = list(features) + [label]
    header = [f"mfcc_{i}" for i in range(len(features))] + ["label"]
    new_df = pd.DataFrame([row], columns=header)

    if not os.path.exists(FEATURES_FILE):
        new_df.to_csv(FEATURES_FILE, index=False)
    else:
        try:
            existing_df = pd.read_csv(FEATURES_FILE, nrows=1)
            if list(existing_df.columns) != header:
                print("[WARNING] Column mismatch. Appending anyway.")
        except Exception as e:
            print(f"[ERROR] Could not validate existing CSV: {e}")
        new_df.to_csv(FEATURES_FILE, mode='a', header=False, index=False)

def retrain_model():
    if not os.path.exists(FEATURES_FILE):
        print("[INFO] No data file found for retraining.")
        return False

    try:
        df = pd.read_csv(FEATURES_FILE)
        if df.empty or "label" not in df.columns:
            print("[INFO] Data file is empty or missing 'label' column.")
            return False

        X = df.drop("label", axis=1).values
        y = df["label"].values

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        print("[SUCCESS] Model retrained and saved.")
        return True
    except Exception as e:
        print(f"[ERROR] Model retraining failed: {e}")
        return False



def append_training_data(features, label):
    row = list(features) + [label]
    header = [f"mfcc_{i}" for i in range(len(features))] + ["label"]
    new_df = pd.DataFrame([row], columns=header)

    if not os.path.exists(FEATURES_FILE):
        new_df.to_csv(FEATURES_FILE, index=False)
    else:
        try:
            existing_df = pd.read_csv(FEATURES_FILE, nrows=1)
            if list(existing_df.columns) != header:
                print("[WARNING] Column mismatch. Appending anyway.")
        except Exception as e:
            print(f"[ERROR] Could not validate existing CSV: {e}")
        new_df.to_csv(FEATURES_FILE, mode='a', header=False, index=False)

def retrain_model():
    if not os.path.exists(FEATURES_FILE):
        print("[INFO] No data file found for retraining.")
        return False

    try:
        df = pd.read_csv(FEATURES_FILE)
        if df.empty or "label" not in df.columns:
            print("[INFO] Data file is empty or missing 'label' column.")
            return False

        X = df.drop("label", axis=1).values
        y = df["label"].values

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        print("[SUCCESS] Model retrained and saved.")
        return True
    except Exception as e:
        print(f"[ERROR] Model retraining failed: {e}")
        return False



