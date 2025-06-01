from flask import Flask, request, render_template
import joblib
import numpy as np
import librosa
import os

app = Flask(__name__)
model = joblib.load("trained_model.joblib")

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' in request.files:
        audio = request.files['audio']
        path = os.path.join("temp_audio.wav")
        audio.save(path)
        features = extract_features(path)
        os.remove(path)
        prediction = model.predict([features])[0]
        return render_template('index.html', prediction_text=f"Predicted Ship Type: {prediction}")

    elif request.form.get('features'):
        try:
            input_features = request.form['features']
            features = np.array([float(x) for x in input_features.strip().split(',')])
            prediction = model.predict([features])[0]
            return render_template('index.html', prediction_text=f"Predicted Ship Type: {prediction}")
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {e}")

    return render_template('index.html', prediction_text="Please upload a file or input features.")

if __name__ == "__main__":
    app.run(debug=True)

