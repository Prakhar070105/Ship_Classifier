from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import os
import librosa
import joblib
from utils import extract_features, append_training_data, retrain_model

app = Flask(__name__)
app.secret_key = 'ship_secret'

model_path = 'trained_model.joblib'

# Load model initially
def load_model():
    return joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        model = load_model()

        if 'predict_btn' in request.form:
            if 'audio_file' in request.files and request.files['audio_file'].filename != '':
                audio = request.files['audio_file']
                audio_path = os.path.join("temp.wav")
                audio.save(audio_path)

                features = extract_features(audio_path)
                os.remove(audio_path)

                if features is not None:
                    prediction = model.predict([features])[0]
                    append_training_data(features, prediction)
                else:
                    flash("Failed to extract features from audio file", "danger")

            elif request.form.get('manual_input'):
                try:
                    values = list(map(float, request.form['manual_input'].strip().split(',')))
                    prediction = model.predict([values])[0]
                    append_training_data(values, prediction)
                except:
                    flash("Invalid manual input", "danger")

        elif 'train_btn' in request.form:
            success = retrain_model()
            if success:
                flash("Model retrained successfully!", "success")
            else:
                flash("Training failed. Check logs or data.", "danger")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
