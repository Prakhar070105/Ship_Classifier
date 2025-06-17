from flask import Flask, render_template, request, flash
import numpy as np
import os
import librosa
import joblib
from utils import extract_features, append_training_data, retrain_model

app = Flask(__name__)
app.secret_key = 'ship_secret'

model_path = 'trained_model.joblib'

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
                audio_path = "temp.wav"
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

        elif 'add_train_btn' in request.form:
            label = request.form.get('label')

            if 'new_audio' in request.files and request.files['new_audio'].filename != '':
                audio = request.files['new_audio']
                audio_path = "temp_train.wav"
                audio.save(audio_path)

                features = extract_features(audio_path)
                os.remove(audio_path)

                if features is not None:
                    append_training_data(features, label)
                    flash("Audio file features saved!", "success")
                else:
                    flash("Failed to extract features from audio.", "danger")

            elif request.form.get('manual_mfcc'):
                try:
                    mfcc_values = list(map(float, request.form['manual_mfcc'].strip().split(',')))
                    append_training_data(mfcc_values, label)
                    flash("Manual MFCC values saved!", "success")
                except:
                    flash("Invalid MFCC values.", "danger")
            else:
                flash("Please upload a file or enter MFCC values.", "warning")

        elif 'train_btn' in request.form:
            success = retrain_model()
            if success:
                flash("Model retrained successfully!", "success")
            else:
                flash("Training failed. Check if data exists in new_data.csv", "danger")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)




