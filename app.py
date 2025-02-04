# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np
from keras.models import load_model

# Load models
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('ica_model.pkl', 'rb') as f:
    ica = pickle.load(f)

ann_model = load_model('ann_model.h5')

app = Flask(__name__)

# Preprocessing function
def preprocess_input(features):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform([features])

# BCM-based Voting (Majority Voting)
def bcm_voting(kmeans_pred, knn_pred, pca_pred, ica_pred, ann_pred):
    predictions = [kmeans_pred, knn_pred, pca_pred, ica_pred, ann_pred]
    return max(set(predictions), key=predictions.count)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        processed_input = preprocess_input(features)

        # Get predictions from each model
        kmeans_pred = kmeans.predict(processed_input)[0]
        knn_pred = knn.predict(processed_input)[0]
        pca_pred = (pca.transform(processed_input)[:, 0] > 0).astype(int)[0]
        ica_pred = (ica.transform(processed_input)[:, 0] > 0).astype(int)[0]
        ann_pred = (ann_model.predict(processed_input) > 0.5).astype(int)[0]

        # Perform BCM-based Voting
        final_prediction = bcm_voting(kmeans_pred, knn_pred, pca_pred, ica_pred, ann_pred)

        if final_prediction == 1:
            result = "The person has a risk of cardiovascular disease."
        else:
            result = "The person does not have a risk of cardiovascular disease."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
