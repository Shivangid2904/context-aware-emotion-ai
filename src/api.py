from fastapi import FastAPI
import joblib
import numpy as np
from scipy.sparse import hstack
import pandas as pd

app = FastAPI()

state_model = joblib.load("models/state_model.pkl")
intensity_model = joblib.load("models/intensity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.post("/predict")

def predict(data: dict):

    text = data["journal_text"]

    X_text = vectorizer.transform([text])

    # simplified metadata
    meta = np.array([[data["stress_level"], data["energy_level"]]])

    X = hstack([X_text, meta])

    state = state_model.predict(X)[0]

    intensity = intensity_model.predict(X)[0]

    return {
        "predicted_state": str(state),
        "predicted_intensity": int(round(intensity))
    }