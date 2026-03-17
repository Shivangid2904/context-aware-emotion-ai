import streamlit as st
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from decision_engine import decide_action
from message_generator import generate_message
from feature_engineering import encode_metadata


# ---------------------------
# Page configuration
# ---------------------------

st.set_page_config(
    page_title="Emotion Insight AI",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Emotion Insight AI")

st.write(
    "Reflect on how you're feeling. "
    "The system will analyze your emotional state and suggest a helpful action."
)


# ---------------------------
# User inputs
# ---------------------------

journal_text = st.text_area(
    "Write a short reflection",
    placeholder="Example: I feel tired and overwhelmed after work today..."
)

col1, col2 = st.columns(2)

with col1:
    stress_level = st.slider("Stress Level", 0, 10, 5)

with col2:
    energy_level = st.slider("Energy Level", 0, 10, 5)

sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

time_of_day = st.selectbox(
    "Time of Day",
    ["morning", "afternoon", "evening", "night"]
)


# ---------------------------
# Load models
# ---------------------------

state_model = joblib.load("models/state_model.pkl")
intensity_model = joblib.load("models/intensity_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
state_encoder = joblib.load("models/state_label_encoder.pkl")


# ---------------------------
# Predict button
# ---------------------------

if st.button("Analyze Emotion"):

    if journal_text.strip() == "":
        st.warning("Please write a reflection first.")
        st.stop()

    # create dataframe with defaults for unused features
    input_df = pd.DataFrame({
        "journal_text": [journal_text],
        "sleep_hours": [sleep_hours],
        "energy_level": [energy_level],
        "stress_level": [stress_level],
        "duration_min": [10],
        "ambience_type": ["neutral"],
        "time_of_day": [time_of_day],
        "previous_day_mood": ["neutral"],
        "face_emotion_hint": ["neutral"],
        "reflection_quality": ["medium"]
    })

    # ---------------------------
    # Text features
    # ---------------------------

    X_text = vectorizer.transform(input_df["journal_text"])

    # ---------------------------
    # Metadata features
    # ---------------------------

    _, input_meta = encode_metadata(input_df, input_df)

    input_meta_sparse = csr_matrix(input_meta.values)

    # combine
    X = hstack([X_text, input_meta_sparse])

    # fix feature size if needed
    expected_features = state_model.n_features_in_

    if X.shape[1] < expected_features:
        padding = csr_matrix((X.shape[0], expected_features - X.shape[1]))
        X = hstack([X, padding])

    # ---------------------------
    # Emotion prediction
    # ---------------------------

    state_probs = state_model.predict_proba(X)
    state_pred = state_probs.argmax(axis=1)

    emotion = state_encoder.inverse_transform(state_pred)[0]
    confidence = state_probs.max()

    # ---------------------------
    # Intensity prediction
    # ---------------------------

    intensity = intensity_model.predict(X)[0]
    intensity = int(np.clip(round(intensity), 1, 5))

    # ---------------------------
    # Decision engine
    # ---------------------------

    action, when = decide_action(
        emotion,
        intensity,
        stress_level,
        energy_level,
        time_of_day
    )

    message = generate_message(emotion, intensity, action)

    # ---------------------------
    # Display results
    # ---------------------------

    st.divider()

    st.subheader("🧠 Emotional Insight")

    col1, col2, col3 = st.columns(3)

    col1.metric("Emotion", emotion)
    col2.metric("Intensity", intensity)
    col3.metric("Confidence", round(confidence, 2))

    st.divider()

    st.subheader("📌 Suggested Action")

    st.success(f"**{action}** — {when}")

    st.subheader("💬 Support Message")

    st.info(message)