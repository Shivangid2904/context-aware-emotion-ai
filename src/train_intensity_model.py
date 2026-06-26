"""
train_intensity_model.py  –  Standalone intensity-model trainer (backward compatible)
======================================================================================
Uses the embedding backend selected in src/config.py.
Requires that train_state_model.py (or train.py) has already been run first
so the fitted embedder is available on disk.
"""

import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import config
from text_embedder import load_text_embedder
from data_loader import load_data
from preprocess import clean_data
from feature_engineering import encode_metadata, stack_features


def train_intensity_model():
    backend = config.TEXT_EMBEDDER
    print(f"Backend: {backend}")

    print("Loading data...")
    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    print("Cleaning data...")
    train_df = clean_data(train_df)
    test_df  = clean_data(test_df)

    # ── Load the fitted embedder saved by train_state_model ───────────────
    model_dir = os.path.join("models", backend)
    vectorizer_key = "vectorizer.pkl" if backend == "tfidf" else "embedder.pkl"
    embedder_path  = os.path.join(model_dir, vectorizer_key)

    # Fallback: root-level tfidf_vectorizer.pkl for backward compat
    if not os.path.exists(embedder_path):
        embedder_path = "models/tfidf_vectorizer.pkl"

    print(f"Loading embedder from {embedder_path}...")
    vectorizer = load_text_embedder(embedder_path)

    print("Transforming text...")
    X_train_text = vectorizer.transform(train_df["journal_text"])

    print("Encoding metadata...")
    train_meta, test_meta = encode_metadata(train_df, test_df)

    print("Combining features...")
    X_train = stack_features(X_train_text, train_meta)
    print(f"  Feature matrix shape: {X_train.shape}")

    y = train_df["intensity"]

    print("Splitting dataset...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, random_state=42
    )

    print("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    print("Evaluating model...")
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print("MAE:", round(mae, 4))

    print("Saving model...")
    os.makedirs(model_dir, exist_ok=True)
    intensity_path = os.path.join(model_dir, "intensity_model.pkl")
    joblib.dump(model, intensity_path)

    # ── Backward-compat alias in models/ root (tfidf only) ────────────────
    if backend == "tfidf":
        joblib.dump(model, "models/intensity_model.pkl")

    print("Intensity model trained and saved successfully!")


if __name__ == "__main__":
    train_intensity_model()