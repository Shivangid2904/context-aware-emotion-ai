"""
train_state_model.py  –  Standalone state-model trainer (backward compatible)
==============================================================================
Uses the embedding backend selected in src/config.py.
Saves artefacts to models/<backend>/ automatically.
"""

import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import config
from data_loader import load_data
from preprocess import clean_data
from feature_engineering import build_text_features, encode_metadata, stack_features


def train_state_model():
    backend = config.TEXT_EMBEDDER
    print(f"Backend: {backend}")

    print("Loading data...")
    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    print("Cleaning data...")
    train_df = clean_data(train_df)
    test_df  = clean_data(test_df)

    print("Building text features...")
    X_train_text, X_test_text, embedder = build_text_features(
        train_df["journal_text"],
        test_df["journal_text"],
    )

    print("Encoding metadata...")
    train_meta, test_meta = encode_metadata(train_df, test_df)

    print("Combining features...")
    X_train = stack_features(X_train_text, train_meta)
    print(f"  Feature matrix shape: {X_train.shape}")

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df["emotional_state"])

    print("Splitting dataset...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_tr, y_tr)

    print("Evaluating model...")
    preds = model.predict(X_val)
    print(classification_report(y_val, preds,
                                target_names=label_encoder.classes_))

    print("Saving models...")
    model_dir = os.path.join("models", backend)
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "state_model.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "state_label_encoder.pkl"))

    vectorizer_key = "vectorizer.pkl" if backend == "tfidf" else "embedder.pkl"
    embedder.save(os.path.join(model_dir, vectorizer_key))

    # ── Backward-compat aliases in models/ root (tfidf only) ──────────────
    if backend == "tfidf":
        joblib.dump(model, "models/state_model.pkl")
        joblib.dump(label_encoder, "models/state_label_encoder.pkl")
        embedder.save("models/tfidf_vectorizer.pkl")

    print("Emotion state model trained and saved successfully!")


if __name__ == "__main__":
    train_state_model()