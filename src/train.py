try:
    import torch
    import sentence_transformers
except ImportError:
    pass

import os
# Disable HuggingFace tokenizer parallelism to prevent Windows multiprocess crashes
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

"""
train.py  -  Dual-backend training pipeline
============================================
Trains both the Emotion-State classifier (XGBoost) and the
Intensity regressor (RandomForest) for whichever embedding
backend is selected in src/config.py.

Usage
-----
# TF-IDF (default)
python src/train.py

# MiniLM
# 1.  Edit src/config.py  ->  TEXT_EMBEDDER = "minilm"
# 2.  python src/train.py

Artefacts are written to:
    models/tfidf/  or  models/minilm/
depending on the active backend.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
)

import config
from data_loader import load_data
from preprocess import clean_data
from feature_engineering import build_text_features, encode_metadata, stack_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_size_mb(filepath):
    """Return file size in MB, or None if the file doesn't exist."""
    try:
        return round(os.path.getsize(filepath) / (1024 * 1024), 3)
    except FileNotFoundError:
        return None


def print_section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# State-model training
# ---------------------------------------------------------------------------

def train_state_pipeline(train_df, test_df, backend):
    """Train XGBoost emotion-state classifier.

    Returns a dict of metrics + artefact paths.
    """
    print_section(f"[{backend.upper()}] Emotion State Classifier")

    # -- Text features ----------------------------------------------------
    print("Building text features...")
    t0 = time.time()
    X_train_text, X_test_text, embedder = build_text_features(
        train_df["journal_text"],
        test_df["journal_text"],
    )
    text_time = time.time() - t0
    print(f"  Text features ready in {text_time:.1f}s")

    # -- Metadata ----------------------------------------------------------
    print("Encoding metadata...")
    train_meta, test_meta = encode_metadata(train_df, test_df)

    # -- Stack ------------------------------------------------------------
    print("Combining features...")
    X_train = stack_features(X_train_text, train_meta)
    feature_dim = X_train.shape[1]
    print(f"  Feature matrix shape: {X_train.shape}")

    # -- Labels -----------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df["emotional_state"])

    # -- Split ------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, random_state=42
    )

    # -- Train ------------------------------------------------------------
    print("Training XGBoost classifier...")
    t0 = time.time()
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
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")

    # -- Evaluate ---------------------------------------------------------
    preds = model.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, preds,
                                target_names=label_encoder.classes_))

    metrics = {
        "accuracy":     round(accuracy_score(y_val, preds), 4),
        "precision":    round(precision_score(y_val, preds, average="macro", zero_division=0), 4),
        "recall":       round(recall_score(y_val, preds, average="macro", zero_division=0), 4),
        "f1_macro":     round(f1_score(y_val, preds, average="macro", zero_division=0), 4),
        "f1_weighted":  round(f1_score(y_val, preds, average="weighted", zero_division=0), 4),
        "train_time_s": round(train_time, 2),
        "feature_dim":  int(feature_dim),
    }

    # -- Save -------------------------------------------------------------
    model_dir = os.path.join("models", backend)
    os.makedirs(model_dir, exist_ok=True)

    state_path     = os.path.join(model_dir, "state_model.pkl")
    encoder_path   = os.path.join(model_dir, "state_label_encoder.pkl")
    vectorizer_key = "vectorizer.pkl" if backend == "tfidf" else "embedder.pkl"
    embedder_path  = os.path.join(model_dir, vectorizer_key)

    joblib.dump(model, state_path)
    joblib.dump(label_encoder, encoder_path)
    embedder.save(embedder_path)

    metrics["state_model_mb"] = model_size_mb(state_path)
    print(f"\nSaved -> {state_path}  ({metrics['state_model_mb']} MB)")

    return metrics, embedder, train_meta.columns.tolist()


# ---------------------------------------------------------------------------
# Intensity-model training
# ---------------------------------------------------------------------------

def train_intensity_pipeline(train_df, test_df, backend, embedder, meta_columns):
    """Train RandomForest intensity regressor.

    Reuses the already-fitted embedder from state-model training.
    Returns a dict of metrics.
    """
    print_section(f"[{backend.upper()}] Intensity Regressor")

    # -- Text features (reuse fitted embedder) ----------------------------
    print("Transforming text with fitted embedder...")
    t0 = time.time()
    X_train_text = embedder.transform(train_df["journal_text"])
    print(f"  Transform complete in {time.time() - t0:.1f}s")

    # -- Metadata ----------------------------------------------------------
    print("Encoding metadata...")
    train_meta, _ = encode_metadata(train_df, test_df)
    # Align to the same columns used during state training
    train_meta = train_meta.reindex(columns=meta_columns, fill_value=0)

    # -- Stack ------------------------------------------------------------
    print("Combining features...")
    X_train = stack_features(X_train_text, train_meta)
    y = train_df["intensity"]

    # -- Split ------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, random_state=42
    )

    # -- Train ------------------------------------------------------------
    print("Training RandomForest regressor...")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.1f}s")

    # -- Evaluate ---------------------------------------------------------
    preds = model.predict(X_val)
    mae  = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    r2   = r2_score(y_val, preds)

    print(f"\n  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}")

    metrics = {
        "mae":          round(mae, 4),
        "rmse":         round(rmse, 4),
        "r2":           round(r2, 4),
        "train_time_s": round(train_time, 2),
    }

    # -- Save -------------------------------------------------------------
    model_dir = os.path.join("models", backend)
    os.makedirs(model_dir, exist_ok=True)
    intensity_path = os.path.join(model_dir, "intensity_model.pkl")
    joblib.dump(model, intensity_path)

    metrics["intensity_model_mb"] = model_size_mb(intensity_path)
    print(f"\nSaved -> {intensity_path}  ({metrics['intensity_model_mb']} MB)")

    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_comparison_report(results):
    """Write a Markdown comparison report to outputs/."""
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/model_comparison.md"

    lines = [
        "# Embedding Backend Comparison Report\n",
        f"Generated automatically by `src/train.py`.\n",
        "",
    ]

    for backend, data in results.items():
        sm = data["state"]
        im = data["intensity"]

        lines += [
            f"## Backend: `{backend}`\n",
            "### Classification (Emotion State)\n",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Accuracy | {sm['accuracy']} |",
            f"| Precision (macro) | {sm['precision']} |",
            f"| Recall (macro) | {sm['recall']} |",
            f"| F1 Macro | {sm['f1_macro']} |",
            f"| F1 Weighted | {sm['f1_weighted']} |",
            f"| Training time | {sm['train_time_s']}s |",
            f"| Feature dimension | {sm['feature_dim']} |",
            f"| Model size | {sm['state_model_mb']} MB |",
            "",
            "### Regression (Intensity)\n",
            "| Metric | Value |",
            "| --- | --- |",
            f"| MAE | {im['mae']} |",
            f"| RMSE | {im['rmse']} |",
            f"| R2 | {im['r2']} |",
            f"| Training time | {im['train_time_s']}s |",
            f"| Model size | {im['intensity_model_mb']} MB |",
            "",
            "---",
            "",
        ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nComparison report saved -> {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training(backend=None):
    """Train both models for the given backend (default: from config.py)."""
    if backend is None:
        backend = config.TEXT_EMBEDDER

    print_section(f"Starting training pipeline  [ backend = {backend} ]")

    # -- Load & clean data -------------------------------------------------
    print("Loading data...")
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    train_df = clean_data(train_df)
    test_df  = clean_data(test_df)

    # -- State model -------------------------------------------------------
    state_metrics, embedder, meta_columns = train_state_pipeline(
        train_df, test_df, backend
    )

    # -- Intensity model ---------------------------------------------------
    intensity_metrics = train_intensity_pipeline(
        train_df, test_df, backend, embedder, meta_columns
    )

    return {"state": state_metrics, "intensity": intensity_metrics}


if __name__ == "__main__":
    results = {}

    # Always train the active backend from config.py
    backend_to_train = config.TEXT_EMBEDDER
    results[backend_to_train] = run_training(backend_to_train)

    save_comparison_report(results)

    print_section("Training complete")
    sm = results[backend_to_train]["state"]
    im = results[backend_to_train]["intensity"]
    print(f"  Backend       : {backend_to_train}")
    print(f"  Accuracy      : {sm['accuracy']}")
    print(f"  F1 Weighted   : {sm['f1_weighted']}")
    print(f"  MAE (intensity): {im['mae']}")
    print(f"  RMSE          : {im['rmse']}")
    print(f"  R2            : {im['r2']}")
