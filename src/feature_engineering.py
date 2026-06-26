import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, csr_matrix
from text_embedder import get_text_embedder


def build_text_features(train_text, test_text):
    """Fit an embedder on train_text and transform both splits.

    Returns (X_train_text, X_test_text, embedder).
    Works for both TF-IDF (sparse) and MiniLM (dense numpy).
    """
    embedder = get_text_embedder()
    X_train_text = embedder.fit_transform(train_text)
    X_test_text = embedder.transform(test_text)
    return X_train_text, X_test_text, embedder


def encode_metadata(train_df, test_df):
    """One-hot encode categorical columns and pass numeric columns through.

    Returns aligned (train_meta, test_meta) DataFrames with identical columns.
    """
    metadata_cols = [
        "ambience_type",
        "time_of_day",
        "previous_day_mood",
        "face_emotion_hint",
        "reflection_quality"
    ]
    numeric_cols = [
        "sleep_hours",
        "energy_level",
        "stress_level",
        "duration_min"
    ]
    train_meta = pd.get_dummies(train_df[metadata_cols + numeric_cols]).astype(float)
    test_meta = pd.get_dummies(test_df[metadata_cols + numeric_cols]).astype(float)
    train_meta, test_meta = train_meta.align(
        test_meta,
        join="left",
        axis=1,
        fill_value=0
    )
    return train_meta, test_meta


def stack_features(text_features, meta_df):
    """Horizontally stack text embeddings with metadata features.

    Handles both:
      - Sparse TF-IDF arrays  → scipy hstack stays sparse
      - Dense MiniLM arrays   → numpy hstack, meta converted to dense
    """
    meta_array = meta_df.values.astype(float)

    if issparse(text_features):
        meta_sparse = csr_matrix(meta_array)
        return hstack([text_features, meta_sparse])
    else:
        # Dense path (MiniLM)
        return np.hstack([text_features, meta_array])