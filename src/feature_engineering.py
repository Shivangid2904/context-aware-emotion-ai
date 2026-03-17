import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_text_features(train_text, test_text):
    vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1,2)
)
    X_train_text = vectorizer.fit_transform(train_text)
    X_test_text = vectorizer.transform(test_text)
    return X_train_text, X_test_text, vectorizer

def encode_metadata(train_df, test_df):
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