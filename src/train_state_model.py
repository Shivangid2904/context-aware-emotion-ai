import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

from data_loader import load_data
from preprocess import clean_data
from feature_engineering import build_text_features, encode_metadata


def train_state_model():

    print("Loading data...")

    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    print("Cleaning data...")

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("Building text features...")

    X_train_text, X_test_text, vectorizer = build_text_features(
        train_df["journal_text"],
        test_df["journal_text"]
    )

    print("Encoding metadata...")

    train_meta, test_meta = encode_metadata(train_df, test_df)

    print("Combining features...")

    X_train = hstack([X_train_text, train_meta])

    print("Encoding labels...")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_df["emotional_state"])

    print("Splitting dataset...")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training model...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_tr, y_tr)

    print("Evaluating model...")

    preds = model.predict(X_val)

    print(classification_report(y_val, preds))

    print("Saving models...")

    joblib.dump(model, "models/state_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "models/state_label_encoder.pkl")

    print("Emotion state model trained and saved successfully!")


if __name__ == "__main__":
    train_state_model()