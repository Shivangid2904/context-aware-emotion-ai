import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

from data_loader import load_data
from preprocess import clean_data
from feature_engineering import encode_metadata
from decision_engine import decide_action
from uncertainty import compute_uncertainty


def run_prediction():

    print("Loading data...")

    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    test_df = clean_data(test_df)

    print("Loading models...")

    state_model = joblib.load("models/state_model.pkl")
    state_encoder = joblib.load("models/state_label_encoder.pkl")
    intensity_model = joblib.load("models/intensity_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    print("Generating features...")

# Text features
    X_test_text = vectorizer.transform(test_df["journal_text"])

# Metadata encoding
    train_meta, test_meta = encode_metadata(train_df, test_df)

# Ensure identical column structure
    test_meta = test_meta.reindex(columns=train_meta.columns, fill_value=0)

    from scipy.sparse import csr_matrix
    test_meta_sparse = csr_matrix(test_meta.values)

# Combine features
    X_test = hstack([X_test_text, test_meta_sparse])

# Ensure feature count matches model expectation
    expected_features = state_model.n_features_in_

    if X_test.shape[1] < expected_features:
     from scipy.sparse import csr_matrix
     padding = csr_matrix((X_test.shape[0], expected_features - X_test.shape[1]))
     X_test = hstack([X_test, padding])

    print("Predicting emotional state...")

    state_probs = state_model.predict_proba(X_test)

    state_preds = state_probs.argmax(axis=1)

    predicted_state = state_encoder.inverse_transform(state_preds)

    confidence, uncertain_flag = compute_uncertainty(state_probs)

    print("Predicting intensity...")

    predicted_intensity = intensity_model.predict(X_test)

    predicted_intensity = np.clip(np.round(predicted_intensity), 1, 5)

    print("Running decision engine...")

    actions = []
    times = []

    for i, row in test_df.iterrows():

        action, when = decide_action(
            predicted_state[i],
            predicted_intensity[i],
            row["stress_level"],
            row["energy_level"],
            row["time_of_day"]
        )

        actions.append(action)
        times.append(when)

    print("Saving predictions...")

    results = pd.DataFrame({
        "id": test_df["id"],
        "predicted_state": predicted_state,
        "predicted_intensity": predicted_intensity,
        "confidence": confidence,
        "uncertain_flag": uncertain_flag,
        "what_to_do": actions,
        "when_to_do": times
    })

    results.to_csv("outputs/predictions.csv", index=False)

    print("predictions.csv created successfully")
if __name__ == "__main__":
    run_prediction()