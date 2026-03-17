import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.sparse import hstack

from data_loader import load_data
from preprocess import clean_data
from feature_engineering import encode_metadata


def train_intensity_model():

    print("Loading data...")

    train_df, test_df = load_data("data/train.csv", "data/test.csv")

    print("Cleaning data...")

    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("Loading TF-IDF vectorizer...")

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    print("Transforming text...")

    X_train_text = vectorizer.transform(train_df["journal_text"])
    X_test_text = vectorizer.transform(test_df["journal_text"])

    print("Encoding metadata...")

    train_meta, test_meta = encode_metadata(train_df, test_df)

    print("Combining features...")

    X_train = hstack([X_train_text, train_meta])

    y = train_df["intensity"]

    print("Splitting dataset...")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Training RandomForest model...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    print("Evaluating model...")

    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)

    print("MAE:", mae)

    joblib.dump(model, "models/intensity_model.pkl")

    print("Intensity model trained and saved successfully!")


if __name__ == "__main__":
    train_intensity_model()