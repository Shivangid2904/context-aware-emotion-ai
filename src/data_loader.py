import pandas as pd
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
if __name__ == "__main__":
    train_df, test_df = load_data(
        "data/train.csv",
        "data/test.csv"
    )
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nColumns:")
    print(train_df.columns)
    print("\nSample rows:")
    print(train_df.head())