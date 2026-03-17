import pandas as pd
def clean_data(df):
    df = df.copy()
    # Fill missing journal text
    df["journal_text"] = df["journal_text"].fillna("")
    # Numeric columns
    numeric_cols = [
        "sleep_hours",
        "energy_level",
        "stress_level",
        "duration_min"
    ]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = [
        "ambience_type",
        "time_of_day",
        "previous_day_mood",
        "face_emotion_hint",
        "reflection_quality"
    ]
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")
    return df