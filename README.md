# Context-Aware Emotion AI

A machine learning system that understands human emotional signals from reflective text and contextual data, predicts emotional intensity, and recommends supportive actions with uncertainty awareness.

The system combines natural language processing with contextual behavioral signals such as sleep, stress, energy level, and time of day to provide meaningful guidance.

---

# Project Goal

The goal of this system is to:

1. Understand a user's emotional state from messy reflective text.
2. Predict emotional intensity on a **1–5 scale**.
3. Recommend a supportive action and the best time to perform it.
4. Estimate uncertainty when the system is unsure.

The system simulates how a real-world **AI wellness assistant** might interpret user reflections and guide them toward healthier mental states.

---

# System Architecture

User Reflection + Context Signals
↓
TF-IDF Text Features + Metadata Features
↓
Emotion Classification Model
↓
Intensity Prediction Model
↓
Decision Engine (Action + Timing)
↓
Uncertainty Estimation
↓
Final Output → `predictions.csv`

This architecture separates **prediction from reasoning**, which makes the system easier to interpret and extend.

---

# Features Used

## Text Feature

**journal_text**

Processed using **TF-IDF vectorization** to convert reflective text into numerical representations.

TF-IDF emphasizes words that are frequent in a reflection but rare across the dataset, helping the model detect emotional cues such as:

* overwhelmed
* calm
* tired
* anxious
* focused

---

## Contextual Metadata Features

The system also uses contextual signals:

* `sleep_hours`
* `energy_level`
* `stress_level`
* `duration_min`
* `ambience_type`
* `time_of_day`
* `previous_day_mood`
* `face_emotion_hint`
* `reflection_quality`

These signals provide **behavioral context** that improves emotion and intensity prediction.

Example:

Low sleep + high stress → higher probability of negative emotional intensity.

---

# Models Used

## 1. Emotional State Prediction

Model: **XGBoost Classifier**

Reason for choosing XGBoost:

* Handles mixed tabular and sparse features well
* Performs strongly on structured + text feature combinations
* Robust to noisy real-world data
* Efficient training for small–medium datasets

Output:

* `predicted_state`
* probability distribution across emotion classes
* `confidence`

Model performance:

Accuracy ≈ **0.66** across 6 emotional states.

Random baseline would be ≈ **0.16**, so the model performs significantly better than chance.

---

## 2. Intensity Prediction

Model: **RandomForest Regressor**

Intensity prediction is treated as a **regression problem**, not classification.

Reason:

Intensity values represent **ordered emotional strength**:

1 → very calm
5 → very intense emotion

Regression preserves this ordering and predicts continuous values.

---

### Why RandomForest Was Selected

Several models were evaluated:

| Model                  | Result         |
| ---------------------- | -------------- |
| XGBoost Regressor      | MAE ≈ 1.26     |
| Ridge Regression       | MAE ≈ 1.31     |
| RandomForest Regressor | **MAE ≈ 1.21** |

RandomForest performed best because:

* Captures **non-linear relationships** between context features
* Handles mixed sparse + tabular features well
* Less sensitive to overfitting than boosting methods for small datasets

Given the **subjective nature of emotional intensity**, an MAE around 1 is acceptable.

---

# Feature Importance

### Text Features

Text features contribute most to **emotion classification**, since emotional expressions appear directly in reflections.

Examples:

"calm", "tired", "overwhelmed", "stressed"

These words strongly influence predicted emotional state.

---

### Metadata Features

Context features are particularly useful for **intensity prediction**.

Examples:

* high stress + low sleep → stronger negative intensity
* high energy + calm state → readiness for deep work

Thus the system benefits from combining **linguistic signals + behavioral context**.

---

# Ablation Study

To understand the impact of contextual metadata, two experiments were compared.

| Model                    | Accuracy  |
| ------------------------ | --------- |
| Text-only features       | ~0.60     |
| Text + metadata features | **~0.66** |

Result:

Adding contextual signals improves prediction performance by allowing the model to interpret emotional expressions with behavioral context.

Example:

"feeling okay"

Text alone is ambiguous, but metadata (stress, sleep, energy) clarifies emotional state.

---

# Decision Engine

The **Decision Engine** converts predictions into practical recommendations.

Inputs used:

* predicted emotional state
* predicted intensity
* stress level
* energy level
* time of day

Example decision rules:

High stress + high intensity → `box_breathing` now

Calm + high energy → `deep_work` within_15_min

Low energy → `rest`

Neutral mood → `light_planning`

This rule-based layer ensures the system produces **consistent and interpretable actions**.

---

# Uncertainty Modeling

Confidence is derived from model probabilities.

```
confidence = max(class_probability)
```

If confidence is below a threshold:

```
confidence < 0.55
```

Then:

```
uncertain_flag = 1
```

Meaning the system is unsure and may recommend **low-risk supportive actions**.

This prevents overly confident guidance when signals are ambiguous.

---

# Robustness Handling

The system is designed to handle several real-world edge cases.

### Very Short Text

Example:

"ok"

In such cases the system relies more heavily on **metadata features**.

---

### Missing Values

Missing numeric values are filled during preprocessing using default strategies (such as mean or neutral values).

---

### Conflicting Signals

Example:

Low sleep but high energy.

The model evaluates multiple signals simultaneously and uses learned feature weights to determine predictions.

---

# How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Train the models:

```
python src/train_state_model.py
python src/train_intensity_model.py
```

Run prediction pipeline:

```
python src/predict.py
```

Output will be generated at:

```
outputs/predictions.csv
```

---

# Output Format

| Column              | Description               |
| ------------------- | ------------------------- |
| id                  | sample id                 |
| predicted_state     | predicted emotional state |
| predicted_intensity | predicted intensity (1–5) |
| confidence          | model confidence          |
| uncertain_flag      | uncertainty indicator     |
| what_to_do          | recommended action        |
| when_to_do          | recommended timing        |

---

## Running the Streamlit Demo

The project includes a simple Streamlit interface for interacting with the model.

Run locally:

streamlit run src/app.py

# Design Philosophy

The system separates:

Prediction → Reasoning → Action

Machine learning models focus on **understanding emotional state**, while a rule-based reasoning layer converts predictions into **supportive guidance**.

This architecture mirrors real-world AI systems where predictive models are combined with interpretable decision policies.

The goal is not just prediction accuracy, but **helpful, safe, and context-aware user guidance**.
