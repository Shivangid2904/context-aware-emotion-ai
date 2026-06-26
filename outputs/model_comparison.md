# Embedding Backend Comparison Report

Generated automatically by `src/train.py`.


## Backend: `tfidf`

### Classification (Emotion State)

| Metric | Value |
| --- | --- |
| Accuracy | 0.6625 |
| Precision (macro) | 0.6621 |
| Recall (macro) | 0.666 |
| F1 Macro | 0.6628 |
| F1 Weighted | 0.6626 |
| Training time | 2.46s |
| Feature dimension | 1622 |
| Model size | 2.338 MB |

### Regression (Intensity)

| Metric | Value |
| --- | --- |
| MAE | 1.2169 |
| RMSE | 1.4273 |
| R2 | -0.0204 |
| Training time | 2.55s |
| Model size | 2.196 MB |

---

## Backend: `minilm`

### Classification (Emotion State)

| Metric | Value |
| --- | --- |
| Accuracy | 0.6 |
| Precision (macro) | 0.6058 |
| Recall (macro) | 0.6081 |
| F1 Macro | 0.5996 |
| F1 Weighted | 0.6026 |
| Training time | 19.1s |
| Feature dimension | 415 |
| Model size | 2.766 MB |

### Regression (Intensity)

| Metric | Value |
| --- | --- |
| MAE | 1.2583 |
| RMSE | 1.4736 |
| R2 | -0.0877 |
| Training time | 36.83s |
| Model size | 5.837 MB |

---
