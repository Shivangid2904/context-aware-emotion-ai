"""
compare.py  –  Train both TF-IDF and MiniLM pipelines and generate a report
=============================================================================
This script trains each backend independently, collects evaluation metrics,
and saves a Markdown comparison report to outputs/model_comparison.md.

Usage:
    python src/compare.py
"""

try:
    import torch
    import sentence_transformers
except ImportError:
    pass

import sys
import os

# Ensure src/ is on the path when called from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train import run_training, save_comparison_report, print_section


BACKENDS = ["tfidf", "minilm"]


def main():
    print_section("Dual-Backend Training & Comparison")
    results = {}

    for backend in BACKENDS:
        print(f"\n>>> Training backend: {backend} <<<\n")
        # Temporarily override config so run_training uses the right backend
        import config as _cfg
        _cfg.TEXT_EMBEDDER = backend

        # Also patch the imported symbol in feature_engineering and text_embedder
        import feature_engineering as _fe
        import text_embedder as _te
        _fe_embedder_backup = None   # not needed, they read config at call time

        results[backend] = run_training(backend)

    report_path = save_comparison_report(results)

    print_section("Summary")
    header = f"{'Metric':<22} {'TF-IDF':>10} {'MiniLM':>10}"
    print(header)
    print("-" * len(header))

    clf_metrics = ["accuracy", "precision", "recall", "f1_macro", "f1_weighted",
                   "train_time_s", "feature_dim"]
    reg_metrics = ["mae", "rmse", "r2", "train_time_s"]

    print("\n-- Classification (Emotion State) --")
    for m in clf_metrics:
        tfidf_val = results["tfidf"]["state"].get(m, "n/a")
        minilm_val = results["minilm"]["state"].get(m, "n/a")
        print(f"  {m:<20} {str(tfidf_val):>10} {str(minilm_val):>10}")

    print("\n-- Regression (Intensity) --")
    for m in reg_metrics:
        tfidf_val = results["tfidf"]["intensity"].get(m, "n/a")
        minilm_val = results["minilm"]["intensity"].get(m, "n/a")
        print(f"  {m:<20} {str(tfidf_val):>10} {str(minilm_val):>10}")

    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
