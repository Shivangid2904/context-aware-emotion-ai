"""Minimal encode diagnostic — run from project root."""
import sys
sys.path.insert(0, "src")

print("Step 1: importing sentence_transformers", flush=True)
from sentence_transformers import SentenceTransformer
print("Step 2: loading model", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Step 3: encoding 3 texts", flush=True)
texts = [
    "I feel overwhelmed and tired.",
    "Today was calm and productive.",
    "Feeling restless and anxious.",
]
vecs = model.encode(texts, show_progress_bar=False)
print(f"Step 4: done — shape={vecs.shape}, dtype={vecs.dtype}", flush=True)
