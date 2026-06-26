try:
    import torch
    import sentence_transformers
except ImportError:
    pass

import joblib
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text
import config

class BaseTextEmbedder(ABC):
    @abstractmethod
    def fit(self, texts):
        """Fit the embedder on the given texts."""
        pass

    @abstractmethod
    def transform(self, texts):
        """Transform the given texts into embeddings."""
        pass

    @abstractmethod
    def fit_transform(self, texts):
        """Fit and transform the given texts."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the embedder state to filepath."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load the embedder state from filepath."""
        pass


class TfidfEmbedder(BaseTextEmbedder):
    def __init__(self, max_features=5000, stop_words="english", ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=ngram_range
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self


class MiniLMEmbedder(BaseTextEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit(self, texts):
        # No-op for pretrained embeddings
        return self

    def transform(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif hasattr(texts, "tolist"):
            texts = texts.tolist()
        else:
            texts = list(texts)
        return self.model.encode(texts, show_progress_bar=False)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def save(self, filepath):
        # Save structural details to avoid pickling bulky PyTorch weights
        joblib.dump({"backend_type": "minilm", "model_name": self.model_name}, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.model_name = data.get("model_name", "all-MiniLM-L6-v2")
        self._model = None
        return self


def get_text_embedder(backend_type=None, **kwargs):
    """Factory function to instantiate text embedder based on backend type."""
    if backend_type is None:
        backend_type = config.TEXT_EMBEDDER
    
    if backend_type == "tfidf":
        return TfidfEmbedder(**kwargs)
    elif backend_type == "minilm":
        return MiniLMEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown text embedding backend: {backend_type}")


def load_text_embedder(filepath, backend_type=None):
    """Loads a saved text embedder, maintaining backward compatibility with raw TfidfVectorizers."""
    if backend_type is None:
        backend_type = config.TEXT_EMBEDDER

    obj = joblib.load(filepath)

    # Check if it's a dict representing MiniLMEmbedder state
    if isinstance(obj, dict) and obj.get("backend_type") == "minilm":
        return MiniLMEmbedder(model_name=obj.get("model_name", "all-MiniLM-L6-v2"))

    # Backward compatibility check for raw TfidfVectorizer
    if isinstance(obj, sklearn.feature_extraction.text.TfidfVectorizer):
        embedder = TfidfEmbedder()
        embedder.vectorizer = obj
        return embedder

    # If it's already an instance of BaseTextEmbedder, return it directly
    if isinstance(obj, BaseTextEmbedder):
        return obj

    # Fallback return
    return obj
