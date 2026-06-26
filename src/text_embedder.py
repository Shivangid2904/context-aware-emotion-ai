import joblib
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text
from config import TEXT_EMBEDDER

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


def get_text_embedder(backend_type=None, **kwargs):
    """Factory function to instantiate text embedder based on backend type."""
    if backend_type is None:
        backend_type = TEXT_EMBEDDER
    
    if backend_type == "tfidf":
        return TfidfEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown text embedding backend: {backend_type}")


def load_text_embedder(filepath, backend_type=None):
    """Loads a saved text embedder, maintaining backward compatibility with raw TfidfVectorizers."""
    if backend_type is None:
        backend_type = TEXT_EMBEDDER

    obj = joblib.load(filepath)

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
