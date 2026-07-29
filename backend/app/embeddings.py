"""Embedding providers.

tfidf   -- default, no API key needed, fit locally on the repo's own text.
openai  -- real dense embeddings via OpenAI's embeddings API, used when
           OPENAI_API_KEY is set and EMBEDDING_PROVIDER=openai.

Both implement the same interface (`fit_transform` / `transform`) so `search.py`
and `vectorstore.py` don't need to know which one is active.
"""
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


class TfidfEmbedder:
    name = "tfidf"
    dim = None  # variable, sized to vocabulary at fit time

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=20000)
        self._fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        self.dim = matrix.shape[1]
        return matrix.toarray().astype("float32")

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit before transform")
        return self.vectorizer.transform(texts).toarray().astype("float32")


class OpenAIEmbedder:
    name = "openai"

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use openai embedding provider")
        import openai
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_EMBEDDING_MODEL
        self.dim = 1536 if "3-small" in self.model else 3072

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        # OpenAI embedding inputs must be non-empty
        texts = [t if t.strip() else " " for t in texts]
        vectors = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors += [d.embedding for d in resp.data]
        return np.array(vectors, dtype="float32")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self._embed_batch(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._embed_batch(texts)


def get_embedder():
    if config.EMBEDDING_PROVIDER == "openai":
        try:
            return OpenAIEmbedder()
        except RuntimeError:
            pass  # fall through to tfidf if key missing
    return TfidfEmbedder()
