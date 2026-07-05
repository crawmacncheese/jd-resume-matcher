import re
from typing import Callable

import numpy as np

from models.datapoints import Skill

_TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class TfidfEmbeddingModel:
    """Lightweight local embeddings for keyword cosine similarity (no external API)."""

    def __init__(self, corpus: list[str]) -> None:
        unique_corpus = list(dict.fromkeys(corpus))
        self.corpus = unique_corpus
        self.doc_tokens = [_tokenize(text) for text in unique_corpus]
        self.vocab = sorted({token for tokens in self.doc_tokens for token in tokens})
        self.vocab_index = {word: index for index, word in enumerate(self.vocab)}

        document_count = max(len(unique_corpus), 1)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for index, word in enumerate(self.vocab):
            document_frequency = sum(1 for tokens in self.doc_tokens if word in tokens)
            self.idf[index] = np.log((1 + document_count) / (1 + document_frequency)) + 1

        self.text_to_vector = {
            text: self._vectorize(tokens)
            for text, tokens in zip(unique_corpus, self.doc_tokens)
        }

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            index = self.vocab_index.get(token)
            if index is None:
                continue
            tf = count / len(tokens)
            vector[index] = tf * self.idf[index]
        return vector

    def embed(self, text: str) -> np.ndarray:
        return self.text_to_vector.get(text, self._vectorize(_tokenize(text)))


def build_local_embedding_fn(skills: list[Skill]) -> Callable[[str], np.ndarray]:
    model = TfidfEmbeddingModel([skill.name for skill in skills])
    return model.embed
