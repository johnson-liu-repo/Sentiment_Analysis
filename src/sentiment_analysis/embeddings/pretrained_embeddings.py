from __future__ import annotations

import os
import io
import numpy as np
from typing import Iterable, Dict, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional: use gensim if available for fast downloads/loading
try:
    import gensim.downloader as api
    from gensim.models.keyedvectors import KeyedVectors
except Exception:
    api = None
    KeyedVectors = None


def build_tfidf_and_vocab(texts: Iterable[str], lowercase: bool = True) -> Tuple[TfidfVectorizer, Set[str]]:
    """
    Fit a TF-IDF vectorizer and return both the fitted vectorizer and the set of feature names (vocabulary).
    """
    vectorizer = TfidfVectorizer(lowercase=lowercase, tokenizer=str.split, token_pattern=None)
    vectorizer.fit(texts)
    vocab = set(vectorizer.get_feature_names_out().tolist())
    return vectorizer, vocab


def _load_glove_txt(glove_path: str, restrict_vocab: Set[str] | None, expected_dim: int) -> Dict[str, np.ndarray]:
    """
    Load a (possibly huge) GloVe .txt file (e.g., glove.6B.300d.txt) while restricting to the given vocabulary.
    Returns a dict word -> vector (np.float32).
    """
    if not os.path.exists(glove_path):
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    vectors: Dict[str, np.ndarray] = {}
    with io.open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            if restrict_vocab is not None and word not in restrict_vocab:
                continue
            vals = parts[1:]
            if expected_dim is not None and len(vals) != expected_dim:
                # Skip rows that don't match expected dim (some files have header rows or anomalies)
                continue
            try:
                vec = np.asarray(vals, dtype=np.float32)
            except ValueError:
                # Corrupted line, skip
                continue
            vectors[word] = vec
    return vectors


class KVWrapper:
    """
    A tiny wrapper so we can treat either gensim KeyedVectors or a dict[str, np.ndarray] the same way.
    """
    def __init__(self, obj, vector_size: int):
        self.obj = obj
        self.vector_size = vector_size

    def __contains__(self, key: str) -> bool:
        if isinstance(self.obj, dict):
            return key in self.obj
        return key in self.obj.key_to_index

    def get_vec(self, key: str) -> np.ndarray | None:
        if isinstance(self.obj, dict):
            return self.obj.get(key, None)
        try:
            return self.obj.get_vector(key)
        except KeyError:
            return None


def load_pretrained_vectors(
    prefer: str = "glove-wiki-gigaword-300",
    glove_path: str | None = None,
    restrict_vocab: Set[str] | None = None,
    expected_dim: int = 300
) -> KVWrapper:
    """
    Try to load gensim 'glove-wiki-gigaword-300' (downloads automatically).
    If that's not available (e.g., offline), fall back to a local GloVe .txt.
    The loader restricts to 'restrict_vocab' to keep RAM in check.
    """
    # 1) Try gensim downloader if available
    if api is not None:
        try:
            kv = api.load(prefer)  # e.g., 'glove-wiki-gigaword-300'
            # If we have a restricted vocab, subset it to save memory
            if restrict_vocab is not None:
                keep = [w for w in restrict_vocab if w in kv.key_to_index]
                # Build a new tiny KeyedVectors with just the kept words
                from gensim.models import KeyedVectors as _KV
                small_kv = _KV(vector_size=kv.vector_size)
                small_kv.add_vectors(keep, [kv.get_vector(w) for w in keep])
                return KVWrapper(small_kv, vector_size=kv.vector_size)
            return KVWrapper(kv, vector_size=kv.vector_size)
        except Exception:
            pass  # fall back to local

    # 2) Fall back to local GloVe .txt
    if glove_path is None:
        raise RuntimeError(
            "Could not load embeddings via gensim and no local --glove_path provided."
        )
    print(f"Falling back to local GloVe at: {glove_path}")
    vecs = _load_glove_txt(glove_path, restrict_vocab=restrict_vocab, expected_dim=expected_dim)
    if not vecs:
        raise RuntimeError(
            "Loaded 0 vectors from the provided GloVe file. Check the path and ensure the expected dimension matches."
        )
    return KVWrapper(vecs, vector_size=expected_dim)


def tfidf_weighted_average_embeddings(
    texts: Iterable[str],
    vectorizer: TfidfVectorizer,
    kv: KVWrapper,
    embedding_dim: int
) -> np.ndarray:
    """
    Compute TF-IDF–weighted average embedding for each text in 'texts' using the fitted 'vectorizer' and embeddings 'kv'.
    Returns an array of shape [num_texts, embedding_dim].
    """
    tfidf = vectorizer.transform(texts)  # sparse CSR: [n_docs, n_terms]
    feature_names = vectorizer.get_feature_names_out()
    feat_index_to_token = np.array(feature_names)

    n_docs = tfidf.shape[0]
    X = np.zeros((n_docs, embedding_dim), dtype=np.float32)

    for i in range(n_docs):
        row = tfidf[i]
        if row.nnz == 0:
            continue
        idxs = row.indices
        weights = row.data
        # Accumulate weighted sum
        total_w = 0.0
        accum = np.zeros(embedding_dim, dtype=np.float32)
        for j, w in zip(idxs, weights):
            token = feat_index_to_token[j]
            if token in kv:
                vec = kv.get_vec(token)
                if vec is not None:
                    accum += (w * vec.astype(np.float32))
                    total_w += w
        if total_w > 0:
            X[i] = accum / total_w  # weighted average
        # else: remains zero vector (no known tokens)
    return X


def save_vectorized_comments(X: np.ndarray, labels: np.ndarray, X_path: str, y_path: str) -> None:
    os.makedirs(os.path.dirname(X_path), exist_ok=True)
    np.save(X_path, X)
    np.save(y_path, labels)