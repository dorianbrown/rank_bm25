import numpy as np

SIMPLE_CORPUS = [
    ["the", "cat", "sat"],
    ["the", "dog", "ran", "fast"],
    ["a", "cat", "and", "a", "dog"],
    ["the", "cat", "sat", "on", "the", "mat"],
]


def build_test_corpus(n_docs=200, n_terms=50):
    import random
    random.seed(42)
    vocab = [f"term_{i}" for i in range(n_terms)]
    corpus = []
    for _ in range(n_docs):
        doc = [random.choice(vocab) for _ in range(random.randint(5, 30))]
        corpus.append(doc)
    return corpus, vocab


def ref_okapi_score(query, doc_freqs, idf, doc_len, avgdl, k1, b):
    N = len(doc_freqs)
    score = np.zeros(N, dtype=np.float64)
    len_norm = k1 * (1 - b + b * doc_len / avgdl)
    for di in range(N):
        for t in query:
            tf = doc_freqs[di].get(t, 0)
            if tf == 0:
                continue
            score[di] += idf.get(t, 0) * tf * (k1 + 1) / (tf + len_norm[di])
    return score.astype(np.float32)


def ref_bm25l_score(query, doc_freqs, idf, doc_len, avgdl, k1, b, delta):
    N = len(doc_freqs)
    score = np.zeros(N, dtype=np.float64)
    _lambda = 1 - b + b * doc_len / avgdl
    for di in range(N):
        for t in query:
            tf = doc_freqs[di].get(t, 0)
            if tf == 0:
                continue
            tidf = idf.get(t, 0)
            score[di] += tidf * (k1 + 1) * (tf + delta * _lambda[di]) / (tf + _lambda[di] * (k1 + delta))
    return score.astype(np.float32)


def ref_bm25l_score_full(query, doc_freqs, idf, doc_len, avgdl, k1, b, delta):
    N = len(doc_freqs)
    score = np.zeros(N, dtype=np.float64)
    _lambda = 1 - b + b * doc_len / avgdl
    for di in range(N):
        for t in query:
            tf = doc_freqs[di].get(t, 0)
            tidf = idf.get(t, 0)
            score[di] += tidf * (k1 + 1) * (tf + delta * _lambda[di]) / (tf + _lambda[di] * (k1 + delta))
    return score.astype(np.float32)


def ref_bm25plus_score(query, doc_freqs, idf, doc_len, avgdl, k1, b, delta):
    N = len(doc_freqs)
    score = np.zeros(N, dtype=np.float64)
    len_norm = k1 * (1 - b + b * doc_len / avgdl)
    bonus = sum(idf.get(t, 0) for t in query) * delta
    for di in range(N):
        for t in query:
            tf = doc_freqs[di].get(t, 0)
            if tf == 0:
                continue
            tidf = idf.get(t, 0)
            score[di] += tidf * tf * (k1 + 1) / (len_norm[di] + tf)
        score[di] += bonus
    return score.astype(np.float32)


def batch_scores_from_full(bm25, query, doc_ids):
    full = bm25.get_scores(query)
    return np.array([full[di] for di in doc_ids], dtype=np.float32)


def _check_c_accel(bm25):
    import pytest
    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")


def _skip_if_no_c_accel(bm25):
    import pytest
    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")
