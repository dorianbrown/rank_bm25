import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from _helpers import SIMPLE_CORPUS, build_test_corpus


def test_get_top_n_returns_n_documents():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    corpus_copy = [" ".join(d) for d in SIMPLE_CORPUS]
    top = bm25.get_top_n(["cat", "the"], corpus_copy, n=2)
    assert len(top) == 2
    assert all(isinstance(d, str) for d in top)


def test_get_top_n_all_docs_when_n_exceeds_corpus():
    bm25 = BM25L(SIMPLE_CORPUS)
    corpus_copy = ["d0", "d1", "d2", "d3"]
    top = bm25.get_top_n(["the"], corpus_copy, n=10)
    assert len(top) == 4


def test_get_top_n_sorted_by_score():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    scores = bm25.get_scores(["cat", "the"])
    corpus_copy = ["d0", "d1", "d2", "d3"]
    top = bm25.get_top_n(["cat", "the"], corpus_copy, n=2)
    top_indices = [corpus_copy.index(d) for d in top]
    top_scores = scores[top_indices]
    assert top_scores[0] >= top_scores[1]
    top_n_indices_from_scores = set(np.argsort(scores)[::-1][:2])
    assert set(top_indices) == top_n_indices_from_scores


def test_get_top_n_exact_corpus_size():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    corpus_copy = ["d0", "d1", "d2", "d3"]
    top = bm25.get_top_n(["the"], corpus_copy, n=4)
    assert len(top) == 4
    scores = bm25.get_scores(["the"])
    top_indices = [corpus_copy.index(d) for d in top]
    top_scores = scores[top_indices]
    for i in range(len(top_scores) - 1):
        assert top_scores[i] >= top_scores[i + 1]


def test_get_top_n_corpus_size_mismatch():
    import pytest
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    with pytest.raises(AssertionError):
        bm25.get_top_n(["the"], ["only", "two"], n=1)


def test_get_top_n_empty_query():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    corpus_copy = ["d0", "d1", "d2", "d3"]
    top = bm25.get_top_n([], corpus_copy, n=2)
    assert len(top) == 2


def test_get_top_n_larger_random():
    import random
    random.seed(99)
    corpus, vocab = build_test_corpus(n_docs=100, n_terms=15)
    bm25 = BM25Okapi(corpus)
    query = [random.choice(vocab) for _ in range(5)]
    corpus_str = [" ".join(d) for d in corpus]
    top = bm25.get_top_n(query, corpus_str, n=7)
    assert len(top) == 7
    scores = bm25.get_scores(query)
    top_indices = [corpus_str.index(d) for d in top]
    top_scores = scores[top_indices]
    assert all(top_scores[i] >= top_scores[i + 1] for i in range(len(top_scores) - 1))
