import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from _helpers import (
    SIMPLE_CORPUS,
    build_test_corpus,
    ref_okapi_score,
    ref_bm25l_score,
    ref_bm25plus_score,
)

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]
tokenized_corpus = [doc.split(" ") for doc in corpus]

algs = [
    BM25Okapi(tokenized_corpus),
    BM25L(tokenized_corpus),
    BM25Plus(tokenized_corpus)
]


def test_corpus_loading():
    for alg in algs:
        assert alg.corpus_size == 3
        assert alg.avgdl == 5
        assert list(alg.doc_len) == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert list(bm25.doc_len) == [4, 6, 5]


def test_c_accel_consistency():
    import pytest
    import random

    random.seed(42)
    corpus, vocab = build_test_corpus(n_docs=200, n_terms=50)
    bm25 = BM25Okapi(corpus)

    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")

    query = [random.choice(vocab) for _ in range(5)]

    scores_c = bm25.get_scores(query)
    bm25._use_c_accel = False
    try:
        scores_py = bm25.get_scores(query)
    finally:
        bm25._use_c_accel = True

    assert scores_c.dtype == np.float32
    assert scores_py.dtype == np.float32
    assert scores_c.shape == scores_py.shape
    assert np.allclose(scores_c, scores_py, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(scores_c - scores_py))}"


def test_bm25l_c_accel_consistency():
    import pytest
    import random

    corpus, vocab = build_test_corpus(n_docs=200, n_terms=50)
    bm25 = BM25L(corpus)

    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")

    query = [random.choice(vocab) for _ in range(5)]

    scores_c = bm25.get_scores(query)
    assert scores_c.dtype == np.float32

    bm25._use_c_accel = False
    try:
        scores_py = bm25.get_scores(query)
    finally:
        bm25._use_c_accel = True

    assert scores_py.dtype == np.float32
    assert np.allclose(scores_c, scores_py, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(scores_c - scores_py))}"


def test_bm25plus_c_accel_consistency():
    import pytest
    import random

    corpus, vocab = build_test_corpus(n_docs=200, n_terms=50)
    bm25 = BM25Plus(corpus)

    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")

    query = [random.choice(vocab) for _ in range(5)]

    scores_c = bm25.get_scores(query)
    assert scores_c.dtype == np.float32

    bm25._use_c_accel = False
    try:
        scores_py = bm25.get_scores(query)
    finally:
        bm25._use_c_accel = True

    assert scores_py.dtype == np.float32
    assert np.allclose(scores_c, scores_py, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(scores_c - scores_py))}"


def test_okapi_scores_match_reference():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    query = ["the", "cat", "dog"]
    got = bm25.get_scores(query)
    expected = ref_okapi_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl, bm25.k1, bm25.b,
    )
    assert got.dtype == np.float32
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(got - expected))}"


def test_bm25l_scores_match_reference():
    bm25 = BM25L(SIMPLE_CORPUS)
    query = ["the", "cat", "dog", "mat"]
    got = bm25.get_scores(query)
    expected = ref_bm25l_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )
    assert got.dtype == np.float32
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(got - expected))}"


def test_bm25plus_scores_match_reference():
    bm25 = BM25Plus(SIMPLE_CORPUS)
    query = ["the", "cat", "sat", "ran"]
    got = bm25.get_scores(query)
    expected = ref_bm25plus_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )
    assert got.dtype == np.float32
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(got - expected))}"


def test_okapi_scores_match_reference_with_nondefault_params():
    bm25 = BM25Okapi(SIMPLE_CORPUS, k1=2.0, b=0.5, epsilon=0.5)
    query = ["the", "fast"]
    got = bm25.get_scores(query)
    expected = ref_okapi_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl, bm25.k1, bm25.b,
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_bm25l_scores_match_reference_with_nondefault_params():
    bm25 = BM25L(SIMPLE_CORPUS, k1=2.0, b=0.5, delta=1.0)
    query = ["dog", "mat"]
    got = bm25.get_scores(query)
    expected = ref_bm25l_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_bm25plus_scores_match_reference_with_nondefault_params():
    bm25 = BM25Plus(SIMPLE_CORPUS, k1=2.0, b=0.5, delta=2.0)
    query = ["a", "on", "cat"]
    got = bm25.get_scores(query)
    expected = ref_bm25plus_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_okapi_scores_with_larger_random_corpus():
    import random
    random.seed(123)
    corpus, vocab = build_test_corpus(n_docs=100, n_terms=20)
    bm25 = BM25Okapi(corpus)
    query = [random.choice(vocab) for _ in range(8)]
    got = bm25.get_scores(query)
    expected = ref_okapi_score(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl, bm25.k1, bm25.b,
    )
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(got - expected))}"
