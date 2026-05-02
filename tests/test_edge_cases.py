import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from _helpers import (
    SIMPLE_CORPUS,
    ref_okapi_score,
    ref_bm25l_score,
    ref_bm25plus_score,
)


def test_empty_corpus_raises():
    with np.testing.assert_raises(ZeroDivisionError):
        BM25Okapi([])


def test_empty_query_returns_zeros():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    scores = bm25.get_scores([])
    assert scores.shape == (bm25.corpus_size,)
    assert scores.dtype == np.float32
    assert np.all(scores == 0.0)


def test_empty_query_bm25l():
    bm25 = BM25L(SIMPLE_CORPUS)
    scores = bm25.get_scores([])
    assert np.all(scores == 0.0)


def test_empty_query_bm25plus():
    bm25 = BM25Plus(SIMPLE_CORPUS)
    scores = bm25.get_scores([])
    assert np.all(scores == 0.0)


def test_single_document_corpus():
    bm25 = BM25Okapi([["hello", "world", "hello"]])
    assert bm25.corpus_size == 1
    assert bm25.avgdl == 3.0
    scores = bm25.get_scores(["hello"])
    assert scores.shape == (1,)
    assert scores.dtype == np.float32
    expected = ref_okapi_score(
        ["hello"], bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl, bm25.k1, bm25.b,
    )
    assert np.allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_single_document_bm25l():
    bm25 = BM25L([["a", "b", "a"]])
    assert bm25.corpus_size == 1
    scores = bm25.get_scores(["a"])
    assert scores.shape == (1,)
    expected = ref_bm25l_score(
        ["a"], bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )
    assert np.allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_single_document_bm25plus():
    bm25 = BM25Plus([["x", "y"]])
    scores = bm25.get_scores(["x"])
    expected = ref_bm25plus_score(
        ["x"], bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )
    assert np.allclose(scores, expected, rtol=1e-5, atol=1e-6)


def test_oov_query_all_terms_unknown():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    scores = bm25.get_scores(["xyzzy", "nonexistent"])
    assert np.all(scores == 0.0)
    assert scores.shape == (bm25.corpus_size,)


def test_oov_query_some_terms_unknown():
    bm25 = BM25L(SIMPLE_CORPUS)
    scores_mixed = bm25.get_scores(["the", "xyzzy", "cat"])
    scores_only_known = bm25.get_scores(["the", "cat"])
    assert np.allclose(scores_mixed, scores_only_known, rtol=1e-5, atol=1e-6)
    assert scores_mixed.dtype == np.float32


def test_duplicate_query_terms():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    scores_once = bm25.get_scores(["the"])
    scores_twice = bm25.get_scores(["the", "the"])
    assert np.allclose(scores_twice, 2 * scores_once, rtol=1e-5, atol=1e-6)


def test_duplicate_query_terms_bm25l():
    bm25 = BM25L(SIMPLE_CORPUS)
    scores_once = bm25.get_scores(["cat"])
    scores_twice = bm25.get_scores(["cat", "cat"])
    assert np.allclose(scores_twice, 2 * scores_once, rtol=1e-5, atol=1e-6)


def test_duplicate_query_terms_bm25plus():
    bm25 = BM25Plus(SIMPLE_CORPUS)
    scores_once = bm25.get_scores(["dog"])
    scores_twice = bm25.get_scores(["dog", "dog"])
    assert np.allclose(scores_twice, 2 * scores_once, rtol=1e-5, atol=1e-6)


def test_corpus_with_empty_document():
    bm25 = BM25Okapi([[], ["hello", "world"]])
    assert bm25.corpus_size == 2
    assert bm25.avgdl == 1.0
    scores = bm25.get_scores(["hello"])
    assert scores.shape == (2,)
    assert scores.dtype == np.float32


def test_corpus_all_same_document():
    bm25 = BM25Okapi([["a", "b"], ["a", "b"], ["a", "b"]])
    assert bm25.corpus_size == 3
    scores = bm25.get_scores(["a"])
    assert scores.shape == (3,)
    assert scores[0] == scores[1] == scores[2]
