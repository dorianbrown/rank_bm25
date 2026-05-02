import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from _helpers import (
    SIMPLE_CORPUS,
    batch_scores_from_full,
    ref_bm25l_score_full,
)


def test_batch_scores_okapi_matches_get_scores():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    query = ["the", "cat", "dog"]
    doc_ids = [0, 2, 3]
    got = np.array(bm25.get_batch_scores(query, doc_ids), dtype=np.float32)
    expected = batch_scores_from_full(bm25, query, doc_ids)
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6), \
        f"Max diff: {np.max(np.abs(got - expected))}"


def test_batch_scores_bm25l_matches_reference():
    bm25 = BM25L(SIMPLE_CORPUS)
    query = ["sat", "on", "mat"]
    doc_ids = [0, 3]
    got = np.array(bm25.get_batch_scores(query, doc_ids), dtype=np.float32)
    expected = ref_bm25l_score_full(
        query, bm25.doc_freqs, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.k1, bm25.b, bm25.delta,
    )[doc_ids]
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_batch_scores_bm25plus_matches_get_scores():
    bm25 = BM25Plus(SIMPLE_CORPUS)
    query = ["dog", "ran", "fast"]
    doc_ids = [1, 2]
    got = np.array(bm25.get_batch_scores(query, doc_ids), dtype=np.float32)
    expected = batch_scores_from_full(bm25, query, doc_ids)
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_batch_scores_all_docs_equals_get_scores():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    query = ["the", "cat"]
    doc_ids = list(range(bm25.corpus_size))
    got = np.array(bm25.get_batch_scores(query, doc_ids), dtype=np.float32)
    expected = bm25.get_scores(query)
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_batch_scores_single_doc():
    bm25 = BM25L(SIMPLE_CORPUS)
    doc_ids = [1]
    got = bm25.get_batch_scores(["dog", "ran"], doc_ids)
    assert len(got) == 1
    assert isinstance(got[0], float)


def test_batch_scores_empty_doc_ids():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    got = bm25.get_batch_scores(["the"], [])
    assert got == []
    assert isinstance(got, list)


def test_batch_scores_empty_query():
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    got = bm25.get_batch_scores([], [0, 1])
    assert got == [0.0, 0.0]


def test_batch_scores_invalid_doc_id_raises():
    import pytest
    bm25 = BM25Okapi(SIMPLE_CORPUS)
    with pytest.raises(AssertionError):
        bm25.get_batch_scores(["the"], [10])


def test_batch_scores_with_nondefault_params():
    bm25 = BM25Okapi(SIMPLE_CORPUS, k1=2.0, b=0.5)
    query = ["cat", "sat"]
    doc_ids = [0, 3]
    got = np.array(bm25.get_batch_scores(query, doc_ids), dtype=np.float32)
    expected = batch_scores_from_full(bm25, query, doc_ids)
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-6)
