import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from _helpers import build_test_corpus


def _test_i64_path(bm25_cls, seed):
    import pytest
    import random

    random.seed(seed)
    corpus, vocab = build_test_corpus(n_docs=200, n_terms=50)
    bm25 = bm25_cls(corpus)

    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")

    query = [random.choice(vocab) for _ in range(8)]
    scores_i32 = bm25.get_scores(query)

    sm = bm25._score_matrix
    sm.indices = sm.indices.astype(np.int64)
    sm.indptr = sm.indptr.astype(np.int64)
    bm25._setup_c_accel()
    assert bm25._use_c_accel

    scores_i64 = bm25.get_scores(query)
    assert np.allclose(scores_i32, scores_i64, rtol=1e-7, atol=1e-7), \
        f"Max diff: {np.max(np.abs(scores_i32 - scores_i64))}"

    sm.indices = sm.indices.astype(np.int32)
    sm.indptr = sm.indptr.astype(np.int32)
    bm25._setup_c_accel()


def test_i64_c_path_matches_i32():
    _test_i64_path(BM25Okapi, seed=77)


def test_i64_c_path_bm25l():
    _test_i64_path(BM25L, seed=88)


def test_i64_c_path_bm25plus():
    _test_i64_path(BM25Plus, seed=99)
