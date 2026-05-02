import numpy as np
from rank_bm25 import BM25Plus

from _helpers import SIMPLE_CORPUS


def test_bm25plus_score_dtype_is_float32():
    bm25 = BM25Plus(SIMPLE_CORPUS)
    scores = bm25.get_scores(["the", "cat"])
    assert scores.dtype == np.float32


def test_bm25plus_score_dtype_with_all_oov():
    bm25 = BM25Plus(SIMPLE_CORPUS)
    scores = bm25.get_scores(["xyzzy", "nope"])
    assert scores.dtype == np.float32
    assert np.all(scores == 0.0)


def test_bm25plus_score_dtype_large_delta():
    bm25 = BM25Plus(SIMPLE_CORPUS, delta=100.0)
    scores = bm25.get_scores(["the"])
    assert scores.dtype == np.float32
