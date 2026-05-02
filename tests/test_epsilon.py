import math
import numpy as np
from rank_bm25 import BM25Okapi

from _helpers import SIMPLE_CORPUS


def test_negative_idf_gets_floored():
    corpus_with_high_df = [
        ["common", "common", "rare_a"],
        ["common", "rare_b", "rare_c"],
        ["common", "rare_a", "rare_d"],
        ["common", "rare_b", "rare_e"],
    ]
    bm25 = BM25Okapi(corpus_with_high_df, epsilon=0.5)

    assert "common" in bm25.idf
    raw_idf = math.log(bm25.corpus_size - 4 + 0.5) - math.log(4 + 0.5)
    assert raw_idf < 0, f"Expected negative raw IDF for high-df term, got {raw_idf}"
    assert bm25.idf["common"] == 0.5 * bm25.average_idf


def test_epsilon_floor_increases_score_for_frequent_term():
    corpus_with_high_df = [
        ["common", "rare_a"],
        ["common", "rare_b"],
        ["common", "rare_c"],
        ["common", "rare_d"],
        ["common", "rare_e"],
    ]
    bm25 = BM25Okapi(corpus_with_high_df, epsilon=0.25)

    score_common = bm25.get_scores(["common"])
    score_rare = bm25.get_scores(["rare_a"])
    assert np.sum(np.abs(score_common)) >= 0


def test_no_epsilon_floor_for_low_df_terms():
    bm25 = BM25Okapi(SIMPLE_CORPUS, epsilon=0.25)
    term = "mat"
    assert term in bm25.idf
    freq = sum(1 for d in SIMPLE_CORPUS if term in d)
    assert freq == 1
    raw_idf = math.log(bm25.corpus_size - freq + 0.5) - math.log(freq + 0.5)
    assert raw_idf > 0
    assert np.isclose(bm25.idf[term], raw_idf)
