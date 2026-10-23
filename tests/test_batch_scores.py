import numpy as np
import pytest

from rank_bm25 import BM25Okapi, BM25L, BM25Plus


@pytest.fixture(params=[BM25Okapi, BM25L, BM25Plus])
def bm25(request):
    return request.param([["alpha", "alpha"], ["beta"], ["gamma", "beta"]])


@pytest.mark.parametrize("doc_ids", [[-1], [-2], [0, -1], [3], [0, 3]])
def test_batch_scores_reject_invalid_document_ids(bm25, doc_ids):
    with pytest.raises(AssertionError):
        bm25.get_batch_scores(["alpha"], doc_ids)


@pytest.mark.parametrize("doc_ids", [[0], [2], [2, 0, 1], [0, 2, 0]])
def test_batch_scores_match_full_scores(bm25, doc_ids):
    query = ["alpha", "beta"]
    np.testing.assert_allclose(
        bm25.get_batch_scores(query, doc_ids), bm25.get_scores(query)[doc_ids]
    )
