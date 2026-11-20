import numpy as np
import pytest

from rank_bm25 import BM25L, BM25Okapi, BM25Plus


class IndexedLengths(list):
    def __iter__(self):
        raise AssertionError("Batch scoring scanned document lengths outside the batch")


@pytest.mark.parametrize("algorithm", [BM25Okapi, BM25L, BM25Plus])
@pytest.mark.parametrize("ids", [[], [2], [3, 0, 3, 1], np.array([1, 2])])
def test_batch_scores_use_only_requested_lengths(algorithm, ids):
    index = algorithm([["one"], ["two", "two"], ["one", "three"], []])
    query = ["one", "missing", "one"]
    expected = [index.get_scores(query)[i] for i in ids]
    index.doc_len = IndexedLengths(index.doc_len)
    assert index.get_batch_scores(query, ids) == expected
