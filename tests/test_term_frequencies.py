import os
import pickle
import random
import sys

import numpy as np

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from rank_bm25 import BM25Okapi, BM25L, BM25Plus  # noqa: E402


def _reference(cls):
    """The same algorithm with the original corpus-wide loop for term frequencies."""
    class Reference(cls):
        def _term_frequencies(self, word):
            return np.array([(doc.get(word) or 0) for doc in self.doc_freqs])
    return Reference


def _corpus(n_docs=300, vocab=400, seed=7):
    rng = random.Random(seed)
    words = ["w%d" % i for i in range(vocab)]
    weights = [1.0 / (i + 1) for i in range(vocab)]  # Zipf-like, so some words are very common
    return [rng.choices(words, weights, k=rng.randint(1, 40)) for _ in range(n_docs)]


def test_scores_identical_to_corpus_loop():
    corpus = _corpus()
    rng = random.Random(1)
    queries = [rng.choices(["w%d" % i for i in range(450)], k=rng.randint(1, 8)) for _ in range(50)]
    queries += [[], ["unseen"], ["w0", "w0", "w0"]]  # empty, unknown and repeated terms
    for cls in (BM25Okapi, BM25L, BM25Plus):
        fast, slow = cls(corpus), _reference(cls)(corpus)
        for query in queries:
            assert np.array_equal(fast.get_scores(query), slow.get_scores(query))


def test_works_after_unpickling_an_object_without_postings():
    bm25 = BM25Okapi(_corpus(n_docs=20))
    expected = bm25.get_scores(["w1", "w2"])
    del bm25._postings  # as if pickled by a version without the index
    restored = pickle.loads(pickle.dumps(bm25))
    assert np.array_equal(restored.get_scores(["w1", "w2"]), expected)
