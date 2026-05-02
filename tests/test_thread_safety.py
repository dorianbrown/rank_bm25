import threading
import numpy as np
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

from _helpers import build_test_corpus


def _run_thread_safety_test(bm25_cls, seed):
    import pytest
    import random

    random.seed(seed)
    corpus, vocab = build_test_corpus(n_docs=200, n_terms=50)
    bm25 = bm25_cls(corpus)

    if not bm25._use_c_accel:
        pytest.skip("C accelerator not available")

    queries = [
        [random.choice(vocab) for _ in range(random.randint(1, 10))]
        for _ in range(8)
    ]
    expected = {i: bm25.get_scores(q).copy() for i, q in enumerate(queries)}

    num_threads = 8
    iterations = 500
    errors = []
    barrier = threading.Barrier(num_threads)

    def worker(thread_id):
        for _ in range(iterations):
            i = (thread_id + _) % len(queries)
            q = queries[i]
            result = bm25.get_scores(q)
            if not np.array_equal(result, expected[i]):
                errors.append((thread_id, i))
            barrier.wait()

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety violation: {len(errors)} mismatches across {num_threads} threads"


def test_thread_safety():
    _run_thread_safety_test(BM25Okapi, seed=42)


def test_thread_safety_bm25l():
    _run_thread_safety_test(BM25L, seed=55)


def test_thread_safety_bm25plus():
    _run_thread_safety_test(BM25Plus, seed=66)
