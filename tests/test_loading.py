import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import threading
import numpy as np
import re


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


def test_thread_safety():
    import pytest
    import random

    random.seed(42)

    n_docs = 200
    vocab = [f"term_{i}" for i in range(50)]
    corpus = []
    for _ in range(n_docs):
        doc = [random.choice(vocab) for _ in range(random.randint(5, 30))]
        corpus.append(doc)

    bm25 = BM25Okapi(corpus)

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
