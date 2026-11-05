import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
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
        assert alg.doc_len == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]


def failing_tokenizer(doc):
    raise ValueError("invalid document")


def test_tokenizer_failure_releases_pool():
    import gc
    import warnings
    import pytest

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        with pytest.raises(ValueError, match="invalid document"):
            BM25Okapi(corpus, tokenizer=failing_tokenizer)
        gc.collect()

    assert not any("unclosed running multiprocessing pool" in str(w.message) for w in caught)
