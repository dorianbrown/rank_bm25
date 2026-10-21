"""
Score regression tests for BM25Okapi, BM25L and BM25Plus.

The expected scores don't come from rank_bm25 itself: a few are worked out by
hand, and the rest come from the small pure-Python reference implementation
below, which shares no code with the library. They pin the current behaviour,
so a refactor or speed-up of get_scores(), get_batch_scores() or get_top_n()
can be checked against it, edge cases included.
"""
from collections import Counter
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from rank_bm25 import BM25Okapi, BM25L, BM25Plus  # noqa: E402


# Loose enough for an implementation that sums scores in float32 (even over a
# 600-term query), tight enough to catch any change to the formulas.
def approx(expected):
    return pytest.approx(expected, rel=1e-5, abs=1e-6)


def as_floats(scores):
    return [float(s) for s in scores]


# Reference implementation: the formulas rank_bm25 uses (Trotman et al., 2014),
# written out one document and one query term at a time.

def _stats(corpus):
    avgdl = sum(len(doc) for doc in corpus) / len(corpus)
    df = Counter(term for doc in corpus for term in set(doc))
    return len(corpus), avgdl, df


def ref_okapi(corpus, query, k1=1.5, b=0.75, epsilon=0.25):
    n_docs, avgdl, df = _stats(corpus)
    idf = {t: math.log((n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    # Negative idfs (terms in more than half of the documents) are replaced by
    # epsilon * the average idf, and that average includes the negative ones.
    floor = epsilon * sum(idf.values()) / len(idf)
    idf = {t: floor if v < 0 else v for t, v in idf.items()}

    scores = []
    for doc in corpus:
        tf = Counter(doc)
        norm = 1 - b + b * len(doc) / avgdl
        scores.append(sum(
            idf.get(q, 0) * tf[q] * (k1 + 1) / (tf[q] + k1 * norm)
            for q in query
        ))
    return scores


def ref_bm25l(corpus, query, k1=1.5, b=0.75, delta=0.5):
    n_docs, avgdl, df = _stats(corpus)
    idf = {t: math.log((n_docs + 1) / (n + 0.5)) for t, n in df.items()}

    scores = []
    for doc in corpus:
        tf = Counter(doc)
        norm = 1 - b + b * len(doc) / avgdl
        # A known term adds idf * (k1 + 1) * delta / (k1 + delta) even to the
        # documents that don't contain it. Unknown terms add nothing.
        scores.append(sum(
            idf.get(q, 0) * (k1 + 1) * (tf[q] / norm + delta) / (k1 + tf[q] / norm + delta)
            for q in query
        ))
    return scores


def ref_bm25plus(corpus, query, k1=1.5, b=0.75, delta=1):
    n_docs, avgdl, df = _stats(corpus)
    idf = {t: math.log((n_docs + 1) / n) for t, n in df.items()}

    scores = []
    for doc in corpus:
        tf = Counter(doc)
        norm = 1 - b + b * len(doc) / avgdl
        # Same as BM25L: a known term adds idf * delta to every document.
        scores.append(sum(
            idf.get(q, 0) * (delta + tf[q] * (k1 + 1) / (k1 * norm + tf[q]))
            for q in query
        ))
    return scores


# An empty dict means the library defaults, which the reference functions
# above repeat, so changing a default fails these tests too.
VARIANTS = [
    pytest.param(BM25Okapi, ref_okapi, {}, id="okapi"),
    pytest.param(BM25Okapi, ref_okapi, {"k1": 1.2, "b": 0.5, "epsilon": 0.1}, id="okapi-params"),
    pytest.param(BM25L, ref_bm25l, {}, id="bm25l"),
    pytest.param(BM25L, ref_bm25l, {"k1": 2.0, "b": 0.3, "delta": 1.0}, id="bm25l-params"),
    pytest.param(BM25Plus, ref_bm25plus, {}, id="bm25plus"),
    pytest.param(BM25Plus, ref_bm25plus, {"k1": 0.9, "b": 0.9, "delta": 0.5}, id="bm25plus-params"),
]

README_CORPUS = [doc.split(" ") for doc in [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?",
]]


def random_corpus(n_docs=300, vocab_size=60, seed=42):
    # Zipf-like term weights, so a few terms are in most documents (negative
    # Okapi idf) and most are rare. Some documents come out empty.
    rng = random.Random(seed)
    vocab = ["w%d" % i for i in range(vocab_size)]
    weights = [1 / (rank + 1) for rank in range(vocab_size)]
    return [rng.choices(vocab, weights, k=rng.randint(0, 40)) for _ in range(n_docs)]


CORPORA = {
    "readme": README_CORPUS,
    "repeated-terms": [
        ["a", "a", "a", "b"],
        ["a", "c", "c"],
        ["b", "b", "d", "e", "f", "a"],
        ["f"],
    ],
    "empty-doc": [["x", "y"], [], ["y", "z", "z"], ["x"]],
    # "a" is in 3 of 4 documents, so Okapi floors its idf.
    "idf-floor": [["a", "b"], ["a", "c"], ["a", "d"], ["e"]],
    # Okapi idf is exactly 0 for a term in half of the documents (#39, #43).
    "half-df": [["windy", "London"], ["windy", "there"], ["good"], ["man"]],
    # Every term is in at least half of the documents, so the Okapi floor
    # itself is negative.
    "negative-floor": [["a", "b"], ["a", "c"]],
    "single-doc": [["only", "one", "doc", "one"]],
    "random": random_corpus(),
}


def queries_for(corpus):
    vocab = sorted(set(term for doc in corpus for term in doc))
    rng = random.Random(len(vocab))
    return [
        [],
        vocab[:1],
        vocab[:3] + ["not-in-corpus"],
        [vocab[0], vocab[0], vocab[-1]],  # repeated query terms count every time
        rng.choices(vocab, k=600),  # long query
    ] + [rng.sample(vocab, min(5, len(vocab))) for _ in range(5)]


@pytest.mark.parametrize("corpus_name", sorted(CORPORA))
@pytest.mark.parametrize("model, reference, params", VARIANTS)
def test_get_scores(model, reference, params, corpus_name):
    corpus = CORPORA[corpus_name]
    bm25 = model(corpus, **params)
    for query in queries_for(corpus):
        scores = bm25.get_scores(query)
        assert len(scores) == len(corpus)
        assert as_floats(scores) == approx(reference(corpus, query, **params)), query[:10]


@pytest.mark.parametrize("corpus_name", sorted(CORPORA))
@pytest.mark.parametrize("model, reference, params", VARIANTS)
def test_get_batch_scores(model, reference, params, corpus_name):
    corpus = CORPORA[corpus_name]
    bm25 = model(corpus, **params)
    doc_ids = random.Random(0).sample(range(len(corpus)), len(corpus) // 2 + 1)
    for query in queries_for(corpus):
        expected = reference(corpus, query, **params)
        scores = bm25.get_batch_scores(query, doc_ids)
        assert as_floats(scores) == approx([expected[i] for i in doc_ids]), query[:10]
        assert bm25.get_batch_scores(query, []) == []


@pytest.mark.parametrize("corpus_name", sorted(CORPORA))
@pytest.mark.parametrize("model, reference, params", VARIANTS)
def test_get_top_n(model, reference, params, corpus_name):
    corpus = CORPORA[corpus_name]
    bm25 = model(corpus, **params)
    labels = ["doc %d" % i for i in range(len(corpus))]
    for query in queries_for(corpus):
        expected = reference(corpus, query, **params)
        for n in (0, 1, 3, len(corpus), len(corpus) + 5):
            top = bm25.get_top_n(query, labels, n=n)
            assert len(set(top)) == len(top) == min(n, len(corpus))
            # Tied documents may come back in any order, so compare scores.
            got = [expected[labels.index(doc)] for doc in top]
            assert got == approx(sorted(expected, reverse=True)[:n]), (n, query[:10])


def test_readme_example_by_hand():
    # Documents of length 4, 6 and 5, so avgdl = 5. "windy" and "London"
    # each occur once, both in document 1.
    query = ["windy", "London"]
    norm = 1 - 0.75 + 0.75 * 6 / 5  # length normalisation for document 1

    scores = BM25Okapi(README_CORPUS).get_scores(query)
    assert as_floats(scores) == approx([0.0, 0.93729472, 0.0])  # as in the README
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5))
    assert as_floats(scores) == approx([0.0, 2 * idf * 2.5 / (1 + 1.5 * norm), 0.0])

    idf = math.log((3 + 1) / (1 + 0.5))
    absent = 2 * idf * 2.5 * 0.5 / (1.5 + 0.5)
    present = 2 * idf * 2.5 * (1 / norm + 0.5) / (1.5 + 1 / norm + 0.5)
    scores = BM25L(README_CORPUS).get_scores(query)
    assert as_floats(scores) == approx([absent, present, absent])

    idf = math.log((3 + 1) / 1)
    absent = 2 * idf * 1
    present = 2 * idf * (1 + 2.5 / (1.5 * norm + 1))
    scores = BM25Plus(README_CORPUS).get_scores(query)
    assert as_floats(scores) == approx([absent, present, absent])


def test_okapi_idf_floor_by_hand():
    corpus = CORPORA["idf-floor"]
    # "a" is in 3 of 4 documents: raw idf log(1.5 / 3.5) < 0. The other four
    # terms are in one document each: idf log(3.5 / 1.5).
    idf = math.log(3.5 / 1.5)
    average_idf = (4 * idf - idf) / 5
    for epsilon in (0.25, 0.5):
        bm25 = BM25Okapi(corpus, epsilon=epsilon)
        assert bm25.idf["a"] == pytest.approx(epsilon * average_idf)
        assert bm25.idf["b"] == pytest.approx(idf)

    # With epsilon=0 the floor is 0, so "a" adds nothing.
    bm25 = BM25Okapi(corpus, epsilon=0)
    assert as_floats(bm25.get_scores(["a"])) == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("model", [BM25Okapi, BM25L, BM25Plus])
def test_later_calls_do_not_change_earlier_results(model):
    bm25 = model(CORPORA["random"])
    first = bm25.get_scores(["w1", "w2"])
    before = as_floats(first)
    bm25.get_scores(["w3", "w40"])
    bm25.get_batch_scores(["w4"], [0, 1, 2])
    bm25.get_top_n(["w5"], CORPORA["random"], n=3)
    assert as_floats(first) == before
