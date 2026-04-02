import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from rank_bm25 import BM25Okapi, BM25L, BM25Plus, BM25F
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
        assert alg.doc_len == [4, 6, 5]


def tokenizer(doc):
    return doc.split(" ")


def test_tokenizer():
    bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
    assert bm25.corpus_size == 3
    assert bm25.avgdl == 5
    assert bm25.doc_len == [4, 6, 5]


# BM25F tests

field_corpus = [
    {"title": ["machine", "learning", "basics"], "body": ["introduction", "to", "machine", "learning", "algorithms"]},
    {"title": ["deep", "neural", "networks"], "body": ["convolutional", "neural", "networks", "for", "image", "recognition"]},
    {"title": ["natural", "language", "processing"], "body": ["text", "classification", "using", "transformers"]},
    {"title": ["reinforcement", "learning"], "body": ["policy", "gradient", "methods", "for", "control"]},
    {"title": ["computer", "vision"], "body": ["object", "detection", "and", "segmentation"]},
    {"title": ["data", "engineering"], "body": ["building", "data", "pipelines", "at", "scale"]},
]


def test_bm25f_corpus_loading():
    bm25f = BM25F(field_corpus)
    assert bm25f.corpus_size == 6
    assert set(bm25f.fields) == {"title", "body"}
    assert bm25f.field_doc_len["title"] == [3, 3, 3, 2, 2, 2]
    assert bm25f.field_doc_len["body"] == [5, 6, 4, 5, 4, 5]


def test_bm25f_field_weights():
    bm25f = BM25F(field_corpus, field_weights={"title": 3.0, "body": 1.0})
    assert bm25f.field_weights == {"title": 3.0, "body": 1.0}


def test_bm25f_default_params():
    bm25f = BM25F(field_corpus)
    for f in bm25f.fields:
        assert bm25f.field_weights[f] == 1.0
        assert bm25f.field_b[f] == 0.75


def test_bm25f_scoring():
    bm25f = BM25F(field_corpus, field_weights={"title": 3.0, "body": 1.0})
    scores = bm25f.get_scores(["learning"])
    # Doc 0 has "learning" in title and body
    # Doc 3 has "learning" in title only
    # Both should score > 0, rest should be 0
    assert scores[0] > 0
    assert scores[3] > 0
    assert scores[1] == 0
    assert scores[2] == 0
    assert scores[4] == 0
    assert scores[5] == 0


def test_bm25f_title_boost_ranking():
    """Documents with query terms in a boosted field should rank higher."""
    bm25f = BM25F(field_corpus, field_weights={"title": 3.0, "body": 1.0})
    scores = bm25f.get_scores(["learning"])
    # Doc 0 has learning in both fields -> highest
    # Doc 3 has learning in title only -> second
    assert scores[0] > scores[3]


def test_bm25f_matches_okapi_single_field():
    """BM25F with a single field should produce identical scores to BM25Okapi."""
    flat_corpus = [
        ["hello", "world", "foo"],
        ["bar", "baz", "qux"],
        ["hello", "bar", "world", "quux"],
        ["foo", "baz"],
        ["hello", "qux", "quux", "bar", "world"],
    ]
    bm25 = BM25Okapi(flat_corpus, k1=1.5, b=0.75)
    okapi_scores = bm25.get_scores(["hello", "world"])

    corpus_f = [{"text": doc} for doc in flat_corpus]
    bm25f = BM25F(corpus_f, field_weights={"text": 1.0}, field_b={"text": 0.75}, k1=1.5)
    f_scores = bm25f.get_scores(["hello", "world"])

    assert np.allclose(okapi_scores, f_scores)


def test_bm25f_batch_scores():
    bm25f = BM25F(field_corpus, field_weights={"title": 2.0, "body": 1.0})
    full_scores = bm25f.get_scores(["learning"])
    batch_scores = bm25f.get_batch_scores(["learning"], [0, 3])
    assert np.isclose(batch_scores[0], full_scores[0])
    assert np.isclose(batch_scores[1], full_scores[3])


def test_bm25f_get_top_n():
    bm25f = BM25F(field_corpus, field_weights={"title": 3.0, "body": 1.0})
    labels = ["doc0", "doc1", "doc2", "doc3", "doc4", "doc5"]
    top = bm25f.get_top_n(["learning"], labels, n=2)
    assert top == ["doc0", "doc3"]


def test_bm25f_sparse_fields():
    """Documents with missing fields should work correctly."""
    sparse = [
        {"title": ["hello"]},
        {"title": ["world"], "body": ["hello", "foo"]},
        {"body": ["bar", "baz"]},
    ]
    bm25f = BM25F(sparse)
    scores = bm25f.get_scores(["hello"])
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0


def test_bm25f_tokenizer():
    str_corpus = [
        {"title": "hello world", "body": "foo bar baz"},
        {"title": "test doc", "body": "hello again"},
    ]
    bm25f = BM25F(str_corpus, tokenizer=str.split)
    assert bm25f.corpus_size == 2
    assert bm25f.field_doc_len["title"] == [2, 2]
    assert bm25f.field_doc_len["body"] == [3, 2]
