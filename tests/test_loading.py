from rank_bm25 import BM25Okapi, BM25L, BM25Plus

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]


def test_corpus_loading():

    algs = [
        BM25Okapi(corpus),
        BM25L(corpus),
        BM25Plus(corpus)
    ]

    for alg in algs:
        assert alg.corpus_size == 3
        assert alg.avgdl