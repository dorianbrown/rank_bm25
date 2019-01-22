from bm25 import BM25

corpus1 = ["A A B B C", "B B C D"]
corpus1 = [s.split(" ") for s in corpus1]

def test_corpus_loading():
    bm25_1 = BM25(corpus1)

    assert(BM25)