from bm25 import BM25

corpus1 = ["A A B B C", "B B C D"]
corpus1 = [s.split(" ") for s in corpus1]

corpus2 = ["The quick brown fox", "The slow brown dog", "Jumping jack flash"]
corpus2 = [s.split(" ") for s in corpus2]


def test_corpus_loading():
    bm25_1 = BM25(corpus1)

    assert (bm25_1.corpus_size == 2)
    assert (bm25_1.avgdl == 4.5)
    assert (bm25_1.doc_freqs == [{'A': 2, 'B': 2, 'C': 1}, {'B': 2, 'C': 1, 'D': 1}])
    assert (bm25_1.idf == {'A': 0.0, 'B': -0.20117973905426256, 'C': -0.20117973905426256, 'D': 0.0})
    assert (bm25_1.doc_len == [5, 4])
