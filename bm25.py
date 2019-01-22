#!/usr/bin/env python

import math
import numpy as np


# 0. Determine what corpus input data should look like. List of lists (containing tokens)

# 1. Calculate corpus-wide statistics
#   a. df for each term
#   b. average document length
#   c. each document's length

# import pandas as pd
# pd.read_pickle("/data/reuters/normalized_text_df.pkl")

class BM25:
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        self._initialize(corpus)

    # TODO: Move avgdl and doc_freqs to BM25. Pass calculated info to new class _calc_idf
    def _initialize(self, corpus):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()


class BM25Atire(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus)

    def _initialize(self, corpus):
        """
        Calculates frequencies of terms in documents and in corpus.
        Also computes inverse document frequencies.
        """
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []

        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            print(f"Query term=: {q}")
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            print(f"IDF of q: {self.idf[q] or 0}")
            print(f"Document frequency: {q_freq}")
            print(f"Document length: {doc_len}")
            print(f"Avg Doc Length: {self.avgdl}")
            score += (self.idf[q] or 0) * (q_freq * (self.k1 + 1) /
                                           (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score


class BM25L(BM25):
    def __init__(self, corpus, k1=1.5, b=0.75, delta=0.1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus)

    def _initialize(self, corpus):
        """
        Calculates frequencies of terms in documents and in corpus.
        Also computes inverse document frequencies.
        """
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avgdl = num_doc / self.corpus_size
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []

        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            print(f"Query term=: {q}")
            print(f"IDF of q: {self.idf[q] or 0}")
            print(f"Document frequency: {q_freq}")
            print(f"Document length: {doc_len}")
            print(f"Avg Doc Length: {self.avgdl}")
            ctd = q_freq / (1 - self.b + self.b * doc_len/self.avgdl)
            score += (self.idf[q] or 0) * q_freq * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score
