#!/usr/bin/env python

import math
from multiprocessing import cpu_count, Pool
import numpy as np


# 0. Determine what corpus input data should look like. List of lists (containing tokens)

# 1. Calculate corpus-wide statistics
#   a. df for each term
#   b. average document length
#   c. each document's length

# import pandas as pd
# pd.read_pickle("/data/reuters/normalized_text_df.pkl")

class BM25:
    def __init__(self, corpus, algorithm="atire"):
        """
        Parameters
        ----------
        corpus : list of list of str
            Given corpus.
        """
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        self.k1 = 1.5
        self.b = 0.75
        self.epsilon = 0.25

        self._initialize(corpus)

    def _initialize(self, corpus, algorithm):
        """Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
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

        if algorithm == "atire":
            self.idf = self.calc_atire_idf

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

    def get_atire_scores(self, query):
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

    def get_bm25l_scores(self, query):
        return

    def get_bm25plus_scores(self, query):
        return

    def get_bm25adpt_scores(self, query):
        return

    def get_bm25t_scores(self, query):
        return

    def get_bm25rousvar_scores(self, query):
        """
        Rousseau, F., M. Vazirgiannis, Composition of TF normalizations: new insights on scoring functions for ad hoc IR
        :param query:
        :return:
        """
        return

class BM25_atire(BM25):
