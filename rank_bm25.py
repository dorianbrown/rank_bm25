#!/usr/bin/env python
from typing import List, Dict
import math
import numpy as np
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0  # total number of document in corpus
        self.avgdl = 0  # average length of a document in corpus
        self.doc_freqs = []  # list of dictionaries of term_frequency of each document
        self.idf = {}  # idf score of each word in whole corpus
        self.doc_len = []  # list of length of each document in corpus
        self.tokenizer = tokenizer  # user input tokenizer, defaults to none

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus) -> Dict:
        """
        Args:
            corpus (List): list of the documents

        Returns:
            dict: number of document that contains the word

        Example:
            corpus = [['ram', 'is', 'a', 'good', 'boy'], ['ram', 'does', 'cycling', 'and', 'racing'], ['ram', 'is', 'healthy'], ['rita', 'likes', 'shyam'], ['good', 'luck']]
            nd = {'ram': 3, 'is': 2, 'a': 1, 'good': 2, 'boy': 1, 'does': 1, 'cycling': 1, 'and': 1, 'racing': 1, 'healthy': 1, 'rita': 1, 'likes': 1, 'shyam': 1, 'luck': 1}

        """
        nd = {}  # word -> number of documents with word
        num_words = 0  # total number of words in whole corpus........... changed num_doc to num_words
        for document in corpus:
            self.doc_len.append(len(document))
            num_words += len(document)  # total number of words in whole corpus

            term_frequencies = (
                {}
            )  # term frequency of each word in a document........ changed frequencies to term_frequencies
            for word in document:
                if word not in term_frequencies:
                    term_frequencies[word] = 0
                term_frequencies[word] += 1
            self.doc_freqs.append(term_frequencies)

            for word, freq in term_frequencies.items():
                """
                increase the size of 'nd' for word present in document of corpus by 1 otherwise set it to 1.
                """
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1  # increases the size of corpus after each loop until the end of document in corpus

        self.avgdl = num_words / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        """tokenizes the corpus according to user input tokenizer

        Args:
            corpus (List): list of documents
        """
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        """sorts the score of documents in descending order to find the most similar document in corpus

        Args:
            query (List): list of token
            documents (List): list of documents
            n (int, optional): Top n documents. Defaults to 5.

        Returns:
            List: list of most similar document in descending order
        """
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        """
        Args:
            corpus (List): list of documents
            tokenizer (optional): user defined tokenizer. Defaults to None.
            k1 (float, optional): free parameter as k1 ∈ [ 1.2 , 2.0 ]. Defaults to 1.5.
            b (float, optional): free parameter. Defaults to 0.75.
            epsilon (float, optional): constant used for negative idf of document in corpus. Defaults to 0.25.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf

        Args:
            nd (Dict): word -> number of documents with word
        """
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
        """The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info

        Args:
            query (list): tokenized document

        Returns:
            Array: score of each token in a corpus
        """
        score = np.zeros(
            self.corpus_size
        )  # array of score of each token in a document in corpus
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array(
                [(doc.get(q) or 0) for doc in self.doc_freqs]
            )  # term freq of a term 'q' in a document
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                * q_freq
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                * q_freq
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score.tolist()


# BM25Adpt and BM25T are a bit more complicated than the previous algorithms here. Here a term-specific k1
# parameter is calculated before scoring is done

# class BM25Adpt(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
#
#
# class BM25T(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
