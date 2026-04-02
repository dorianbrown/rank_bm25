#!/usr/bin/env python

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
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
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
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
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

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
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
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
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
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
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
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
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
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
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
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
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
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()


class BM25F:
    """BM25F: field-aware BM25 that combines term frequencies across fields
    before applying saturation, avoiding the over-estimation that occurs when
    scoring each field independently and summing.

    Reference: Robertson, S., Zaragoza, H., & Taylor, M. (2004).
    Simple BM25 extension to multiple weighted fields.
    """

    def __init__(self, corpus, field_weights=None, field_b=None, k1=1.5, epsilon=0.25, tokenizer=None):
        """
        Parameters
        ----------
        corpus : list of dicts
            Each document is a dict mapping field names to token lists, e.g.
            [{"title": ["hello", "world"], "body": ["foo", "bar", "baz"]}]
        field_weights : dict, optional
            Boost weight per field, e.g. {"title": 2.0, "body": 1.0}.
            Fields not listed default to 1.0.
        field_b : dict, optional
            Length-normalization parameter per field (0.0-1.0), e.g. {"title": 0.75, "body": 0.75}.
            Fields not listed default to 0.75.
        k1 : float
            Saturation parameter (shared across fields). Default 1.5.
        epsilon : float
            Floor for IDF values, as a fraction of average IDF. Default 0.25.
        tokenizer : callable, optional
            If provided, each field value is passed through this function.
        """
        self.k1 = k1
        self.epsilon = epsilon
        self.tokenizer = tokenizer
        self.corpus_size = 0
        self.idf = {}

        if tokenizer:
            corpus = [{f: tokenizer(v) if isinstance(v, str) else v
                       for f, v in doc.items()} for doc in corpus]

        # Discover all field names from corpus
        self.fields = sorted({f for doc in corpus for f in doc})
        self.field_weights = {f: (field_weights or {}).get(f, 1.0) for f in self.fields}
        self.field_b = {f: (field_b or {}).get(f, 0.75) for f in self.fields}

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents containing word (in any field)
        self.corpus_size = len(corpus)

        # Per-field: doc lengths and term frequencies
        self.field_doc_len = {f: [] for f in self.fields}
        self.field_avgdl = {}
        self.field_doc_freqs = {f: [] for f in self.fields}

        for doc in corpus:
            doc_terms = set()
            for field in self.fields:
                tokens = doc.get(field, [])
                self.field_doc_len[field].append(len(tokens))

                frequencies = {}
                for word in tokens:
                    if word not in frequencies:
                        frequencies[word] = 0
                    frequencies[word] += 1
                    doc_terms.add(word)
                self.field_doc_freqs[field].append(frequencies)

            for word in doc_terms:
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

        for field in self.fields:
            total = sum(self.field_doc_len[field])
            self.field_avgdl[field] = total / self.corpus_size if self.corpus_size else 0

        return nd

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """Score all documents against a tokenized query.

        For each query term, computes a combined term frequency across fields:
            tf_combined = sum over fields of: weight_f * tf(t,d,f) / (1 - b_f + b_f * dl_f / avgdl_f)
        Then applies BM25 saturation once:
            score += idf(t) * tf_combined * (k1 + 1) / (tf_combined + k1)
        """
        score = np.zeros(self.corpus_size)
        for q in query:
            # Compute combined TF across all fields
            tf_combined = np.zeros(self.corpus_size)
            for field in self.fields:
                w = self.field_weights[field]
                b = self.field_b[field]
                avgdl = self.field_avgdl[field]

                q_freq = np.array([(doc.get(q) or 0) for doc in self.field_doc_freqs[field]])
                dl = np.array(self.field_doc_len[field])

                if avgdl > 0:
                    tf_combined += w * q_freq / (1 - b + b * dl / avgdl)
                else:
                    tf_combined += w * q_freq

            idf = self.idf.get(q) or 0
            score += idf * (tf_combined * (self.k1 + 1)) / (tf_combined + self.k1)
        return score

    def get_batch_scores(self, query, doc_ids):
        """Score a subset of documents against a tokenized query."""
        assert all(di < self.corpus_size for di in doc_ids)
        score = np.zeros(len(doc_ids))
        for q in query:
            tf_combined = np.zeros(len(doc_ids))
            for field in self.fields:
                w = self.field_weights[field]
                b = self.field_b[field]
                avgdl = self.field_avgdl[field]

                q_freq = np.array([(self.field_doc_freqs[field][di].get(q) or 0) for di in doc_ids])
                dl = np.array(self.field_doc_len[field])[doc_ids]

                if avgdl > 0:
                    tf_combined += w * q_freq / (1 - b + b * dl / avgdl)
                else:
                    tf_combined += w * q_freq

            idf = self.idf.get(q) or 0
            score += idf * (tf_combined * (self.k1 + 1)) / (tf_combined + self.k1)
        return score.tolist()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


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
