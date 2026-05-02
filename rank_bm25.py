#!/usr/bin/env python

import ctypes
import math
from multiprocessing import Pool, cpu_count

import numpy as np

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned.
"""

# ---------------------------------------------------------------------------
# Optional C-accelerated CSC column accumulation via ctypes
# The shared library is compiled at install time by setup.py.
# If the .so is not found, falls back to np.add.at at runtime.
# ---------------------------------------------------------------------------


def _load_csc_accel():
    import ctypes
    import os
    import sys

    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib_path = os.path.join(os.path.dirname(__file__), "_csc_accum" + ext)
    if not os.path.exists(lib_path):
        return None
    VP = ctypes.c_void_p
    try:
        lib = ctypes.CDLL(lib_path)
        lib.csc_accumulate_i32_score_f32.restype = None
        lib.csc_accumulate_i32_score_f32.argtypes = [
            VP,
            VP,
            VP,
            VP,
            ctypes.c_int64,
            VP,
            ctypes.c_int64,
        ]
        lib.csc_accumulate_i64.restype = None
        lib.csc_accumulate_i64.argtypes = [
            VP,
            VP,
            VP,
            VP,
            ctypes.c_int64,
            VP,
            ctypes.c_int64,
        ]
        return lib
    except Exception:
        return None


_csc_accel = _load_csc_accel()


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
        self._vocab = {}  # word -> int id
        _rows = []
        _cols = []
        _data = []

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
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1
                wid = self._vocab.get(word)
                if wid is None:
                    wid = len(self._vocab)
                    self._vocab[word] = wid
                _rows.append(self.corpus_size)
                _cols.append(wid)
                _data.append(freq)

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        self.doc_len = np.array(self.doc_len)

        try:
            from scipy.sparse import csc_array as csc_sparse
        except ImportError:
            from scipy.sparse import csc_matrix as csc_sparse
        self._tf_matrix = csc_sparse(
            (
                np.array(_data, dtype=np.float64),
                (np.array(_rows, dtype=np.int32), np.array(_cols, dtype=np.int32)),
            ),
            shape=(self.corpus_size, len(self._vocab)),
        )
        return nd

    def _tokenize_corpus(self, corpus):
        with Pool(cpu_count()) as pool:
            tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def _setup_c_accel(self):
        self._use_c_accel = False
        if _csc_accel is not None:
            sm = self._score_matrix
            idx_dtype = sm.indices.dtype
            if idx_dtype == np.int32:
                self._c_fn = _csc_accel.csc_accumulate_i32_score_f32
                self._c_indptr = np.ascontiguousarray(sm.indptr, dtype=np.int32)
                self._c_indices = np.ascontiguousarray(sm.indices, dtype=np.int32)
                self._c_use_f32_score = True
            else:
                self._c_fn = _csc_accel.csc_accumulate_i64
                self._c_indptr = np.ascontiguousarray(sm.indptr, dtype=np.int64)
                self._c_indices = np.ascontiguousarray(sm.indices, dtype=np.int64)
                self._c_use_f32_score = False
            self._c_data = np.ascontiguousarray(sm.data, dtype=np.float64)
            self._c_score_dtype = np.float32 if self._c_use_f32_score else np.float64
            self._c_n_rows = ctypes.c_int64(self.corpus_size)
            self._c_int64 = ctypes.c_int64
            self._use_c_accel = True

    def get_scores(self, query):
        if self._use_c_accel:
            return self._get_scores_c(query)
        score = np.zeros(self.corpus_size, dtype=np.float32)
        sm = self._score_matrix
        indptr = sm.indptr
        indices = sm.indices
        data = sm.data
        vocab_get = self._vocab.get
        _add_at = np.add.at
        for q in query:
            wid = vocab_get(q)
            if wid is None:
                continue
            s = indptr[wid]
            e = indptr[wid + 1]
            if s < e:
                _add_at(score, indices[s:e], data[s:e])
        return score

    def _get_scores_c(self, query):
        vocab_get = self._vocab.get
        wid_buf = np.empty(len(query), dtype=np.int64)
        n = 0
        for q in query:
            wid = vocab_get(q)
            if wid is not None:
                wid_buf[n] = wid
                n += 1
        if n == 0:
            return np.zeros(self.corpus_size, dtype=np.float32)
        score_buf = np.empty(self.corpus_size, dtype=self._c_score_dtype)
        self._c_fn(
            self._c_indptr.ctypes.data,
            self._c_indices.ctypes.data,
            self._c_data.ctypes.data,
            wid_buf.ctypes.data,
            self._c_int64(n),
            score_buf.ctypes.data,
            self._c_n_rows,
        )
        return score_buf

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), (
            "The documents given don't match the index corpus!"
        )

        scores = self.get_scores(query)
        if n >= len(scores):
            top_n = np.argsort(scores)[::-1][:n]
        else:
            top_n_unsorted = np.argpartition(scores, -n)[-n:]
            top_n = top_n_unsorted[np.argsort(scores[top_n_unsorted])[::-1]]
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

        self._len_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

        # Build IDF array indexed by vocab ID
        vocab_size = len(self._vocab)
        idf_arr = np.zeros(vocab_size)
        for word, wid in self._vocab.items():
            idf_arr[wid] = self.idf.get(word, 0)

        # Precompute full BM25 weights vectorized: idf * tf*(k1+1) / (tf + len_norm)
        tfm = self._tf_matrix.copy()
        # Step 1: compute tf*(k1+1) / (tf + len_norm[row]) for all nonzero entries
        tf = tfm.data
        new_data = tf * (self.k1 + 1) / (tf + self._len_norm[tfm.indices])
        # Step 2: multiply each column's entries by its IDF (vectorized, no matrix multiply)
        col_idf = np.repeat(idf_arr, np.diff(tfm.indptr))
        new_data *= col_idf
        tfm.data = new_data
        # Remove zero-IDF entries
        tfm.eliminate_zeros()
        # Force int32 indices if dimensions fit (halves index bandwidth)
        if max(tfm.shape) < 2**31:
            tfm.indices = tfm.indices.astype(np.int32)
            tfm.indptr = tfm.indptr.astype(np.int32)
        self._score_matrix = tfm
        self._setup_c_accel()

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        len_norm = self._len_norm[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq * (self.k1 + 1) / (q_freq + len_norm)
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

        _lambda = 1 - self.b + self.b * self.doc_len / self.avgdl
        vocab_size = len(self._vocab)
        idf_arr = np.zeros(vocab_size)
        for word, wid in self._vocab.items():
            idf_arr[wid] = self.idf.get(word, 0)

        tfm = self._tf_matrix.copy()
        tf = tfm.data
        lambda_row = _lambda[tfm.indices]
        new_data = (
            (self.k1 + 1)
            * (tf + self.delta * lambda_row)
            / (tf + lambda_row * (self.k1 + self.delta))
        )
        col_idf = np.repeat(idf_arr, np.diff(tfm.indptr))
        new_data *= col_idf
        tfm.data = new_data
        tfm.eliminate_zeros()
        if max(tfm.shape) < 2**31:
            tfm.indices = tfm.indices.astype(np.int32)
            tfm.indptr = tfm.indptr.astype(np.int32)
        self._score_matrix = tfm
        self._setup_c_accel()

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = self.doc_len[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
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
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

        self._len_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        vocab_size = len(self._vocab)
        idf_arr = np.zeros(vocab_size)
        for word, wid in self._vocab.items():
            idf_arr[wid] = self.idf.get(word, 0)

        tfm = self._tf_matrix.copy()
        tf = tfm.data
        new_data = tf * (self.k1 + 1) / (tf + self._len_norm[tfm.indices])
        col_idf = np.repeat(idf_arr, np.diff(tfm.indptr))
        new_data *= col_idf
        tfm.data = new_data
        tfm.eliminate_zeros()
        if max(tfm.shape) < 2**31:
            tfm.indices = tfm.indices.astype(np.int32)
            tfm.indptr = tfm.indptr.astype(np.int32)
        self._score_matrix = tfm
        self._setup_c_accel()

    def get_scores(self, query):
        score = super().get_scores(query)
        bonus = sum((self.idf.get(q) or 0) for q in query) * self.delta
        score += bonus
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = self.doc_len[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score.tolist()
