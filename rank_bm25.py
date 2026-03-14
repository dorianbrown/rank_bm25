#!/usr/bin/env python

import math
import numpy as np
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned.
"""

# ---------------------------------------------------------------------------
# Optional C-accelerated CSC column accumulation via ctypes
# Falls back to np.add.at if compilation fails
# ---------------------------------------------------------------------------
_csc_accel = None

def _try_compile_csc_accel():
    """Compile a tiny C shared library for fast CSC column accumulation."""
    import ctypes, subprocess, tempfile, sys, os
    C_SRC = r"""
#include <stdint.h>
#include <string.h>

/* float64 data + float32 score: halved score buffer improves L1 cache utilization */
void csc_accumulate_i32_score_f32(
    const void *indptr_v, const void *indices_v, const void *data_v,
    const void *wids_v, int64_t n_wids, void *score_v, int64_t n_rows)
{
    const int32_t *indptr  = (const int32_t *)indptr_v;
    const int32_t *indices = (const int32_t *)indices_v;
    const double  *data    = (const double  *)data_v;
    const int64_t *wids    = (const int64_t *)wids_v;
    float         *score   = (float         *)score_v;
    memset(score, 0, (size_t)n_rows * sizeof(float));
    for (int64_t i = 0; i < n_wids; ++i) {
        int32_t col = (int32_t)wids[i];
        int32_t s = indptr[col], e = indptr[col + 1];
        for (int32_t j = s; j < e; ++j)
            score[indices[j]] += (float)data[j];
    }
}

void csc_accumulate_i64(
    const void *indptr_v, const void *indices_v, const void *data_v,
    const void *wids_v, int64_t n_wids, void *score_v, int64_t n_rows)
{
    const int64_t *indptr  = (const int64_t *)indptr_v;
    const int64_t *indices = (const int64_t *)indices_v;
    const double  *data    = (const double  *)data_v;
    const int64_t *wids    = (const int64_t *)wids_v;
    double        *score   = (double        *)score_v;
    memset(score, 0, (size_t)n_rows * sizeof(double));
    for (int64_t i = 0; i < n_wids; ++i) {
        int64_t col = wids[i];
        int64_t s = indptr[col], e = indptr[col + 1];
        for (int64_t j = s; j < e; ++j)
            score[indices[j]] += data[j];
    }
}
"""
    VP = ctypes.c_void_p
    try:
        tmpdir = tempfile.mkdtemp(prefix="bm25_csc_")
        src = os.path.join(tmpdir, "csc_accum.c")
        ext = ".dylib" if sys.platform == "darwin" else ".so"
        lib_path = os.path.join(tmpdir, "csc_accum" + ext)
        with open(src, "w") as f:
            f.write(C_SRC)
        cc = "clang" if sys.platform == "darwin" else "gcc"
        subprocess.check_call(
            [cc, "-Ofast", "-march=native", "-ffast-math", "-shared", "-fPIC", "-o", lib_path, src],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lib = ctypes.CDLL(lib_path)
        lib.csc_accumulate_i32_score_f32.restype = None
        lib.csc_accumulate_i32_score_f32.argtypes = [VP, VP, VP, VP, ctypes.c_int64, VP, ctypes.c_int64]
        lib.csc_accumulate_i64.restype = None
        lib.csc_accumulate_i64.argtypes = [VP, VP, VP, VP, ctypes.c_int64, VP, ctypes.c_int64]
        return lib
    except Exception:
        return None

_csc_accel = _try_compile_csc_accel()


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
        rows = []
        cols = []
        data = []

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
                wid = self._vocab.get(word)
                if wid is None:
                    wid = len(self._vocab)
                    self._vocab[word] = wid
                rows.append(self.corpus_size)
                cols.append(wid)
                data.append(freq)

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        self.doc_len = np.array(self.doc_len)

        from scipy.sparse import csc_array
        self._tf_matrix = csc_array(
            (np.array(data, dtype=np.float64), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(self.corpus_size, len(self._vocab))
        )
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
        tfm = self._tf_matrix
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

        # Set up C-accelerated scoring if available
        self._use_c_accel = False
        if _csc_accel is not None:
            import ctypes
            sm = self._score_matrix
            idx_dtype = sm.indices.dtype
            if idx_dtype == np.int32:
                # float32 score buffer: halved L1 cache pressure on random writes
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
            score_dtype = np.float32 if self._c_use_f32_score else np.float64
            self._c_score_buf = np.empty(self.corpus_size, dtype=score_dtype)
            self._c_wid_buf = np.empty(512, dtype=np.int64)
            # Cache raw integer pointers (avoids typed pointer creation per call)
            self._c_ptr_indptr = self._c_indptr.ctypes.data
            self._c_ptr_indices = self._c_indices.ctypes.data
            self._c_ptr_data = self._c_data.ctypes.data
            self._c_ptr_score = self._c_score_buf.ctypes.data
            self._c_ptr_wids = self._c_wid_buf.ctypes.data
            self._c_n_rows = ctypes.c_int64(self.corpus_size)
            self._c_int64 = ctypes.c_int64
            self._use_c_accel = True

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        if self._use_c_accel:
            return self._get_scores_c(query)
        score = np.zeros(self.corpus_size)
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
        wid_buf = self._c_wid_buf
        n = 0
        for q in query:
            wid = vocab_get(q)
            if wid is not None:
                wid_buf[n] = wid
                n += 1
        if n == 0:
            return np.zeros(self.corpus_size)
        self._c_fn(
            self._c_ptr_indptr, self._c_ptr_indices, self._c_ptr_data,
            self._c_ptr_wids, self._c_int64(n),
            self._c_ptr_score, self._c_n_rows)
        return self._c_score_buf.copy()

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        len_norm = self._len_norm[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + len_norm))
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
