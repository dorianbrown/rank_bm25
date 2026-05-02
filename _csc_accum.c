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
