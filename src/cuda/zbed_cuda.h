#ifndef ZBED_CUDA_H
#define ZBED_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int zbed_cuda_device_count(void);
int zbed_cuda_get_device_name(int device, char *out, int out_len);
const char *zbed_cuda_last_error_message(void);

void *zbed_cuda_embedder_create(const int8_t *weights, const float *scales, int vocab_size, int dim);
int zbed_cuda_embedder_embed(void *ctx, const int16_t *tokens, int num_tokens, float *output);
void zbed_cuda_embedder_destroy(void *ctx);

void *zbed_cuda_search_create(const int8_t *embeddings, const float *norms, int n_docs, int dim);
int zbed_cuda_search_scores(void *ctx, const int8_t *query, float query_norm, float *scores_out);
void zbed_cuda_search_destroy(void *ctx);

#ifdef __cplusplus
}
#endif

#endif
