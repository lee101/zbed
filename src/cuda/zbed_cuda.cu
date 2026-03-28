#include "zbed_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char g_last_error[256];

static int set_error_msg(const char *msg) {
    snprintf(g_last_error, sizeof(g_last_error), "%s", msg);
    return 0;
}

static int set_error_cuda(const char *prefix, cudaError_t err) {
    snprintf(g_last_error, sizeof(g_last_error), "%s: %s", prefix, cudaGetErrorString(err));
    return 0;
}

const char *zbed_cuda_last_error_message(void) { return g_last_error; }

int zbed_cuda_device_count(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        set_error_cuda("cudaGetDeviceCount failed", err);
        return 0;
    }
    return count;
}

int zbed_cuda_get_device_name(int device, char *out, int out_len) {
    if (out == NULL || out_len <= 0) return set_error_msg("invalid output buffer");
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return set_error_cuda("cudaGetDeviceProperties failed", err);
    snprintf(out, (size_t)out_len, "%s", prop.name);
    return 1;
}

typedef struct {
    int8_t *d_weights;
    float *d_scales;
    int16_t *d_tokens;
    float *d_output;
    int vocab_size;
    int dim;
    int max_tokens;
    cudaStream_t stream;
} zbed_cuda_embedder_t;

typedef struct {
    int8_t *d_embeddings;
    float *d_norms;
    int8_t *d_query;
    float *d_scores;
    int n_docs;
    int dim;
    cudaStream_t stream;
} zbed_cuda_search_t;

__global__ void embed_tokens_kernel(
    const int8_t *weights,
    const float *scales,
    const int16_t *tokens,
    int num_tokens,
    int vocab_size,
    float *output
) {
    int dim = threadIdx.x + blockIdx.x * blockDim.x;
    if (dim >= 512) return;

    float sum = 0.0f;
    int valid = 0;
    for (int i = 0; i < num_tokens; ++i) {
        int16_t tok = tokens[i];
        if (tok < 0 || tok >= vocab_size) continue;
        sum += (float)weights[tok * 512 + dim] * scales[tok];
        valid += 1;
    }

    output[dim] = valid > 0 ? sum / (float)valid : 0.0f;
}

__device__ __forceinline__ int dot_i8_512(const int8_t *a, const int8_t *b) {
    int acc = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    #pragma unroll
    for (int i = 0; i < 512; i += 4) {
        int pa = *reinterpret_cast<const int *>(a + i);
        int pb = *reinterpret_cast<const int *>(b + i);
        acc = __dp4a(pa, pb, acc);
    }
#else
    #pragma unroll 8
    for (int i = 0; i < 512; ++i) {
        acc += (int)a[i] * (int)b[i];
    }
#endif
    return acc;
}

__global__ void cosine_scores_kernel(
    const int8_t *embeddings,
    const float *norms,
    const int8_t *query,
    float query_norm,
    int n_docs,
    float *scores
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_docs) return;

    float norm = norms[idx];
    if (norm <= 0.0f || query_norm <= 0.0f) {
        scores[idx] = 0.0f;
        return;
    }

    const int8_t *row = embeddings + (size_t)idx * 512;
    int dot = dot_i8_512(query, row);
    scores[idx] = (float)dot / (query_norm * norm);
}

void *zbed_cuda_embedder_create(const int8_t *weights, const float *scales, int vocab_size, int dim) {
    if (weights == NULL || scales == NULL) return NULL;
    if (dim != 512) {
        set_error_msg("CUDA backend currently supports dim=512 only");
        return NULL;
    }

    zbed_cuda_embedder_t *ctx = (zbed_cuda_embedder_t *)calloc(1, sizeof(zbed_cuda_embedder_t));
    if (!ctx) {
        set_error_msg("calloc failed");
        return NULL;
    }

    ctx->vocab_size = vocab_size;
    ctx->dim = dim;
    ctx->max_tokens = 256;

    cudaError_t err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        free(ctx);
        set_error_cuda("cudaStreamCreate failed", err);
        return NULL;
    }

    size_t weights_size = (size_t)vocab_size * 512 * sizeof(int8_t);
    size_t scales_size = (size_t)vocab_size * sizeof(float);

    err = cudaMalloc(&ctx->d_weights, weights_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_scales, scales_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_tokens, (size_t)ctx->max_tokens * sizeof(int16_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_output, 512 * sizeof(float));
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpy(ctx->d_weights, weights, weights_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(ctx->d_scales, scales, scales_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    return ctx;

fail:
    if (ctx->d_weights) cudaFree(ctx->d_weights);
    if (ctx->d_scales) cudaFree(ctx->d_scales);
    if (ctx->d_tokens) cudaFree(ctx->d_tokens);
    if (ctx->d_output) cudaFree(ctx->d_output);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
    set_error_cuda("embedder init failed", err);
    return NULL;
}

int zbed_cuda_embedder_embed(void *ctx_ptr, const int16_t *tokens, int num_tokens, float *output) {
    if (ctx_ptr == NULL || tokens == NULL || output == NULL) return set_error_msg("invalid embedder args");
    if (num_tokens <= 0) {
        memset(output, 0, 512 * sizeof(float));
        return 1;
    }

    zbed_cuda_embedder_t *ctx = (zbed_cuda_embedder_t *)ctx_ptr;
    if (num_tokens > ctx->max_tokens) {
        cudaFree(ctx->d_tokens);
        ctx->max_tokens = num_tokens + 64;
        cudaError_t err = cudaMalloc(&ctx->d_tokens, (size_t)ctx->max_tokens * sizeof(int16_t));
        if (err != cudaSuccess) return set_error_cuda("cudaMalloc d_tokens failed", err);
    }

    cudaError_t err = cudaMemcpyAsync(ctx->d_tokens, tokens, (size_t)num_tokens * sizeof(int16_t), cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) return set_error_cuda("cudaMemcpyAsync tokens failed", err);

    embed_tokens_kernel<<<1, 512, 0, ctx->stream>>>(ctx->d_weights, ctx->d_scales, ctx->d_tokens, num_tokens, ctx->vocab_size, ctx->d_output);
    err = cudaGetLastError();
    if (err != cudaSuccess) return set_error_cuda("embed_tokens_kernel launch failed", err);

    err = cudaMemcpyAsync(output, ctx->d_output, 512 * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) return set_error_cuda("cudaMemcpyAsync output failed", err);

    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) return set_error_cuda("cudaStreamSynchronize failed", err);
    return 1;
}

void zbed_cuda_embedder_destroy(void *ctx_ptr) {
    zbed_cuda_embedder_t *ctx = (zbed_cuda_embedder_t *)ctx_ptr;
    if (!ctx) return;
    if (ctx->d_weights) cudaFree(ctx->d_weights);
    if (ctx->d_scales) cudaFree(ctx->d_scales);
    if (ctx->d_tokens) cudaFree(ctx->d_tokens);
    if (ctx->d_output) cudaFree(ctx->d_output);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    free(ctx);
}

void *zbed_cuda_search_create(const int8_t *embeddings, const float *norms, int n_docs, int dim) {
    if (embeddings == NULL || norms == NULL) return NULL;
    if (dim != 512) {
        set_error_msg("CUDA search currently supports dim=512 only");
        return NULL;
    }

    zbed_cuda_search_t *ctx = (zbed_cuda_search_t *)calloc(1, sizeof(zbed_cuda_search_t));
    if (!ctx) {
        set_error_msg("calloc failed");
        return NULL;
    }

    ctx->n_docs = n_docs;
    ctx->dim = dim;
    cudaError_t err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        free(ctx);
        set_error_cuda("cudaStreamCreate failed", err);
        return NULL;
    }

    size_t emb_size = (size_t)n_docs * 512 * sizeof(int8_t);
    size_t norm_size = (size_t)n_docs * sizeof(float);

    err = cudaMalloc(&ctx->d_embeddings, emb_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_norms, norm_size);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_query, 512 * sizeof(int8_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&ctx->d_scores, (size_t)n_docs * sizeof(float));
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpy(ctx->d_embeddings, embeddings, emb_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(ctx->d_norms, norms, norm_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    return ctx;

fail:
    if (ctx->d_embeddings) cudaFree(ctx->d_embeddings);
    if (ctx->d_norms) cudaFree(ctx->d_norms);
    if (ctx->d_query) cudaFree(ctx->d_query);
    if (ctx->d_scores) cudaFree(ctx->d_scores);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
    set_error_cuda("search init failed", err);
    return NULL;
}

int zbed_cuda_search_scores(void *ctx_ptr, const int8_t *query, float query_norm, float *scores_out) {
    if (ctx_ptr == NULL || query == NULL || scores_out == NULL) return set_error_msg("invalid search args");

    zbed_cuda_search_t *ctx = (zbed_cuda_search_t *)ctx_ptr;
    cudaError_t err = cudaMemcpyAsync(ctx->d_query, query, 512 * sizeof(int8_t), cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) return set_error_cuda("cudaMemcpyAsync query failed", err);

    int threads = 256;
    int blocks = (ctx->n_docs + threads - 1) / threads;
    cosine_scores_kernel<<<blocks, threads, 0, ctx->stream>>>(
        ctx->d_embeddings, ctx->d_norms, ctx->d_query, query_norm, ctx->n_docs, ctx->d_scores
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) return set_error_cuda("cosine_scores_kernel launch failed", err);

    err = cudaMemcpyAsync(scores_out, ctx->d_scores, (size_t)ctx->n_docs * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) return set_error_cuda("cudaMemcpyAsync scores failed", err);

    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) return set_error_cuda("cudaStreamSynchronize failed", err);
    return 1;
}

void zbed_cuda_search_destroy(void *ctx_ptr) {
    zbed_cuda_search_t *ctx = (zbed_cuda_search_t *)ctx_ptr;
    if (!ctx) return;
    if (ctx->d_embeddings) cudaFree(ctx->d_embeddings);
    if (ctx->d_norms) cudaFree(ctx->d_norms);
    if (ctx->d_query) cudaFree(ctx->d_query);
    if (ctx->d_scores) cudaFree(ctx->d_scores);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    free(ctx);
}
