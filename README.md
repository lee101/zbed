# zbed

Fast semantic code search, powered by static embeddings. Pure Zig port of [gobed](https://github.com/lee101/gobed).

> **Primary repository**: [codex-infinity.com/lee101/zbed](https://codex-infinity.com/lee101/zbed) | GitHub mirror: [github.com/lee101/zbed](https://github.com/lee101/zbed)

Uses int8 quantized embeddings from [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) with SIMD-accelerated cosine similarity search via Zig's `@Vector`.

## Features

- **WordPiece tokenizer** — loads vocabulary from HuggingFace `tokenizer.json`
- **Int8 quantized embeddings** — safetensors format, ~8x memory savings vs float32
- **SIMD cosine similarity** — uses `@Vector` for vectorized dot products and embedding accumulation
- **Flat brute-force search** — top-k results with min-heap and precomputed norms
- **Persistent index** — `.zbed/index.bin` binary format for fast reload
- **.gitignore-aware** — respects `.gitignore` patterns when walking directories
- **Pure Zig stdlib** — no external dependencies

## Quick Start

```bash
# 1. Download model weights (~8 MB quantized)
./setup.sh

# 2. Build
zig build

# 3. Index your codebase
./zig-out/bin/zbed index .

# 4. Search
./zig-out/bin/zbed "error handling"
```

## Usage

```
zbed <query>            Search indexed files for semantically similar lines
zbed index [path]       Build semantic index for a directory (default: .)
zbed bench              Run embedding benchmark
zbed status [path]      Show index statistics
zbed help               Show help message
```

### Options

```
-l, --limit N           Maximum results to show (default: 10)
-t, --threshold F       Similarity threshold 0.0-1.0 (default: 0.3)
-m, --model-dir DIR     Path to model directory (default: model/)
```

### Environment Variables

- `ZBED_MODEL_PATH` — override model directory location

## Architecture

```
src/
  tokenizer.zig   WordPiece BPE tokenizer from tokenizer.json
  embed.zig       Safetensors int8 embeddings with SIMD @Vector
  search.zig      Flat cosine similarity index with top-k heap
  index.zig       .zbed/index.bin persistence + .gitignore-aware walk
  main.zig        CLI: zbed QUERY, zbed index, zbed bench, zbed status
  lib.zig         Public library API
```

### How It Works

1. **Tokenization**: Text is lowercased, split into words, and each word is broken into subword tokens using the WordPiece algorithm (same as BERT).

2. **Embedding**: Each token ID maps to a row in the int8 embedding table. The row is dequantized (`int8 * scale`) and accumulated. Mean pooling averages across all tokens to produce a single vector.

3. **Indexing**: Files are walked (respecting `.gitignore`), each line is embedded, and the resulting vectors plus metadata are serialized to `.zbed/index.bin`.

4. **Search**: The query is embedded, then cosine similarity is computed against every indexed line using SIMD-accelerated dot products. A min-heap maintains the top-k results.

### SIMD

Embedding accumulation and cosine similarity both use Zig's `@Vector(8, f32)` for 8-wide SIMD operations. This maps to SSE/AVX on x86 and NEON on ARM, with automatic scalar fallback.

## Model

The default model is [static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1), a static embedding model (no neural network inference — just table lookups + mean pooling). The setup script downloads and quantizes it to int8 with per-token scaling.

| Property | Value |
|----------|-------|
| Dimensions | 256 (truncated from 1024) |
| Vocab size | 30,522 |
| Format | safetensors (int8 + f32 scales) |
| Size | ~8 MB |

## Building

Requires Zig 0.13.0+.

```bash
zig build              # debug build
zig build -Doptimize=ReleaseFast  # optimized build
zig build test         # run unit tests
```

## Testing

```bash
zig build test
```

Tests cover:
- WordPiece tokenization and subword splitting
- Int8 quantization round-trip accuracy
- Embedding computation with SIMD
- Cosine similarity (identical, orthogonal, opposite vectors)
- Flat index search correctness
- Index save/load round-trip
- .gitignore pattern matching
- Text file detection

## License

MIT
