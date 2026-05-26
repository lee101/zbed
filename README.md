# zbed

`zbed` is a pure Zig port of the core filesystem search path from
[`lee101/gobed`](https://github.com/lee101/gobed). It uses the
`sentence-transformers/static-retrieval-mrl-en-v1` static embedding model,
quantized to int8 safetensors, to embed file paths and text lines into a flat
cosine similarity index.

## Features

- WordPiece tokenization from `tokenizer.json`
- int8 safetensors embedding table loading
- mean-pooled embeddings with `@Vector` SIMD accumulation
- flat top-k cosine similarity search
- persisted `.zbed/index.bin` index files
- `.gitignore`-aware directory walking
- text files indexed by filename and matching content lines
- optional binary/media filename indexing with `--search-binaries`
- only Zig standard library dependencies

## Setup

Install Zig 0.15 or newer, then download and quantize the model:

```bash
./setup.sh
```

The setup script downloads:

- `0_StaticEmbedding/tokenizer.json`
- `0_StaticEmbedding/model.safetensors`

It writes a compact quantized model to:

```text
model/modelint8_512dim.safetensors
```

You can also set `ZBED_MODEL_PATH` to a directory containing `tokenizer.json`
and a supported safetensors file.

## Build And Test

```bash
zig build
zig build test
```

The binary is installed at:

```text
zig-out/bin/zbed
```

## CLI

```text
zbed QUERY                  Search the index in the current directory
zbed index [PATH]           Build PATH/.zbed/index.bin
zbed status [PATH]          Print index statistics
zbed bench                  Run embedding and search benchmarks
```

Useful flags:

```text
-p, --path PATH             Search/index path
-l, --limit N               Maximum results
-t, --threshold F           Similarity threshold
-m, --model-dir DIR         Model directory
    --search-binaries       Index binary/media filenames
```

Example:

```bash
./zig-out/bin/zbed index . --search-binaries
./zig-out/bin/zbed "database connection pooling"
./zig-out/bin/zbed status .
./zig-out/bin/zbed bench
```

## Index Format

`zbed index PATH` writes `PATH/.zbed/index.bin` with:

- magic/version header
- embedding dimension and document count
- document metadata: kind, line number, path, content, scale, norm
- contiguous int8 embedding rows

The persisted vectors are searched directly without rebuilding the embedding
store.
