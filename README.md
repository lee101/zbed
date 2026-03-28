# zbed

`zbed` is a Zig port of the fast static-embedding path from `gobed`, now paired with a Zig `bed` CLI for semantic filesystem search.

The core model is `sentence-transformers/static-retrieval-mrl-en-v1` in quantized `int8/512` form. Tokenization is WordPiece over `tokenizer.json`; inference is table lookup + mean pooling; search runs directly over quantized vectors.

## Binaries

- `zbed`: model-oriented CLI for embedding, indexing, status, and benchmarks
- `bed`: filesystem search CLI built on the same engine

## Features

- `int8/512` static embedding model loading from safetensors
- low-allocation tokenizer and embedding scratch buffers
- quantized flat search over persisted int8 vectors
- text files indexed by filename and line content
- binary/media files indexed by filename only when `--search-binaries` is enabled
- `.gitignore`-aware directory walking
- cross-platform Zig build for Linux, macOS, and Windows

## Quick start

```bash
./setup.sh
zig build

./zig-out/bin/bed index . --search-binaries
./zig-out/bin/bed "opus audio"
./zig-out/bin/zbed embed "semantic file search"
```

If `../gobed/model` already exists, `setup.sh` reuses that quantized model before downloading anything.

## Commands

```text
bed <query>                    Search indexed files and filenames
bed index [path]               Build a quantized search index
bed status [path]              Show index statistics

zbed embed <text>              Run one embedding inference
zbed bench                     Run embedding/search benchmarks
zbed status [path]             Show index statistics
```

Useful flags:

```text
-p, --path PATH
-l, --limit N
-t, --threshold F
-m, --model-dir DIR
    --search-binaries
    --gpu
```

`--gpu` is currently a request flag with CPU fallback. The build is structured so a Zig CUDA bridge can be linked in without changing the CLI surface.

## Index semantics

- Text files produce one filename document plus one document per qualifying content line.
- Binary files produce one filename-only document when `--search-binaries` is enabled.
- Filename search text is normalized from the relative path, so `a.opus` becomes searchable via terms like `a` and `opus`.

## Build

Requires Zig `0.15.x`.

```bash
zig build --global-cache-dir .zig-global-cache
zig build test --global-cache-dir .zig-global-cache
```

## CI

The included CI workflow installs Zig, runs `./setup.sh`, builds both binaries, runs unit tests, and performs real model-backed smoke inference/search on Linux, macOS, and Windows.
