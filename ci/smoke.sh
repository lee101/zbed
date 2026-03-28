#!/usr/bin/env bash
set -euo pipefail

export ZIG_GLOBAL_CACHE_DIR="${ZIG_GLOBAL_CACHE_DIR:-.zig-global-cache}"

if [[ -z "${ZBED_MODEL_PATH:-}" ]]; then
  if [[ -d "model" ]]; then
    export ZBED_MODEL_PATH="model"
  elif [[ -d "../gobed/model" ]]; then
    export ZBED_MODEL_PATH="../gobed/model"
  fi
fi

if [[ ! -d "${ZBED_MODEL_PATH:-}" ]]; then
  echo "ZBED_MODEL_PATH is not set to a valid model directory" >&2
  exit 1
fi

BED_BIN="./zig-out/bin/bed"
ZBED_BIN="./zig-out/bin/zbed"
if [[ -x "./zig-out/bin/bed.exe" ]]; then
  BED_BIN="./zig-out/bin/bed.exe"
  ZBED_BIN="./zig-out/bin/zbed.exe"
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmpdir/readme.txt" <<'EOF'
hello world
semantic search line
EOF
printf 'binary-ish' >"$tmpdir/a.opus"

"$ZBED_BIN" embed "audio file search mp3 opus mp4" >/dev/null
"$BED_BIN" index "$tmpdir" --search-binaries >/dev/null

semantic_out="$("$BED_BIN" "semantic search" --path "$tmpdir" --search-binaries)"
audio_out="$("$BED_BIN" "opus audio" --path "$tmpdir" --search-binaries)"

printf '%s\n' "$semantic_out" | grep -q "semantic search line"
printf '%s\n' "$audio_out" | grep -q "binary filename: a.opus"

echo "smoke ok"
