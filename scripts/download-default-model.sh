#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/models/Qwen3.5-0.8B-Q8_0.gguf"
url="https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"
expected="37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f"
mkdir -p "$root/models"
curl -L --fail --continue-at - -o "$dst" "$url"
actual="$(shasum -a 256 "$dst" | awk '{print $1}')"
[[ "$actual" == "$expected" ]] || { echo "checksum mismatch: $actual" >&2; exit 1; }
echo "$dst"
