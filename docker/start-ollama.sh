#!/bin/sh
set -e

echo "[Ollama Service] Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!

echo "[Ollama Service] Waiting for Ollama API to be reachable..."
until curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done

echo "[Ollama Service] Ollama API is active."

# Check if qwen3:4b-instruct is already present in local cache
if ! curl -s http://127.0.0.1:11434/api/tags | grep -q "qwen3:4b-instruct"; then
    echo "[Ollama Service] Model 'qwen3:4b-instruct' not found in cache. Pulling now..."
    ollama pull qwen3:4b-instruct
    echo "[Ollama Service] Successfully pulled qwen3:4b-instruct."
else
    echo "[Ollama Service] Model 'qwen3:4b-instruct' is already cached."
fi

echo "[Ollama Service] Ready to serve inference requests on port 11434."
wait "$OLLAMA_PID"
