# Sentinel Ollama Private Service Dockerfile
FROM ollama/ollama:latest

# Install curl for API polling
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy startup script that ensures qwen3:4b-instruct is downloaded on first boot
COPY docker/start-ollama.sh /usr/local/bin/start-ollama.sh
RUN chmod +x /usr/local/bin/start-ollama.sh

EXPOSE 11434

ENTRYPOINT ["/usr/local/bin/start-ollama.sh"]
