# Sentinel Local Open-Source LLM Setup Guide

This guide walks you through setting up **Ollama** and **Qwen3 4B Instruct** for Sentinel's natural-language video investigation engine. Sentinel runs completely offline with **zero proprietary AI API keys** (no Gemini, OpenAI, or Anthropic keys required).

---

## Architecture Overview

```
Natural Language Investigator Query
                 ↓
Local LLM (Qwen3 4B via Ollama) / Deterministic Parser Fallback
                 ↓
Validated Structured Query Parameters
                 ↓
Sentinel Ground-Truth Database (SQLite / PostgreSQL)
                 ↓
Retrieved Detection & Evidence Records
                 ↓
Local LLM (Grounded Synthesis) / Deterministic Synthesizer Fallback
                 ↓
Evidence-Grounded Investigation Findings (with citations)
```

The database remains the sole source of truth. The local LLM translates questions and synthesizes grounded answers without ever fabricating evidence, timestamps, or object counts.

---

## Step 1: Install Ollama

Download and install Ollama for your operating system:

- **Windows:** Download the Windows installer from [https://ollama.com/download/windows](https://ollama.com/download/windows) and run the installer.
- **macOS:** Download the macOS app from [https://ollama.com/download/mac](https://ollama.com/download/mac) or install via Homebrew:
  ```bash
  brew install ollama
  ```
- **Linux:** Run the official install script:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Once installed, ensure the Ollama background service is running. By default, it listens on `http://127.0.0.1:11434`.

---

## Step 2: Pull the Qwen3 4B Instruct Model

Open your terminal or PowerShell and run:

```bash
ollama pull qwen3:4b-instruct
```

> **Note:** If `qwen3:4b-instruct` is not yet available in your local registry, or if you prefer an alternative high-performance compact model, Sentinel also supports `qwen2.5:3b`, `llama3.2:3b`, or any model configured via the `LLM_MODEL` environment variable.

---

## Step 3: Verify Model Installation

Confirm that the model is loaded into your local Ollama library:

```bash
ollama list
```

You should see an entry similar to:
```
NAME                    ID              SIZE      MODIFIED
qwen3:4b-instruct       latest          2.5 GB    Just now
```

You can test a quick interactive prompt:
```bash
ollama run qwen3:4b-instruct "Hello Sentinel"
```

---

## Step 4: Run Sentinel

1. Ensure your `.env` or environment variables are configured (optional, defaults are already pre-configured):
   ```env
   LLM_PROVIDER=local_ollama
   LLM_MODEL=qwen3:4b-instruct
   OLLAMA_BASE_URL=http://127.0.0.1:11434
   LLM_TEMPERATURE=0.1
   ```
   *Notice: No `LLM_API_KEY` or `GEMINI_API_KEY` is required.*

2. Start the Sentinel Backend:
   ```bash
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

3. Start the Sentinel Frontend:
   ```bash
   cd frontend
   npm run dev
   ```

---

## Step 5: Verify Sentinel Communication with Ollama

You can verify that Sentinel is connected and communicating with Ollama in three ways:

### A. Run the Sentinel Automated Verification Script
```bash
python scripts/verify_local_llm.py
```
This tests:
- Ollama endpoint reachability
- Model availability
- Structured query translation (JSON extraction)
- Grounded answer synthesis
- API key independence

### B. Query the Sentinel Health Endpoint
Open `http://127.0.0.1:8000/api/health` in your browser or curl:
```bash
curl http://127.0.0.1:8000/api/health
```
Expected output:
```json
{
  "status": "ok",
  "service": "Sentinel Backend",
  "version": "0.1.0",
  "environment": "development",
  "message": "Sentinel backend is running successfully.",
  "llm": {
    "provider": "Local Ollama",
    "model": "qwen3:4b-instruct",
    "api_key_required": false,
    "provider_status": "Available"
  }
}
```

For deeper diagnostics:
```bash
curl http://127.0.0.1:8000/api/health/llm
```

### C. Test Natural-Language Investigation in the UI
In the Sentinel Forensic Investigation Workspace:
1. Open a processed video.
2. Enter an investigation query, e.g.:
   - *"Did any cars appear between 5 and 15 seconds?"*
   - *"Give me a summary of what happened."*
   - *"Were there any suspicious events or takeaway behaviors?"*
3. Sentinel will interpret your query using local Ollama, query the database, and display evidence citations (`[08.01s]`, `[EVENT-XXXX]`, `[EV-XXXX]`).

---

## Fault Tolerance & Offline Fallback

If Ollama is stopped or unreachable at any time:
- Sentinel **does not crash**.
- Sentinel automatically engages its built-in **deterministic investigation engine**.
- Timeline filtering, bounding box citations, and evidence links continue to operate seamlessly.
