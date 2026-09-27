"""
Local Ollama LLM Provider for Sentinel

Connects Sentinel to a locally hosted Ollama instance running Qwen3 4B Instruct
(or any configured open-source model) via Ollama's local HTTP REST API.
Operates fully offline with NO proprietary AI API keys required.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
import requests

from ai.investigation.provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class LocalOllamaProvider(LLMProvider):
    """
    Local Ollama implementation of Sentinel's LLMProvider interface.
    Communicates with Ollama through its native HTTP REST API:
    - GET /api/tags for model inspection & availability checks
    - POST /api/chat for structured JSON intent & grounded synthesis
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "qwen3:4b-instruct"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: float = 30.0,
        availability_timeout: float = 2.0,
    ):
        raw_url = (base_url or self.DEFAULT_BASE_URL).strip()
        self.base_url = raw_url.rstrip("/")
        self.model = (model or self.DEFAULT_MODEL).strip()
        self.temperature = temperature
        self.timeout = timeout
        self.availability_timeout = availability_timeout

    def is_available(self) -> bool:
        """
        Check if the local Ollama instance is reachable and responding.
        Uses a short timeout to prevent blocking if Ollama is not running.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=self.availability_timeout)
            if resp.status_code == 200:
                return True
            logger.debug(f"Ollama health check returned status code {resp.status_code}")
            return False
        except Exception as e:
            logger.debug(f"Ollama is unreachable at {self.base_url}: {e}")
            return False

    def _call_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        json_format: bool = False,
    ) -> str:
        """
        Execute chat completion against Ollama's /api/chat endpoint.
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        if json_format:
            payload["format"] = "json"

        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise LLMProviderError(f"Ollama API error ({resp.status_code}): {resp.text}")

            data = resp.json()
            message = data.get("message", {})
            content = message.get("content", "")
            if not content:
                raise LLMProviderError("Empty content returned by Ollama.")

            return content
        except requests.RequestException as e:
            logger.warning(f"Ollama HTTP request failed: {e}")
            raise LLMProviderError(f"Ollama connection failure: {str(e)}")

    def generate_structured_intent(
        self, user_query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Translate user natural language query into structured retrieval parameters
        using Qwen3 4B Instruct via Ollama with JSON mode enabled.
        Preserves the exact Sentinel intent schema.
        """
        system_instruction = (
            "You are Sentinel's query translation engine. Your job is to convert investigator questions "
            "into a structured JSON filter object for querying the CCTV detections database.\n"
            "DO NOT answer the question. Only output valid JSON matching this schema:\n"
            "{\n"
            '  "intent": "investigate" | "summarize" | "activity_analysis" | "guardrail",\n'
            '  "object_classes": ["car", "person", ...] or null,\n'
            '  "start_time": float or null,\n'
            '  "end_time": float or null,\n'
            '  "min_confidence": float or null,\n'
            '  "result_type": "detections" | "events" | "count" | "security_events",\n'
            '  "is_summary_request": bool,\n'
            '  "is_activity_request": bool,\n'
            '  "guardrail_category": "identity" | "criminal_attribution" | null\n'
            "}\n"
            "Rules:\n"
            "- Only use recognized classes: person, car, bicycle, motorcycle, bus, truck, backpack, handbag, suitcase, bottle, cell phone, chair, traffic light, stop sign.\n"
            "- Map 'vehicle' to object_classes=['car','bus','truck','motorcycle','bicycle'].\n"
            "- Do not infer colors.\n"
            "- If user asks who someone is or to identify someone by name, mark intent='guardrail', guardrail_category='identity'.\n"
            "- If user asks if someone is a criminal/thief/guilty, mark intent='guardrail', guardrail_category='criminal_attribution'.\n"
            "- If user asks about summary/overview/what happened, set is_summary_request=true.\n"
            "- If user asks about activity/movement density/suspicious patterns, set is_activity_request=true.\n"
            "- If user asks about theft/takeaway, set event_type='POTENTIAL_THEFT', result_type='security_events'.\n"
            "Output JSON only."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]

        if history:
            for item in history[-4:]:
                messages.append({
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                })

        messages.append({
            "role": "user",
            "content": f"Investigator query: {user_query}",
        })

        raw_text = self._call_ollama_chat(messages, json_format=True)

        try:
            cleaned = raw_text.strip()
            # Strip markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Find outer JSON braces if extra text exists
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                cleaned = cleaned[start_idx : end_idx + 1]

            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise LLMProviderError(f"Expected JSON object, got {type(parsed)}")
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse Ollama JSON output: {raw_text}. Error: {e}")
            raise LLMProviderError(f"Malformed structured output from provider: {e}")

    def generate_grounded_response(
        self,
        user_query: str,
        retrieved_data: Dict[str, Any],
        history: Optional[List[Dict[str, str]]] = None,
        context_notes: Optional[str] = None,
    ) -> str:
        """
        Synthesize a natural language investigation answer strictly grounded
        in the provided retrieved Sentinel database records and evidence.
        The database remains the sole source of truth.
        """
        system_instruction = (
            "You are Sentinel AI, an evidence-grounded video investigation assistant.\n"
            "CORE DIRECTIVE: The retrieved database records and evidence are the ABSOLUTE SOURCE OF TRUTH.\n"
            "NEVER invent timestamps, object counts, confidence scores, evidence, or events.\n"
            "NEVER perform facial recognition, claim identities, or label individuals as criminals or thieves.\n"
            "Distinguish weak/isolated raw model detections from persistent visual intelligence: "
            "If only an isolated low-confidence (<0.50) detection is present for an object, describe it as an isolated low-confidence model prediction that did not persist across frames, rather than claiming the object was present. "
            "Only describe an object as confirmed or active when supported by persistent, validated detections.\n"
            "For potential theft behavioral patterns, refer to 'Potential Theft Pattern (Human verification required)' and never claim legal confirmation or call a person a thief.\n"
            "Always cite evidence and events using tags like [08.01s], [DET-XXXX], [EVENT-XXXX], or [EV-XXXX] where available.\n"
            "If no records were found, clearly state that no matching data was detected.\n"
            "Be professional, concise, and structured."
        )

        data_summary = json.dumps(retrieved_data, indent=2, default=str)
        user_prompt = (
            f"User Question: {user_query}\n\n"
            f"Retrieved Sentinel Ground Truth Records:\n{data_summary}\n\n"
        )
        if context_notes:
            user_prompt += f"Investigator Notes:\n{context_notes}\n\n"
        user_prompt += "Synthesize a clear, grounded response citing relevant timestamps and records."

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]

        if history:
            for item in history[-4:]:
                messages.append({
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                })

        messages.append({"role": "user", "content": user_prompt})

        return self._call_ollama_chat(messages, json_format=False)


def check_local_llm_health(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """
    Internal health check utility for verifying local Ollama availability:
    1. Verifies Ollama endpoint is reachable.
    2. Verifies configured model is pulled / exists.
    3. Verifies model can generate a small response.
    Never raises an uncaught exception; returns diagnostic dictionary.
    """
    from backend.app.core.config import settings

    target_base = (base_url or getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    target_model = (model or getattr(settings, "LLM_MODEL", "qwen3:4b-instruct")).strip()

    health_status: Dict[str, Any] = {
        "reachable": False,
        "model_exists": False,
        "can_generate": False,
        "available": False,
        "base_url": target_base,
        "model": target_model,
        "error": None,
        "models_found": [],
    }

    # Step 1: Reachability & model list
    try:
        resp = requests.get(f"{target_base}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            health_status["error"] = f"Endpoint returned status {resp.status_code}"
            return health_status

        health_status["reachable"] = True
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "") for m in models if isinstance(m, dict)]
        health_status["models_found"] = model_names

        # Check model existence (exact match or tag match e.g. qwen3:4b-instruct)
        target_clean = target_model.lower()
        model_matched = any(
            target_clean == name.lower()
            or name.lower().startswith(f"{target_clean}:")
            or target_clean.startswith(f"{name.lower().split(':')[0]}:")
            for name in model_names
        )
        health_status["model_exists"] = model_matched

        if not model_matched:
            health_status["error"] = f"Configured model '{target_model}' not found in Ollama models list."
            return health_status

    except Exception as e:
        health_status["error"] = f"Connection error: {str(e)}"
        return health_status

    # Step 2: Mini-inference generation check
    try:
        gen_url = f"{target_base}/api/generate"
        gen_payload = {
            "model": target_model,
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 3, "temperature": 0.0},
        }
        gen_resp = requests.post(gen_url, json=gen_payload, timeout=timeout + 5.0)
        if gen_resp.status_code == 200:
            health_status["can_generate"] = True
            health_status["available"] = True
        else:
            health_status["error"] = f"Inference test returned HTTP {gen_resp.status_code}"
    except Exception as e:
        health_status["error"] = f"Inference test failed: {str(e)}"

    return health_status
