"""
Unit tests for LocalOllamaProvider

Verifies:
1. Provider initialization with custom and default configurations
2. Unavailable Ollama gracefully indicates is_available() == False
3. Orchestrator gracefully falls back when Ollama is unavailable
4. Valid structured JSON is properly parsed into Sentinel intent
5. Malformed JSON or non-JSON output from Ollama is safely caught and handled
6. Grounded response sends only retrieved Sentinel data and preserves evidence rules
7. No API key is required for provider initialization or operation
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.investigation.local_provider import LocalOllamaProvider, check_local_llm_health
from ai.investigation.provider import get_llm_provider, LLMProviderError, MockLLMProvider
from ai.investigation.orchestrator import InvestigationOrchestrator
from ai.investigation.validator import StructuredIntentValidator, ValidationError
from backend.app.core.config import settings


from database.session import init_db


class TestLocalOllamaProvider(unittest.TestCase):
    """Test suite for LocalOllamaProvider and local open-source LLM integration."""

    @classmethod
    def setUpClass(cls):
        init_db()

    def test_1_provider_initialization_defaults(self):
        """Test LocalOllamaProvider initializes with proper defaults and no API key."""
        prov = LocalOllamaProvider()
        self.assertEqual(prov.base_url, "http://127.0.0.1:11434")
        self.assertEqual(prov.model, "qwen3:4b-instruct")
        self.assertEqual(prov.temperature, 0.1)

    def test_2_provider_initialization_custom(self):
        """Test LocalOllamaProvider with custom base_url and model parameters."""
        prov = LocalOllamaProvider(
            base_url="http://192.168.1.100:11434/",
            model="custom-qwen:7b",
            temperature=0.2,
        )
        self.assertEqual(prov.base_url, "http://192.168.1.100:11434")
        self.assertEqual(prov.model, "custom-qwen:7b")
        self.assertEqual(prov.temperature, 0.2)

    def test_3_factory_defaults_to_local_ollama(self):
        """Test get_llm_provider factory instantiates LocalOllamaProvider by default."""
        prov = get_llm_provider()
        self.assertIsInstance(prov, LocalOllamaProvider)
        self.assertEqual(prov.model, settings.LLM_MODEL)

    def test_4_no_api_key_required(self):
        """Verify provider does NOT require or check for any API keys."""
        with patch.dict(os.environ, {"LLM_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
            prov = get_llm_provider("local_ollama")
            self.assertIsInstance(prov, LocalOllamaProvider)
            self.assertFalse(hasattr(prov, "api_key"))

    @patch("requests.get")
    def test_5_is_available_when_ollama_reachable(self, mock_get):
        """Verify is_available returns True when Ollama /api/tags returns 200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        prov = LocalOllamaProvider()
        self.assertTrue(prov.is_available())
        mock_get.assert_called_once_with("http://127.0.0.1:11434/api/tags", timeout=2.0)

    @patch("requests.get")
    def test_6_is_available_false_when_unreachable(self, mock_get):
        """Verify is_available returns False when Ollama connection fails or times out."""
        mock_get.side_effect = Exception("Connection refused")

        prov = LocalOllamaProvider()
        self.assertFalse(prov.is_available())

    def test_7_unavailable_ollama_gracefully_falls_back_in_orchestrator(self):
        """Verify orchestrator uses deterministic fallback without crashing when Ollama is unavailable."""
        prov = LocalOllamaProvider()
        prov.is_available = MagicMock(return_value=False)

        orch = InvestigationOrchestrator(provider=prov)
        result = orch.process_investigation(
            video_id="any_video_id",
            user_query="Show cars between 8 and 12 seconds",
        )
        self.assertEqual(result["mode"], "deterministic_fallback")
        self.assertTrue(result["is_supported"])
        self.assertIn("structured_query", result)

    @patch("requests.post")
    def test_8_valid_structured_json_parsing(self, mock_post):
        """Verify valid structured JSON from Ollama is parsed and validated."""
        valid_llm_json = {
            "intent": "investigate",
            "object_classes": ["car"],
            "start_time": 8.0,
            "end_time": 12.0,
            "min_confidence": 0.75,
            "result_type": "detections",
            "is_summary_request": False,
            "is_activity_request": False,
            "guardrail_category": None,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": json.dumps(valid_llm_json)}
        }
        mock_post.return_value = mock_resp

        prov = LocalOllamaProvider()
        intent = prov.generate_structured_intent("Were there cars between 8 and 12 seconds?")

        self.assertEqual(intent["intent"], "investigate")
        self.assertEqual(intent["object_classes"], ["car"])
        self.assertEqual(intent["start_time"], 8.0)
        self.assertEqual(intent["end_time"], 12.0)

        # Validate with Sentinel's intent validator
        sanitized = StructuredIntentValidator.validate_and_sanitize(intent)
        self.assertEqual(sanitized["object_class"], "car")
        self.assertEqual(sanitized["start_time"], 8.0)
        self.assertEqual(sanitized["end_time"], 12.0)

    @patch("requests.post")
    def test_9_markdown_wrapped_json_parsing(self, mock_post):
        """Verify Ollama markdown-wrapped JSON (```json ... ```) is cleaned and parsed."""
        valid_llm_json = {
            "intent": "investigate",
            "object_classes": ["person"],
            "start_time": 2.0,
            "end_time": 5.0,
            "min_confidence": 0.8,
            "result_type": "detections",
        }
        markdown_content = f"```json\n{json.dumps(valid_llm_json)}\n```"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": markdown_content}
        }
        mock_post.return_value = mock_resp

        prov = LocalOllamaProvider()
        intent = prov.generate_structured_intent("Show persons between 2 and 5 seconds")
        self.assertEqual(intent["object_classes"], ["person"])

    @patch("requests.post")
    def test_10_malformed_json_raises_llm_provider_error(self, mock_post):
        """Verify non-JSON response from Ollama raises LLMProviderError for orchestrator fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "I am an AI, I cannot provide JSON."}
        }
        mock_post.return_value = mock_resp

        prov = LocalOllamaProvider()
        with self.assertRaises(LLMProviderError):
            prov.generate_structured_intent("Show cars")

    @patch("requests.post")
    def test_11_grounded_response_receives_only_retrieved_data(self, mock_post):
        """Verify generate_grounded_response passes only retrieved database records to Ollama."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "At [08.00s], Sentinel detected car with 85% confidence [EVENT-001].",
            }
        }
        mock_post.return_value = mock_resp

        prov = LocalOllamaProvider()
        retrieved_data = {
            "query": "Show cars",
            "count": 1,
            "results": [
                {
                    "event_id": "001",
                    "object_class": "car",
                    "timestamp": 8.0,
                    "confidence": 0.85,
                }
            ],
            "evidence": [],
        }

        resp = prov.generate_grounded_response(
            user_query="Show cars",
            retrieved_data=retrieved_data,
        )
        self.assertIn("[08.00s]", resp)

        # Inspect the payload sent to Ollama
        call_args = mock_post.call_args
        sent_json = call_args[1]["json"]
        user_message = [m for m in sent_json["messages"] if m["role"] == "user"][0]["content"]
        self.assertIn("Retrieved Sentinel Ground Truth Records", user_message)
        self.assertIn("001", user_message)
        self.assertIn("0.85", user_message)

    def test_12_unknown_provider_safely_falls_back(self):
        """Verify unknown provider configuration produces safe fallback instead of defaulting to Gemini."""
        prov = get_llm_provider("completely_unknown_llm_xyz")
        self.assertIsInstance(prov, MockLLMProvider)
        self.assertFalse(prov.is_available())

    @patch("requests.post")
    @patch("requests.get")
    def test_13_check_local_llm_health_utility(self, mock_get, mock_post):
        """Test internal health check utility with mocked Ollama endpoints."""
        # Mock GET /api/tags
        mock_tags = MagicMock()
        mock_tags.status_code = 200
        mock_tags.json.return_value = {
            "models": [{"name": "qwen3:4b-instruct:latest"}]
        }
        mock_get.return_value = mock_tags

        # Mock POST /api/generate
        mock_gen = MagicMock()
        mock_gen.status_code = 200
        mock_post.return_value = mock_gen

        health = check_local_llm_health(model="qwen3:4b-instruct")
        self.assertTrue(health["reachable"])
        self.assertTrue(health["model_exists"])
        self.assertTrue(health["can_generate"])
        self.assertTrue(health["available"])


if __name__ == "__main__":
    unittest.main()
