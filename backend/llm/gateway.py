"""Single optional OpenAI-compatible LLM gateway; never fails core deterministic analysis."""
from dataclasses import dataclass
import json
import os
import time
import requests
from pydantic import BaseModel, ValidationError


@dataclass
class GatewayResult:
    value: BaseModel | None
    provider: str
    model: str
    status: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost: float | None = None
    error: str | None = None


class LLMGateway:
    """Provider-independent interface with one configured OpenAI-compatible adapter."""
    def __init__(self):
        self.provider = os.getenv("AURA_LLM_PROVIDER", "openai_compatible")
        self.model = os.getenv("AURA_LLM_MODEL", "")
        self.base_url = os.getenv("AURA_LLM_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.api_key = os.getenv("AURA_LLM_API_KEY", "")
        self.timeout_seconds = float(os.getenv("AURA_LLM_TIMEOUT_SECONDS", "20"))

    def status(self, tenant_settings: dict | None = None) -> dict:
        enabled = bool((tenant_settings or {}).get("llm_enabled", False))
        configured = self.provider == "openai_compatible" and bool(self.model and self.api_key and self.base_url)
        available = enabled and configured
        return {"enabled_for_tenant": enabled, "provider_configured": configured,
                "provider": self.provider if available else None, "model": self.model if available else None,
                "available": available, "fallback": "deterministic"}

    def _cost(self, prompt_tokens, completion_tokens):
        if prompt_tokens is None or completion_tokens is None: return None
        if "AURA_LLM_INPUT_COST_PER_1K" not in os.environ or "AURA_LLM_OUTPUT_COST_PER_1K" not in os.environ: return None
        input_rate = float(os.environ["AURA_LLM_INPUT_COST_PER_1K"])
        output_rate = float(os.environ["AURA_LLM_OUTPUT_COST_PER_1K"])
        return round(prompt_tokens*input_rate/1000 + completion_tokens*output_rate/1000, 8)

    def complete_structured(self, *, tenant_settings: dict, system_prompt: str, payload: dict,
                            response_model: type[BaseModel]) -> GatewayResult:
        provider = self.provider if self.provider else "unconfigured"
        if not self.status(tenant_settings)["available"]:
            return GatewayResult(None, provider, self.model or "", "fallback_no_authorized_provider")
        schema_prompt = system_prompt + "\nOutput JSON Schema:\n" + json.dumps(response_model.model_json_schema())
        body = {"model": self.model, "temperature": 0.1, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": schema_prompt},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(3):
            try:
                response = requests.post(self.base_url, headers=headers, json=body, timeout=self.timeout_seconds)
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < 2:
                        time.sleep(0.25 * (2**attempt)); continue
                    return GatewayResult(None, provider, self.model, "fallback_retry_limit", error=f"http_{response.status_code}")
                if response.status_code == 401:
                    return GatewayResult(None, provider, self.model, "fallback_invalid_key", error="provider rejected configured credentials")
                if response.status_code >= 400:
                    return GatewayResult(None, provider, self.model, "fallback_provider_error", error=f"http_{response.status_code}")
                response.raise_for_status()
                raw = response.json()
                content = raw["choices"][0]["message"]["content"]
                parsed = response_model.model_validate_json(content)
                usage = raw.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                return GatewayResult(parsed, provider, self.model, "success", prompt_tokens, completion_tokens,
                                     self._cost(prompt_tokens, completion_tokens))
            except (requests.RequestException, KeyError, ValueError, TypeError, ValidationError) as exc:
                if attempt < 2 and isinstance(exc, requests.RequestException):
                    time.sleep(0.25 * (2**attempt)); continue
                return GatewayResult(None, provider, self.model, "fallback_provider_error", error=type(exc).__name__)
        return GatewayResult(None, provider, self.model, "fallback_provider_error", error="retry limit reached")
