"""LLM provider: Ollama locally, Amazon Bedrock in cloud mode.

Only cool-path agents (sentiment, documents, reports, chat) use this. The
hot path that makes trading decisions never calls an LLM.
"""

from __future__ import annotations

from typing import Protocol

from app.config import AppMode, Settings


class LLM(Protocol):
    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system or "",
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class BedrockLLM:
    def __init__(self, model_id: str, region: str) -> None:
        import boto3

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        resp = self.client.converse(**kwargs)
        return resp["output"]["message"]["content"][0]["text"]


def get_llm(settings: Settings) -> LLM:
    if settings.app_mode is AppMode.CLOUD:
        if not settings.bedrock_model_id:
            raise RuntimeError("BEDROCK_MODEL_ID is not set")
        return BedrockLLM(settings.bedrock_model_id, settings.aws_region)
    return OllamaLLM(settings.ollama_url, settings.ollama_model)
