"""
GLM Vision Provider Implementation
Uses Zhipu/Z.AI PAAS API (GLM-4.6V-Flash) with OpenAI-compatible multimodal chat completions.
"""

import os
import base64
import httpx
from typing import Optional, Dict, Any, List

from .base_provider import (
    VisionProvider,
    VisionProviderError,
    VisionProviderAuthError,
    VisionProviderRateLimitError,
)
from .prompt_builder import build_satellite_analysis_prompt
from .response_parser import SatelliteAnalysisStructured, parse_structured_response


def _load_env_if_missing():
    if not os.getenv("ZAI_API_KEY") and not os.getenv("GLM_API_KEY"):
        from pathlib import Path
        for p in [
            Path.cwd() / ".env",
            Path.cwd() / "backend" / ".env",
            Path(__file__).resolve().parent.parent.parent / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]:
            if p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() not in os.environ:
                                    os.environ[k.strip()] = v.strip().strip("\"'")
                except Exception:
                    pass


class GLMVisionProvider(VisionProvider):
    provider_name: str = "glm"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    @property
    def effective_api_key(self) -> Optional[str]:
        if not self.api_key and not os.getenv("ZAI_API_KEY") and not os.getenv("GLM_API_KEY"):
            _load_env_if_missing()
        return self.api_key or os.getenv("ZAI_API_KEY") or os.getenv("GLM_API_KEY")

    @property
    def effective_model(self) -> str:
        return self.model or os.getenv("GLM_MODEL") or "glm-4.6v-flash"

    async def analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_query: str,
        analysis_mode: str = "general",
        detection_context: Optional[Dict[str, Any]] = None,
        change_context: Optional[Dict[str, Any]] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
        second_image_bytes: Optional[bytes] = None,
        second_mime_type: Optional[str] = None,
        temperature: float = 0.1,
        spatial_tile_label: Optional[str] = None,
    ) -> SatelliteAnalysisStructured:
        api_key = self.effective_api_key
        if not api_key:
            raise VisionProviderAuthError(
                "ZAI_API_KEY or GLM_API_KEY is not configured.",
                status_code=500,
                provider="glm",
            )
        model = self.effective_model

        has_second = second_image_bytes is not None and len(second_image_bytes) > 0
        system_prompt, user_prompt = build_satellite_analysis_prompt(
            user_query=user_query,
            detection_context=detection_context,
            change_context=change_context,
            image_metadata=image_metadata,
            analysis_mode=analysis_mode,
            has_second_image=has_second,
            spatial_tile_label=spatial_tile_label,
        )

        b64_primary = base64.b64encode(image_bytes).decode("utf-8")
        primary_url = f"data:{mime_type or 'image/jpeg'};base64,{b64_primary}"

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": primary_url}},
        ]

        if has_second and second_image_bytes:
            b64_secondary = base64.b64encode(second_image_bytes).decode("utf-8")
            secondary_url = f"data:{second_mime_type or 'image/jpeg'};base64,{b64_secondary}"
            user_content.append({"type": "text", "text": "The following is Image 2 (AFTER / Latest observation state):"})
            user_content.append({"type": "image_url", "image_url": {"url": secondary_url}})

        endpoint = "https://api.z.ai/api/paas/v4/chat/completions"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
            except httpx.TimeoutException:
                raise VisionProviderError(
                    f"GLM API request timed out after 45 seconds.",
                    status_code=504,
                    provider="glm",
                )
            except Exception as e:
                raise VisionProviderError(
                    f"GLM network connection error: {e}",
                    status_code=503,
                    provider="glm",
                )

        if response.status_code == 401 or response.status_code == 403:
            raise VisionProviderAuthError(
                f"GLM authentication failed (HTTP {response.status_code}): {response.text[:200]}",
                status_code=response.status_code,
                provider="glm",
            )
        elif response.status_code == 429:
            raise VisionProviderRateLimitError(
                f"GLM rate limit reached (HTTP 429): {response.text[:200]}",
                status_code=429,
                provider="glm",
            )
        elif response.status_code != 200:
            raise VisionProviderError(
                f"GLM API returned error {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                provider="glm",
            )

        try:
            res_json = response.json()
            raw_text = (
                res_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception as parse_err:
            raise VisionProviderError(
                f"Failed to read GLM response content: {parse_err}",
                status_code=502,
                provider="glm",
            )

        return parse_structured_response(
            raw_text=raw_text,
            query=user_query,
            detection_used=detection_context is not None and bool(detection_context.get("detections")),
            change_used=change_context is not None,
        )
