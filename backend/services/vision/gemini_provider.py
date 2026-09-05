"""
Gemini Vision Provider Implementation
Uses Google Generative Language REST API with structured JSON response mode and low temperature.
"""

import os
import base64
import httpx
from typing import Optional, Dict, Any

from .base_provider import (
    VisionProvider,
    VisionProviderError,
    VisionProviderAuthError,
    VisionProviderRateLimitError,
)
from .prompt_builder import build_satellite_analysis_prompt
from .response_parser import SatelliteAnalysisStructured, parse_structured_response


def _load_env_if_missing():
    if not os.getenv("GEMINI_API_KEY"):
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


class GeminiVisionProvider(VisionProvider):
    provider_name: str = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    @property
    def effective_api_key(self) -> Optional[str]:
        if not self.api_key and not os.getenv("GEMINI_API_KEY"):
            _load_env_if_missing()
        return self.api_key or os.getenv("GEMINI_API_KEY")

    @property
    def effective_model(self) -> str:
        return self.model or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash"

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
        temperature: float = 0.15,
        spatial_tile_label: Optional[str] = None,
    ) -> SatelliteAnalysisStructured:
        api_key = self.effective_api_key
        if not api_key:
            raise VisionProviderAuthError(
                "GEMINI_API_KEY is not configured.",
                status_code=500,
                provider="gemini",
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

        # Build parts for Gemini
        parts = [{"text": user_prompt}]

        b64_primary = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": mime_type or "image/jpeg",
                "data": b64_primary,
            }
        })

        if has_second and second_image_bytes:
            parts.append({
                "text": "The following is Image 2 (AFTER / Latest observation state):"
            })
            b64_secondary = base64.b64encode(second_image_bytes).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": second_mime_type or "image/jpeg",
                    "data": b64_secondary,
                }
            })

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.TimeoutException:
                raise VisionProviderError(
                    f"Gemini API request timed out after 45 seconds.",
                    status_code=504,
                    provider="gemini",
                )
            except Exception as e:
                raise VisionProviderError(
                    f"Gemini network connection error: {e}",
                    status_code=503,
                    provider="gemini",
                )

        if response.status_code == 401 or response.status_code == 403:
            raise VisionProviderAuthError(
                f"Gemini authentication failed (HTTP {response.status_code}): {response.text[:200]}",
                status_code=response.status_code,
                provider="gemini",
            )
        elif response.status_code == 429:
            raise VisionProviderRateLimitError(
                f"Gemini rate limit reached (HTTP 429): {response.text[:200]}",
                status_code=429,
                provider="gemini",
            )
        elif response.status_code != 200:
            raise VisionProviderError(
                f"Gemini API returned error {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                provider="gemini",
            )

        try:
            res_json = response.json()
            raw_text = (
                res_json.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        except Exception as parse_err:
            raise VisionProviderError(
                f"Failed to read Gemini response candidate: {parse_err}",
                status_code=502,
                provider="gemini",
            )

        return parse_structured_response(
            raw_text=raw_text,
            query=user_query,
            detection_used=detection_context is not None and bool(detection_context.get("detections")),
            change_used=change_context is not None,
        )
