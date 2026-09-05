"""
Satellite Vision Service Orchestrator
Coordinates image preprocessing, tiling, single/two-stage depth, provider routing, fallback, and ensemble.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List, Tuple

from .base_provider import VisionProvider, VisionProviderError
from .gemini_provider import GeminiVisionProvider
from .glm_provider import GLMVisionProvider
from .response_parser import ObservationItem, SatelliteAnalysisStructured
from .image_processor import preprocess_image, generate_tiles, decode_data_url

logger = logging.getLogger("satquery.vision")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


def _load_env_if_missing():
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("ZAI_API_KEY") and not os.getenv("GLM_API_KEY"):
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


class VisionService:
    def __init__(self):
        self.gemini = GeminiVisionProvider()
        self.glm = GLMVisionProvider()

    def get_available_providers(self) -> List[str]:
        _load_env_if_missing()
        providers = []
        if self.gemini.effective_api_key:
            providers.append("gemini")
        if self.glm.effective_api_key:
            providers.append("glm")
        return providers

    async def analyze_image(
        self,
        image_data: str,
        user_query: str,
        provider: str = "auto",
        analysis_mode: str = "general",
        analysis_depth: str = "standard",
        use_detections: bool = True,
        use_change_context: bool = True,
        use_tiles: bool = False,
        detection_context: Optional[Dict[str, Any]] = None,
        change_context: Optional[Dict[str, Any]] = None,
        image_metadata: Optional[Dict[str, Any]] = None,
        second_image_data: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for satellite image analysis.
        Returns standardized dict containing:
          - 'success': bool
          - 'provider_used': str
          - 'fallback_used': bool
          - 'analysis_mode': str
          - 'query': str
          - 'structured_analysis': SatelliteAnalysisStructured
          - 'processing_time_ms': int
        """
        start_time = time.time()
        logger.info(f"[Vision] Starting analysis (provider={provider}, mode={analysis_mode}, depth={analysis_depth}, tiles={use_tiles})")

        # 1. Image preprocessing
        raw_bytes, mime = decode_data_url(image_data)
        processed_bytes, clean_mime, (w, h) = preprocess_image(raw_bytes)
        logger.info(f"[Vision] Primary image normalized: {w}x{h} ({clean_mime})")

        sec_bytes: Optional[bytes] = None
        sec_mime: Optional[str] = None
        if second_image_data:
            s_raw, s_mime = decode_data_url(second_image_data)
            sec_bytes, sec_mime, (sw, sh) = preprocess_image(s_raw)
            logger.info(f"[Vision] Secondary image normalized: {sw}x{sh} ({sec_mime})")

        # Clean contexts based on flags
        active_detection = detection_context if use_detections else None
        active_change = change_context if use_change_context else None

        if active_detection and active_detection.get("detections"):
            logger.info(f"[Vision] Attached {len(active_detection.get('detections', []))} Grounding DINO detection hints")
        if active_change:
            logger.info("[Vision] Attached bi-temporal change context")

        # 2. Provider execution (Ensemble vs Single/Fallback)
        chosen_provider = provider.lower().strip() if provider else "auto"

        if chosen_provider == "ensemble":
            result = await self._run_ensemble(
                processed_bytes,
                clean_mime,
                user_query,
                analysis_mode=analysis_mode,
                analysis_depth=analysis_depth,
                detection_context=active_detection,
                change_context=active_change,
                image_metadata=image_metadata,
                second_image_bytes=sec_bytes,
                second_mime_type=sec_mime,
                use_tiles=use_tiles,
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = elapsed_ms
            return result

        structured_res, provider_used, fallback_used = await self._run_with_fallback(
            chosen_provider=chosen_provider,
            image_bytes=processed_bytes,
            mime_type=clean_mime,
            user_query=user_query,
            analysis_mode=analysis_mode,
            analysis_depth=analysis_depth,
            detection_context=active_detection,
            change_context=active_change,
            image_metadata=image_metadata,
            second_image_bytes=sec_bytes,
            second_mime_type=sec_mime,
            use_tiles=use_tiles,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"[Vision] Analysis completed in {elapsed_ms}ms using {provider_used} (fallback={fallback_used})")

        return {
            "success": True,
            "provider_used": provider_used,
            "fallback_used": fallback_used,
            "analysis_mode": analysis_mode,
            "query": user_query,
            "structured_analysis": structured_res,
            "processing_time_ms": elapsed_ms,
        }

    async def _execute_provider_analysis(
        self,
        prov: VisionProvider,
        image_bytes: bytes,
        mime_type: str,
        user_query: str,
        analysis_mode: str,
        analysis_depth: str,
        detection_context: Optional[Dict[str, Any]],
        change_context: Optional[Dict[str, Any]],
        image_metadata: Optional[Dict[str, Any]],
        second_image_bytes: Optional[bytes],
        second_mime_type: Optional[str],
        use_tiles: bool,
    ) -> SatelliteAnalysisStructured:
        """Executes analysis for a specific provider, supporting optional 2x2 tiling and deep two-stage."""
        # 1. Optional Tiling Pipeline
        tile_observations: List[ObservationItem] = []
        if use_tiles:
            logger.info("[Vision] Generating 2x2 spatial sub-tiles for detailed quadrant inspection...")
            tiles = generate_tiles(image_bytes, grid=(2, 2))
            for t in tiles:
                try:
                    tile_res = await prov.analyze(
                        image_bytes=t["bytes"],
                        mime_type=t["mime"],
                        user_query=user_query,
                        analysis_mode=analysis_mode,
                        detection_context=None,
                        change_context=None,
                        image_metadata=None,
                        second_image_bytes=None,
                        second_mime_type=None,
                        spatial_tile_label=t["label"],
                    )
                    for obs in tile_res.observations:
                        obs.location = t["label"]
                        tile_observations.append(obs)
                except Exception as te:
                    logger.warning(f"[Vision] Tile [{t['label']}] analysis skipped: {te}")

        # 2. Global Full-Scene Analysis (Stage 1)
        base_res = await prov.analyze(
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_query=user_query,
            analysis_mode=analysis_mode,
            detection_context=detection_context,
            change_context=change_context,
            image_metadata=image_metadata,
            second_image_bytes=second_image_bytes,
            second_mime_type=second_mime_type,
        )

        if tile_observations:
            # Merge tile observations into base observations
            base_res.observations.extend(tile_observations[:8])

        # 3. Optional Deep Two-Stage Synthesis (Stage 2)
        if analysis_depth == "deep" and base_res.observations:
            logger.info("[Vision] Executing Stage 2 synthesis pass for deep grounded reasoning...")
            obs_summary = "\n".join([f"- [{o.location}] {o.finding}: {o.evidence}" for o in base_res.observations[:10]])
            deep_query = (
                f"{user_query}\n\n"
                f"STAGE 1 OBSERVATIONS:\n{obs_summary}\n\n"
                "SYNTHESIS INSTRUCTION: Synthesize these visible observations directly answering the query."
            )
            try:
                stage2_res = await prov.analyze(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    user_query=deep_query,
                    analysis_mode=analysis_mode,
                    detection_context=detection_context,
                    change_context=change_context,
                    image_metadata=image_metadata,
                    second_image_bytes=second_image_bytes,
                    second_mime_type=second_mime_type,
                )
                stage2_res.observations = base_res.observations
                return stage2_res
            except Exception as se:
                logger.warning(f"[Vision] Deep synthesis pass error (using Stage 1 result): {se}")

        return base_res

    async def _run_with_fallback(
        self,
        chosen_provider: str,
        image_bytes: bytes,
        mime_type: str,
        user_query: str,
        analysis_mode: str,
        analysis_depth: str,
        detection_context: Optional[Dict[str, Any]],
        change_context: Optional[Dict[str, Any]],
        image_metadata: Optional[Dict[str, Any]],
        second_image_bytes: Optional[bytes],
        second_mime_type: Optional[str],
        use_tiles: bool,
    ) -> Tuple[SatelliteAnalysisStructured, str, bool]:
        """Runs preferred provider with automatic fallback if available."""
        # Determine preference order
        if chosen_provider == "glm":
            providers_to_try = [("glm", self.glm), ("gemini", self.gemini)]
        elif chosen_provider == "gemini":
            providers_to_try = [("gemini", self.gemini), ("glm", self.glm)]
        else:  # auto
            # Prefer Gemini if key present, else GLM
            if self.gemini.effective_api_key:
                providers_to_try = [("gemini", self.gemini), ("glm", self.glm)]
            else:
                providers_to_try = [("glm", self.glm), ("gemini", self.gemini)]

        last_err: Optional[Exception] = None
        for idx, (p_name, p_instance) in enumerate(providers_to_try):
            try:
                logger.info(f"[Vision] Attempting provider: {p_name}")
                res = await self._execute_provider_analysis(
                    prov=p_instance,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    user_query=user_query,
                    analysis_mode=analysis_mode,
                    analysis_depth=analysis_depth,
                    detection_context=detection_context,
                    change_context=change_context,
                    image_metadata=image_metadata,
                    second_image_bytes=second_image_bytes,
                    second_mime_type=second_mime_type,
                    use_tiles=use_tiles,
                )
                fallback_used = idx > 0
                return res, p_name, fallback_used
            except Exception as e:
                logger.warning(f"[Vision] Provider [{p_name}] failed: {e}")
                last_err = e

        raise VisionProviderError(f"All vision providers failed. Last error: {last_err}")

    async def _run_ensemble(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_query: str,
        analysis_mode: str,
        analysis_depth: str,
        detection_context: Optional[Dict[str, Any]],
        change_context: Optional[Dict[str, Any]],
        image_metadata: Optional[Dict[str, Any]],
        second_image_bytes: Optional[bytes],
        second_mime_type: Optional[str],
        use_tiles: bool,
    ) -> Dict[str, Any]:
        """Runs both Gemini and GLM and computes consensus and disagreement findings."""
        logger.info("[Vision] Running dual-provider ensemble (Gemini + GLM)...")
        gemini_res: Optional[SatelliteAnalysisStructured] = None
        glm_res: Optional[SatelliteAnalysisStructured] = None

        try:
            gemini_res = await self._execute_provider_analysis(
                self.gemini, image_bytes, mime_type, user_query, analysis_mode, analysis_depth,
                detection_context, change_context, image_metadata, second_image_bytes, second_mime_type, use_tiles
            )
        except Exception as ge:
            logger.warning(f"[Vision Ensemble] Gemini call failed: {ge}")

        try:
            glm_res = await self._execute_provider_analysis(
                self.glm, image_bytes, mime_type, user_query, analysis_mode, analysis_depth,
                detection_context, change_context, image_metadata, second_image_bytes, second_mime_type, use_tiles
            )
        except Exception as gle:
            logger.warning(f"[Vision Ensemble] GLM call failed: {gle}")

        if not gemini_res and not glm_res:
            raise VisionProviderError("Ensemble failed: neither Gemini nor GLM returned a response.")
        if gemini_res and not glm_res:
            return {"success": True, "provider_used": "gemini (ensemble fallback)", "fallback_used": True, "structured_analysis": gemini_res, "query": user_query}
        if glm_res and not gemini_res:
            return {"success": True, "provider_used": "glm (ensemble fallback)", "fallback_used": True, "structured_analysis": glm_res, "query": user_query}

        # Both returned results -> Calculate consensus
        consensus_findings: List[Dict[str, Any]] = []
        disagreements: List[str] = []

        gemini_findings = {o.finding.lower(): o for o in gemini_res.observations}
        glm_findings = {o.finding.lower(): o for o in glm_res.observations}

        for g_k, g_obs in gemini_findings.items():
            matched = False
            for m_k, m_obs in glm_findings.items():
                if g_k in m_k or m_k in g_k:
                    consensus_findings.append({
                        "finding": g_obs.finding,
                        "location": g_obs.location if g_obs.location == m_obs.location else f"{g_obs.location} / {m_obs.location}",
                        "confidence": "high" if (g_obs.confidence == "high" or m_obs.confidence == "high") else "medium",
                        "consensus_evidence": f"Gemini: {g_obs.evidence}; GLM: {m_obs.evidence}",
                    })
                    matched = True
                    break
            if not matched:
                disagreements.append(f"Gemini reported: '{g_obs.finding}' in {g_obs.location} (not reported by GLM)")

        for m_k, m_obs in glm_findings.items():
            if not any(m_k in g_k or g_k in m_k for g_k in gemini_findings):
                disagreements.append(f"GLM reported: '{m_obs.finding}' in {m_obs.location} (not reported by Gemini)")

        combined_obs: List[ObservationItem] = [
            ObservationItem(
                finding=c["finding"],
                location=c["location"],
                confidence=c["confidence"],
                evidence=c["consensus_evidence"],
            ) for c in consensus_findings
        ] or gemini_res.observations

        ensemble_summary = (
            f"Consensus Analysis (Gemini + GLM): {gemini_res.summary} "
            f"Cross-verified {len(consensus_findings)} common visual patterns."
        )

        structured_ensemble = SatelliteAnalysisStructured(
            summary=ensemble_summary,
            answer_to_query=gemini_res.answer_to_query,
            observations=combined_obs,
            uncertainties=gemini_res.uncertainties + glm_res.uncertainties,
            model_notes={
                "ensemble": True,
                "consensus_count": len(consensus_findings),
                "disagreements": disagreements[:5],
            }
        )

        return {
            "success": True,
            "provider_used": "ensemble (gemini+glm)",
            "fallback_used": False,
            "analysis_mode": analysis_mode,
            "query": user_query,
            "structured_analysis": structured_ensemble,
            "consensus_findings": consensus_findings,
            "provider_disagreements": disagreements,
        }


vision_service = VisionService()
