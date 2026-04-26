# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NeuroPitch environment implementation with layered reward verification."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import requests
from ollama import Client as OllamaClient
from openai import OpenAI
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from tribev2.demo_utils import TribeModel

try:
    from ..models import NeuropitchAction, NeuropitchObservation
except ImportError:
    from models import NeuropitchAction, NeuropitchObservation


PERSONA_MODEL_MAP: dict[str, tuple[str, str]] = {
    "Skeptical Millennial": ("llama3.2:3b", "llama3.2:3b"),
    "Value-Driven Shopper": ("qwen2.5:3b", "qwen2.5:3b"),
    "Tech-Savvy Gen Z": ("phi4-mini:3.8b", "phi4-mini:3.8b"),
    "Eco-Conscious Consumer": ("gemma3:4b", "gemma3:4b"),
    "Impulse Buyer": ("ministral-3:3b", "ministral-3:3b"),
}

DEFAULT_BRIEFS = [
    (
        "Pitch a new plant-based energy drink to urban professionals.",
        "Our drink is clean and tasty, but premium priced for lifestyle consumers.",
    ),
    (
        "Pitch a compact AI note-taking device for students.",
        "Competitor says they are cheaper but never explains battery reliability.",
    ),
    (
        "Pitch eco-friendly detergent pods for families.",
        "Competitor ad highlights low cost but uses generic, bland messaging.",
    ),
]


@dataclass
class ComplianceResult:
    compliant: bool
    log: str
    violations: list[str]


import logging as _logging
_compliance_log = _logging.getLogger(__name__)


class ComplianceDirector:
    """OpenAI-based compliance checker with optional Tavily search evidence."""

    def __init__(self, api_key: str, tavily_api_key: str, model: str):
        self._client = OpenAI(api_key=api_key)
        self._tavily_key = tavily_api_key
        self._model = model

    def _web_search(self, query: str) -> str:
        """Return Tavily search snippets, or a fallback string on any failure.

        Tavily errors (bad key, quota, 4xx/5xx, timeout) must NOT propagate —
        the compliance check can still run without external evidence; it just
        relies on the model's own knowledge instead.
        """
        if not self._tavily_key:
            return "Web evidence unavailable (TAVILY_API_KEY not configured)."
        try:
            # Tavily current API: Bearer token in Authorization header (no api_key in body).
            # requests sets Content-Type automatically when json= is used; do not duplicate it.
            truncated_query = query[:300]
            response = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self._tavily_key}"},
                json={"query": truncated_query, "max_results": 5},
                timeout=20,
            )
            if not response.ok:
                _compliance_log.warning(
                    "Tavily %s: %s — proceeding without web evidence.",
                    response.status_code,
                    response.text[:300],
                )
                return "Web search unavailable; evaluate based on known advertising standards only."
            data = response.json()
            snippets = []
            for item in data.get("results", []):
                title = item.get("title", "")
                content = item.get("content", "")
                url = item.get("url", "")
                snippets.append(f"- {title} ({url}): {content}")
            return "\n".join(snippets) if snippets else "No evidence returned by search."
        except Exception as exc:
            _compliance_log.warning("Tavily request failed (%s); proceeding without web evidence.", exc)
            return "Web search unavailable; evaluate based on known advertising standards only."

    _SYSTEM_WITH_EVIDENCE = (
        "You are a Compliance Director for advertising. "
        "Return JSON only with keys: compliant (bool), violations (list[str]), log (str). "
        "Mark compliant=false if the pitch contains clearly illegal claims: "
        "guaranteed medical cures, fabricated statistics, dangerous safety claims, "
        "misleading price comparisons, or regulated terms (e.g. 'FDA approved') used falsely. "
        "General aspirational language ('energize your day', 'feel refreshed', 'high quality') "
        "is standard marketing and should be marked compliant."
    )
    _SYSTEM_WITHOUT_EVIDENCE = (
        "You are a Compliance Director for advertising with no web evidence available. "
        "Return JSON only with keys: compliant (bool), violations (list[str]), log (str). "
        "Without external evidence, mark compliant=false ONLY for blatantly illegal claims: "
        "guaranteed disease cures, life-threatening safety falsehoods, or explicitly fraudulent "
        "statements. Common aspirational phrases, product benefits, and lifestyle claims are "
        "acceptable standard advertising copy — do NOT reject them without hard evidence."
    )

    def evaluate(self, product_brief: str, pitch_text: str) -> ComplianceResult:
        evidence = self._web_search(
            f"Advertising compliance and claims for: {product_brief[:150]}. "
            f"Verify: {pitch_text[:150]}"
        )
        has_evidence = "unavailable" not in evidence.lower()
        system_prompt = self._SYSTEM_WITH_EVIDENCE if has_evidence else self._SYSTEM_WITHOUT_EVIDENCE
        user_prompt = (
            f"Product brief:\n{product_brief}\n\nPitch:\n{pitch_text}\n\n"
            f"External evidence:\n{evidence}\n\n"
            "Respond with strict JSON."
        )
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        payload = completion.choices[0].message.content or "{}"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return ComplianceResult(
                compliant=False,
                log=f"Invalid compliance output: {payload}",
                violations=["Mediator returned non-JSON output"],
            )
        violations = parsed.get("violations", []) or []
        return ComplianceResult(
            compliant=bool(parsed.get("compliant", False)),
            log=str(parsed.get("log", "Compliance check finished.")),
            violations=[str(v) for v in violations],
        )


class OllamaFocusGroup:
    """Runs persona-based BUY/PASS voting across local Ollama models."""

    def __init__(self, host: str):
        self._host = host
        self._client = OllamaClient(host=host)

    @staticmethod
    def _extract_field(item: object, key: str, default: str = "") -> str:
        if isinstance(item, dict):
            return str(item.get(key, default) or default)
        return str(getattr(item, key, default) or default)

    def _vote_once(self, persona: str, model_name: str, pitch: str) -> tuple[str, str]:
        prompt = (
            f"You are {persona}. Read this ad pitch:\n{pitch}\n\n"
            "Would you buy this product? Answer only with BUY or PASS."
        )
        response = self._client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        if isinstance(response, dict):
            text = str((response.get("message", {}) or {}).get("content", "")).upper()
        else:
            message = getattr(response, "message", None)
            text = str(getattr(message, "content", "") or "").upper()
        vote = "BUY" if "BUY" in text else "PASS"
        return persona, vote

    def vote(self, pitch: str) -> dict[str, str]:
        results = [
            self._vote_once(persona, model_name, pitch)
            for persona, (model_name, _) in PERSONA_MODEL_MAP.items()
        ]
        return {persona: vote for persona, vote in results}

    def _available_model_names(self) -> set[str]:
        models_resp = self._client.list()
        models = getattr(models_resp, "models", None)
        if models is None and isinstance(models_resp, dict):
            models = models_resp.get("models")
        models = models or []

        names: set[str] = set()
        for item in models:
            model_name = self._extract_field(item, "model").strip()
            if model_name:
                names.add(model_name)
                names.add(model_name.split(":")[0])
            name = self._extract_field(item, "name").strip()
            if name:
                names.add(name)
                names.add(name.split(":")[0])
        return names

    def ensure_models_ready(self) -> None:
        """
        Ensure required focus-group models exist locally.

        Uses runtime `ollama.pull()` to fetch any missing model tags.
        """
        available = self._available_model_names()
        required_models = [model_name for model_name, _ in PERSONA_MODEL_MAP.values()]
        for model_name in required_models:
            base_name = model_name.split(":")[0]
            if model_name in available or base_name in available:
                continue
            self._client.pull(model=model_name)
            available = self._available_model_names()

        missing = sorted(
            model_name
            for model_name in required_models
            if model_name not in available and model_name.split(":")[0] not in available
        )
        if missing:
            raise RuntimeError(f"Missing Ollama models after runtime pull: {missing}")


class TribeNeuromarketer:
    """
    TRIBE v2 wrapper for biological shaping reward.

    ROI extraction uses fixed fsaverage5 vertex ranges as a practical approximation.
    """

    _REGION_VERTEX_SLICES = {
        "STS": (2400, 3200),
        "TPJ": (8700, 9600),
        "Broca_45": (12900, 13600),
    }

    def __init__(self, model_id: str, cache_folder: str, anchor_audio_path: str):
        self._anchor_audio_path = Path(anchor_audio_path).expanduser().resolve()
        if not self._anchor_audio_path.exists():
            raise FileNotFoundError(
                f"TRIBE audio anchor not found at '{self._anchor_audio_path}'."
            )
        self._model = TribeModel.from_pretrained(model_id, cache_folder=Path(cache_folder))

    def _region_score(self, preds: np.ndarray, region: str) -> float:
        start, end = self._REGION_VERTEX_SLICES[region]
        capped_end = min(end, preds.shape[1])
        if start >= capped_end:
            return 0.0
        roi_values = preds[:, start:capped_end].reshape(-1)
        if roi_values.size == 0:
            return 0.0
        std = float(np.std(roi_values)) or 1.0
        return float((np.mean(roi_values) - np.mean(preds)) / std)

    def _predict_from_text(self, text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="neuropitch-tribe-") as tmpdir:
            text_path = Path(tmpdir) / "pitch.txt"
            text_path.write_text(text)
            events = self._model.get_events_dataframe(text_path=text_path)
            preds, _ = self._model.predict(events=events, verbose=False)
        return preds

    def _predict_from_anchor_audio(self) -> np.ndarray:
        events = self._model.get_events_dataframe(audio_path=self._anchor_audio_path)
        preds, _ = self._model.predict(events=events, verbose=False)
        return preds

    def get_biological_reward(self, ad_copy: str) -> dict[str, float]:
        text_preds = self._predict_from_text(ad_copy)
        audio_preds = self._predict_from_anchor_audio()
        min_steps = min(len(text_preds), len(audio_preds))
        if min_steps == 0:
            raise RuntimeError("TRIBE returned empty predictions.")
        blended = (text_preds[:min_steps] * 0.8) + (audio_preds[:min_steps] * 0.2)
        z_sts = self._region_score(blended, "STS")
        z_tpj = self._region_score(blended, "TPJ")
        z_broca = self._region_score(blended, "Broca_45")
        biological_reward = (z_sts + z_tpj) - z_broca
        return {
            "z_sts": float(z_sts),
            "z_tpj": float(z_tpj),
            "z_broca_45": float(z_broca),
            "biological_reward": float(biological_reward),
        }


@dataclass
class RuntimeServices:
    compliance: ComplianceDirector
    focus_group: OllamaFocusGroup
    tribe: TribeNeuromarketer


_RUNTIME: RuntimeServices | None = None


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _resolve_anchor_audio_path() -> str:
    configured = os.environ.get("TRIBE_AUDIO_ANCHOR_PATH", "").strip()
    if configured:
        return configured

    candidates = [
        "./server/reference.wav",
        "/data/commercial_anchor.wav",
        "./commercial_anchor.wav",
    ]
    for candidate in candidates:
        if Path(candidate).expanduser().resolve().exists():
            return candidate
    # Preserve strict behavior: caller will fail fast if this doesn't exist.
    return "./server/reference.wav"


def _init_runtime() -> RuntimeServices:
    compliance = ComplianceDirector(
        api_key=_required_env("OPENAI_API_KEY"),
        tavily_api_key=_required_env("TAVILY_API_KEY"),
        model=os.environ.get("OPENAI_MEDIATOR_MODEL", "gpt-4o-mini"),
    )
    focus_group = OllamaFocusGroup(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    tribe = TribeNeuromarketer(
        model_id=os.environ.get("TRIBE_MODEL_ID", "facebook/tribev2"),
        cache_folder=os.environ.get("TRIBE_CACHE_DIR", "./cache"),
        anchor_audio_path=_resolve_anchor_audio_path(),
    )
    focus_group.ensure_models_ready()
    return RuntimeServices(compliance=compliance, focus_group=focus_group, tribe=tribe)


def _get_runtime() -> RuntimeServices:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _init_runtime()
    return _RUNTIME


class NeuropitchEnvironment(Environment):
    """Single-step NeuroPitch environment with layered verification."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._brief_idx = 0
        self._runtime = _get_runtime()
        self._word_limit = int(os.environ.get("NEUROPITCH_WORD_LIMIT", "120"))

        self._product_brief = ""
        self._competitor_ad = ""

    def _next_prompt(self) -> tuple[str, str]:
        pair = DEFAULT_BRIEFS[self._brief_idx % len(DEFAULT_BRIEFS)]
        self._brief_idx += 1
        return pair

    def reset(self, **kwargs) -> NeuropitchObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        product_brief, competitor_ad = self._next_prompt()
        self._product_brief = kwargs.get("product_brief", product_brief)
        self._competitor_ad = kwargs.get("competitor_ad", competitor_ad)

        return NeuropitchObservation(
            product_brief=self._product_brief,
            competitor_ad=self._competitor_ad,
            compliance_status="READY",
            compliance_log="Awaiting learner action.",
            panel_votes={persona: "PENDING" for persona in PERSONA_MODEL_MAP},
            buy_votes=0,
            tribe_scores={},
            reward_components={},
            final_reward=0.0,
            pitch_text="",
            done=False,
            reward=0.0,
            metadata={"episode_id": self._state.episode_id, "phase": "reset"},
        )

    def step(self, action: NeuropitchAction) -> NeuropitchObservation:  # type: ignore[override]
        started = time.time()
        self._state.step_count += 1
        pitch_text = action.pitch_text.strip()

        formatting_penalty = 0.0
        if len(pitch_text.split()) > self._word_limit:
            formatting_penalty = -0.5

        components: dict[str, float] = {"formatting_penalty": formatting_penalty}
        compliance = self._runtime.compliance.evaluate(self._product_brief, pitch_text)

        if not compliance.compliant:
            components["compliance_penalty"] = -1.0
            total_reward = components["compliance_penalty"] + formatting_penalty
            return NeuropitchObservation(
                product_brief=self._product_brief,
                competitor_ad=self._competitor_ad,
                compliance_status="NON_COMPLIANT",
                compliance_log=compliance.log,
                panel_votes={persona: "SKIPPED" for persona in PERSONA_MODEL_MAP},
                buy_votes=0,
                tribe_scores={},
                reward_components=components,
                final_reward=total_reward,
                pitch_text=pitch_text,
                done=True,
                reward=total_reward,
                metadata={
                    "episode_id": self._state.episode_id,
                    "violations": compliance.violations,
                    "execution_ms": int((time.time() - started) * 1000),
                },
            )

        votes = self._runtime.focus_group.vote(pitch_text)
        buy_votes = sum(1 for vote in votes.values() if vote == "BUY")
        vote_reward = buy_votes * 0.2
        tribe_scores = self._runtime.tribe.get_biological_reward(pitch_text)
        biological_reward = tribe_scores["biological_reward"]
        components["vote_reward"] = float(vote_reward)
        components["biological_reward"] = float(biological_reward)

        total_reward = formatting_penalty + vote_reward + biological_reward

        return NeuropitchObservation(
            product_brief=self._product_brief,
            competitor_ad=self._competitor_ad,
            compliance_status="COMPLIANT",
            compliance_log=compliance.log,
            panel_votes=votes,
            buy_votes=buy_votes,
            tribe_scores=tribe_scores,
            reward_components=components,
            final_reward=float(total_reward),
            pitch_text=pitch_text,
            done=True,
            reward=float(total_reward),
            metadata={
                "episode_id": self._state.episode_id,
                "execution_ms": int((time.time() - started) * 1000),
            },
        )

    @property
    def state(self) -> State:
        return self._state
