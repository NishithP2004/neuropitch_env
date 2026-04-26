# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NeuroPitch environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import NeuropitchAction, NeuropitchObservation
except ImportError:  # pragma: no cover
    from models import NeuropitchAction, NeuropitchObservation


class NeuropitchEnv(
    EnvClient[NeuropitchAction, NeuropitchObservation, State]
):
    """
    Client for the Neuropitch Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with NeuropitchEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.product_brief)
        ...
        ...     result = client.step(NeuropitchAction(pitch_text="Try our new drink today."))
        ...     print(result.observation.final_reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = NeuropitchEnv.from_docker_image("neuropitch_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(NeuropitchAction(pitch_text="A crisp, clean energy boost."))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: NeuropitchAction) -> Dict:
        """
        Convert NeuropitchAction to JSON payload for step message.

        Args:
            action: NeuropitchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "pitch_text": action.pitch_text,
            "pitch_title": action.pitch_title,
            "target_product": action.target_product,
        }

    def _parse_result(self, payload: Dict) -> StepResult[NeuropitchObservation]:
        """
        Parse server response into StepResult[NeuropitchObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with NeuropitchObservation
        """
        obs_data = payload.get("observation", {})
        observation = NeuropitchObservation(
            product_brief=obs_data.get("product_brief", ""),
            competitor_ad=obs_data.get("competitor_ad", ""),
            compliance_status=obs_data.get("compliance_status", "PENDING"),
            compliance_log=obs_data.get("compliance_log", ""),
            panel_votes=obs_data.get("panel_votes", {}),
            buy_votes=obs_data.get("buy_votes", 0),
            tribe_scores=obs_data.get("tribe_scores", {}),
            reward_components=obs_data.get("reward_components", {}),
            final_reward=obs_data.get("final_reward", payload.get("reward") or 0.0),
            pitch_text=obs_data.get("pitch_text", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
