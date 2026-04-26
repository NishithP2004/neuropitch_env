# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the NeuroPitch OpenEnv environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class NeuropitchAction(Action):
    """Learner action payload for a single episode step."""

    pitch_text: str = Field(
        ...,
        description="Generated ad pitch/script text from the learner model.",
        min_length=1,
    )
    pitch_title: str = Field(
        default="",
        description="Optional short title/headline for the ad pitch.",
    )
    target_product: str = Field(
        default="",
        description="Optional product override (defaults to reset() brief product).",
    )


class NeuropitchObservation(Observation):
    """Structured telemetry returned after reset and step."""

    product_brief: str = Field(default="", description="Prompt for the current episode.")
    competitor_ad: str = Field(
        default="",
        description="Competitor ad script/context shown to the learner.",
    )
    compliance_status: str = Field(
        default="PENDING",
        description="Compliance status for the generated pitch.",
    )
    compliance_log: str = Field(
        default="",
        description="Mediator logs and reasoning trace.",
    )
    panel_votes: dict[str, str] = Field(
        default_factory=dict,
        description="Persona name -> BUY/PASS.",
    )
    buy_votes: int = Field(default=0, description="Total BUY votes (0-5).")
    tribe_scores: dict[str, float] = Field(
        default_factory=dict,
        description="TRIBE region metrics including z_sts, z_tpj, z_broca_45.",
    )
    reward_components: dict[str, float] = Field(
        default_factory=dict,
        description="Component-level reward decomposition.",
    )
    final_reward: float = Field(default=0.0, description="Total reward for this episode.")
    pitch_text: str = Field(default="", description="Pitch text submitted by the learner.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostics and execution metadata.",
    )
