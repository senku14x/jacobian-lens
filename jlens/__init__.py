# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.cot import CoTTrace, chat_prompt, generate_cot
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.interventions import (
    coordinate_swap,
    project_out,
    residual_edits,
    steer,
    token_directions,
    workspace_band,
)
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "CoTTrace",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "chat_prompt",
    "configure_logging",
    "coordinate_swap",
    "fit",
    "from_hf",
    "generate_cot",
    "jacobian_for_prompt",
    "project_out",
    "residual_edits",
    "steer",
    "token_directions",
    "workspace_band",
]
