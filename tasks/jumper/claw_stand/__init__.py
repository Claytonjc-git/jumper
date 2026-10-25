"""A grounded two-claw stand and two push-ups, tracked from hand keyframes."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.claw_stand",
    assets=JUMPER_ASSETS,
    description="grounded claw stand, 1.5 s hold and two push-ups; hand-keyframed reference",
    tags=("imitation", "jumper", "gesture", "flat"),
)
