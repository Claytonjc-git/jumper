"""A physics-reference-guided frontflip; a qualified reference is required."""

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.frontflip",
    assets=JUMPER_ASSETS,
    description="Frontflip tracking a qualified physics rollout",
    tags=("jumper", "jump", "reference-tracking"),
)
