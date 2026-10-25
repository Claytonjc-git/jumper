"""A physics-reference-guided forward somersault; a qualified reference is required."""

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.front_somersault",
    assets=JUMPER_ASSETS,
    description="forward somersault tracking a qualified physics rollout",
    tags=("jumper", "jump", "reference-tracking"),
)
