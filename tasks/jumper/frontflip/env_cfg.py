"""200 Hz tracking of a qualified, torque-limited frontflip reference."""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.managers import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..common.constants import HOME
from ..common.assets import JUMPER_XML
from ..common.dance.rewards import peak_actuator_force, torque_headroom_cost
from ..jump.env_cfg import env_cfg as jump_env_cfg
from ..jump.mdp.observations import FUTURE_OFFSETS
from . import mdp
from .physics import (
    CHASSIS_FORCE_LIMIT, CONTROL_DT, LIMB_FORCE_LIMIT, PHYSICS_DT, SUPPORT_GEOMS, T_GO, robot_cfg,
    sensors,
)

REFERENCE = Path(__file__).resolve().parent / "ref" / "frontflip.npz"


def env_cfg(asset: Path | None = None, play: bool = False):
    if asset is not None and Path(asset).resolve() != JUMPER_XML.resolve():
        raise ValueError("Frontflip reference is qualified against the default jumper.xml only")
    path = Path(os.environ.get("MJRL_FRONTFLIP_REFERENCE", str(REFERENCE))).expanduser()
    diagnostic = os.environ.get("MJRL_FRONTFLIP_ALLOW_DIAGNOSTIC") == "1"
    _, arrays = mdp.read_reference(path, diagnostic)
    cfg = jump_env_cfg(asset=asset, play=play)
    cfg.sim.mujoco.timestep = PHYSICS_DT
    cfg.decimation = round(CONTROL_DT / PHYSICS_DT)
    cfg.scene.entities["robot"] = robot_cfg()
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
            "robot", geom_names=SUPPORT_GEOMS,
        )

    # A flip may invert freely; shell or hard limb impacts and inter-leg contacts end an attempt.
    cfg.scene.sensors = tuple(
        sensor for sensor in cfg.scene.sensors or ()
        if sensor.name not in ("body_ground_contact", "feet_ground_contact")
    ) + sensors(cfg.decimation) + (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("force",), reduce="maxforce", num_slots=1,
            history_length=cfg.decimation,
        ),
    )
    cfg.commands["jump"] = mdp.FrontflipCommandCfg(
        entity_name="robot", motion_file=str(path), allow_diagnostic=diagnostic,
        command_delay_range=(T_GO, T_GO), rsi_fraction=0.0 if play else 0.8,
    )
    reset = cfg.events["reset_from_reference"]
    reset.func = mdp.reset_from_reference_phase
    reset.params = {"command_name": "jump", "rsi_fraction": 0.0 if play else 0.8,
                    "phase_range": (0.0, 0.9)}
    # External pushes obscure the initial physics-reference feasibility measurement.
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("fell_over", None)
    for name, limit in (("chassis_ground_contact", CHASSIS_FORCE_LIMIT),
                        ("body_ground_contact", LIMB_FORCE_LIMIT), ("self_collision", 1.0)):
        cfg.terminations[name] = TerminationTermCfg(
            func=mdp.collision_failure, params={"sensor_name": name, "force_threshold": limit}
        )

    # Starting scales in rad² and (rad/s)², pending measured policy tracking runs.
    cfg.rewards["ref_base_ori"] = RewardTermCfg(
        func=mdp.ref_full_orientation, weight=2.0, params={"sigma_sq": 0.15}
    )
    cfg.rewards["ref_base_ang_vel"] = RewardTermCfg(
        func=mdp.ref_angular_velocity, weight=2.0, params={"sigma_sq": 4.0}
    )
    cfg.rewards["ref_forward_rotation"] = RewardTermCfg(
        func=mdp.ref_forward_rotation, weight=2.0, params={"sigma_sq": 0.25}
    )
    cfg.rewards["ref_contact"] = RewardTermCfg(
        func=mdp.ref_contact_found, weight=2.5,
        params={"command_name": "jump"},
    )
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=torque_headroom_cost, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    cfg.metrics["peak_torque"] = MetricsTermCfg(
        func=peak_actuator_force, reduce="max",
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    # Include the HOME lead-in for non-RSI attempts; 20% train from HOME as well.
    cfg.episode_length_s = float(arrays["t"][-1]) + 0.5
    cfg.viewer.distance = 1.2

    def reference_contract():
        reference = mdp.FrontflipReference(path, "cpu", list(HOME), diagnostic)
        return reference.deploy_contract(FUTURE_OFFSETS, CONTROL_DT, starts_on_entry=True)

    cfg.reference_contract = reference_contract
    return cfg
