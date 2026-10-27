"""200 Hz tracking of a qualified, torque-limited frontflip reference."""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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
from .rl_cfg import agent_cfg
from .physics import (
    CHASSIS_FORCE_LIMIT, CONTROL_DT, PHYSICS_DT, SUPPORT_GEOMS, T_GO, robot_cfg,
    sensors,
)

REFERENCE = Path(__file__).resolve().parent / "ref" / "frontflip.npz"
# Limb impacts end an attempt only above 100 N; between 35 N (the reference's own
# landing peak) and 100 N they cost reward instead. LIMB_FORCE_LIMIT (50 N) stays
# the pass/fail line, read from the peak_limb_force metric.
LIMB_TERMINATION_N = 100.0
LIMB_PENALTY_FROM_N = 35.0
# Limb-on-limb contact in the tuck is penalized above 5 N and ends an attempt at 15 N.
SELF_TERMINATION_N = 15.0
SELF_PENALTY_FROM_N = 5.0
# Every non-timeout termination costs this much, so ending in the air cannot beat a landing.
# Probe, 64 HOME starts, Windows AMD CPU, native (2026-10-26): iteration 1950 earned 21.9 by
# ending on self-contact at 1.19 s, zero residual 12.8 by landing (41/64 then end at touchdown);
# landing wins only above 25. At 50 the same probe gives -47.4 against -35.0.
TERMINATION_PENALTY = 50.0
ROTATION_LAG_RAD = 1.5
RSI_FRACTION = 0.5
# PPO updates with nominal physics, then over which friction/COM/bias widen to full.
# With zero residual, 64/64 HOME attempts qualify without randomization and 4/64 with it
# (same probe); dropping any one source still leaves 10/64 or fewer.
RANDOMIZATION_START_UPDATES, RANDOMIZATION_RAMP_UPDATES = 500, 1000
RANDOMIZED_EVENTS = ("foot_friction", "encoder_bias", "base_com")
# Clean, dry floor: the silicone estimate in common/constants.py, not the dusty or wet low end.
# Zero residual at full randomization (192 HOME attempts, same probe) lands 3/63 below 0.8
# and 79/129 from 0.8 to 1.5; sampled over this range it lands 107/192.
FOOT_FRICTION_RANGE = (0.8, 1.5)


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
        cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION_RANGE

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
        command_delay_range=(T_GO, T_GO), rsi_fraction=0.0 if play else RSI_FRACTION,
    )
    reset = cfg.events["reset_from_reference"]
    reset.func = mdp.reset_from_reference_phase
    reset.params = {"command_name": "jump", "rsi_fraction": 0.0 if play else RSI_FRACTION,
                    "phase_range": (0.0, 0.9)}
    # External pushes obscure the initial physics-reference feasibility measurement.
    cfg.events.pop("push_robot", None)
    if not play:
        steps = agent_cfg().num_steps_per_env
        for name in RANDOMIZED_EVENTS:
            cfg.events[name].mode = "reset"
        cfg.curriculum["randomization"] = CurriculumTermCfg(
            func=mdp.RandomizationRamp,
            params={"events": RANDOMIZED_EVENTS,
                    "start_step": RANDOMIZATION_START_UPDATES * steps,
                    "ramp_steps": RANDOMIZATION_RAMP_UPDATES * steps,
                    "nominal_friction": next(iter(
                        cfg.scene.entities["robot"].collisions[0].friction.values()))[0]},
        )
    cfg.terminations.pop("fell_over", None)
    for name, limit in (("chassis_ground_contact", CHASSIS_FORCE_LIMIT),
                        ("body_ground_contact", LIMB_TERMINATION_N),
                        ("self_collision", SELF_TERMINATION_N)):
        cfg.terminations[name] = TerminationTermCfg(
            func=mdp.collision_failure, params={"sensor_name": name, "force_threshold": limit}
        )
    # Up to 1.10 s, standing still lags 2.6 rad (run 2, update 450) and real flips lag 1.05 rad at
    # most (run 1 checkpoints, zero residual with full randomization; same probe as
    # TERMINATION_PENALTY). Later, under-rotated flips near touchdown would end too.
    cfg.terminations["rotation_lag"] = TerminationTermCfg(
        func=mdp.rotation_lag,
        params={"command_name": "jump", "threshold": ROTATION_LAG_RAD, "until_s": 1.10},
    )

    # Starting scales in rad² and (rad/s)², pending measured policy tracking runs.
    cfg.rewards["ref_base_ori"] = RewardTermCfg(
        func=mdp.ref_full_orientation, weight=2.0, params={"sigma_sq": 0.15}
    )
    cfg.rewards["ref_base_ang_vel"] = RewardTermCfg(
        func=mdp.ref_angular_velocity, weight=2.0, params={"sigma_sq": 4.0}
    )
    cfg.rewards["ref_forward_rotation"] = RewardTermCfg(
        func=mdp.ref_forward_rotation, weight=4.0, params={"sigma_sq": 0.1}
    )
    cfg.rewards["ref_contact"] = RewardTermCfg(
        func=mdp.ref_contact_found, weight=2.5,
        params={"command_name": "jump"},
    )
    # A 30 ms landing costs about 9 at 50 N and 39 at 100 N; a 1.27 s termination forfeits ~38.
    cfg.rewards["limb_impact"] = RewardTermCfg(
        func=mdp.contact_force_excess, weight=-20.0,
        params={"sensor_name": "body_ground_contact", "threshold": LIMB_PENALTY_FROM_N},
    )
    cfg.rewards["self_impact"] = RewardTermCfg(
        func=mdp.contact_force_excess, weight=-20.0,
        params={"sensor_name": "self_collision", "threshold": SELF_PENALTY_FROM_N},
    )
    # At most 25 x 2.2 s = 55 from the reference touchdown to the episode end, about the
    # termination penalty: a completed, upright landing is worth what a crash costs.
    cfg.rewards["flip_landing"] = RewardTermCfg(
        func=mdp.flip_landing, weight=25.0, params={"command_name": "jump"},
    )
    cfg.rewards["termination"] = RewardTermCfg(
        func=envs_mdp.is_terminated, weight=-TERMINATION_PENALTY / cfg.sim.mujoco.timestep
        / cfg.decimation,
    )
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=torque_headroom_cost, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    cfg.metrics["peak_torque"] = MetricsTermCfg(
        func=peak_actuator_force, reduce="max",
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    cfg.metrics["peak_limb_force"] = MetricsTermCfg(
        func=mdp.contact_force, reduce="max", params={"sensor_name": "body_ground_contact"},
    )
    cfg.metrics["peak_self_force"] = MetricsTermCfg(
        func=mdp.contact_force, reduce="max", params={"sensor_name": "self_collision"},
    )
    # Include the HOME lead-in for attempts that start from HOME instead of a reference phase.
    cfg.episode_length_s = float(arrays["t"][-1]) + 0.5
    cfg.viewer.distance = 1.2

    def reference_contract():
        reference = mdp.FrontflipReference(path, "cpu", list(HOME), diagnostic)
        return reference.deploy_contract(FUTURE_OFFSETS, CONTROL_DT, starts_on_entry=True)

    cfg.reference_contract = reference_contract
    return cfg
