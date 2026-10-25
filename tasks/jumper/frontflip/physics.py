"""Nominal physics, bounded target schedule and measured rollout qualification.

Search and training use the same Entity/ServoCurveActuator, including thermal
derating. No external forces or root-state writes occur after a rollout reset.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from ..common.actuator import MOTOR_CONFIG
from ..common.assets import JUMPER_XML
from ..common.constants import (
    CHASSIS_LINKS, EFFORT_LIMIT, FEET, FOOT_GEOMS, HOME, LIMB_LINKS, TIP_LINKS,
    get_jumper_robot_cfg,
)
from .momentum import MomentumMeter

KP, KD = 10.0, 0.5
# Peak terrain force [N] allowed on the shell and on limb links (palms, calves, fingers).
CHASSIS_FORCE_LIMIT, LIMB_FORCE_LIMIT = 10.0, 50.0
# Forward pitch momentum [N*m*s] the claw PUSH stage must add about the whole-robot COM.
CLAW_SPIN_MIN = 0.05
PHYSICS_DT, CONTROL_DT = 0.0025, 0.005
T_GO, DURATION = 0.30, 3.0
FRONT_PADS = ("LF_palm_pad_f_link_meshcol", "RF_palm_pad_f_link_meshcol")
SUPPORT_GEOMS = FOOT_GEOMS + FRONT_PADS

# Seven left-side joints per pose; counterparts are resolved by name/sign below.
JOINT_PAIRS = (
    ("LF_J1_joint", "RF_J1_joint", 1),
    ("LF_J2_joint", "RF_J2_joint", -1),
    ("LF_J3_joint", "RF_J3_joint", -1),
    ("LM_J1_joint", "RM_J1_joint", -1),
    ("LM_J2_joint", "RM_J2_joint", -1),
    ("LR_J1_joint", "RR_J1_joint", -1),
    ("LR_J2_joint", "RR_J2_joint", -1),
)
# crouch / push / tuck, then four durations and front/rear timing offsets [s].
SEED = np.array([
    -1.26, -0.46, -1.57, 1.10, -2.20, 1.10, -2.20,
    -1.50, 0.12, -1.85, 0.00, -0.15, -0.10, -0.15,
    -0.20, -1.50, -0.70, 1.20, -2.15, 1.30, -2.15,
    0.30, 0.08, 0.08, 0.25, 0.025, -0.025,
])
LOW = np.r_[np.tile([-2.0, -1.90, -1.96, -0.5, -2.20, -0.5, -2.20], 3),
            .18, .04, .04, .08, -.12, -.12]
HIGH = np.r_[np.tile([.6, .8, 1.2, 1.35, -.10, 1.4, -.10], 3),
             .55, .18, .18, .65, .12, .12]


def fingerprint():
    """Hash model XML, referenced meshes, motor spec and this task's physics code."""
    root = ET.parse(JUMPER_XML).getroot()
    meshdir = root.find("compiler").get("meshdir", "")
    common = Path(__file__).resolve().parent.parent / "common"
    paths = [JUMPER_XML, MOTOR_CONFIG, Path(__file__), Path(__file__).with_name("staged.py"),
             common / "constants.py", common / "actuator.py"]
    paths += [JUMPER_XML.parent / meshdir / node.get("file")
              for node in root.findall("./asset/mesh")]
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda p: p.as_posix()):
        # Relative labels keep a Colab checkout's fingerprint identical to Windows.
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n")
                      if path.suffix in (".py", ".xml", ".yaml") else path.read_bytes())
    return digest.hexdigest()


def sensors(history=2):
    nonfoot = tuple(f"{n}_meshcol" for n in CHASSIS_LINKS + LIMB_LINKS + TIP_LINKS
                    if f"{n}_meshcol" not in SUPPORT_GEOMS)
    return (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            history_length=history,
        ),
        ContactSensorCfg(
            name="claw_front_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FRONT_PADS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            history_length=history,
        ),
        ContactSensorCfg(
            name="chassis_ground_contact",
            primary=ContactMatch(mode="geom", pattern=tuple(f"{n}_meshcol" for n in CHASSIS_LINKS),
                                 entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            history_length=history,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=nonfoot, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            history_length=history,
        ),
    )


def robot_cfg():
    """Both silicone pads carry the claws here; walking chooses only the rear pad."""
    robot = get_jumper_robot_cfg()
    collision = robot.collisions[0]
    support_re = "|".join(f"^{name}$" for name in SUPPORT_GEOMS)
    # Replace complete mappings; do not mutate the shared HYBRID_COLLISION object.
    robot.collisions = (replace(
        collision, condim={support_re: 3, ".*": 1}, priority={support_re: 1, ".*": 0},
        friction={support_re: next(iter(collision.friction.values()))},
    ),)
    for actuator in robot.articulation.actuators:
        actuator.stiffness, actuator.damping = KP, KD
    return robot


def support_contacts(env):
    rear = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, 6).bool()
    front = env.scene.sensors["claw_front_ground_contact"].data.found.reshape(env.num_envs, 2).bool()
    return torch.cat((rear[:, :2] | front, rear[:, 2:]), dim=1)


def support_forces(env):
    rear = env.scene.sensors["feet_ground_contact"].data.force.reshape(env.num_envs, 6, 3)
    front = env.scene.sensors["claw_front_ground_contact"].data.force.reshape(env.num_envs, 2, 3)
    return torch.cat((rear[:, :2] + front, rear[:, 2:]), dim=1)


def hands_center_x(env):
    robot = env.scene["robot"]
    if not hasattr(env, "_frontflip_pad_ids"):
        names = list(robot.body_names)
        env._frontflip_pad_ids = [names.index(name) for name in
                                   FEET[:2] + ("LF_palm_pad_f_link", "RF_palm_pad_f_link")]
    x = robot.data.body_link_pos_w[:, env._frontflip_pad_ids, 0]
    back = env.scene.sensors["feet_ground_contact"].data.found.reshape(env.num_envs, 6)[:, :2]
    front = env.scene.sensors["claw_front_ground_contact"].data.found.reshape(env.num_envs, 2)
    weights = torch.cat((back, front), dim=1).bool().float()
    count = weights.sum(dim=1)
    return torch.where(count > 0, (x * weights).sum(dim=1) / count.clamp_min(1), x.mean(dim=1))


def scene_cfg(num_envs):
    robot = robot_cfg()
    sensor_cfgs = list(sensors())
    sensor_cfgs.append(ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"), reduce="maxforce", num_slots=1, history_length=2,
    ))
    cfg = ManagerBasedRlEnvCfg(
        decimation=2, episode_length_s=DURATION,
        scene=SceneCfg(num_envs=num_envs, entities={"robot": robot},
                       terrain=TerrainEntityCfg(terrain_type="plane"),
                       sensors=tuple(sensor_cfgs)),
    )
    cfg.sim.mujoco.timestep = PHYSICS_DT
    cfg.sim.njmax, cfg.sim.nconmax, cfg.sim.contact_sensor_maxmatch = 512, 128, 128
    cfg.sim.mujoco.ccd_iterations = 50
    return cfg


def targets(params, times, names):
    """[batch,time,joint] ZOH targets, bounded before physics; fingers stay at HOME."""
    params = np.atleast_2d(params)
    if params.shape[1] != len(SEED) or not np.isfinite(params).all():
        raise ValueError("expected finite 27-parameter rows")
    if ((params < LOW) | (params > HIGH)).any():
        raise ValueError("search parameters outside their bounds")
    home = np.array([HOME[n] for n in names])
    result = np.broadcast_to(home, (len(params), len(times), len(names))).copy()
    for k, p in enumerate(params):
        tc, tp, tt, hold, front_delay, rear_delay = p[21:]
        knots = np.array([T_GO, T_GO + tc, T_GO + tc + tp,
                          T_GO + tc + tp + tt, T_GO + tc + tp + tt + hold,
                          T_GO + tc + tp + tt + hold + .4])
        for j, (left, right, sign) in enumerate(JOINT_PAIRS):
            delay = front_delay if left.startswith("LF") else rear_delay if left.startswith("LR") else 0
            ts = times - delay
            idx = np.clip(np.searchsorted(knots, ts, side="right") - 1, 0, len(knots) - 2)
            u = np.clip((ts - knots[idx]) / (knots[idx + 1] - knots[idx]), 0, 1)
            u = u * u * (3 - 2 * u)
            values = np.array([HOME[left], p[j], p[j + 7], p[j + 14], p[j + 14], HOME[left]])
            delta = values[idx] + (values[idx + 1] - values[idx]) * u - HOME[left]
            result[k, :, names.index(left)] += delta
            result[k, :, names.index(right)] += sign * delta
    return result


def signed_rotation(quaternions):
    """Accumulate small world-Y quaternion increments; positive means nose down."""
    q = np.asarray(quaternions)
    a, b = q[1:], q[:-1].copy()
    b[..., 1:] *= -1
    w = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(axis=-1)
    v = (a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
         + np.cross(a[..., 1:], b[..., 1:]))
    negative = w < 0
    w = np.where(negative, -w, w)
    v = np.where(negative[..., None], -v, v)
    norm = np.linalg.norm(v, axis=-1)
    scale = 2 * np.arctan2(norm, w) / np.maximum(norm, 1e-12)
    return np.r_[0.0, np.cumsum(v[..., 1] * scale)]


def qualify(data):
    """Flight/turn/landing gates; grounded rolling cannot earn qualification."""
    t, rotation = data["t"], data["forward_rotation"]
    airborne = (data["contact"].sum(axis=1) == 0) & (data["body_force"] < 1.0)
    run, longest, start = 0, 0, 0
    flight_turn = 0.0
    for i, value in enumerate(airborne):
        if value and run == 0:
            start = i
        run = run + 1 if value else 0
        longest = max(longest, run)
        if value:
            flight_turn = max(flight_turn, float(rotation[i] - rotation[start]))
    last = t >= t[-1] - .30
    upright = 1 - 2 * (data["base_quat"][:, 1] ** 2 + data["base_quat"][:, 2] ** 2)
    home = np.array([HOME[n] for n in data["joint_names"]])
    home_error = float(np.sqrt(np.mean((data["q"][last] - home) ** 2)))
    checks = {
        "airborne": longest * CONTROL_DT >= .08,
        "full_forward_turn": abs(float(rotation[-1]) - 2 * np.pi) < .35,
        "turn_in_flight": flight_turn >= 1.5 * np.pi,
        "no_chassis_impact": float(data["chassis_force"].max()) < CHASSIS_FORCE_LIMIT,
        "limb_impact_bounded": float(data["body_force"].max()) < LIMB_FORCE_LIMIT,
        "no_leg_collision": float(data["self_force"].max()) < 1.0,
        "upright_landing": bool((upright[last] > np.cos(np.radians(15))).all()),
        "feet_landed": bool((data["contact"][last].sum(axis=1) >= 3).all()),
        "home_recovered": home_error < .18,
        "settled": bool((np.linalg.norm(data["base_lin_vel"][last], axis=1) < .25).all()
                        and (np.linalg.norm(data["base_ang_vel"][last], axis=1) < 1.0).all()),
        "torque_bounded": float(np.abs(data["tau"]).max()) <= EFFORT_LIMIT + 1e-5,
        "joint_limits": bool(data["joint_limit_violation"] < .01),
    }
    staged = {}
    if data.get("schedule") == "rear-claw":
        from .staged import summarize

        staged = summarize(data)
        # 0.02 N*s is measured forward reaction, not merely visiting a phase.
        checks["rear_propulsion"] = staged["rear_forward_impulse_Ns"] > .02
        # Braking at the claws, below the COM, is what spins the robot nose-down.
        checks["claw_spin"] = (staged["claw_push_executed"]
                               and staged["claw_pitch_momentum_Nms"] >= CLAW_SPIN_MIN)
    return {
        "qualified": all(checks.values()), "checks": checks,
        "longest_flight_s": longest * CONTROL_DT,
        "forward_turn_deg": float(np.degrees(rotation[-1])),
        "flight_turn_deg": float(np.degrees(flight_turn)),
        "peak_height_m": float(data["base_pos"][:, 2].max()),
        "home_rms_rad": home_error,
        "peak_torque_Nm": float(np.abs(data["tau"]).max()),
        "peak_body_force_N": float(data["body_force"].max()),
        "peak_chassis_force_N": float(data["chassis_force"].max()),
        **staged,
    }


def save_reference(path, data, params, *, diagnostic=False, variant=None):
    report = qualify(data)
    if not report["qualified"] and not diagnostic:
        raise ValueError("unqualified rollout may only be saved as a diagnostic candidate")
    meta = {
        "joint_order": data["joint_names"], "rec_hz": 1 / CONTROL_DT,
        "control_hz": 1 / CONTROL_DT, "t_go": T_GO, "go_frame": round(T_GO / CONTROL_DT),
        "stand_base_z": float(data["base_pos"][0, 2]),
        "model_fingerprint": fingerprint(), "KP": KP, "KD": KD,
        "qualified": report["qualified"], "diagnostic": diagnostic,
        "qualification": report, "params": np.asarray(params).tolist(),
        "angular_velocity_frame": "world", "source": "native physics rollout; no state animation",
        "schedule": data.get("schedule", "jump"),
        "foot_force_convention": "world-frame terrain reaction acting on each foot",
        "momentum_frame": "world [Px Py Pz Hx Hy Hz]; angular momentum about whole-robot COM",
        "variant": variant,
    }
    out = {k: np.asarray(data[k], dtype=np.float32) for k in (
        "t", "q", "qd", "q_cmd", "tau", "base_pos", "base_quat", "base_lin_vel",
        "base_ang_vel", "foot_pos", "contact", "forward_rotation", "body_force", "self_force",
        "chassis_force",
    )}
    for key in ("stage", "foot_force", "com_ahead", "momentum"):
        if key in data:
            out[key] = np.asarray(data[key], dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **out)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return report


class PhysicsRunner:
    """Batched native rollouts of targets, with the training entity's actual servos."""

    def __init__(self, num_envs):
        from mjrl.backend.resolve import resolve
        from mjrl.backend.select import use_backend
        from mjlab.envs import ManagerBasedRlEnv

        use_backend(resolve(backend="native", device="cpu", num_envs=num_envs, training=False))
        self.env = ManagerBasedRlEnv(cfg=scene_cfg(num_envs), device="cpu")
        self.robot = self.env.scene.entities["robot"]
        self.names = list(self.robot.joint_names)
        self.meter = MomentumMeter(self.env.sim.mj_model, self.robot.indexing.root_body_id)
        self.feet = [list(self.robot.body_names).index(n) for n in FEET]
        self.times = np.arange(round(DURATION / CONTROL_DT) + 1) * CONTROL_DT

    def rollout(self, params, video=None, schedule="jump"):
        env, robot = self.env, self.robot
        env.reset(seed=0)
        params = np.atleast_2d(params)
        if len(params) != env.num_envs:
            raise ValueError("parameter batch must match num_envs")
        controller = None
        if schedule == "rear-claw":
            from .staged import RearClawController

            controller = RearClawController(params, self.names)
            command = np.broadcast_to([HOME[n] for n in self.names],
                                      (len(params), len(self.times), len(self.names))).copy()
        elif schedule == "jump":
            command = targets(params, self.times, self.names)
        else:
            raise ValueError(f"Unknown search schedule: {schedule}")
        limits = robot.data.joint_pos_limits[0].cpu().numpy()
        if ((command < limits[:, 0] + .08) | (command > limits[:, 1] - .08)).any():
            raise ValueError("target schedule violates an XML joint limit/margin")
        command_t = torch.tensor(command, dtype=torch.float32)
        records = {k: [] for k in ("q", "qd", "tau", "base_pos", "base_quat",
                                   "base_lin_vel", "base_ang_vel", "foot_pos", "contact",
                                   "body_force", "self_force", "chassis_force", "qpos",
                                   "qvel", "stage",
                                   "foot_force", "com_ahead")}
        context = nullcontext(None)
        if video is not None:
            from mjrl.viewer.video import ReplayVideo

            context = ReplayVideo(env, video, fps=50, width=960, height=720,
                                  camera_distance=1.2, azimuth=145, elevation=-15)
        with context as recorder:
            return self._integrate(command, command_t, records, limits, recorder, controller)

    def _integrate(self, command, command_t, records, limits, recorder, controller):
        env, robot = self.env, self.robot
        for k in range(len(self.times)):
            if k:
                target = command_t[:, k]
                if controller is not None:
                    contact = support_contacts(env)
                    hands_x = hands_center_x(env)
                    com_x = env.sim.data.subtree_com[:, robot.indexing.root_body_id, 0]
                    ahead = (com_x >= hands_x + .005).cpu().numpy()
                    command[:, k] = controller.target(
                        self.times[k], robot.data.root_link_quat_w.cpu().numpy(),
                        contact.cpu().numpy(), ahead,
                    )
                    if ((command[:, k] < limits[:, 0] + .08)
                            | (command[:, k] > limits[:, 1] - .08)).any():
                        raise ValueError("rear-claw schedule violates a joint limit/margin")
                    target = torch.tensor(command[:, k], dtype=torch.float32)
                robot.set_joint_position_target(target)
                for _ in range(2):
                    env.scene.write_data_to_sim()
                    env.sim.step()
                    env.scene.update(PHYSICS_DT)
                env.sim.forward()
                env.sim.sense()
                if recorder is not None:
                    recorder.update()
            state = robot.data
            values = dict(
                q=state.joint_pos, qd=state.joint_vel,
                tau=env.sim.data.actuator_force,
                base_pos=state.root_link_pos_w - env.scene.env_origins,
                base_quat=state.root_link_quat_w,
                base_lin_vel=state.root_link_lin_vel_w, base_ang_vel=state.root_link_ang_vel_w,
                foot_pos=state.body_link_pos_w[:, self.feet] - env.scene.env_origins[:, None, :],
                qpos=env.sim.data.qpos, qvel=env.sim.data.qvel,
            )
            values["contact"] = support_contacts(env).float()
            # The sensor stores primary -> terrain; reaction on the robot is opposite.
            values["foot_force"] = -support_forces(env)
            hands_x = hands_center_x(env)
            com_x = env.sim.data.subtree_com[:, robot.indexing.root_body_id, 0]
            values["com_ahead"] = (com_x >= hands_x + .005).float()
            values["stage"] = torch.as_tensor(
                controller.command_stage if controller is not None and k else
                np.full(env.num_envs, -1), dtype=torch.float32,
            )
            for name, key in (("body_ground_contact", "body_force"), ("self_collision", "self_force"),
                              ("chassis_ground_contact", "chassis_force")):
                force = env.scene.sensors[name].data.force_history
                values[key] = force.norm(dim=-1).reshape(env.num_envs, -1).amax(dim=1)
            for key, value in values.items():
                records[key].append(value.detach().cpu().numpy().copy())
        arrays = {k: np.stack(v, axis=1) for k, v in records.items()}
        results = []
        for i in range(env.num_envs):
            data = {k: v[i] for k, v in arrays.items()}
            data.update(t=self.times.copy(), q_cmd=command[i], joint_names=self.names)
            data["schedule"] = "rear-claw" if controller is not None else "jump"
            data["forward_rotation"] = signed_rotation(data["base_quat"])
            data["momentum"] = self.meter.read(data["qpos"], data["qvel"])
            data["joint_limit_violation"] = float(np.maximum(
                limits[:, 0] - data["q"], data["q"] - limits[:, 1]).max())
            if not all(np.isfinite(v).all() for v in data.values() if isinstance(v, np.ndarray)):
                raise ValueError("nonfinite physics rollout")
            results.append(data)
        return results

    def close(self):
        self.env.close()
