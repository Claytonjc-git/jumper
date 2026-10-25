"""Task-local flip reference, angular-state reset, tracking and failure checks."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.utils.lab_api.math import (
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)

from ..jump.mdp.commands import JumpMotionCommand, JumpMotionCommandCfg
from ..jump.mdp.reference import JumpReference, ref_state
from ..common.assets import JUMPER_XML
from ..common.constants import HOME
from .physics import CONTROL_DT, KD, KP, T_GO, fingerprint, qualify, signed_rotation, support_contacts


def read_reference(path: Path, allow_diagnostic: bool = False) -> tuple[dict, dict]:
    """Reject incompatible, malformed or unqualified references before building a scene."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Somersault reference missing: {path}. Run tools/search_front_somersault.py "
            "first; train only after its physics rollout qualifies. See the task README."
        )
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"].item()))
        keys = (
            "t", "q", "qd", "q_cmd", "base_pos", "base_quat", "base_lin_vel",
            "base_ang_vel", "foot_pos", "contact", "forward_rotation", "tau",
            "body_force", "self_force",
        )
        arrays = {key: archive[key] for key in keys}
        if meta.get("schedule") == "rear-claw":
            arrays.update({key: archive[key] for key in ("stage", "foot_force", "com_ahead")})
            if meta.get("foot_force_convention") != "world-frame terrain reaction acting on each foot":
                raise ValueError("Rear-claw reference must record terrain reaction acting on the feet")
    names = meta["joint_order"]
    if len(names) != len(set(names)) or set(names) != set(HOME):
        raise ValueError("Reference joint_order must name every robot joint exactly once")
    n, joints = len(arrays["t"]), len(names)
    shapes = {
        "t": (n,), "q": (n, joints), "qd": (n, joints), "q_cmd": (n, joints),
        "base_pos": (n, 3), "base_quat": (n, 4), "base_lin_vel": (n, 3),
        "base_ang_vel": (n, 3), "foot_pos": (n, 6, 3), "contact": (n, 6),
        "forward_rotation": (n,), "tau": (n, joints), "body_force": (n,), "self_force": (n,),
    }
    if meta.get("schedule") == "rear-claw":
        shapes.update(stage=(n,), foot_force=(n, 6, 3), com_ahead=(n,))
    for key, array in arrays.items():
        if array.shape != shapes[key] or array.dtype != np.float32:
            raise ValueError(f"{key}: expected float32 {shapes[key]}, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains non-finite samples")
    if n < 2 or abs(float(arrays["t"][0])) > 1e-6:
        raise ValueError("Reference must have at least two samples and start at t=0")
    if not np.allclose(np.diff(arrays["t"]), CONTROL_DT, atol=1e-6, rtol=0):
        raise ValueError("Reference times must increase uniformly at the 200 Hz control rate")
    if not np.allclose(np.linalg.norm(arrays["base_quat"], axis=1), 1.0, atol=1e-3):
        raise ValueError("Reference base_quat must contain unit wxyz quaternions")
    if not np.isin(arrays["contact"], (0.0, 1.0)).all():
        raise ValueError("Reference contact must contain six binary foot-contact channels")
    if meta.get("schedule") == "rear-claw":
        if not np.isin(arrays["stage"], np.arange(-1, 7)).all():
            raise ValueError("Rear-claw reference has an unknown controller stage")
        if not np.isin(arrays["com_ahead"], (0., 1.)).all():
            raise ValueError("Rear-claw COM feedback must be binary")
    if meta.get("model_fingerprint") != fingerprint():
        raise ValueError("Reference model/motor fingerprint differs; regenerate it for this checkout")
    for key, expected in (("KP", KP), ("KD", KD), ("rec_hz", 1 / CONTROL_DT),
                          ("control_hz", 1 / CONTROL_DT)):
        if meta.get(key) != expected:
            raise ValueError(f"Reference {key}={meta.get(key)!r}; expected {expected}")
    go_frame = meta["go_frame"]
    if (isinstance(go_frame, bool) or not isinstance(go_frame, int)
            or not 0 <= go_frame < n - 1):
        raise ValueError("Reference go_frame must select a sample before its endpoint")
    if not np.isclose(meta["t_go"], arrays["t"][go_frame], atol=1e-6):
        raise ValueError("Reference t_go must agree with go_frame")
    if (go_frame != round(T_GO / CONTROL_DT)
            or not np.isclose(meta["t_go"], T_GO, atol=1e-6, rtol=0)):
        raise ValueError(f"Reference go timing must match this task's {T_GO:.2f} s launch clock")
    if not np.isfinite(meta["stand_base_z"]):
        raise ValueError("Reference stand_base_z must be finite")
    if not isinstance(meta.get("qualification"), dict):
        raise ValueError("Reference must carry its measured qualification report")
    rotation = signed_rotation(arrays["base_quat"])
    if not np.allclose(arrays["forward_rotation"], rotation, atol=1e-4, rtol=0):
        raise ValueError("Reference accumulated rotation disagrees with its quaternion trajectory")
    limits = {joint.get("name"): np.fromstring(joint.get("range"), sep=" ")
              for joint in ET.parse(JUMPER_XML).iter("joint") if joint.get("range")}
    ranges = np.stack([limits[name] for name in names])
    if ((arrays["q_cmd"] < ranges[:, 0] + .08)
            | (arrays["q_cmd"] > ranges[:, 1] - .08)).any():
        raise ValueError("Reference commanded targets violate the XML limits or 0.08 rad margin")
    violation = float(np.maximum(ranges[:, 0] - arrays["q"], arrays["q"] - ranges[:, 1]).max())
    measured = qualify(dict(arrays, joint_names=names, joint_limit_violation=violation,
                            schedule=meta.get("schedule", "jump")))
    if meta["qualification"].get("checks") != measured["checks"]:
        raise ValueError("Reference qualification report disagrees with measured trajectory")
    qualified = meta.get("qualified") is True and meta.get("diagnostic") is False
    if qualified:
        report = meta["qualification"]
        if report.get("qualified") is not True or not measured["qualified"]:
            raise ValueError("Qualified metadata disagrees with its qualification checks")
    if meta.get("angular_velocity_frame") != "world":
        raise ValueError("Somersault angular velocity must be recorded in world coordinates")
    if not qualified and not allow_diagnostic:
        raise ValueError(
            "Reference is not a qualified forward somersault. Inspect its qualification report "
            "and continue the physics search. MJRL_FRONT_SOMERSAULT_ALLOW_DIAGNOSTIC=1 "
            "allows an explicit environment smoke test only."
        )
    return meta, arrays


class FrontSomersaultReference(JumpReference):
    """Jump's reference API with measured contacts and angular state for a full revolution."""

    def __init__(self, path: Path, device: str, joint_names: list[str],
                 allow_diagnostic: bool = False):
        meta, arrays = read_reference(path, allow_diagnostic)
        source = meta["joint_order"]
        missing = set(joint_names) - set(source)
        if missing:
            raise ValueError(f"Somersault reference missing robot joints: {sorted(missing)}")
        perm = [source.index(name) for name in joint_names]
        self.path, self.meta, self.joint_names = path, meta, list(joint_names)
        self.n = len(arrays["t"])
        self.rec_hz, self.control_hz = float(meta["rec_hz"]), float(meta["control_hz"])
        self.dt = 1.0 / self.rec_hz
        self.t_go, self.go_frame = float(meta["t_go"]), int(meta["go_frame"])
        self.duration = float(arrays["t"][-1])
        self.span = self.duration - self.t_go
        self.stand_base_z = float(meta["stand_base_z"])
        for key in arrays:
            value = arrays[key][:, perm] if key in ("q", "qd", "q_cmd") else arrays[key]
            setattr(self, key, torch.as_tensor(value, dtype=torch.float32, device=device))
        rel = self.foot_pos - self.base_pos.unsqueeze(1)
        q = self.base_quat.unsqueeze(1).expand(-1, 6, -1).reshape(-1, 4)
        self.foot_b = quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(rel.shape)
        gravity = self.base_pos.new_tensor([0.0, 0.0, -1.0]).expand(self.n, 3)
        self.proj_g = quat_apply_inverse(self.base_quat, gravity)

    def sample(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        sample = super().sample(idx)
        lo = idx.floor().long().clamp(0, self.n - 1)
        hi = (lo + 1).clamp(0, self.n - 1)
        w = (idx - lo.float()).unsqueeze(-1)
        q0, q1 = self.base_quat[lo], self.base_quat[hi]
        # Antipodal quaternions describe the same pose; interpolating through zero does not.
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        quat = q0 * (1 - w) + q1 * w
        sample["base_quat"] = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sample["base_ang_vel"] = self.base_ang_vel[lo] * (1 - w) + self.base_ang_vel[hi] * w
        scalar_w = w.squeeze(-1)
        sample["forward_rotation"] = (
            self.forward_rotation[lo] * (1 - scalar_w) + self.forward_rotation[hi] * scalar_w
        )
        return sample


@dataclass(kw_only=True)
class FrontSomersaultCommandCfg(JumpMotionCommandCfg):
    allow_diagnostic: bool = False

    def __post_init__(self):
        self.class_type = FrontSomersaultCommand

    def build(self, env):
        return FrontSomersaultCommand(self, env)


class FrontSomersaultCommand(JumpMotionCommand):
    def __init__(self, cfg: FrontSomersaultCommandCfg, env):
        # Reuse jump machinery through its per-environment cache, without changing module globals.
        env._jump_reference = FrontSomersaultReference(
            Path(cfg.motion_file), env.device, list(env.scene[cfg.entity_name].joint_names),
            cfg.allow_diagnostic,
        )
        super().__init__(cfg, env)
        self.rotation = torch.zeros(self.num_envs, device=self.device)
        self.previous_quat = self.robot.data.root_link_quat_w.clone()
        self._rotation_stamp = -1
        self.metrics["forward_rotation_rad"] = self.rotation

    def _resample_command(self, env_ids):
        phase = getattr(self._env, "_jump_rsi_phase", torch.zeros_like(self.rotation))
        phase = phase[env_ids].clone()
        sample = self.reference.sample(self.reference.index_of(phase * self.reference.span))
        super()._resample_command(env_ids)
        self.rotation[env_ids] = sample["forward_rotation"]
        self.previous_quat[env_ids] = sample["base_quat"]
        # A reset changes go_step without advancing common_step_counter.
        self._env._jump_ref_stamp = None

    def measured_rotation(self):
        # Rewards run before command.compute(); update once at the reward's current state.
        stamp = int(self._env.common_step_counter)
        if stamp != self._rotation_stamp:
            quat = self.robot.data.root_link_quat_w
            delta = quat_mul(quat, quat_inv(self.previous_quat))
            delta = torch.where(delta[:, :1] < 0, -delta, delta)
            length = delta[:, 1:].norm(dim=-1)
            angle = 2 * torch.atan2(length, delta[:, 0])
            self.rotation += angle * delta[:, 2] / length.clamp_min(1e-8)
            self.previous_quat.copy_(quat)
            self._rotation_stamp = stamp
        return self.rotation

    def _update_command(self, env_ids=None):
        if env_ids is not None:
            return
        z = self.robot.data.root_link_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        torch.maximum(self._apex_rise, z - self._stance_z, out=self._apex_rise)
        airborne = ~support_contacts(self._env).any(dim=1)
        self._flight_s += airborne.float() * self._env.step_dt
        grounded = ~airborne
        vz = self.robot.data.root_link_lin_vel_w[:, 2]
        self._takeoff_vz[grounded] = torch.maximum(self._takeoff_vz[grounded], vz[grounded])
        self.measured_rotation()


def reset_from_reference_phase(env, env_ids, command_name="jump", rsi_fraction=1.0,
                               phase_range=(0.0, 0.9)):
    """Restore pose and both world-frame root velocities, including the flip's spin."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term(command_name)
    phase = torch.empty(len(env_ids), device=env.device).uniform_(*phase_range)
    use_rsi = torch.rand(len(env_ids), device=env.device) < rsi_fraction
    if not hasattr(env, "_jump_rsi_phase"):
        env._jump_rsi_phase = torch.zeros(env.num_envs, device=env.device)
    env._jump_rsi_phase[env_ids] = torch.where(use_rsi, phase, torch.zeros_like(phase))
    ids = env_ids[use_rsi]
    if not len(ids):
        return
    sample = cmd.reference.sample(cmd.reference.index_of(phase[use_rsi] * cmd.reference.span))
    root = torch.cat((sample["base_pos"] + env.scene.env_origins[ids], sample["base_quat"],
                      sample["base_lin_vel"], sample["base_ang_vel"]), dim=-1)
    robot = env.scene["robot"]
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(sample["q"], sample["qd"], env_ids=ids)


def ref_full_orientation(env, command_name="jump", sigma_sq=0.15):
    _, sample, _, _ = ref_state(env, command_name)
    error = quat_error_magnitude(env.scene["robot"].data.root_link_quat_w, sample["base_quat"])
    return torch.exp(-error.square() / sigma_sq)


def ref_angular_velocity(env, command_name="jump", sigma_sq=4.0):
    _, sample, _, _ = ref_state(env, command_name)
    error = env.scene["robot"].data.root_link_ang_vel_w - sample["base_ang_vel"]
    return torch.exp(-error.square().sum(dim=-1) / sigma_sq)


def ref_forward_rotation(env, command_name="jump", sigma_sq=0.25):
    _, sample, _, _ = ref_state(env, command_name)
    cmd = env.command_manager.get_term(command_name)
    error = cmd.measured_rotation() - sample["forward_rotation"]
    return torch.exp(-error.square() / sigma_sq)


def ref_contact_found(env, command_name="jump"):
    """Match the geometric contacts recorded by the physics search, including unloaded feet."""
    _, sample, _, _ = ref_state(env, command_name)
    actual = support_contacts(env)
    return (actual == sample["contact"].bool()).float().mean(dim=1)


def collision_failure(env, sensor_name, force_threshold=1.0):
    sensor = env.scene[sensor_name]
    history = sensor.data.force_history
    if history is None:
        raise ValueError(f"{sensor_name} must retain force history across physics substeps")
    return (history.norm(dim=-1) > force_threshold).flatten(1).any(dim=1)
