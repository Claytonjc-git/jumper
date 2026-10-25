"""Tracking configuration for the 8.9 s claw stand; measurements in media/claw_stand.json.

Windows CPU, MuJoCo 3.11.0, 50 Hz: position RMS amplitude 0.26209 rad,
velocity RMS 0.58308 rad/s, peak speed 2.507 rad/s; base 106.47..135.00 mm,
maximum tilt 37.242 degrees. These weights are starting points, not trained results.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

MEDIA = Path(__file__).resolve().parent / "media"
# Short-clip rule: half of 8.9 s, rounded down to a control tick.
EPISODE_S = 4.44


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # Same hardware-scaled RSI and sensor noise as the other gestures.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # Bow's rules: .44 * position RMS = .11532 rad; (2/3)*velocity RMS=.38872.
        joint_pos_std=0.115,
        joint_pos_weight=1.0,
        joint_vel_std=0.389,
        joint_vel_weight=0.5,
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        power_weight=-0.02,
        # Keep the real servo's continuous/peak torque headroom objective.
        torque_headroom_weight=-2.0,
        # Every reference joint stays inside the 0.9 soft limits, including fingers.
        joint_limit_weight=-10.0,
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # Height error relative to reference: 28.53 mm band plus 16.47 mm margin.
        anchor_height_error=0.045,
        # Tilt is reference-relative |cos(ref tilt)-cos(actual tilt)|, not an
        # absolute tilt cutoff. 1-cos(37.242 deg)=.204; .33 corresponds to 47.93
        # deg from upright, leaving >10 deg over the clip's maximum tilt.
        anchor_tilt_error=0.33,
        # Actual claw/rear collision contacts stay within 0.220 mm of the floor.
        # 25 mm gives RSI/tracking room while detecting a lost support limb.
        support_foot_error=0.025,
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
    # This gesture stands on both claws and the rear pair. All six tips still
    # receive tracking rewards; only this task's termination support set changes.
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "LF_palm_pad_b_link", "RF_palm_pad_b_link",
        "LR_foot_tip_link", "RR_foot_tip_link",
    )
    return cfg
