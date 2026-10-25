"""Rear-leg drive, pitch-triggered claw plant, then a push after COM crosses the hands."""

from __future__ import annotations

import numpy as np

from ..common.constants import HOME

# Left joint targets [rad], mirrored by HOME-relative deltas. FK checks found
# these rear load -> drive endpoints move the feet backward and down, not up.
# Front tuck folds the hand near the body with floor clearance at HOME height.
SEED = np.array([
    .26, -1.23722, -1.87970,     # LF J1/J2/J3: tucked close to the body
    .010935, -.768270, -1.738357,  # LF J1/J2/J3: plant 40 mm ahead, 130 mm down
    .120679, -1.946263, -.085052,  # LF J1/J2/J3: drive 20 mm back, 40 mm down
    .5534, -1.8648,             # LR J1/J2: keep rear support during preparation
    -.65, -.20,                 # LR J1/J2: drive backward/down
    1.20, -1.80,                # LM J1/J2: lift the middle feet
    1.20, -2.10,                # LR J1/J2: flight tuck
    .90,                       # LR J0: rear drive yaw; RR resolved by name
    .35, .10, .20, .06, .08, .35,  # prep/rear drive/pitch trigger/reach/push/tuck hold
])
LOW = np.array([
    .00, -1.50, -1.96, -.95, -1.10, -1.96, -1.1, -2.00, -1.90,
    .20, -2.10, -1.10, -1., .85, -2.10, .80, -2.20, .65,
    .25, .04, .10, .025, .035, .12,
])
HIGH = np.array([
    .60, -1.00, -1.50, .10, .05, -1.40, .35, -.30, .10,
    .80, -1.50, -.15, .15, 1.40, -1.30, 1.40, -1.60, 1.15,
    .50, .18, .50, .12, .15, .55,
])
STAGE_NAMES = ("load", "rear_drive", "claw_reach", "claw_push", "flight_tuck", "recover", "home")


def summarize(data):
    stage, force = data["stage"], data["foot_force"]
    dt = float(data["t"][1] - data["t"][0])
    return {
        "peak_forward_rotation_deg": float(np.degrees(data["forward_rotation"].max())),
        "forward_travel_m": float(data["base_pos"][:, 0].max() - data["base_pos"][0, 0]),
        "claw_push_executed": bool((stage == RearClawController.PUSH).any()),
        "rear_forward_impulse_Ns": float(force[stage == RearClawController.REAR, 4:, 0].sum() * dt),
        "claw_forward_impulse_Ns": float(force[stage == RearClawController.PUSH, :2, 0].sum() * dt),
        "claw_pitch_momentum_Nms": claw_spin(data),
    }


def claw_spin(data):
    """World-Y angular momentum gained from the frame before PUSH to its last frame."""
    push = np.flatnonzero(data["stage"] == RearClawController.PUSH)
    if "momentum" not in data or not len(push):
        return 0.0
    return float(data["momentum"][push[-1], 4] - data["momentum"][max(push[0] - 1, 0), 4])


class RearClawController:
    LOAD, REAR, REACH, PUSH, TUCK, RECOVER, HOME = range(7)

    def __init__(self, params, names):
        from .physics import T_GO

        self.params = np.atleast_2d(params)
        if (self.params.shape[1] != len(SEED) or not np.isfinite(self.params).all()
                or ((self.params < LOW) | (self.params > HIGH)).any()):
            raise ValueError("rear-claw schedule requires 24 bounded finite parameters")
        self.names = list(names)
        if set(names) != set(HOME) or len(names) != len(HOME):
            raise ValueError("rear-claw schedule must receive all robot joints by name")
        n = len(self.params)
        home = np.array([HOME[name] for name in names])
        self.poses = np.broadcast_to(home, (n, 7, len(names))).copy()

        def pair(stage, leg, joints, values):
            right = {"LF": "RF", "LM": "RM", "LR": "RR"}[leg]
            for joint, value in zip(joints, values.T, strict=True):
                left_name, right_name = f"{leg}_J{joint}_joint", f"{right}_J{joint}_joint"
                sign = 1 if leg == "LF" and joint == 1 else -1
                self.poses[:, stage, names.index(left_name)] = value
                self.poses[:, stage, names.index(right_name)] = HOME[right_name] + sign * (value - HOME[left_name])

        p = self.params
        for stage in (self.LOAD, self.REAR, self.REACH, self.PUSH, self.TUCK):
            pair(stage, "LF", (1, 2, 3), p[:, :3] if stage in (self.LOAD, self.REAR, self.TUCK)
                 else p[:, 3:6] if stage == self.REACH else p[:, 6:9])
            if stage == self.LOAD:
                # Keep middle support during preparation; lift it only when
                # the rear drive starts. A small forward stance widens support.
                pair(stage, "LM", (0,), np.full((n, 1), -.40))
            else:
                pair(stage, "LM", (1, 2), p[:, 13:15])
            pair(stage, "LR", (1, 2), p[:, 9:11] if stage == self.LOAD
                 else p[:, 11:13] if stage in (self.REAR, self.REACH) else p[:, 15:17])
            if stage in (self.REAR, self.REACH):
                pair(stage, "LR", (0,), p[:, 17:18])
        self.stage = np.zeros(n, dtype=np.int32)
        self.started = np.full(n, T_GO)
        self.current = np.tile(home, (n, 1))
        self.stance = self.current.copy()
        for name in ("LM_J0_joint", "RM_J0_joint"):
            self.stance[:, names.index(name)] = self.poses[:, self.LOAD, names.index(name)]
        self.start_pose = self.current.copy()
        self.pitch_triggered = np.zeros(n, dtype=bool)
        self.claw_pushed = np.zeros(n, dtype=bool)
        self.rear_contacted = np.zeros(n, dtype=bool)

    def target(self, time, quat, contact, com_ahead):
        self.command_stage = self.stage.copy()
        # atan2 keeps a nose-down lean positive past 90 degrees, unlike asin(pitch).
        w, x, y, z = np.asarray(quat).T
        pitch = np.arctan2(2 * (w * y - x * z), 1 - 2 * (y * y + z * z))
        elapsed = np.maximum(time - self.started, 0)
        p = self.params
        durations = np.choose(self.stage, (p[:, 18], p[:, 19], p[:, 21], p[:, 22],
                                           np.full(len(p), .08), np.full(len(p), .4),
                                           np.ones(len(p))))
        u = np.clip(elapsed / durations, 0, 1)
        u = (u * u * (3 - 2 * u))[:, None]
        goals = self.poses[np.arange(len(p)), self.stage]
        self.current = self.start_pose + (goals - self.start_pose) * u
        loading = self.stage == self.LOAD
        # Establish the support polygon before taking the claws off the floor.
        stance_u = np.clip(elapsed / .20, 0, 1)
        tuck_u = np.clip((elapsed - .20) / p[:, 18], 0, 1)
        stance_u = (stance_u * stance_u * (3 - 2 * stance_u))[:, None]
        tuck_u = (tuck_u * tuck_u * (3 - 2 * tuck_u))[:, None]
        prepared = (self.start_pose + (self.stance - self.start_pose) * stance_u
                    + (self.poses[:, self.LOAD] - self.stance) * tuck_u)
        self.current[loading] = prepared[loading]
        next_stage = self.stage.copy()
        next_stage[loading & (elapsed >= .20 + p[:, 18])] = self.REAR
        self.rear_contacted |= (self.stage == self.REAR) & np.asarray(contact)[:, 4:].all(axis=1)
        triggered = ((self.stage == self.REAR) & (elapsed >= .5 * p[:, 19])
                     & self.rear_contacted & (pitch >= p[:, 20]))
        self.pitch_triggered |= triggered
        next_stage[triggered] = self.REACH
        next_stage[(self.stage == self.REAR) & (elapsed >= .4) & ~triggered] = self.RECOVER
        planted = np.asarray(contact)[:, :2].all(axis=1)
        # Front ground forces can brake the roll while the hands are ahead of
        # the COM. Wait for the body to pass over the planted hands before pushing.
        push = ((self.stage == self.REACH) & (elapsed >= p[:, 21])
                & planted & np.asarray(com_ahead))
        self.claw_pushed |= push
        next_stage[push] = self.PUSH
        next_stage[(self.stage == self.REACH) & (elapsed >= .35) & ~push] = self.RECOVER
        next_stage[(self.stage == self.PUSH) & (elapsed >= p[:, 22])] = self.TUCK
        next_stage[(self.stage == self.TUCK) & (elapsed >= .08 + p[:, 23])] = self.RECOVER
        next_stage[(self.stage == self.RECOVER) & (elapsed >= .4)] = self.HOME
        changed = next_stage != self.stage
        self.start_pose[changed] = self.current[changed]
        self.started[changed] = time
        self.stage = next_stage
        return self.current.copy()
