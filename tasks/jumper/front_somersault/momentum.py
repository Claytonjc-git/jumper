"""Original short handoff with optional rearward middle drive and causal peak release.

Optional `flight` settings change the tuck pose per leg, hold the claws extended, or open
chosen limbs to HOME at a set time after TUCK starts, before touchdown.
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from ..common.constants import HOME

LIMBS = {"claw": ("LF", "RF"), "middle": ("LM", "RM"), "rear": ("LR", "RR")}


class MomentumMeter:
    """Read nominal state into an independent model; never change live physics."""

    def __init__(self, model, root_body):
        self.model = copy.copy(model)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(self.model)
        self.root = int(root_body)
        self.mass = float(self.model.body_subtreemass[self.root])

    def read(self, qpos, qvel):
        result = []
        for q, v in zip(np.atleast_2d(qpos), np.atleast_2d(qvel), strict=True):
            self.data.qpos[:] = q
            self.data.qvel[:] = v
            mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mj_fwdVelocity(self.model, self.data)
            mujoco.mj_subtreeVel(self.model, self.data)
            result.append(np.r_[self.mass*self.data.subtree_linvel[self.root],
                                self.data.subtree_angmom[self.root]])
        return np.array(result)


def controller_type(base, variants):
    class MomentumController(base):
        def __init__(self, params, names):
            super().__init__(params, names)
            self.options = [v["momentum"] for v in variants]
            n = len(params)
            self.middle_ids = [names.index(f"{leg}_J{j}_joint")
                               for leg in ("LM", "RM") for j in (0, 1, 2)]
            self.rear_started = np.full(n, np.nan)
            self.middle_start = np.zeros((n, 6))
            self.middle_release = np.full((n, 6), np.nan)
            self.middle_release_time = np.full(n, np.nan)
            self.planted_since = np.full(n, np.nan)
            self.running_peak = np.zeros(n)
            self.declining = np.zeros(n, dtype=int)
            self.launch_momentum = np.full((n, 6), np.nan)
            self.launch_time = np.full(n, np.nan)
            self.release_reason = [None]*n
            self.tuck_time = np.full(n, np.nan)
            self.land_time = np.full(n, np.nan)
            self.land_from = np.zeros((n, len(names)))
            self.home = np.array([HOME[name] for name in names])
            for i, o in enumerate(self.options):
                f = o.get("flight", {})
                if f.get("claw", "tuck") == "extended":
                    claw = [k for k, name in enumerate(names) if name[:2] in LIMBS["claw"]]
                    self.poses[i, self.TUCK, claw] = self.poses[i, self.PUSH, claw]
                # Absolute left-leg tuck targets from J0; the right leg mirrors by name.
                for left_leg, values in f.get("tuck", {}).items():
                    right_leg = "R"+left_leg[1]
                    for j, value in enumerate(values):
                        left, right = f"{left_leg}_J{j}_joint", f"{right_leg}_J{j}_joint"
                        sign = 1 if left == "LF_J1_joint" else -1
                        self.poses[i, self.TUCK, names.index(left)] = value
                        self.poses[i, self.TUCK, names.index(right)] = (
                            HOME[right]+sign*(value-HOME[left]))
            for i, o in enumerate(self.options):
                if not o.get("middle_enabled", True):
                    continue
                for leg, sign in (("LM", 1), ("RM", -1)):
                    for j in (0, 1, 2):
                        name, left = f"{leg}_J{j}_joint", f"LM_J{j}_joint"
                        load = o["load"][j]
                        drive = o["drive"][j]
                        load = load if sign == 1 else HOME[name]-(load-HOME[left])
                        drive = drive if sign == 1 else HOME[name]-(drive-HOME[left])
                        self.stance[i, names.index(name)] = load
                        self.poses[i, self.LOAD, names.index(name)] = load
                        for stage in (self.REAR, self.REACH):
                            self.poses[i, stage, names.index(name)] = drive

        def target(self, time, quat, contact, com_ahead):
            if not hasattr(self, "measured_momentum"):
                raise ValueError("momentum launch requires measured whole-robot momentum")
            before = self.stage.copy()
            elapsed = np.maximum(time-self.started, 0)
            start_pose = self.start_pose.copy()
            previous_middle = self.current[:, self.middle_ids].copy()
            # Preserve the original pitch-triggered deployment. Only take over
            # the final planted-palm handoff; it must remain causal and short.
            ahead = np.asarray(com_ahead).copy()
            ahead[before == self.REACH] = False
            super().target(time, quat, contact, ahead)
            reached_rear = (before == self.LOAD) & (self.stage == self.REAR)
            self.rear_started[reached_rear] = time
            self.middle_start[reached_rear] = self.current[reached_rear][:, self.middle_ids]
            for i, o in enumerate(self.options):
                if not o.get("middle_enabled", True):
                    continue
                active = before[i] in (self.REAR, self.REACH)
                if before[i] == self.PUSH and elapsed[i] < o["middle_hold_s"]:
                    active = True
                if active:
                    u = np.clip((time-self.rear_started[i]-o["middle_start_s"])
                                /o["middle_ramp_s"], 0, 1)
                    u = u*u*(3-2*u)
                    goal = self.poses[i, self.REAR, self.middle_ids]
                    self.current[i, self.middle_ids] = (
                        self.middle_start[i]+(goal-self.middle_start[i])*u)
                elif before[i] == self.PUSH and o["middle_hold_s"] > 0:
                    if np.isnan(self.middle_release_time[i]):
                        self.middle_release_time[i] = time
                        self.middle_release[i] = previous_middle[i]
                    u = np.clip((time-self.middle_release_time[i])/.08, 0, 1)
                    u = u*u*(3-2*u)
                    drive = self.middle_release[i]
                    self.current[i, self.middle_ids] = drive+(
                        self.poses[i, self.PUSH, self.middle_ids]-drive)*u
            measured = np.asarray(self.measured_momentum)
            # Normalize by original launch scales; this is a timing signal,
            # not kinetic energy. Upward momentum is checked separately.
            signal = np.maximum(measured[:, 0], 0)/.63+np.maximum(measured[:, 4], 0)/.063
            watching = (before == self.REAR) | (before == self.REACH)
            self.running_peak[watching] = np.maximum(self.running_peak[watching], signal[watching])
            falling = watching & (signal < self.running_peak*.98)
            self.declining = np.where(falling, self.declining+1, 0)
            planted = np.asarray(contact)[:, :2].all(axis=1)
            ready = ((before == self.REACH) & planted & np.asarray(com_ahead)
                     & (elapsed >= self.params[:, 21]) & (self.stage == self.REACH))
            first = ready & np.isnan(self.planted_since)
            self.planted_since[first] = time
            self.planted_since[~ready] = np.nan
            for i, o in enumerate(self.options):
                if not ready[i]:
                    continue
                waited = time-self.planted_since[i]
                fire = waited+1e-9 >= o["claw_delay_s"]
                if o.get("trigger", "delay") == "peak":
                    fire &= (self.declining[i] >= 2 and measured[i, 2] > 0
                             or waited >= o.get("max_wait_s", .03))
                if fire:
                    self.stage[i] = self.PUSH
                    self.started[i] = time
                    self.claw_pushed[i] = True
                    self.launch_time[i] = time
                    self.launch_momentum[i] = measured[i]
                    self.release_reason[i] = (
                        "delay" if o.get("trigger", "delay") == "delay" else
                        "peak_decline" if self.declining[i] >= 2 and measured[i, 2] > 0 else "max_wait")
            self.tuck_time[(before == self.PUSH) & (self.stage == self.TUCK)] = time
            for i, o in enumerate(self.options):
                f = o.get("flight", {})
                if f.get("land_s") is None or np.isnan(self.tuck_time[i]):
                    continue
                # Open the landing limbs to HOME before touchdown, from the live command.
                # ponytail: timed from TUCK start; a measured-rotation trigger if launches vary.
                ids = [k for k, name in enumerate(self.names)
                       if name[:2] in sum((LIMBS[g] for g in f["land"]), ())]
                if np.isnan(self.land_time[i]):
                    if time-self.tuck_time[i] < f["land_s"]:
                        continue
                    self.land_time[i] = time
                    self.land_from[i] = self.current[i]
                u = np.clip((time-self.land_time[i])/f["land_ramp_s"], 0, 1)
                u = u*u*(3-2*u)
                self.current[i, ids] = (self.land_from[i, ids]
                                        + (self.home[ids]-self.land_from[i, ids])*u)
            changed = before != self.stage
            # The middle override changes the outgoing command, so save that
            # command at every transition rather than the parent's earlier copy.
            self.start_pose[changed] = self.current[changed]
            # start_pose is otherwise owned by the original interpolation.
            self.start_pose[~changed] = start_pose[~changed]
            return self.current.copy()

    return MomentumController
