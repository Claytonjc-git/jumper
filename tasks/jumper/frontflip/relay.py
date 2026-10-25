"""Experimental rear -> middle -> palm launch, with measured delayed jaw opening.

Used only by tools/probe_underbody_claw.py --relay. No shared configuration or
training controller changes. Stage IDs retain the original rollout schema;
the additional middle drive is 7.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import HOME

MIDDLE = 7
ALIGN = 8


def controller_type(base, variants):
    """Extend the diagnostic controller without changing the existing recipe."""
    class RelayController(base):
        def __init__(self, params, names):
            super().__init__(params, names)
            n = len(params)
            # Preplant the palms during middle drive; the large PUSH extension
            # remains blocked until both palms contact at the requested pitch.
            self.poses = np.concatenate((self.poses, self.poses[:, self.REACH:self.REACH+1],
                                         self.poses[:, self.LOAD:self.LOAD+1]), axis=1)
            self.options = [v["relay"] for v in variants]
            self.launch_angle = np.radians([o["launch_angle_deg"] for o in self.options])
            self.middle_ramp = np.array([o["middle_ramp"] for o in self.options])
            self.jaw_fraction = np.array([o["jaw_after_fraction"] for o in self.options])
            self.overlap_enabled = np.array(["overlap_fraction" in o for o in self.options])
            self.overlap_fraction = np.array([o.get("overlap_fraction", 1.) for o in self.options])
            self.overlap_angle = np.radians([o.get("overlap_min_angle_deg", 20)
                                            for o in self.options])
            self.middle_hold = np.array([o.get("middle_hold_s", .06) for o in self.options])
            self.middle_ids = [names.index(f"{leg}_J{j}_joint")
                               for leg in ("LM", "RM") for j in (0, 1, 2)]
            self.middle_started = np.full(n, np.nan)
            self.middle_start = np.zeros((n, 6))
            self.effective_hold = np.zeros(n)
            self.align_enabled = np.array([o.get("shoulder_after_tuck", False) for o in self.options])
            self.safe_retract = np.array([o.get("safe_retract", self.align_enabled[i])
                                          for i, o in enumerate(self.options)])
            self.align_duration = np.array([o.get("align_seconds", .4) for o in self.options])
            self.align_ids = [names.index(f"{leg}_J{j}_joint")
                              for leg in ("LF", "RF") for j in (0, 1, 2, 3)]
            self.aligned_at = np.full(n, np.nan)
            self.fold_started = np.full(n, np.nan)
            self.rear_clear_count = np.zeros(n, dtype=int)
            self.middle_seen = np.zeros(n, dtype=bool)
            self.jaw_started = np.full(n, np.nan)
            self.jaw_progress = np.full(n, np.nan)
            self.jaw_grounded = np.zeros(n, dtype=bool)
            self.relay_pitch = np.full(n, np.nan)
            self.middle_grounded_at_launch = np.zeros(n, dtype=bool)
            self.rear_clear_at = np.full(n, np.nan)
            self.arm_ids = [names.index(f"{leg}_J{j}_joint")
                            for leg in ("LF", "RF") for j in (1, 2, 3)]
            self.arm_start = np.zeros((n, 6))
            for i, option in enumerate(self.options):
                if option.get("shoulder_after_tuck", False):
                    for leg in ("LF", "RF"):
                        idx = names.index(f"{leg}_J0_joint")
                        self.poses[i, self.LOAD, idx] = HOME[names[idx]]
                        # Folding directly into the aligned elbow angles before
                        # turning J0 crosses the claws. Reach the original tuck
                        # first, then turn all four joints together in ALIGN.
                        for j, value in enumerate(params[i, :3], start=1):
                            name, left = f"{leg}_J{j}_joint", f"LF_J{j}_joint"
                            mirrored = (value if leg == "LF" else HOME[name]
                                        + (1 if j == 1 else -1)*(value-HOME[left]))
                            self.poses[i, self.LOAD, names.index(name)] = mirrored
                if "preplant_joints" in option:
                    for leg in ("LF", "RF"):
                        for j, value in enumerate(option["preplant_joints"], start=1):
                            name, left = f"{leg}_J{j}_joint", f"LF_J{j}_joint"
                            value = (value if leg == "LF" else HOME[name]
                                     + (1 if j == 1 else -1)*(value-HOME[left]))
                            for stage in (MIDDLE, self.REACH):
                                self.poses[i, stage, names.index(name)] = value
                if "push_joints" in option:
                    for leg in ("LF", "RF"):
                        for j, value in enumerate(option["push_joints"], start=1):
                            name, left = f"{leg}_J{j}_joint", f"LF_J{j}_joint"
                            self.poses[i, self.PUSH, names.index(name)] = (
                                value if leg == "LF" else HOME[name]
                                + (1 if j == 1 else -1)*(value-HOME[left]))
                for leg, sign in (("LM", 1), ("RM", -1)):
                    for j, value in enumerate((option["middle_yaw"],
                                               option["middle_hip"], option["middle_knee"])):
                        name, left = f"{leg}_J{j}_joint", f"LM_J{j}_joint"
                        value = value if sign == 1 else HOME[name] - (value-HOME[left])
                        for stage in (MIDDLE, self.REACH):
                            self.poses[i, stage, names.index(name)] = value
                        # Retain middle support through rear drive. J0 already
                        # moved to its forward stance in the LOAD preparation.
                        self.poses[i, self.REAR, names.index(name)] = (
                            self.poses[i, self.LOAD, names.index(name)])
                for leg in ("LR", "RR"):
                    for j in (0, 1, 2):
                        idx = names.index(f"{leg}_J{j}_joint")
                        for stage in (MIDDLE, self.REACH):
                            self.poses[i, stage, idx] = self.poses[i, self.TUCK, idx]
                for leg in ("LF", "RF"):
                    self.poses[i, :, names.index(f"{leg}_J4_joint")] = 0
                    if self.safe_retract[i]:
                        self.poses[i, self.TUCK, names.index(f"{leg}_J0_joint")] = HOME[
                            f"{leg}_J0_joint"]
                        for j, value in enumerate(params[i, :3], start=1):
                            name, left = f"{leg}_J{j}_joint", f"LF_J{j}_joint"
                            self.poses[i, self.TUCK, names.index(name)] = (
                                value if leg == "LF" else HOME[name]
                                + (1 if j == 1 else -1)*(value-HOME[left]))

        def target(self, time, quat, contact, com_ahead):
            if not hasattr(self, "measured_q"):
                raise ValueError("relay jaw gate requires measured joint positions")
            before = self.stage.copy()
            loading = before == self.LOAD
            elapsed = np.maximum(time-self.started, 0)
            # Reuse the original HOME lead-in and support-before-tuck ramp.
            if loading.any():
                held = self.stage.copy()
                self.stage[~loading] = self.HOME
                super().target(time, quat, contact, com_ahead)
                self.stage[~loading] = held[~loading]
                self.stage[loading & self.align_enabled & (self.stage == self.REAR)] = ALIGN
            self.command_stage = before.copy()
            w, x, y, z = np.asarray(quat).T
            pitch = np.arctan2(2*(w*y-x*z), 1-2*(y*y+z*z))
            duration = np.choose(before, (self.params[:, 18], self.params[:, 19],
                                         self.params[:, 21], self.params[:, 22],
                                         np.full(len(before), .08), np.full(len(before), .4),
                                         np.ones(len(before)), self.middle_ramp, self.align_duration))
            u = np.clip(elapsed/duration, 0, 1)
            u = (u*u*(3-2*u))[:, None]
            goal = self.poses[np.arange(len(before)), before]
            target = self.start_pose + (goal-self.start_pose)*u
            self.current[~loading] = target[~loading]
            retracting = (before == self.TUCK) & self.safe_retract
            if retracting.any():
                grip_ids = [self.names.index(f"{leg}_J4_joint") for leg in ("LF", "RF")]
                closed = (np.abs(np.asarray(self.measured_q)[:, grip_ids]) <= .08).all(axis=1)
                begin_fold = retracting & closed & np.isnan(self.fold_started)
                self.fold_started[begin_fold] = time
                fold_u = np.clip(np.nan_to_num((time-self.fold_started)/.10, nan=0), 0, 1)
                fold_u = (fold_u*fold_u*(3-2*fold_u))[:, None]
                folded = (self.start_pose[:, self.align_ids]
                          + (goal[:, self.align_ids]-self.start_pose[:, self.align_ids])*fold_u)
                self.current[np.ix_(retracting, self.align_ids)] = folded[retracting]
                for leg in ("LF", "RF"):
                    j = self.names.index(f"{leg}_J4_joint")
                    self.current[retracting, j] = 0
            planting = (before == MIDDLE) & self.overlap_enabled
            if planting.any():
                # Palms must arrive before the middle downstroke is over.
                # A single shared ramp otherwise consumes the whole overlap.
                plant_time = np.array([o.get("preplant_ramp", self.params[i, 21])
                                       for i, o in enumerate(self.options)])
                plant_u = np.clip(elapsed/plant_time, 0, 1)
                plant_u = (plant_u*plant_u*(3-2*plant_u))[:, None]
                planted_target = (self.start_pose[:, self.arm_ids]
                                  + (goal[:, self.arm_ids]-self.start_pose[:, self.arm_ids])*plant_u)
                self.current[np.ix_(planting, self.arm_ids)] = planted_target[planting]
            # Continue the middle stroke while the palms start their push,
            # then lift smoothly. Switching straight to the old PUSH pose would
            # retract the middle feet and silently remove the requested overlap.
            overlapping = (before == self.PUSH) & self.overlap_enabled
            if overlapping.any():
                middle_u = np.clip(np.nan_to_num((time-self.middle_started)/self.middle_ramp,
                                                 nan=0), 0, 1)
                middle_u = (middle_u*middle_u*(3-2*middle_u))[:, None]
                drive = self.poses[:, MIDDLE, self.middle_ids]
                continued = self.middle_start+(drive-self.middle_start)*middle_u
                lift_u = np.clip((elapsed-self.effective_hold)/.08, 0, 1)
                lift_u = (lift_u*lift_u*(3-2*lift_u))[:, None]
                lifted = drive+(self.poses[:, self.PUSH, self.middle_ids]-drive)*lift_u
                continued = np.where((elapsed > self.effective_hold)[:, None], lifted, continued)
                self.current[np.ix_(overlapping, self.middle_ids)] = continued[overlapping]
            next_stage = self.stage.copy()
            contact = np.asarray(contact)
            aligning = before == ALIGN
            align_error = np.abs(np.asarray(self.measured_q)[:, self.align_ids]
                                 - self.poses[:, ALIGN, self.align_ids]).max(axis=1)
            aligned = aligning & (elapsed >= self.align_duration) & (align_error < .15)
            next_stage[aligned] = self.REAR
            self.aligned_at[aligned] = time
            next_stage[aligning & (elapsed >= .8) & ~aligned] = self.RECOVER
            rear = before == self.REAR
            self.rear_contacted |= rear & contact[:, 4:].all(axis=1)
            clear = rear & ~contact[:, 4:].any(axis=1)
            self.rear_clear_count = np.where(clear, self.rear_clear_count+1, 0)
            handoff = (rear & self.rear_contacted & (self.rear_clear_count >= 2)
                       & (elapsed >= .5*self.params[:, 19]))
            next_stage[handoff] = MIDDLE
            self.rear_clear_at[handoff] = time
            self.middle_started[handoff] = time
            self.middle_start[handoff] = self.current[handoff][:, self.middle_ids]
            next_stage[rear & (elapsed >= .6) & ~handoff] = self.RECOVER
            middle = before == MIDDLE
            self.middle_seen |= middle & contact[:, 2:4].all(axis=1)
            # The middle feet may leave after delivering their impulse; the
            # following REACH gate still requires planted palms before launch.
            reached = middle & self.middle_seen & (pitch >= self.launch_angle)
            next_stage[reached] = self.REACH
            next_stage[middle & (elapsed >= .45) & ~reached] = self.RECOVER
            reach = before == self.REACH
            planted = contact[:, :2].all(axis=1)
            ready = reach & planted & (pitch >= self.launch_angle)
            ahead = np.asarray(com_ahead) | np.array([
                v.get("push_trigger", "com") == "contact" for v in variants])
            ready &= ahead
            early = (middle & self.overlap_enabled & contact[:, 2:4].all(axis=1)
                     & planted & ahead & (pitch >= self.overlap_angle)
                     & (elapsed >= self.middle_ramp*self.overlap_fraction))
            ready |= early
            next_stage[ready] = self.PUSH
            self.claw_pushed |= ready
            self.relay_pitch[ready] = pitch[ready]
            self.middle_grounded_at_launch[ready] = contact[ready, 2:4].all(axis=1)
            self.effective_hold[ready] = np.maximum(
                self.middle_hold[ready], self.middle_ramp[ready]
                - np.nan_to_num(time-self.middle_started[ready], nan=0))
            next_stage[reach & (elapsed >= .35) & ~ready] = self.RECOVER
            measured = np.asarray(self.measured_q)[:, self.arm_ids]
            self.arm_start[ready] = measured[ready]
            delta = self.poses[:, self.PUSH, self.arm_ids]-self.arm_start
            progress = ((measured-self.arm_start)*delta).sum(axis=1) / np.maximum(
                (delta*delta).sum(axis=1), 1e-8)
            pushing = before == self.PUSH
            open_now = pushing & np.isnan(self.jaw_started) & (progress >= self.jaw_fraction)
            self.jaw_started[open_now] = time
            self.jaw_progress[open_now] = progress[open_now]
            self.jaw_grounded[open_now] = planted[open_now]
            jaw_u = np.clip(np.nan_to_num((time-self.jaw_started)/self.open_seconds,
                                         nan=0), 0, 1)
            jaw_u = jaw_u*jaw_u*(3-2*jaw_u)
            for leg, sign in (("LF", 1), ("RF", -1)):
                active = pushing | ((before == self.TUCK) & ~self.safe_retract)
                self.current[active, self.names.index(f"{leg}_J4_joint")] = (
                    sign*self.opening*jaw_u)[active]
            grip_ids = [self.names.index(f"{leg}_J4_joint") for leg in ("LF", "RF")]
            aperture = np.abs(np.asarray(self.measured_q)[:, grip_ids])
            jaw_ready = (aperture >= .7*np.abs(self.opening[:, None])).all(axis=1)
            finished = (elapsed >= self.params[:, 22]) & jaw_ready
            # Keep the arms extended while the delayed jaws actually move;
            # a command ramp alone must not immediately trigger flight tuck.
            next_stage[pushing & (finished | (elapsed >= .30))] = self.TUCK
            next_stage[(before == self.TUCK) & (elapsed >= .08+self.params[:, 23])] = self.RECOVER
            next_stage[(before == self.RECOVER) & (elapsed >= .4)] = self.HOME
            changed = next_stage != self.stage
            self.start_pose[changed] = self.current[changed]
            self.started[changed] = time
            self.stage = next_stage
            return self.current.copy()

    return RelayController
