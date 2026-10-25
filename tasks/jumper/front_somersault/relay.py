"""Experimental rear -> middle -> palm launch, with measured delayed jaw opening.

Used only by tools/probe_underbody_claw.py --relay. No shared configuration or
training controller changes. Stage IDs retain the original rollout schema;
the additional middle drive is 7.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import HOME

MIDDLE = 7


def controller_type(base, variants):
    """Extend the diagnostic controller without changing the existing recipe."""
    class RelayController(base):
        def __init__(self, params, names):
            super().__init__(params, names)
            n = len(params)
            # Preplant the palms during middle drive; the large PUSH extension
            # remains blocked until both palms contact at the requested pitch.
            self.poses = np.concatenate((self.poses, self.poses[:, self.REACH:self.REACH+1]), axis=1)
            self.options = [v["relay"] for v in variants]
            self.launch_angle = np.radians([o["launch_angle_deg"] for o in self.options])
            self.middle_ramp = np.array([o["middle_ramp"] for o in self.options])
            self.jaw_fraction = np.array([o["jaw_after_fraction"] for o in self.options])
            self.rear_clear_count = np.zeros(n, dtype=int)
            self.middle_seen = np.zeros(n, dtype=bool)
            self.jaw_started = np.full(n, np.nan)
            self.jaw_progress = np.full(n, np.nan)
            self.jaw_grounded = np.zeros(n, dtype=bool)
            self.relay_pitch = np.full(n, np.nan)
            self.rear_clear_at = np.full(n, np.nan)
            self.arm_ids = [names.index(f"{leg}_J{j}_joint")
                            for leg in ("LF", "RF") for j in (1, 2, 3)]
            self.arm_start = np.zeros((n, 6))
            for i, option in enumerate(self.options):
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

        def target(self, time, quat, contact, com_ahead):
            before = self.stage.copy()
            loading = before == self.LOAD
            elapsed = np.maximum(time-self.started, 0)
            # Reuse the original HOME lead-in and support-before-tuck ramp.
            if loading.any():
                held = self.stage.copy()
                self.stage[~loading] = self.HOME
                super().target(time, quat, contact, com_ahead)
                self.stage[~loading] = held[~loading]
            self.command_stage = before.copy()
            w, x, y, z = np.asarray(quat).T
            pitch = np.arctan2(2*(w*y-x*z), 1-2*(y*y+z*z))
            duration = np.choose(before, (self.params[:, 18], self.params[:, 19],
                                         self.params[:, 21], self.params[:, 22],
                                         np.full(len(before), .08), np.full(len(before), .4),
                                         np.ones(len(before)), self.middle_ramp))
            u = np.clip(elapsed/duration, 0, 1)
            u = (u*u*(3-2*u))[:, None]
            goal = self.poses[np.arange(len(before)), before]
            target = self.start_pose + (goal-self.start_pose)*u
            self.current[~loading] = target[~loading]
            next_stage = self.stage.copy()
            contact = np.asarray(contact)
            rear = before == self.REAR
            self.rear_contacted |= rear & contact[:, 4:].all(axis=1)
            clear = rear & ~contact[:, 4:].any(axis=1)
            self.rear_clear_count = np.where(clear, self.rear_clear_count+1, 0)
            handoff = (rear & self.rear_contacted & (self.rear_clear_count >= 2)
                       & (elapsed >= .5*self.params[:, 19]))
            next_stage[handoff] = MIDDLE
            self.rear_clear_at[handoff] = time
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
            ready &= np.asarray(com_ahead) | np.array([
                v.get("push_trigger", "com") == "contact" for v in variants])
            next_stage[ready] = self.PUSH
            self.claw_pushed |= ready
            self.relay_pitch[ready] = pitch[ready]
            next_stage[reach & (elapsed >= .35) & ~ready] = self.RECOVER
            if not hasattr(self, "measured_q"):
                raise ValueError("relay jaw gate requires measured joint positions")
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
                active = pushing | (before == self.TUCK)
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
