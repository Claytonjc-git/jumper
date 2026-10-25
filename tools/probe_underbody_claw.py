#!/usr/bin/env python3
"""Compare the original underbody roll with straighter arms and loaded jaw opening.

CPU physics only; never promotes an experiment to a training reference.
python tools/probe_underbody_claw.py --output outputs/underbody_claw
python tools/probe_underbody_claw.py --yaw -0.8 --opening -1 --video outputs/underbody.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mjlab.sensor import ContactMatch, ContactSensorCfg
from tasks.jumper.common.constants import HOME
from tasks.jumper.front_somersault import physics, staged

ROOT = Path(__file__).resolve().parents[1]
# Absolute LF angles [rad]. The original-alignment grid changes only yaw/J4
# from ref/diagnostic.json; optional FK profiles also change J1-J3 below.
YAWS = (HOME["LF_J0_joint"], -.8, -1.05, -1.3)
OPENINGS = (0., -.6, -1.2)
OPEN_SECONDS = .03
FINGER_GEOMS = tuple(f"{leg}_{part}_meshcol" for leg in ("LF", "RF")
                     for part in ("finger_link", "finger_tip_link", "finger_grip_insert_link"))
# FK solutions keep the original tucked palm center beneath the body, rather
# than lifting the arms overhead. Columns are LF J0,J1,J2,J3 [rad].
ALIGNMENTS = {
    "original": None,
    "underbody_a": (-.8, .193799, -1.177099, -1.918445),
}
# Matched palm positions at the reach/push endpoints. Compensating elbow and
# wrist angles prevents yaw alone from changing the ground stroke's footprint.
STROKES = {
    "underbody_a": ((.146142366, -.412971688, -1.542585996),
                    (-.908669467, -1.412483713, .741971086)),
}


def controller_type(variants):
    """Create one isolated controller type for this experiment's batch."""
    class UnderbodyController(staged.RearClawController):
        def __init__(self, params, names):
            super().__init__(params, names)
            self.finger_samples = []
            self.opening = np.array([v["opening"] for v in variants])
            self.open_seconds = np.array([v["open_seconds"] for v in variants])
            for i, variant in enumerate(variants):
                tuck = ALIGNMENTS[variant.get("alignment", "original")]
                if tuck is not None:
                    for leg, sign in (("LF", 1), ("RF", -1)):
                        for j, value in enumerate(tuck[1:], start=1):
                            name, left = f"{leg}_J{j}_joint", f"LF_J{j}_joint"
                            mirrored = (value if leg == "LF" else HOME[name]
                                        + (1 if j == 1 else sign)*(value-HOME[left]))
                            for stage in (self.LOAD, self.REAR, self.TUCK):
                                self.poses[i, stage, names.index(name)] = mirrored
                        for stage, values in zip((self.REACH, self.PUSH),
                                                 STROKES[variant["alignment"]], strict=True):
                            for j, value in enumerate(values, start=1):
                                name, left = f"{leg}_J{j}_joint", f"LF_J{j}_joint"
                                self.poses[i, stage, names.index(name)] = (
                                    value if leg == "LF" else HOME[name]
                                    + (1 if j == 1 else sign)*(value-HOME[left]))
                for leg, sign in (("LF", 1), ("RF", -1)):
                    yaw_name = f"{leg}_J0_joint"
                    yaw = HOME[yaw_name] + sign * (variant["yaw"] - HOME["LF_J0_joint"])
                    for stage in (self.LOAD, self.REAR, self.REACH, self.PUSH, self.TUCK):
                        self.poses[i, stage, names.index(yaw_name)] = yaw
                    for stage in (self.PUSH, self.TUCK):
                        self.poses[i, stage, names.index(f"{leg}_J4_joint")] = (
                            sign * variant["opening"])

        def target(self, time, quat, contact, com_ahead):
            before = self.stage.copy()
            elapsed = np.maximum(time - self.started, 0)
            ahead = np.asarray(com_ahead).copy()
            for i, variant in enumerate(variants):
                if variant.get("push_trigger", "com") == "contact":
                    ahead[i] = True
            super().target(time, quat, contact, ahead)
            # PUSH requires both palms planted. The default also requires COM
            # beyond the hands; the explicit contact experiment tests earlier drive.
            # Open the finger at the same instant arm extension begins, not in tuck.
            pushing = before == self.PUSH
            u = np.clip(elapsed / self.open_seconds, 0, 1)
            u = u*u*(3-2*u)
            for leg, sign in (("LF", 1), ("RF", -1)):
                j = self.names.index(f"{leg}_J4_joint")
                self.current[pushing, j] = (sign * self.opening * u)[pushing]
            changed = before != self.stage
            self.start_pose[changed] = self.current[changed]
            if hasattr(self, "env"):
                self.finger_samples.append(finger_force(self.env))
            return self.current.copy()

    return UnderbodyController


def finger_force(env):
    # Match the recorded palm convention: terrain reaction acting on the robot.
    return -env.scene.sensors["finger_ground_contact"].data.force.reshape(
        env.num_envs, 2, 3, 3).sum(dim=2).cpu().numpy().copy()


def self_test():
    """Pin the silent trap: opening a jaw before the palms actually bear weight."""
    params = json.loads((ROOT / "tasks/jumper/front_somersault/ref/diagnostic.json").read_text())
    names = list(HOME)[::-1]
    kind = controller_type([dict(yaw=-.8, opening=-.6, open_seconds=.03,
                                 alignment="underbody_a")])
    controller = kind(np.array(params["params"])[None], names)
    controller.stage[:] = controller.REACH
    controller.started[:] = 0
    quat = np.array([[1., 0, 0, 0]])
    contact = np.zeros((1, 6), dtype=bool)
    for time, planted, ahead in ((.10, False, True), (.11, True, False)):
        contact[:, :2] = planted
        command = controller.target(time, quat, contact, np.array([ahead]))
        assert controller.stage[0] == controller.REACH
        assert command[0, names.index("LF_J4_joint")] == 0
    controller.target(.12, quat, contact, np.array([True]))
    assert controller.stage[0] == controller.PUSH
    command = controller.target(.14, quat, contact, np.array([True]))
    assert -.6 < command[0, names.index("LF_J4_joint")] < -.3
    np.testing.assert_allclose(command[0, names.index("RF_J4_joint")],
                               -command[0, names.index("LF_J4_joint")])
    closed = controller_type([dict(yaw=-.8, opening=0., open_seconds=.03)])(
        np.array(params["params"])[None], names)
    closed.stage[:] = closed.PUSH
    closed.started[:] = 0
    assert closed.target(.03, quat, contact, np.array([True]))[
        0, names.index("LF_J4_joint")] == 0
    print("Self-check passed: unloaded/COM-ahead controls, named mirroring, closed-jaw control.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/underbody_claw")
    parser.add_argument("--yaw", type=float, help="one LF shoulder yaw, instead of comparison grid")
    parser.add_argument("--opening", type=float, help="one LF finger opening in radians")
    parser.add_argument("--open-seconds", type=float, default=OPEN_SECONDS)
    parser.add_argument("--alignment", choices=ALIGNMENTS, default="original")
    parser.add_argument("--push-trigger", choices=("com", "contact"), default="com")
    parser.add_argument("--video", type=Path, help="record first variant; use --yaw for a chosen variant")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    opening = -1.2 if args.opening is None else args.opening
    single = args.yaw is not None or args.opening is not None
    if (not np.isfinite([opening, args.open_seconds]).all()
            or not -1.52 <= opening <= 0 or not .01 <= args.open_seconds <= .2):
        parser.error("opening must be [-1.52,0] rad and open-seconds [.01,.2] s")
    if args.yaw is not None and (not np.isfinite(args.yaw) or not -2.67 <= args.yaw <= .62):
        parser.error("yaw must be finite within [-2.67,.62] rad")
    if args.alignment != "original":
        aligned_yaw = ALIGNMENTS[args.alignment][0]
        if args.yaw is not None and args.yaw != aligned_yaw:
            parser.error("alignment requires its FK-solved yaw")
        yaws = (aligned_yaw,)
    else:
        yaws = ((HOME["LF_J0_joint"] if args.yaw is None else args.yaw,)
                if single else YAWS)
    variants = [dict(yaw=y, opening=o, open_seconds=args.open_seconds, alignment=args.alignment,
                     push_trigger=args.push_trigger)
                for y in yaws
                for o in ((opening,) if single else OPENINGS)]
    original_controller, original_scene = staged.RearClawController, physics.scene_cfg
    controller = controller_type(variants)

    def scene(num_envs):
        cfg = original_scene(num_envs)
        cfg.scene.sensors += (ContactSensorCfg(
            name="finger_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FINGER_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1, history_length=2,
        ),)
        return cfg

    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((ROOT / "tasks/jumper/front_somersault/ref/diagnostic.json").read_text())
    params = np.tile(baseline["params"], (len(variants), 1))
    runner = None
    instances = []

    class Instrumented(controller):
        def __init__(self, p, names):
            super().__init__(p, names)
            self.env = runner.env
            instances.append(self)

    try:
        staged.RearClawController, physics.scene_cfg = Instrumented, scene
        runner = physics.PhysicsRunner(len(variants))
        results = runner.rollout(params, video=args.video, schedule="rear-claw")
        finger_samples = np.stack(instances[-1].finger_samples + [finger_force(runner.env)], axis=1)
        summaries = []
        for i, (variant, data) in enumerate(zip(variants, results, strict=True)):
            report = physics.qualify(data)
            push = data["stage"] == controller.PUSH
            finger = finger_samples[i]
            report.update(
                finger_forward_impulse_Ns=float(finger[push, :, 0].sum()*physics.CONTROL_DT),
                finger_upward_impulse_Ns=float(finger[push, :, 2].sum()*physics.CONTROL_DT),
                palm_upward_impulse_Ns=float(data["foot_force"][push, :2, 2].sum()
                                             * physics.CONTROL_DT),
                finger_peak_force_N=float(np.linalg.norm(finger, axis=-1).max()),
                peak_self_force_N=float(data["self_force"].max()),
            )
            if push.any():
                finger_joints = [runner.names.index(f"{leg}_J4_joint") for leg in ("LF", "RF")]
                report["finger_actual_excursion_rad"] = np.ptp(
                    data["q"][push][:, finger_joints], axis=0).tolist()
            metadata = dict(variant=variant, report=report, baseline_params=baseline["params"],
                            experiment_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            training_reference=False,
                            contact_model="unchanged palm support; fingers retain impact classification")
            stem = f"variant_{i:02d}"
            (args.output / f"{stem}.json").write_text(json.dumps(metadata, indent=2)+"\n")
            arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(args.output / f"{stem}.npz", **arrays, finger_force=finger,
                                meta=np.array(json.dumps(metadata)))
            summaries.append(metadata)
            print(json.dumps(dict(variant=variant, **report)), flush=True)
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    finally:
        staged.RearClawController, physics.scene_cfg = original_controller, original_scene
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
