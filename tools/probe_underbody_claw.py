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
            return self.current.copy()

    return UnderbodyController


def finger_force(env):
    # Match the recorded palm convention: terrain reaction acting on the robot.
    return -env.scene.sensors["finger_ground_contact"].data.force.reshape(
        env.num_envs, 2, 3, 3).sum(dim=2).cpu().numpy().copy()


def self_test(include_relay=False):
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
    if include_relay:
        from tasks.jumper.front_somersault.relay import MIDDLE
        from tasks.jumper.front_somersault.relay import controller_type as relay_type

        variant = dict(yaw=HOME["LF_J0_joint"], opening=-.3, open_seconds=.03,
                       alignment="original", relay=dict(launch_angle_deg=65,
                       middle_hip=-.3, middle_knee=-.8, middle_yaw=-.4,
                       middle_ramp=.08, jaw_after_fraction=.55))
        relay = relay_type(controller_type([variant]), [variant])(
            np.array(params["params"])[None], names)
        relay.measured_q = np.array([[HOME[n] for n in names]])
        relay.stage[:] = relay.REAR
        relay.started[:] = 0
        relay.rear_contacted[:] = True
        contact[:] = False
        contact[:, 2:4] = True
        relay.target(.05, quat, contact, np.array([True]))
        assert relay.stage[0] == relay.REAR, "one missing-contact sample caused a handoff"
        relay.target(.055, quat, contact, np.array([True]))
        assert relay.stage[0] == MIDDLE
        contact[:] = False
        angle = np.radians(70)
        tilted = np.array([[np.cos(angle/2), 0, np.sin(angle/2), 0]])
        relay.target(.06, tilted, contact, np.array([True]))
        assert relay.stage[0] == MIDDLE, "middle stage completed without middle support"
        contact[:, 2:4] = True
        relay.target(.065, quat, contact, np.array([True]))
        assert relay.stage[0] == MIDDLE, "launch angle was ignored"
        relay.target(.07, tilted, contact, np.array([True]))
        assert relay.stage[0] == relay.REACH
        relay.target(.075, tilted, contact, np.array([True]))
        assert relay.stage[0] == relay.REACH, "unplanted palms launched"
        contact[:, :2] = True
        relay.target(.08, tilted, contact, np.array([True]))
        assert relay.stage[0] == relay.PUSH
        relay.target(.12, tilted, contact, np.array([True]))
        assert np.isnan(relay.jaw_started[0]), "command time opened an unextended arm"
        delta = relay.poses[0, relay.PUSH, relay.arm_ids]-relay.arm_start[0]
        relay.measured_q[0, relay.arm_ids] = relay.arm_start[0]+.7*delta
        relay.target(.125, tilted, contact, np.array([True]))
        np.testing.assert_allclose(relay.jaw_progress, [.7])
        assert relay.jaw_started[0] == .125
        command = relay.target(.14, tilted, contact, np.array([True]))
        np.testing.assert_allclose(command[0, names.index("LF_J4_joint")], -.15)
        relay.target(.25, tilted, contact, np.array([True]))
        assert relay.stage[0] == relay.PUSH, "tuck started before the delayed jaws actually moved"
        relay.measured_q[0, names.index("LF_J4_joint")] = -.3
        relay.measured_q[0, names.index("RF_J4_joint")] = .3
        relay.target(.26, tilted, contact, np.array([True]))
        assert relay.stage[0] == relay.TUCK
        print("Relay checks passed: rear-contact debounce, middle support, pitch/contact gates, "
              "measured-arm delay with a command-clock negative control.")
    print("Self-check passed: unloaded/COM-ahead controls, named mirroring, closed-jaw control.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/underbody_claw")
    parser.add_argument("--yaw", type=float, help="one LF shoulder yaw, instead of comparison grid")
    parser.add_argument("--opening", type=float, help="one LF finger opening in radians")
    parser.add_argument("--open-seconds", type=float, default=OPEN_SECONDS)
    parser.add_argument("--alignment", choices=ALIGNMENTS, default="original")
    parser.add_argument("--push-trigger", choices=("com", "contact"), default="com")
    parser.add_argument("--relay", action="store_true", help="rear -> middle -> claw launch")
    parser.add_argument("--launch-angle", type=float, help="nose-down launch threshold [deg]")
    parser.add_argument("--middle-knee", type=float, help="LM drive J2 [rad]; -.8 or -1.0")
    parser.add_argument("--middle-ramp", type=float, default=.08)
    parser.add_argument("--jaw-after", type=float, default=.55,
                        help="measured arm-extension fraction before opening")
    parser.add_argument("--resume", type=Path, help="replay a complete variant JSON from this tool")
    parser.add_argument("--video", type=Path, help="record first variant; use --yaw for a chosen variant")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.relay)
        return
    opening = -1.2 if args.opening is None else args.opening
    single = any(v is not None for v in (args.yaw, args.opening, args.launch_angle, args.middle_knee))
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
                if single or args.relay else YAWS)
    variants = [dict(yaw=y, opening=o, open_seconds=args.open_seconds, alignment=args.alignment,
                     push_trigger=args.push_trigger)
                for y in yaws
                for o in ((opening,) if single else ((0., -.3, -.6) if args.relay else OPENINGS))]
    if args.relay:
        if (args.launch_angle is not None and not 35 <= args.launch_angle <= 85
                or args.middle_knee is not None and args.middle_knee not in (-.8, -1.)
                or not np.isfinite([args.middle_ramp, args.jaw_after]).all()
                or not .03 <= args.middle_ramp <= .2 or not .1 <= args.jaw_after <= .95):
            parser.error("relay needs angle [35,85], knee -.8/-1, ramp [.03,.2], jaw-after [.1,.95]")
        variants = [dict(v, relay=dict(launch_angle_deg=angle, middle_hip=-.3,
                                      middle_knee=knee, middle_yaw=-.4 if knee == -.8 else -.6,
                                      middle_ramp=args.middle_ramp,
                                      jaw_after_fraction=args.jaw_after))
                    for v in variants
                    for angle in ((65 if args.launch_angle is None else args.launch_angle,)
                                  if single else (50, 65, 80))
                    for knee in ((-.8 if args.middle_knee is None else args.middle_knee,)
                                 if single else (-.8, -1.))]
    elif args.launch_angle is not None or args.middle_knee is not None:
        parser.error("launch-angle and middle-knee require --relay")
    resumed = None
    if args.resume:
        if single or args.relay or args.alignment != "original":
            parser.error("resume supplies the motion; do not combine it with motion-selection flags")
        resumed = json.loads(args.resume.read_text())
        variants = [resumed["variant"]]
        args.relay = "relay" in variants[0]
    for v in variants:
        if (v.get("alignment") not in ALIGNMENTS or v.get("push_trigger") not in ("com", "contact")
                or not np.isfinite([v["yaw"], v["opening"], v["open_seconds"]]).all()
                or not -2.67 <= v["yaw"] <= .62 or not -1.52 <= v["opening"] <= 0
                or not .01 <= v["open_seconds"] <= .2):
            parser.error("invalid diagnostic variant settings")
        if "relay" in v:
            o = v["relay"]
            bounds = {"launch_angle_deg": (35, 85), "middle_hip": (-.8, 1.),
                      "middle_knee": (-2.2, .2), "middle_yaw": (-.67, .92),
                      "middle_ramp": (.03, .2), "jaw_after_fraction": (.1, .95)}
            if any(not np.isfinite(o[k]) or not lo <= o[k] <= hi
                   for k, (lo, hi) in bounds.items()):
                parser.error("relay settings outside bounded diagnostic ranges")
    original_controller, original_scene = staged.RearClawController, physics.scene_cfg
    controller = controller_type(variants)
    if args.relay:
        from tasks.jumper.front_somersault.relay import controller_type as relay_type

        controller = relay_type(controller, variants)

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
    source_params = baseline["params"] if resumed is None else resumed["baseline_params"]
    params = np.tile(source_params, (len(variants), 1))
    runner = None
    instances = []

    class Instrumented(controller):
        def __init__(self, p, names):
            super().__init__(p, names)
            self.env = runner.env
            instances.append(self)

        def target(self, time, quat, contact, com_ahead):
            self.measured_q = runner.robot.data.joint_pos.cpu().numpy().copy()
            result = super().target(time, quat, contact, com_ahead)
            self.finger_samples.append(finger_force(self.env))
            return result

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
                finger_full_roll_upward_impulse_Ns=float(finger[:, :, 2].sum()*physics.CONTROL_DT),
                peak_self_force_N=float(data["self_force"].max()),
            )
            if args.relay:
                instance = instances[-1]
                middle = data["stage"] == 7
                report.update(
                    middle_drive_executed=bool(middle.any()),
                    middle_upward_impulse_Ns=float(data["foot_force"][middle, 2:4, 2].sum()
                                                  * physics.CONTROL_DT),
                    middle_forward_impulse_Ns=float(data["foot_force"][middle, 2:4, 0].sum()
                                                   * physics.CONTROL_DT),
                    rear_clear_handoff_s=(None if np.isnan(instance.rear_clear_at[i])
                                          else float(instance.rear_clear_at[i])),
                    claw_launch_pitch_deg=(None if np.isnan(instance.relay_pitch[i])
                                           else float(np.degrees(instance.relay_pitch[i]))),
                    jaw_open_arm_fraction=(None if np.isnan(instance.jaw_progress[i])
                                           else float(instance.jaw_progress[i])),
                    jaw_open_trigger_s=(None if np.isnan(instance.jaw_started[i])
                                        else float(instance.jaw_started[i])),
                    palms_planted_at_jaw_open=bool(instance.jaw_grounded[i]),
                )
            if push.any():
                finger_joints = [runner.names.index(f"{leg}_J4_joint") for leg in ("LF", "RF")]
                report["finger_actual_excursion_rad"] = np.ptp(
                    data["q"][push][:, finger_joints], axis=0).tolist()
            experiment_paths = [Path(__file__)]
            if args.relay:
                experiment_paths.append(ROOT / "tasks/jumper/front_somersault/relay.py")
            experiment_hash = hashlib.sha256()
            for path in experiment_paths:
                experiment_hash.update(path.read_bytes().replace(b"\r\n", b"\n"))
            metadata = dict(variant=variant, report=report, baseline_params=params[i].tolist(),
                            experiment_sha256=experiment_hash.hexdigest(),
                            physics_fingerprint=physics.fingerprint(),
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
