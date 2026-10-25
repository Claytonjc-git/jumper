#!/usr/bin/env python3
"""Test the original fast launch with rearward middle-leg drive; CPU diagnostics only.

python tools/probe_momentum_launch.py --compare --output outputs/momentum_compare
python tools/probe_momentum_launch.py --resume <candidate.json> --video outputs/momentum.mp4
python tools/probe_momentum_launch.py --resume <candidate.json> --tuck inward --land-at .18
python tools/probe_momentum_launch.py --resume <candidate.json> --compare-flight --output <dir>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_underbody_claw import controller_type as original_type
from tasks.jumper.common.constants import HOME
from tasks.jumper.frontflip import physics, staged
from tasks.jumper.frontflip.momentum import MomentumMeter, controller_type

ROOT = Path(__file__).resolve().parents[1]
# LM absolute J0/J1/J2 targets [rad]. RM uses HOME-relative named mirroring.
# Positive J0 sweeps middle feet backward. FK only suggested these endpoints;
# every sample is still integrated with the original limited servos.
DRIVES = {"modest": (.2, -1.25, -.35), "medium": (.45, -1.30, -.35),
          "rear_like": (.65, -1.35, -.35)}
# Absolute left-leg flight-tuck targets from J0 [rad], right legs mirrored. "inward" turns
# 269 degrees in flight but its rear calf touches the middle calf (2.3 N). "tuned": 8 mm
# clearance search, then simulated cross-entropy on flight turn: 289.7 degrees, no self-contact.
TUCKS = {"original": {}, "inward": {"LR": (-.197, .925, -.671)},
         "tuned": {"LF": (-.4645, .5933, -1.2077, -1.9667), "LM": (-.1994, .6752, -1.2797),
                   "LR": (-.1831, .3906, -.8993)}}
LAND_LIMBS = ("claw", "middle", "rear")
# Landing groups, absolute left-leg landing pose from J0 [rad] (right mirrored) and hold
# before HOME. "tuned": cross-entropy on landing impact with the "tuned" tuck.
LANDINGS = {"tuned": dict(
    land=[dict(limbs=["claw"], at_s=.2023, ramp_s=.04),
          dict(limbs=["middle", "rear"], at_s=.2947, ramp_s=.04)],
    land_pose={"LF": (-.5507, -1.166, -.2417, -1.3452), "LM": (-.0927, .3828, -2.1408),
               "LR": (.5701, .2846, -1.7418)},
    hold_s=.1755)}


def variant(preset, delay=0., load_back=False, trigger="delay", opening=0.):
    yaw = DRIVES.get(preset, DRIVES["modest"])[0]
    return dict(yaw=HOME["LF_J0_joint"], opening=opening, open_seconds=.03, alignment="original",
                push_trigger="com", momentum=dict(
                    middle_enabled=preset != "original", preset=preset,
                    load=(yaw if load_back else -.4, HOME["LM_J1_joint"], HOME["LM_J2_joint"]),
                    drive=DRIVES.get(preset, DRIVES["modest"]),
                    middle_start_s=0., middle_ramp_s=.07, middle_hold_s=.025,
                    claw_delay_s=delay, trigger=trigger, max_wait_s=.03))


def landing(data, momentum):
    """Takeoff spin and, per limb pair, first touchdown and peak terrain force after it."""
    air = (data["contact"].sum(axis=1) == 0) & (data["body_force"] < 1.)
    run, longest, start, best = 0, 0, 0, 0
    for k, value in enumerate(air):
        start = k if value and run == 0 else start
        run = run+1 if value else 0
        if run > longest:
            longest, best = run, start
    down = best+longest
    after = slice(down, down+round(.25/physics.CONTROL_DT))
    out = dict(takeoff_s=float(data["t"][best]), takeoff_Hy_Nms=float(momentum[best, 4]),
               touchdown_s=float(data["t"][min(down, len(air)-1)]),
               landing_body_force_N=float(data["body_force"][after].max()))
    for name, ids in (("claw", [0, 1]), ("middle", [2, 3]), ("rear", [4, 5])):
        hit = np.flatnonzero(data["contact"][down:, ids].any(axis=1))
        out[f"{name}_touchdown_s"] = float(data["t"][down+hit[0]]) if len(hit) else None
        out[f"{name}_landing_force_N"] = float(
            np.linalg.norm(data["foot_force"][after][:, ids], axis=-1).max())
    return out


def self_test():
    import mujoco

    from tasks.jumper.common.assets import JUMPER_XML

    model = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
    meter = MomentumMeter(model, model.body("base_link").id)
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, .10647, 1, 0, 0, 0]
    for name, value in HOME.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    data.qvel[:3] = [1.25, -.5, .25]
    measured = meter.read(data.qpos, data.qvel)[0]
    np.testing.assert_allclose(measured[:3], meter.mass*data.qvel[:3], atol=1e-10)
    np.testing.assert_allclose(measured[3:], 0, atol=1e-10)
    data.qvel[:] = 0
    data.qvel[4] = 2
    assert meter.read(data.qpos, data.qvel)[0, 4] > 0, "positive pitch spin lost its sign"
    params = json.loads((ROOT/"tasks/jumper/frontflip/ref/diagnostic.json").read_text())["params"]
    v = variant("modest", delay=.01, trigger="peak")
    names = list(HOME)[::-1]
    kind = controller_type(original_type([v]), [v])
    controller = kind(np.array(params)[None], names)
    controller.stage[:] = controller.REACH
    controller.started[:] = 0
    controller.rear_started[:] = -.05
    controller.middle_start[:] = [[HOME[names[j]] for j in controller.middle_ids]]
    controller.measured_momentum = np.array([[.63, 0, 1.2, 0, .063, 0]])
    contact = np.ones((1, 6), dtype=bool)
    quat = np.array([[1., 0, 0, 0]])
    controller.target(.05, quat, contact, np.array([False]))
    assert controller.stage[0] == controller.REACH, "momentum overrode the COM gate"
    controller.target(.055, quat, contact, np.array([True]))
    assert controller.stage[0] == controller.REACH, "first planted sample launched before delay"
    controller.measured_momentum *= .8
    controller.target(.06, quat, contact, np.array([True]))
    assert controller.stage[0] == controller.REACH
    controller.target(.065, quat, contact, np.array([True]))
    assert controller.stage[0] == controller.PUSH, "falling measured momentum did not release"
    assert controller.release_reason[0] == "peak_decline"
    # A lost palm must restart the continuous-contact delay; rising momentum
    # must still release at the bounded timeout rather than deadlock.
    controller = kind(np.array(params)[None], names)
    controller.stage[:] = controller.REACH
    controller.started[:] = 0
    controller.rear_started[:] = 0
    controller.measured_momentum = np.array([[.63, 0, 1.2, 0, .063, 0]])
    controller.target(.05, quat, contact, np.array([True]))
    contact[0, 0] = False
    controller.target(.055, quat, contact, np.array([True]))
    assert np.isnan(controller.planted_since[0]), "lost palm retained its contact timer"
    contact[0, 0] = True
    controller.target(.06, quat, contact, np.array([True]))
    controller.target(.07, quat, contact, np.array([True]))
    assert controller.stage[0] == controller.REACH, "rising momentum bypassed peak gate"
    controller.target(.095, quat, contact, np.array([True]))
    assert controller.release_reason[0] == "max_wait"
    # An unfinished middle stroke must lift from its actual command, not jump
    # to the full drive endpoint when the overlap hold expires.
    v["momentum"].update(middle_start_s=.1, middle_ramp_s=.2)
    controller = kind(np.array(params)[None], names)
    controller.stage[:] = controller.PUSH
    controller.started[:] = 0
    controller.rear_started[:] = 0
    controller.middle_start[:] = controller.current[:, controller.middle_ids]
    controller.measured_momentum = np.zeros((1, 6))
    q0 = controller.target(.02, quat, contact, np.array([True]))
    q1 = controller.target(.03, quat, contact, np.array([True]))
    np.testing.assert_allclose(q0[:, controller.middle_ids], q1[:, controller.middle_ids])
    # Landing must start from the live tuck command, not jump, and finish at HOME.
    v["momentum"]["flight"] = dict(tuck=TUCKS["inward"],
                                   land=[dict(limbs=list(LAND_LIMBS), at_s=.05, ramp_s=.04)])
    controller = kind(np.array(params)[None], names)
    rr = names.index("RR_J1_joint")
    assert np.isclose(controller.poses[0, controller.TUCK, rr],
                      HOME["RR_J1_joint"]-(.925-HOME["LR_J1_joint"])), "rear tuck lost its mirror"
    controller.stage[:] = controller.TUCK
    controller.started[:] = controller.tuck_time[:] = 0
    controller.start_pose[:] = controller.poses[:, controller.TUCK]
    controller.measured_momentum = np.zeros((1, 6))
    q0 = controller.target(.045, quat, contact, np.array([True]))
    q1 = controller.target(.05, quat, contact, np.array([True]))
    np.testing.assert_allclose(q0, q1, err_msg="landing extension jumped at its start")
    q2 = controller.target(.10, quat, contact, np.array([True]))
    np.testing.assert_allclose(q2[0], controller.home, err_msg="landing did not reach HOME")
    # A landing pose must be held, then released to HOME.
    v["momentum"]["flight"].update(land_pose={"LM": (0., 0., -1.)}, hold_s=.02, release_s=.04)
    controller = kind(np.array(params)[None], names)
    controller.stage[:] = controller.TUCK
    controller.started[:] = controller.tuck_time[:] = 0
    controller.start_pose[:] = controller.poses[:, controller.TUCK]
    controller.measured_momentum = np.zeros((1, 6))
    lm = names.index("LM_J2_joint")
    held = [controller.target(t, quat, contact, np.array([True]))[0, lm]
            for t in (.05, .10, .105, .17)]
    np.testing.assert_allclose(held[1:3], -1., err_msg="landing pose was not held")
    assert np.isclose(held[3], HOME["LM_J2_joint"]), "landing pose was not released to HOME"
    print("Checks passed: centroidal momentum, contact/COM/peak gates, timeout, continuity "
          "and landing extension.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("original", *DRIVES))
    parser.add_argument("--delay", type=float)
    parser.add_argument("--trigger", choices=("delay", "peak"))
    parser.add_argument("--load-back", action="store_true")
    parser.add_argument("--opening", type=float)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/momentum_launch")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tuck", choices=TUCKS)
    parser.add_argument("--claw-flight", choices=("tuck", "extended"))
    parser.add_argument("--land-at", type=float, help="s after TUCK starts to open limbs to HOME")
    parser.add_argument("--land-ramp", type=float, default=.06)
    parser.add_argument("--land-limbs", nargs="+", choices=LAND_LIMBS, default=list(LAND_LIMBS))
    parser.add_argument("--landing", choices=LANDINGS, help="named landing preset")
    parser.add_argument("--reference", type=Path,
                        help="save a single qualified run here as the training reference")
    parser.add_argument("--compare-flight", action="store_true",
                        help="grid of rear tuck, claw flight and landing time on --resume")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.land_at is not None and args.landing is not None:
        parser.error("choose --land-at or --landing")
    flight_flags = (args.tuck, args.claw_flight, args.land_at, args.landing)
    if args.compare_flight and (not args.resume or any(x is not None for x in flight_flags)):
        parser.error("--compare-flight needs --resume and supplies the flight settings itself")
    if args.reference and (args.compare or args.compare_flight):
        parser.error("--reference saves one run; drop --compare/--compare-flight")
    if args.video and (args.compare or args.compare_flight):
        parser.error("select a single candidate before recording video")
    if ((args.resume or args.compare) and (args.load_back or any(x is not None for x in
            (args.preset, args.delay, args.trigger, args.opening)))):
        parser.error("resume/compare supply the motion; do not combine with motion-selection flags")
    source = json.loads((ROOT/"tasks/jumper/frontflip/ref/diagnostic.json").read_text())
    params = source["params"]
    variants = [variant(args.preset or "modest", args.delay or 0., args.load_back,
                        args.trigger or "delay", args.opening or 0.)]
    if args.compare:
        variants = [variant("original")]+[
            variant(preset, delay, load_back) for preset in DRIVES
            for load_back in (False, True) for delay in (0., .01, .03)]
    if args.resume:
        if args.compare:
            parser.error("resume selects a single candidate")
        source = json.loads(args.resume.read_text())
        variants, params = [source["variant"]], source["params"]
    if any(x is not None for x in flight_flags):
        for v in variants:
            f = v["momentum"].setdefault("flight", {})
            if args.tuck is not None:
                f["tuck"] = TUCKS[args.tuck]
            if args.claw_flight is not None:
                f["claw"] = args.claw_flight
            if args.land_at is not None:
                f["land"] = [dict(limbs=args.land_limbs, at_s=args.land_at, ramp_s=args.land_ramp)]
            if args.landing is not None:
                f.update(copy.deepcopy(LANDINGS[args.landing]))
    if args.compare_flight:
        base, variants = variants[0], []
        for tuck in TUCKS:
            for claw in ("tuck", "extended"):
                for land in (None, .14, .18, .22):
                    v = copy.deepcopy(base)
                    v["momentum"]["flight"] = dict(tuck=TUCKS[tuck], claw=claw)
                    if land is not None:
                        v["momentum"]["flight"]["land"] = [dict(
                            limbs=list(LAND_LIMBS), at_s=land, ramp_s=args.land_ramp)]
                    variants.append(v)
    for v in variants:
        o = v["momentum"]
        f = o.get("flight", {})
        poses = (f.get("tuck", {}), f.get("land_pose", {}))
        if (any(leg not in ("LF", "LM", "LR") or not 1 <= len(values) <= (4 if leg == "LF" else 3)
                or not np.isfinite(values).all() for p in poses for leg, values in p.items())
                or f.get("claw", "tuck") not in ("tuck", "extended")
                or not 0 <= f.get("hold_s", 0) <= 1 or not .02 <= f.get("release_s", .3) <= 1
                or any(not 0 <= e["at_s"] <= .4 or not .02 <= e["ramp_s"] <= .2
                       or not e["limbs"] or not set(e["limbs"]) <= set(LAND_LIMBS)
                       for e in f.get("land", ()))):
            parser.error("invalid bounded flight settings")
        numbers = [*o["load"], *o["drive"], o["middle_start_s"], o["middle_ramp_s"],
                   o["middle_hold_s"], o["claw_delay_s"], o.get("max_wait_s", .03),
                   v["opening"], v["yaw"], v["open_seconds"]]
        if (len(o["load"]) != 3 or len(o["drive"]) != 3 or not np.isfinite(numbers).all()
                or not 0 <= o["claw_delay_s"] <= .1 or not .02 <= o["middle_ramp_s"] <= .2
                or not 0 <= o["middle_start_s"] <= .1 or not 0 <= o["middle_hold_s"] <= .15
                or not 0 <= o.get("max_wait_s", .03) <= .1 or not -1.52 <= v["opening"] <= 0
                or not -2.67 <= v["yaw"] <= .62 or not .01 <= v["open_seconds"] <= .2
                or v.get("alignment") != "original" or v.get("push_trigger") != "com"
                or o.get("trigger", "delay") not in ("delay", "peak")):
            parser.error("invalid bounded momentum variant")
    args.output.mkdir(parents=True, exist_ok=True)
    original = staged.RearClawController
    kind = controller_type(original_type(variants), variants)
    runner = None
    samples, instances = [], []

    class Instrumented(kind):
        def __init__(self, *values):
            super().__init__(*values)
            instances.append(self)

        def target(self, *values):
            self.measured_momentum = measure()
            samples.append(self.measured_momentum)
            return super().target(*values)

    def measure():
        return meter.read(runner.env.sim.data.qpos.cpu().numpy(),
                          runner.env.sim.data.qvel.cpu().numpy())

    try:
        staged.RearClawController = Instrumented
        runner = physics.PhysicsRunner(len(variants))
        meter = MomentumMeter(runner.env.sim.mj_model, runner.robot.indexing.root_body_id)
        results = runner.rollout(np.tile(params, (len(variants), 1)), video=args.video,
                                 schedule="rear-claw")
        momentum = np.stack(samples+[measure()], axis=1)
        summaries = []
        for i, (v, data) in enumerate(zip(variants, results, strict=True)):
            report = physics.qualify(data)
            first = np.flatnonzero(data["stage"] == kind.PUSH)
            report["peak_self_force_N"] = float(data["self_force"].max())
            report["joint_limit_overshoot_rad"] = max(0., data["joint_limit_violation"])
            report.update(landing(data, momentum[i]))
            if len(first):
                k = first[0]
                report.update(first_push_frame_s=float(data["t"][k]),
                              momentum_at_first_push=momentum[i, k].tolist(),
                              command_time_s=float(instances[-1].launch_time[i]),
                              observation_time_s=float(instances[-1].launch_time[i]
                                                       -physics.CONTROL_DT),
                              decision_momentum=instances[-1].launch_momentum[i].tolist(),
                              release_reason=instances[-1].release_reason[i])
            paths = [Path(__file__), ROOT/"tasks/jumper/frontflip/momentum.py",
                     ROOT/"tools/probe_underbody_claw.py"]
            digest = hashlib.sha256()
            for path in paths:
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
            meta = dict(variant=v, params=params, report=report, mass_kg=meter.mass,
                        joint_order=list(runner.names), control_dt_s=physics.CONTROL_DT,
                        physics_dt_s=physics.PHYSICS_DT,
                        momentum_order=["Px", "Py", "Pz", "Hx", "Hy", "Hz"],
                        momentum_frame="world; angular momentum about whole-robot COM",
                        source_sha256=digest.hexdigest(), physics_fingerprint=physics.fingerprint(),
                        training_reference=bool(args.reference) and report["qualified"])
            stem = f"variant_{i:02d}"
            (args.output/f"{stem}.json").write_text(json.dumps(meta, indent=2)+"\n")
            np.savez_compressed(args.output/f"{stem}.npz", **{
                **{k: a for k, a in data.items() if isinstance(a, np.ndarray)},
                "momentum": momentum[i]})
            if args.reference:
                if not report["qualified"]:
                    raise SystemExit(f"not qualified; reference not saved: {report['checks']}")
                physics.save_reference(args.reference, data, params, variant=v)
                print(f"QUALIFIED reference saved: {args.reference}", flush=True)
            summaries.append(meta)
            print(json.dumps(dict(variant=v, **report)), flush=True)
        (args.output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    finally:
        staged.RearClawController = original
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
