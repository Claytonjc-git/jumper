#!/usr/bin/env python3
"""Test the original fast launch with rearward middle-leg drive; CPU diagnostics only.

python tools/probe_momentum_launch.py --compare --output outputs/momentum_compare
python tools/probe_momentum_launch.py --resume <candidate.json> --video outputs/momentum.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_underbody_claw import controller_type as original_type
from tasks.jumper.common.constants import HOME
from tasks.jumper.front_somersault import physics, staged
from tasks.jumper.front_somersault.momentum import MomentumMeter, controller_type

ROOT = Path(__file__).resolve().parents[1]
# LM absolute J0/J1/J2 targets [rad]. RM uses HOME-relative named mirroring.
# Positive J0 sweeps middle feet backward. FK only suggested these endpoints;
# every sample is still integrated with the original limited servos.
DRIVES = {"modest": (.2, -1.25, -.35), "medium": (.45, -1.30, -.35),
          "rear_like": (.65, -1.35, -.35)}


def variant(preset, delay=0., load_back=False, trigger="delay", opening=0.):
    yaw = DRIVES.get(preset, DRIVES["modest"])[0]
    return dict(yaw=HOME["LF_J0_joint"], opening=opening, open_seconds=.03, alignment="original",
                push_trigger="com", momentum=dict(
                    middle_enabled=preset != "original", preset=preset,
                    load=(yaw if load_back else -.4, HOME["LM_J1_joint"], HOME["LM_J2_joint"]),
                    drive=DRIVES.get(preset, DRIVES["modest"]),
                    middle_start_s=0., middle_ramp_s=.07, middle_hold_s=.025,
                    claw_delay_s=delay, trigger=trigger, max_wait_s=.03))


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
    params = json.loads((ROOT/"tasks/jumper/front_somersault/ref/diagnostic.json").read_text())["params"]
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
    print("Checks passed: centroidal momentum, contact/COM/peak gates, timeout and continuity.")


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
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.video and args.compare:
        parser.error("select a single candidate before recording video")
    if ((args.resume or args.compare) and (args.load_back or any(x is not None for x in
            (args.preset, args.delay, args.trigger, args.opening)))):
        parser.error("resume/compare supply the motion; do not combine with motion-selection flags")
    source = json.loads((ROOT/"tasks/jumper/front_somersault/ref/diagnostic.json").read_text())
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
    for v in variants:
        o = v["momentum"]
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
            if len(first):
                k = first[0]
                report.update(first_push_frame_s=float(data["t"][k]),
                              momentum_at_first_push=momentum[i, k].tolist(),
                              command_time_s=float(instances[-1].launch_time[i]),
                              observation_time_s=float(instances[-1].launch_time[i]
                                                       -physics.CONTROL_DT),
                              decision_momentum=instances[-1].launch_momentum[i].tolist(),
                              release_reason=instances[-1].release_reason[i])
            paths = [Path(__file__), ROOT/"tasks/jumper/front_somersault/momentum.py",
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
                        training_reference=False)
            stem = f"variant_{i:02d}"
            (args.output/f"{stem}.json").write_text(json.dumps(meta, indent=2)+"\n")
            np.savez_compressed(args.output/f"{stem}.npz", momentum=momentum[i],
                                **{k: a for k, a in data.items() if isinstance(a, np.ndarray)})
            summaries.append(meta)
            print(json.dumps(dict(variant=v, **report)), flush=True)
        (args.output/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    finally:
        staged.RearClawController = original
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
