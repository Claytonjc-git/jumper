#!/usr/bin/env python3
"""Search actual CPU physics rollouts for a forward flip, with a strict promotion gate.

python tools/search_front_somersault.py --self-test
python tools/search_front_somersault.py --generations 8 --population 64 --num-envs 16
python tools/search_front_somersault.py --resume outputs/front_somersault_search/best_candidate.json

Exit 2 means search completed without a qualified flip. --diagnostic permits that
outcome for setup experiments; it never promotes a failed rollout for training.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from tasks.jumper.common.constants import HOME
from tasks.jumper.front_somersault.physics import (
    CONTROL_DT, HIGH, LOW, SEED, PhysicsRunner, qualify, save_reference, signed_rotation, targets,
)

ROOT = Path(__file__).resolve().parents[1]


def score(data, report):
    """Rank launch/air rotation/recovery; an in-place ground roll gets no air-turn credit."""
    progress = np.clip(report["flight_turn_deg"] / 360, -1, 1)
    rise = max(0, report["peak_height_m"] - float(data["base_pos"][0, 2]))
    impact_fraction = float((data["body_force"] > 1).mean())
    self_fraction = float((data["self_force"] > 1).mean())
    last_quat = data["base_quat"][-1]
    upright = 1 - 2 * (last_quat[1] ** 2 + last_quat[2] ** 2)
    settled = np.linalg.norm(data["base_ang_vel"][-1])
    value = (100 * report["qualified"] + 8 * progress + 8 * min(rise, .35)
            + 3 * report["longest_flight_s"] - 4 * impact_fraction - 3 * self_fraction
            - report["home_rms_rad"] - .2 * settled + .25 * upright
            - 2 * max(0, data["joint_limit_violation"] - .01))
    if data.get("schedule") == "rear-claw":
        value += (6 * np.clip(report["peak_forward_rotation_deg"] / 360, 0, 1)
                  + 3 * report["checks"]["claw_spin"]
                  + np.clip(report["rear_forward_impulse_Ns"], 0, .5)
                  + 4 * np.clip(report["forward_travel_m"], 0, .4))
    return value


def self_test():
    """Synthetic positive/negative controls test validation, not physical feasibility."""
    from tasks.jumper.front_somersault.staged import RearClawController, SEED as STAGED_SEED

    controller = RearClawController(STAGED_SEED[None], list(HOME))
    quat = np.array([[1., 0., 0., 0.]])
    contacts = np.zeros((1, 6), dtype=bool)
    contacts[:, 4:] = True
    ahead = np.zeros(1, dtype=bool)
    start = .30 + .20 + STAGED_SEED[-6]
    controller.target(0., quat, contacts, ahead)
    controller.target(start, quat, contacts, ahead)
    command = controller.target(start + .01, quat, contacts, ahead)
    assert controller.stage[0] == controller.REAR, "rear drive did not precede claw deployment"
    np.testing.assert_allclose(command[0, [list(HOME).index(f"LF_J{j}_joint") for j in (1, 2, 3)]],
                               STAGED_SEED[:3])
    angle = STAGED_SEED[-4] + .05
    quat[:] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    trigger = start + STAGED_SEED[-5] + .01
    controller.target(trigger, quat, contacts, ahead)
    assert controller.stage[0] == controller.REACH, "forward pitch did not deploy claws"
    reach = trigger + STAGED_SEED[-3] + .01
    controller.target(reach, quat, contacts, ahead)
    assert controller.stage[0] == controller.REACH, "claws pushed before touching the floor"
    contacts[:, :2] = True
    controller.target(reach + .01, quat, contacts, ahead)
    assert controller.stage[0] == controller.REACH, "claws pushed while still ahead of the COM"
    ahead[:] = True
    controller.target(reach + .02, quat, contacts, ahead)
    assert controller.stage[0] == controller.PUSH, "planted claws did not trigger the push"
    for stamp in np.arange(reach + .025, 3.1, CONTROL_DT):
        final = controller.target(stamp, quat, contacts, ahead)
    np.testing.assert_allclose(final[0], list(HOME.values()))
    t = np.arange(401) * CONTROL_DT
    angle = np.clip((t - .4) / .8, 0, 1) * 2 * np.pi
    quat = np.column_stack([np.cos(angle / 2), np.zeros(len(t)),
                            np.sin(angle / 2), np.zeros(len(t))])
    rotation = signed_rotation(quat)
    np.testing.assert_allclose(rotation, angle, atol=1e-12)
    names = list(HOME)
    contact = np.ones((len(t), 6))
    contact[(t >= .39) & (t <= 1.21)] = 0
    good = dict(
        t=t, q=np.tile(list(HOME.values()), (len(t), 1)), joint_names=names,
        base_quat=quat, forward_rotation=rotation, contact=contact,
        body_force=np.zeros(len(t)), self_force=np.zeros(len(t)), chassis_force=np.zeros(len(t)),
        tau=np.zeros((len(t), len(HOME))), base_lin_vel=np.zeros((len(t), 3)),
        base_ang_vel=np.zeros((len(t), 3)), base_pos=np.tile([0., 0., .2], (len(t), 1)),
        joint_limit_violation=0.,
    )
    assert qualify(good)["qualified"], "positive validation control failed"
    rear_only = dict(good, schedule="rear-claw", stage=np.ones(len(t)),
                     foot_force=np.zeros((len(t), 6, 3)))
    assert not qualify(rear_only)["qualified"], "rear-only flip passed the required claw spin gate"
    staged_good = copy.deepcopy(rear_only)
    staged_good["stage"][(t >= .30) & (t < .35)] = RearClawController.REAR
    staged_good["stage"][(t >= .35) & (t < .39)] = RearClawController.PUSH
    staged_good["foot_force"][(t >= .30) & (t < .35), 4:, 0] = 1.
    staged_good["momentum"] = np.zeros((len(t), 6))
    staged_good["momentum"][t >= .35, 4] = np.minimum((t[t >= .35] - .345) * 2, .08)
    assert qualify(staged_good)["qualified"], "positive rear-plus-claw spin control failed"
    staged_good["momentum"][:, 4] *= -1
    assert not qualify(staged_good)["qualified"], "backward claw spin passed the claw spin gate"
    for key, value in (("contact", np.ones_like(contact)),
                       ("forward_rotation", -rotation),
                       ("forward_rotation", np.zeros_like(rotation)),
                       ("body_force", np.where(t == t[20], 60., 0.)),
                       ("chassis_force", np.where(t == t[20], 20., 0.))):
        bad = copy.copy(good)
        bad[key] = value
        assert not qualify(bad)["qualified"], f"invalid {key} passed qualification"
    # A grounded limb brush under 50 N is allowed; only the gates above may reject.
    assert qualify(dict(good, body_force=np.where(t == t[20], 40., 0.)))["qualified"]
    multiple_hops = copy.copy(good)
    multiple_hops["contact"] = contact.copy()
    multiple_hops["contact"][(t >= .795) & (t <= .805)] = 1
    assert not qualify(multiple_hops)["qualified"], "separate partial hops passed as one flip"
    normal = targets(SEED, t, names)
    reverse = targets(SEED, t, names[::-1])
    np.testing.assert_allclose(normal, reverse[:, :, ::-1])
    np.testing.assert_allclose(normal[0, 0], list(HOME.values()))
    np.testing.assert_allclose(normal[0, -1], list(HOME.values()))
    print("Self-checks passed: +360 rotation, grounded/reverse/partial-hop/impact rejection, name mapping.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--schedule", choices=("rear-claw", "jump"), default="rear-claw",
                        help="rear drive -> pitch-triggered claw plant/push, or the older jump schedule")
    parser.add_argument("--resume", type=Path, help="best_candidate.json from a previous search")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/front_somersault_search")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "tasks/jumper/front_somersault/ref/front_somersault.npz")
    parser.add_argument("--video", type=Path, help="rerun best candidate through actual physics and record")
    parser.add_argument("--diagnostic", action="store_true", help="allow exit 0 without a qualified flip")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if min(args.generations, args.population, args.num_envs) < 1:
        parser.error("generations, population and num-envs must be positive")
    if args.population % args.num_envs:
        parser.error("population must be a multiple of num-envs")
    args.output.mkdir(parents=True, exist_ok=True)
    initial, low, high = SEED, LOW, HIGH
    if args.schedule == "rear-claw":
        from tasks.jumper.front_somersault.staged import HIGH as high, LOW as low, SEED as initial
    mean = initial.copy()
    if args.resume:
        metadata = json.loads(args.resume.read_text(encoding="utf-8"))
        mean = np.asarray(metadata["params"], dtype=float)
        if mean.shape != initial.shape or not np.isfinite(mean).all():
            parser.error(f"resume file must contain {len(initial)} finite params for {args.schedule}")
    mean = np.clip(mean, low, high)
    sigma = (high - low) * .14
    rng = np.random.default_rng(args.seed)
    runner = PhysicsRunner(args.num_envs)
    best_score, best_params, best_report = -np.inf, None, None
    start = time.monotonic()
    log = []
    try:
        for generation in range(args.generations):
            population = np.clip(rng.normal(mean, sigma, (args.population, len(initial))), low, high)
            population[0] = mean if best_params is None else best_params
            scores = []
            for first in range(0, args.population, args.num_envs):
                batch = population[first:first + args.num_envs]
                for params, data in zip(batch, runner.rollout(batch, schedule=args.schedule), strict=True):
                    report = qualify(data)
                    value = score(data, report)
                    scores.append(value)
                    if value > best_score:
                        best_score, best_params, best_report = value, params.copy(), report
                        save_reference(args.output / "best_candidate.npz", data, params, diagnostic=True)
                    if report["qualified"]:
                        save_reference(args.reference, data, params)
            elite = population[np.argsort(scores)[-max(2, args.population // 8):]]
            mean = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), (high - low) * .025)
            row = dict(generation=generation + 1, evaluations=(generation + 1) * args.population,
                       seconds=time.monotonic() - start, score=float(best_score), **best_report)
            log.append(row)
            (args.output / "search.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
            if args.reference.is_file() and best_report["qualified"]:
                break
        if args.video:
            runner.rollout(np.tile(best_params, (args.num_envs, 1)), video=args.video,
                           schedule=args.schedule)
    finally:
        runner.close()
    print("QUALIFIED reference saved." if best_report["qualified"] else
          "No qualified somersault found. Best rollout is diagnostic; training stays gated.")
    print(f"Candidate/report: {args.output}; elapsed {time.monotonic() - start:.1f} s")
    return 0 if best_report["qualified"] or args.diagnostic else 2


if __name__ == "__main__":
    raise SystemExit(main())
