# Forward somersault

`jumper.front_somersault` is a physics-search and residual-tracking setup for
crouch → launch → tuck → extend → land → HOME. It uses the existing jump task's
actions and observations, with task-local orientation, angular-velocity and
accumulated-forward-rotation tracking. Existing common and jump files are unchanged.

**A complete somersault has not been found.** The shipped
[`candidate_physics.mp4`](candidate_physics.mp4) is a failed physics experiment,
not a trained policy or a successful flip. Its matching `ref/diagnostic.npz` is
available only for explicit smoke tests. There is deliberately no qualified
`ref/front_somersault.npz`; ordinary training refuses to start without one.

## 1. Find a physically achievable reference

Run from the repository root, using the installed environment. On Windows,
replace `python` with `.venv/Scripts/python.exe`.

```sh
python tools/search_front_somersault.py --self-test
python tools/search_front_somersault.py --generations 32 --population 128 --num-envs 16 --video outputs/front_somersault_search/candidate.mp4
```

This runs on the CPU. It varies bounded joint targets and timings with a
cross-entropy search; each candidate is integrated in native MuJoCo. After the
initial HOME reset, only servo targets are written. Base motion, contacts and
rotation result from physics. The search uses the robot's real actuator class,
including its torque-speed curve and thermal derating, at kp=10, kd=0.5,
200 Hz control and 400 Hz physics. This differs from the claw-stand MP4, whose
reference was animated by assigning joint and base poses.

`physics.py` contains the readable schedule: three seven-joint poses in `SEED`
(crouch, push, tuck), four phase durations and front/rear timing offsets.
`JOINT_PAIRS` maps every counterpart by name and sign; unused joints hold HOME.
`LOW`/`HIGH` bound exploration. Every commanded sample must retain at least
0.08 rad of XML joint-limit margin. Smoothstep blends the poses; extension
returns to HOME over 0.4 s, with a nominal 0.3 s lead-in (leg timing offsets
can advance preparation) and settling at the end.

Each generation saves `outputs/front_somersault_search/best_candidate.npz`,
its JSON report and `search.json`. Exit code **2** means no candidate qualified;
it is a completed search, not a simulator crash. `--diagnostic` permits exit 0
for an unsuccessful experiment without promoting it. Continue exploration with
a different random seed and the previous best parameters:

```sh
python tools/search_front_somersault.py --resume outputs/front_somersault_search/best_candidate.json --seed 23 --generations 32 --population 128 --num-envs 16 --video outputs/front_somersault_search/candidate.mp4
```

`--resume` warm-starts the parameter mean; it does not restore the optimizer's
random state. More search is not guaranteed to find a flip. If repeated runs
plateau, revise the launch/tuck schedule and bounds rather than raising the
servo limit or animating a rotation into the reference. Changing physics,
hardware constants, actuator implementation, XML or meshes invalidates old
references; rerun their parameters through current physics.

Promotion requires all of the following measured gates:

| Gate | Requirement |
|---|---|
| Flight | At least 0.08 s continuously without foot support |
| Forward turn | Final accumulated world-Y rotation within 0.35 rad of +2π |
| Air rotation | At least 270° within one continuous flight, not summed across hops |
| Impacts | Non-foot ground and self-contact forces below 1 N throughout |
| Landing | Last 0.3 s within 15° of upright, at least three feet planted |
| Recovery | HOME joint RMS below 0.18 rad; speed below 0.25 m/s and 1 rad/s |
| Hardware | Torque within the shared modeled cap (currently 1.7464 N·m, below the XML's 2 N·m ceiling); joint-limit overshoot below 0.01 rad |

Only passing rollouts are written to `ref/front_somersault.npz`. Loading it
recomputes rotation and qualification from the stored measurements, checks
float32 channels and joint names, and verifies the model/motor fingerprint.
Collision checks retain both physics substeps and the strongest self-contact.

## 2. Train on Colab after qualification

Open [the task notebook](front_somersault_colab.ipynb) in Colab, select a T4 GPU,
and run cells in order. It preserves the main notebook's smoke, resume, replay,
telemetry and backup flow and adds a strict reference-upload step.

```python
REPOSITORY_URL = "https://github.com/Claytonjc-git/jumper.git"
SOURCE_REVISION = "feat/front-somersault"
TASK = "jumper.front_somersault"
NUM_ENVS = 256
ITERATIONS = 3000
```

256 environments is a conservative starting point for a free T4; GPU memory
and throughput are unmeasured here. After its five-update smoke test, increase
to 512 or 1024 only if memory and throughput allow, using a new run. Start with
3,000 updates, inspect replay, and continue in blocks if the curves are still
improving. No iteration count guarantees convergence.

Upload the **qualified** NPZ when prompted. The diagnostic fixture is rejected.
Keep the reference in the downloaded backup and reupload the identical file
when continuing: the reference bytes, source revision and runtime configuration
are part of the continuation checks. Set `SIM_COMMAND = "Sampled commands"`,
`RUN_EVALUATION = False` (the standard evaluator measures walking velocity),
`RECORD_JOINTS = True`, and `SIM_SECONDS = 20` for repeated attempts and a clip
longer than 15 seconds. The notebook already supplies these defaults.

Tracking starts with the jump recipe's scales, plus orientation error scale
0.15 rad², angular velocity scale 4 (rad/s)² and accumulated-turn scale
0.25 rad², each weighted 2. The torque-headroom penalty is -2. These are
starting settings, not convergence measurements. Episode duration is the
reference length plus 0.5 s. Training uses 80% random reference-state starts
(including angular velocity) and 20% HOME starts; replay starts from HOME.
Inversion is allowed; non-foot impacts and self-collisions terminate attempts.

Watch the individual joint, orientation, angular-velocity, forward-turn and
contact tracking rewards together with episode length and impact terminations.
Increasing total reward alone can conceal learning only the easy settling
phase. Random reference starts can also inflate apparent rotation progress;
judge complete attempts from HOME in replay. Require repeatable launch, one
forward revolution and stable foot landing, without body contact, sideways
rolling or an early reset. Check joint telemetry and torque headroom, then
repeat under the Colab Warp backend: native feasibility does not establish
GPU-policy success or physical-robot readiness.

## Verification and current measurements

On this Windows AMD CPU machine (October 25, 2026), the initial 128-candidate
search, a further 512-candidate search, and an extended 2,048-candidate search
did not find a qualified flip. The shipped candidate's final strict report
measures about **0.255 s** of continuous flight, **55.28°** forward rotation
within one flight, **0.211 m** peak base height, **1.746 N·m** peak sampled
torque, and **116.97 N** peak non-foot impact. It fails turn and collision
gates. The MP4 integrates that candidate through physics.

The explicit diagnostic CPU test builds and trains 16 environments for five
updates; it proves stepping and PPO integration, not somersault learning:

```powershell
$env:MJRL_FRONT_SOMERSAULT_REFERENCE=(Resolve-Path tasks/jumper/front_somersault/ref/diagnostic.npz).Path
$env:MJRL_FRONT_SOMERSAULT_ALLOW_DIAGNOSTIC='1'
.venv/Scripts/python.exe scripts/train.py --task jumper.front_somersault --backend native --device cpu --num_envs 16 --max-iterations 5 --headless --no-tensorboard
Remove-Item Env:MJRL_FRONT_SOMERSAULT_REFERENCE,Env:MJRL_FRONT_SOMERSAULT_ALLOW_DIAGNOSTIC
```

The task appears in `scripts/train.py --list`; registry/parity and replay
contracts pass. Full-suite result with the diagnostic fixture: **1,200 passed,
18 failed, 27 skipped**, with one Windows SIGTERM test deselected because it
terminates the test process. The 18 failures match the pre-change Windows
baseline (Linux input ABI, encoding/line endings, symlink privileges and socket
reuse). Colab/T4 execution and a successful somersault policy remain unverified.
