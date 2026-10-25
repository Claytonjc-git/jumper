# Forward somersault

`jumper.front_somersault` is a physics-search and residual-tracking setup for
rear-leg drive → claw plant/push → tuck → land → HOME. It uses the existing jump task's
actions and observations, with task-local orientation, angular-velocity and
accumulated-forward-rotation tracking. Existing common and jump files are unchanged.

**A complete somersault has not been found.** The shipped
[`candidate_physics.mp4`](candidate_physics.mp4) is a failed physics experiment,
not a trained policy or a successful flip. Its matching `ref/diagnostic.npz` is
available only for explicit smoke tests. There is deliberately no qualified
`ref/front_somersault.npz`; ordinary training refuses to start without one.
The [half-speed video](rear_claw_half_speed.mp4) makes the two propulsion stages
easier to inspect; it contains the same simulated motion with slower playback.

### Middle-leg lift and forward-arm tuck experiment

This earlier preview interpreted the tuck as an above-body pose. The corrected
underbody experiment below follows the clarified motion instead.

The [new preparation preview](middle_lift_preview.mp4), also available at
[half speed](middle_lift_half_speed.mp4), shows an experimental CPU physics
rollout, not a trained policy or completed flip. It uses a 0.4 s support
preparation, LM J1/J2 targets of 0.2/-1.5 rad, and LF J0..J4 targets of
[-pi/2, -pi, 0, -1.97, -1.0] rad, with HOME-relative right-side mirroring.
Shoulder yaw is delayed until the last 35% of the tuck ramp so the claws fold
upward before rotating forward. The original rear drive, deployment targets,
contact triggers and servo limits are retained. These temporary controller
overrides are not enabled in the shipped search/training controller.

On the Windows AMD CPU, the opened-finger variant had 75.65 mm minimum claw-mesh
height at the end of preparation (52.23 mm without the middle-leg extension).
The later tuck interval, from 0.85 s until rear drive, had 62.93 mm minimum
clearance and no measured self-contact force. Across the full preparation,
non-foot ground force briefly reached 1.98 N. The attempt reaches only about
34 degrees of forward rotation and aborts the reach without a claw push.
It is a clearance experiment; the previous diagnostic reference remains the
better rolling attempt.

### Corrected underbody tuck with opening during arm extension

The [aligned underbody preview](underbody_aligned_open_half_speed.mp4) keeps
the palms folded back underneath the body for the original rear-driven roll.
The fingers start closed and receive a 30 ms opening-target ramp at the same
contact/COM-triggered PUSH transition as the arms extend. Its LF opening target
is -0.3 rad (RF mirrored). This is a new CPU physics rollout, slowed to half
speed for viewing, not a trained policy or successful somersault.

`tools/probe_underbody_claw.py` contains the readable `ALIGNMENTS`/`STROKES`
tables. The `underbody_a` solution preserves the original tucked palm center
and reach/push palm endpoints while rotating the forearms toward forward.
All targets are mirrored by name and checked against XML limits with the same
0.08 rad margin. The tool saves diagnostic NPZ/JSON measurements separately;
it does not replace a training reference or alter the shared physics/controller.
Fingers retain their original collision/friction and impact classification.
An extra sensor measures all moving finger collision geoms independently.

On the Windows AMD CPU, the aligned/-0.3 rad preview reaches 192.45 degrees
peak forward rotation, 0.270 m peak base height and -0.078 N*s net forward
palm impulse during extension (a braking impulse). Peak self-contact is
0.301 N. With this same alignment, closed fingers reach 191.02 degrees.
Opening does not yet create finger-ground force. The original closed-jaw
diagnostic remains better at 224.67 degrees and +0.215 N*s palm impulse.

The [original tuck with timed opening](underbody_original_open_half_speed.mp4)
is an additional comparison: opening to -0.6 rad reaches 226.13 degrees and
+0.218 N*s palm impulse, without measured self-contact. Neither comparison
records finger-ground force or establishes a spring-assisted launch. Actual
finger excursions during the 105 ms arm-extension phase are about 0.254 rad
for the aligned preview and 0.509 rad for the original-tuck comparison;
a 30 ms command ramp does not imply instantaneous servo motion. Larger
-1.2 rad openings caused self-contact and are not used in these previews.
Both attempts still land on the body and fail the somersault qualification.

```sh
python tools/probe_underbody_claw.py --self-test
python tools/probe_underbody_claw.py --alignment underbody_a --opening -.3 --video outputs/underbody_aligned.mp4
python tools/probe_underbody_claw.py --opening -.6 --video outputs/underbody_open.mp4
```

Use separate `--output` directories to retain comparisons. Without `--yaw` or
`--opening`, the tool compares an opening grid; without `--alignment` it also
compares shoulder yaws. The optional `--push-trigger contact` tests extension
as soon as both palms plant, without the default COM condition; it did not
advance the transition in the measured trials. All outputs remain experiments.

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

The default `--schedule rear-claw` follows the requested two-stage mechanism.
`staged.py` contains the readable 24-parameter `SEED` table and search bounds:
claw tuck/plant/push poses, rear support/drive/tuck poses, middle-foot lift,
rear yaw and phase timings. Every counterpart is mapped by name and a
HOME-relative delta. Unused joints hold HOME, and every commanded sample must
retain at least 0.08 rad of XML joint-limit margin.

After a 0.3 s HOME lead-in, it establishes middle-leg support for 0.2 s,
tucks the claws while keeping rear support, then drives the rear feet backward
and down while lifting the middle feet. Claw deployment waits for forward
pitch, recorded rear-foot contact and at least half the rear-drive ramp.
The claw push waits for both claws to touch and the whole robot's center of
mass to pass 5 mm beyond the mean hand position. Front forces ahead of that
point can brake the forward rotation. Failure to reach either trigger causes
a smooth HOME recovery; it does not force a fake push. The subsequent flight
tuck and 0.4 s HOME recovery are time based. The table gives initial settings,
not a successful maneuver.

Both front silicone palm pads are support surfaces in this task. The walking
configuration counts only the rear pad; using it unchanged missed real front-pad
contacts and prevented the claw-push trigger. Search and training therefore share
a task-local collision configuration giving both pads the existing foot friction,
contact dimension and priority. Friction randomization includes both pads. The
six-channel contact/force schema combines the two pads per claw; the COM trigger
uses the centers of the pads actually touching. Rigid palm, arm and chassis
contacts remain impacts. Shared hardware configuration is not mutated. The extra
pad's friction remains the same material estimate as the existing foot pad,
not a new measurement.

`--schedule jump` retains the older 27-parameter schedule in `physics.py` for
comparison. Its JSON parameter files require that explicit flag when resuming.

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
| Rear-claw mechanism | Both rear-drive and claw-push phases deliver more than 0.02 N·s of net forward terrain reaction; the claw-push phase must execute |

Only passing rollouts are written to `ref/front_somersault.npz`. Loading it
recomputes rotation and qualification from the stored measurements, checks
float32 channels and joint names, and verifies the model/motor fingerprint.
Collision checks retain both physics substeps and the strongest self-contact.
Rear-claw recordings also store controller stage, COM feedback and world-frame
ground reaction on each foot. The sensor's primary-to-terrain force is negated
before recording, so positive X impulse means forward propulsion on the robot.
An assisted ground roll does not count as an aerial somersault: the flight,
rotation, impact and landing gates above still apply.

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

On this Windows AMD CPU machine (October 25, 2026), the new rear/claw mechanism
passed both propulsion checks after correcting the tuck/drive geometry and
front-pad contact classification. The shipped candidate came from a
1,024-candidate refinement following a 256-candidate both-pad search. It has
**224.67°** peak forward rotation, **149.26°** during one flight, **0.265 s**
continuous flight, **0.248 m** peak base height and **0.254 m** forward travel.
The measured net forward impulses are **0.604 N·s** from the rear drive and
**0.215 N·s** from the subsequent claw-push phase. Peak sampled torque remains
**1.746 N·m**. It lands on its back, finishes at about **187.76°**, and records
**213.52 N** peak non-foot impact. It is an assisted-roll prototype, not a
qualified somersault or a motion ready for physical hardware.

A zero-residual native replay of the complete three-second initial both-pad
reference matched its recorded joints within **2.93e-6 rad** and base height
within **1.36e-6 m**. Randomization and failure terminations were disabled for
that comparison so the entire failed trajectory could be checked.

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
