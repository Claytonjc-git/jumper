# Frontflip

`jumper.frontflip` is a physics-search and residual-tracking setup for
rear-leg drive → claw plant/push → tuck → land → HOME. It uses the existing jump task's
actions and observations, with task-local orientation, angular-velocity and
accumulated-forward-rotation tracking. Existing common and jump files are unchanged.

`ref/diagnostic.json` keeps the base rear-claw parameters the probes start from.
Its NPZ predates the 2026-10-25 gate change and no longer loads as a reference.

**A qualified physics reference exists:** [`ref/frontflip.npz`](ref/frontflip.npz)
passes all 14 qualification gates in nominal CPU physics. The
[video](frontflip.mp4) shows it. It is an open-loop servo schedule, not a trained
policy, and it has not run on the robot. The reference is the `tuned` tuck with
the `tuned` landing (see below), saved by:

```sh
python tools/probe_momentum_launch.py --resume tasks/jumper/frontflip/momentum_launch.json --tuck tuned --landing tuned --reference tasks/jumper/frontflip/ref/frontflip.npz --video tasks/jumper/frontflip/frontflip.mp4
```

| Measurement | Value |
|---|---:|
| Flight turn / total turn | 272.9 / 360.0 deg |
| Longest flight | 0.31 s |
| Claw spin (pitch momentum added during PUSH) | 0.145 N*m*s |
| Peak chassis / limb terrain force | 0 / 34.9 N |
| Joint overshoot, HOME error | 0 rad, 0.019 rad RMS |

The impact and claw gates changed on 2026-10-25. The robot weighs 2.54 kg (25 N):

- `no_chassis_impact`: shell terrain force below 10 N.
- `limb_impact_bounded`: terrain force on palms, calves, fingers and other
  non-pad limb links below 50 N. The old gate allowed 1 N on any non-pad geom.
  It failed on the palm housing, which carries most of the claw push.
- `claw_spin` replaces `claw_propulsion`. The claws must add at least
  0.05 N*m*s of forward pitch momentum during PUSH. The PUSH brakes forward
  (-0.326 N*s), and that braking below the COM produces 70% of the takeoff spin.
  The old gate required forward impulse, which reduces the spin.

Training ends an attempt at 10 N shell force, 100 N limb force or 15 N self-contact force;
limb force between 35 N and 100 N and self-contact above 5 N are penalized instead (see below). A negative control,
the original tuck with claws extended, lands on the shell at 214.7 N and fails both
gates. In the training environment with zero residual, 1 of 8 randomized
environments (foot friction, COM offset) completed the flip. The other 7 failed
at landing, about 1.28 s. Training must close that gap.

### Rearward middle drive with the original short claw handoff

The [new MP4](momentum_launch_preview.mp4) plays the same CPU rollout first at
normal speed, then at one-quarter speed. Base motion comes entirely from
servo/contact physics. It uses the original underbody front-arm alignment,
with the middle feet swept backward and driven down alongside the rear legs.
The front shoulders retain their original yaw; the earlier forward-elbow
experiment below remains a separate comparison.

[`momentum_launch.json`](momentum_launch.json) stores the exact targets,
timings, source/model hashes, joint order and measured qualification report.
The `momentum` table in it is editable: `load` and `drive` are absolute LM
J0/J1/J2 angles, with RM mirrored by HOME-relative joint names. The selected
load is [0.07870, 0.5356, -1.8271] rad and drive is
[0.36808, -1.29038, -0.44253] rad. The drive starts with the rear push, ramps
over 84 ms, continues for 35 ms of claw extension, then lifts smoothly.
Actual middle/palm support overlap above 1 N is 15 ms.

`tools/probe_momentum_launch.py` reuses the original seven-stage controller,
keeps both-palm contact and COM gates, and measures whole-robot momentum
with an independent model reading the live state. No base pose, velocity,
external impulse, mass, friction or actuator limit is changed. The selected
handoff adds zero delay after the original contact/COM/reach gates. It was
chosen by nominal-physics trials, not by predicting a future momentum peak.
The optional causal `--trigger peak` waits for declining momentum with a
bounded timeout; that extra wait performed worse here and is not selected.

```sh
python tools/probe_momentum_launch.py --self-test
python tools/probe_momentum_launch.py --compare --output outputs/momentum_compare
python tools/probe_momentum_launch.py --resume tasks/jumper/frontflip/momentum_launch.json --output outputs/momentum_replay --video outputs/momentum_replay.mp4
```

The 19-case geometry/timing comparison and 256 bounded refinements were
followed by 12 delay/peak/opening comparisons on the Windows AMD CPU.
Opening the fingers to -0.3/-0.6 rad did not improve airborne rotation for
this selected stroke; the preview keeps them closed. No added spring force
has been established. The original control reproduced its recorded actual
joint trajectory exactly. The chosen final run measured:

| Measurement | Original rear/claw | Rearward middle drive |
|---|---:|---:|
| Peak forward rotation, including ground roll | 224.67 degrees | 361.23 degrees |
| Turn in one continuous flight | 149.26 degrees | 187.86 degrees |
| Longest flight | 0.265 s | 0.275 s |
| Peak base height | 0.248 m | 0.268 m |
| Forward travel | 0.254 m | 0.561 m |
| Rear-foot clearance to first PUSH frame | 65 ms | 35 ms |
| Peak non-foot impact | 213.52 N | 113.69 N |

The release observation at 0.855 s coincides with this trial's largest
forward momentum during the prelaunch leg-drive window: 1.968 N*s, with
1.285 N*s upward momentum. The transition is decided at command time 0.860 s;
the first recorded PUSH frame is 0.865 s. Those times differ because the
controller observes the previous state and a stage transition first emits
the outgoing stage's command. Angular momentum is about the whole robot's
COM in world coordinates. The [momentum plot](momentum_launch.png) compares
the original and new launches; matching peak forward momentum does not
mean peak angular or upward momentum occurs at the same instant.

Middle feet contribute +0.538 N*s forward and +2.004 N*s upward terrain
impulse across REAR/REACH/PUSH. The palms' net forward impulse during PUSH
is **-0.326 N*s**, so that part still brakes forward translation. The full
roll finishes upright and settled, but includes ground contact rather than
a complete airborne frontflip. Self-contact force is zero and peak torque
stays 1.7464 N*m. All commanded targets retain at least 0.10 rad XML margin;
actual RR J2 overshoots its limit by **0.0540 rad** at 1.295 s during the
landing sequence. Body impact, insufficient airborne rotation, the actual
joint overshoot and braking palm impulse fail qualification. This run is
not promoted to the training reference or enabled in the training controller.

Validation includes native CPU integration/rendering, momentum/contact/COM
self-checks with negative controls, timeout and interrupted-drive continuity
checks, registry/parity tests, Ruff and independent code review. No new
policy training or physical-robot test was performed for this experiment.

### Flight tuck shape and landing extension

The `momentum` table accepts an optional `flight` table. `tuck` sets absolute
flight-tuck targets per left leg, from J0 (`{"LF": [...], "LM": [...], "LR": [...]}`);
the right legs mirror by name. `claw: "extended"` holds the
PUSH arm pose through flight. Each `land` entry moves its `limbs` to `land_pose`
(HOME where unset), starting `at_s` after TUCK begins, over `ramp_s`. Each ramp
starts from the live command. After `hold_s` the limbs return to HOME over
`release_s` (0.3 s). Without `flight`, the replay matches the run above exactly.

```sh
python tools/probe_momentum_launch.py --resume tasks/jumper/frontflip/momentum_launch.json --tuck tuned --landing tuned --video outputs/flight_landing.mp4
python tools/probe_momentum_launch.py --resume tasks/jumper/frontflip/momentum_launch.json --tuck tuned --land-at .20 --land-ramp .04 --land-limbs claw
python tools/probe_momentum_launch.py --resume tasks/jumper/frontflip/momentum_launch.json --compare-flight --output outputs/flight_compare
```

In flight the angular momentum stays fixed (0.191 N*m*s at takeoff for every
variant below). Limb poses change only the pitch inertia and, through it, the spin
rate. At takeoff the claws carry 46% of the 21.0 g*m^2 pitch inertia. Measured
on the Windows AMD CPU:

| Flight setting | Flight turn | Flight | Claw/middle/rear touchdown | Peak non-foot impact | Joint overshoot |
|---|---:|---:|---|---:|---:|
| Original tuck | 187.9 deg | 0.275 s | 1.690/1.690/1.250 s | 113.7 N | 0.054 rad |
| Original rear tuck, claws extended | 146.4 deg | 0.285 s | lands on body | 214.7 N | 0 |
| `inward` rear tuck | 269.0 deg | 0.315 s | 1.290/1.560/1.490 s | 139.8 N | 0.058 rad |
| `inward`, claws extended | 161.7 deg | 0.280 s | lands on body | 386.5 N | 0 |
| `inward`, all limbs open at 0.24 s over 0.04 s | 243.6 deg | 0.290 s | 1.290/1.290/1.265 s | 64.1 N | 0 |
| `tuned` tuck | 289.7 deg | 0.315 s | 1.290/1.300/1.305 s | 212.3 N | 0 |
| `tuned`, claws open at 0.20 s over 0.04 s | 276.7 deg | 0.315 s | 1.290/1.300/1.290 s | 104.2 N | 0 |
| `tuned` tuck, `tuned` landing | 272.9 deg | 0.310 s | 1.290/1.295/1.285 s | 34.9 N | 0 |

The `inward` rear tuck alone turns the robot 269 degrees in flight. The claws
touch down first, about 0.2 s before the rear feet, and the LF/RF J3 joints
overshoot their limits. Holding the claws extended removes about 40-110 degrees
of flight turn, and the robot lands on its body. If the robot tucks and then
opens all limbs at 0.24 s, the three limb pairs touch down within 25 ms and the
J3 overshoot is gone. This setting costs 26 degrees of flight turn. In every
setting, touchdown occurs at about 1.29 s with the nose still down, because the
total turn at touchdown is 310-336 degrees. A clean landing needs more spin
before touchdown, not only a different landing pose. The `inward` tuck also
shows 2.3 N of self-contact in flight (1.14-1.20 s), above the 1 N gate: its
rear calf touches the middle calf.

The `tuned` tuck moves the claw, middle and rear legs together. A static search
kept 8 mm between limbs along the takeoff -> tuck -> HOME joint paths. It found
poses near 9.5 g*m^2 (original tuck 12.7, `inward` 9.7 but not clear). A
10-generation cross-entropy search then maximized simulated flight turn. The
servos reach 13-22 rad/s during the tuck, where the torque-speed curve limits
them, so the lowest static inertia does not always give the largest turn.

Unchanged, the `tuned` tuck lands the hard palm links first (212 N). If the claws
open at 0.20 s, all three limb pairs touch down within 10 ms and the robot
passes 11 of 13 gates, including the 270-degree flight turn and joint limits.
The remaining failures are the claw-propulsion gate (palm impulse -0.326 N*s,
which is the braking that makes the spin) and 104 N non-foot impact at 1.31 s.
At that time the robot has turned 344.6 degrees and squats onto its fingers,
palms and leg links: the landing speed is the cause, not the flip shape.

The robot touches down at 1.8 m/s vertical, after a 0.2 m fall from peak height.
With only the claws open, the claw and rear servos reach their 1.75 N*m limit
within 10 ms, but the folded middle legs use only 0.5-0.8 N*m. The `tuned`
landing came from a 12-generation cross-entropy search on peak non-foot force
after 1.1 s. It opens the claws at 0.202 s and the middle and rear legs at
0.295 s into a landing pose, holds it 0.18 s, then releases to HOME. The peak
landing force falls to 34.9 N with 272.9 degrees of flight turn and no self-contact.
The remaining contacts are flat claw fingers, palms and rear calves (up to 18 N
during recovery). Standing at HOME records no non-foot force. Separately, the hard
palm links and middle calves scrape the floor during the launch (30.3 N at
0.815 s). Under the revised limb limit of 50 N, this run qualifies (see the top).

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
speed for viewing, not a trained policy or successful frontflip.

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
Both attempts still land on the body and fail the frontflip qualification.

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

### Rear -> middle -> claw relay with delayed opening

The [stage-labelled slow-motion preview](rear_middle_claw_slow.mp4) shows the
next CPU physics experiment at one-quarter speed. The matching
[`relay_launch.json`](relay_launch.json) stores all settings, the original
24-parameter table, source hashes and measured report for exact replay:

```sh
python tools/probe_underbody_claw.py --relay --self-test
python tools/probe_underbody_claw.py --resume tasks/jumper/frontflip/relay_launch.json --output outputs/relay_replay --video outputs/relay_replay.mp4
```

`relay.py` adds an experimental middle-drive stage (recorded as stage 7),
without modifying `staged.py`, shared code, the trained task or its reference:

1. Retain middle support while the rear legs drive. Handoff requires prior
   rear contact, then two control samples with both rear feet off the floor.
2. Push the middle legs toward LM J0/J1/J2 = [-0.4, -0.3, -0.8] rad,
   mirrored by name, over 0.08 s. Preplant the palms during this stage; the
   large arm-extension stroke still waits for the launch gate.
3. After middle support has been observed, wait for 65 degrees nose-down.
   Arm launch additionally requires both palms planted and COM beyond them.
   The middle feet may depart after contributing their push.
4. Open the fingers toward LF/RF J4 = -0.6/+0.6 rad only after actual arm
   joints pass 55% of the launch stroke. This is measured joint-space
   progress, not elapsed command time. Keep the arms extended until the
   fingers reach 70% of their opening target, with a 0.30 s timeout.
5. Tuck and request HOME recovery. A recovery target does not mean the robot
   has landed upright; this recorded attempt lands on the body.

On the Windows AMD CPU, rear-foot departure handed off at **0.860 s**.
Middle drive contributed **0.991 N*s upward impulse**. Arm launch began at
**67.10 degrees** nose-down. Finger opening triggered at **1.185 s**, after
**57.03%** measured arm progress; actual finger travel during PUSH was about
**0.441 rad**. Peak forward rotation is **206.85 degrees**, peak base height
**0.155 m**, and longest unsupported interval **0.070 s**. Forward palm impulse
is **0.085 N*s**. Peak torque remains **1.7464 N*m**, all commanded targets
keep the XML margin, and measured self-contact force is zero.

The handoff works, but rotation still trails the original 224.67-degree
attempt. Both palms had already lost contact when delayed opening triggered,
and the moving fingers recorded zero ground impulse throughout. This is not
evidence of an additional spring-powered launch. The attempt records
**178.38 N** peak non-foot impact and fails the existing frontflip gates.
No failed recording is promoted for training. The required next improvement
is maintaining useful claw contact through more of the extension stroke.

For new comparisons, `--relay` sweeps 50/65/80-degree launch thresholds,
two middle strokes and closed/-0.3/-0.6 rad jaw targets. A single variant can
be selected with `--launch-angle`, `--middle-knee` and `--opening`;
`--middle-ramp` and `--jaw-after` tune middle timing and the measured opening
threshold. The JSON replay path also preserves arbitrary bounded middle
targets and arm parameters from an experiment.

Validation includes CPU rollouts and MP4 replay, relay feedback self-checks,
25 registry/parity checks, and Ruff. A closed-jaw, non-relay control reproduced
the original recorded joint trajectory exactly on this machine. No policy
training or hardware test was performed for this experimental sequence.

### Earlier overlap with forward-facing elbows

The [new slow-motion recording](overlap_forward_elbows_slow.mp4) and
[`overlap_forward_elbows.json`](overlap_forward_elbows.json) show post-tuck
shoulder rotation and an earlier middle/claw handoff. This remains a failed
physics prototype; the existing training task/reference is unchanged.

```sh
python tools/probe_underbody_claw.py --resume tasks/jumper/frontflip/overlap_forward_elbows.json --output outputs/overlap_replay --video outputs/overlap_replay.mp4
```

After the original underbody tuck, ALIGN (stage 8) turns all four arm joints
together toward LF J0..J3 = [-1.57, -0.025, -1.3, -1.85] rad, RF mirrored by
name. It allows 0.4 s and requires measured joint error below 0.15 rad before
rear drive. Turning J0 alone after changing elbow/wrist angles crossed the
claws. The coordinated path retained 11.75 mm minimum claw-mesh floor clearance
in FK at nominal HOME base height. The final forearm axis is
[0.2674, 0.0001, -0.9636]: forward/down with almost no sideways component.

Shallow planting targets place the palms at [0.04, +/-0.05, -0.14] m relative
to the base. The earliest permitted launch is after 25% of the 0.06 s middle
ramp and 20 degrees nose-down, with current middle/palm contact and the COM
gate satisfied. Middle drive continues through the initial palm push rather
than immediately lifting those feet. Its requested shared-drive hold is
0.08 s; actual support overlap is measured separately from forces. Fingers
start opening at 25% measured arm progress. During TUCK, folding waits for
measured finger aperture below 0.08 rad. The timed HOME-recovery fallback
remains, so this is not a global guarantee for every experimental setting.

A 24-case comparison covered three arm profiles, overlap/no-overlap,
deep/shallow planting and two middle ramps. Another 32 cases varied launch
stroke and middle hip/ramp settings. Short strokes reduced requested wrist
travel from about 2.9 rad to 0.6-0.8 rad, but did not improve the flip in these
trials. The selected recording retains the longer target with earlier
opening and measured-closure flight folding.

Measured on the Windows AMD CPU:

| Measurement | Previous late relay | Forward-elbow overlap | Original rear/claw |
|---|---:|---:|---:|
| Peak forward rotation | 206.85 degrees | 200.85 degrees | 224.67 degrees |
| Turn in one flight | 31.90 degrees | 131.11 degrees | 149.26 degrees |
| Longest flight | 0.070 s | 0.250 s | 0.265 s |
| Peak base height | 0.155 m | 0.266 m | 0.248 m |

The new shoulders settle at 1.185 s; rear-foot departure occurs at 1.275 s.
Claw launch starts at **30.17 degrees**, with middle support still present.
Middle/palm pair forces both exceed 1 N for **0.020 s** during PUSH. Middle
upward impulse is **1.576 N*s** during middle drive plus **0.120 N*s** during
PUSH. Opening triggers at 1.570 s, after **26.66%** arm progress, but palms have
already lost contact and moving fingers still record zero ground force.
Forward palm impulse is only **0.0097 N*s**, below the existing propulsion
gate. Simultaneous support does not establish extra propulsion or positive work.
Peak torque is **1.7464 N*m**, self-contact force zero and peak non-foot impact
**225.59 N**. The attempt still lands on the body and is not promoted for training.

Optional `--overlap`, `--middle-hold`, `--shoulder-after-tuck`, `--shallow-plant`
and `--short-stroke A/B/C` select these experiments. `ALIGNMENTS`,
`SHALLOW_PLANTS`, `STROKES` and `SHORT_PUSHES` are editable named tables in
the probe tool; full JSON settings replay an exact combination.

Validation includes CPU trials/replay, feedback/overlap self-checks, 25
registry/parity tests, Ruff and an independent mixed-stage batch comparison
against four individual controllers. The non-relay control still reproduces
the original actual joint trajectory exactly. No shared code was modified.

## 1. Find a physically achievable reference

Run from the repository root, using the installed environment. On Windows,
replace `python` with `.venv/Scripts/python.exe`.

```sh
python tools/search_frontflip.py --self-test
python tools/search_frontflip.py --generations 32 --population 128 --num-envs 16 --video outputs/frontflip_search/candidate.mp4
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

Each generation saves `outputs/frontflip_search/best_candidate.npz`,
its JSON report and `search.json`. Exit code **2** means no candidate qualified;
it is a completed search, not a simulator crash. `--diagnostic` permits exit 0
for an unsuccessful experiment without promoting it. Continue exploration with
a different random seed and the previous best parameters:

```sh
python tools/search_frontflip.py --resume outputs/frontflip_search/best_candidate.json --seed 23 --generations 32 --population 128 --num-envs 16 --video outputs/frontflip_search/candidate.mp4
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

Only passing rollouts are written to `ref/frontflip.npz`. Loading it
recomputes rotation and qualification from the stored measurements, checks
float32 channels and joint names, and verifies the model/motor fingerprint.
Collision checks retain both physics substeps and the strongest self-contact.
Rear-claw recordings also store controller stage, COM feedback and world-frame
ground reaction on each foot. The sensor's primary-to-terrain force is negated
before recording, so positive X impulse means forward propulsion on the robot.
An assisted ground roll does not count as an aerial frontflip: the flight,
rotation, impact and landing gates above still apply.

## 2. Train on Colab after qualification

Open [the task notebook](frontflip_colab.ipynb) in Colab, select a T4 GPU,
and run cells in order. It preserves the main notebook's smoke, resume, replay,
telemetry and backup flow and adds a strict reference-upload step.

```python
REPOSITORY_URL = "https://github.com/Claytonjc-git/jumper-crab-robot.git"
SOURCE_REVISION = "feat/frontflip"
TASK = "jumper.frontflip"
NUM_ENVS = 256
ITERATIONS = 3000
```

256 environments is a conservative starting point for a free T4; GPU memory
and throughput are unmeasured here. After its five-update smoke test, increase
to 512 or 1024 only if memory and throughput allow, using a new run. Start with
3,000 updates, inspect replay, and continue in blocks if the curves are still
improving. No iteration count guarantees convergence.

The notebook validates the qualified NPZ from the checkout; nothing needs uploading.
Enable `UPLOAD_REFERENCE` only to train on a different reference. The reference
bytes, source revision and runtime configuration are part of the continuation checks. Set `SIM_COMMAND = "Sampled commands"`,
`RUN_EVALUATION = False` (the standard evaluator measures walking velocity),
`RECORD_JOINTS = True`, and `SIM_SECONDS = 20` for repeated attempts and a clip
longer than 15 seconds. The notebook already supplies these defaults.

Tracking starts with the jump recipe's scales, plus orientation error scale
0.15 rad², angular velocity scale 4 (rad/s)² and accumulated-turn scale
0.1 rad², weighted 2, 2 and 4. The torque-headroom penalty is -2. These are
starting settings, not convergence measurements. Episode duration is the
reference length plus 0.5 s. Training uses 50% random reference-state starts
(including angular velocity) and 50% HOME starts; replay starts from HOME.
Inversion is allowed. Shell force above 10 N, limb-link force above 100 N and
self-contact force above 15 N terminate attempts. Self-contact above 5 N costs 20 per N per
second. Limb force above 35 N (the reference's own
landing peak) costs 20 per N per second, so a 30 ms landing at 50 N costs about
9 and one at 100 N about 39. The 50 N qualification limit stays the pass/fail
line: read it from the `peak_limb_force` episode metric.

Every termination except the time-out costs 50. Without it, ending an attempt was
free and a landing was not, so PPO ended every HOME attempt in the air. A 64-attempt
probe of the 2,000-update Kaggle run (iteration 1950, Windows AMD CPU, native backend,
training randomization, every attempt from HOME, state read before reset) measured this:

| HOME attempt | Return | Ends |
|---|---:|---|
| Policy, 1 N self-contact gate | 21.9 | 64/64 on self-contact at 1.19 s, in flight |
| Zero residual | 12.8 | 41/64 at touchdown (1.29 s); landing costs -12.4 |
| Policy, no impact terminations | -58 from 1.19 to 1.5 s | shell lands at 1.23 s, 237 N |

The contact is the policy's own: it holds `RF_J0` 0.41 rad off the reference and
closes the 27 mm gap between the right claw's upper arm and the right middle hip. The
zero residual never touches that pair. With the 15 N gate the same policy ends on
shell contact at 1.225 s instead. A landing beats the exit only when a termination
costs more than 25. The policy also launches with less pitch momentum and vertical speed than the reference
(0.168 against 0.191 N*m*s, 0.92 against 1.13 m/s at 0.98 s) and lands 65 ms early,
1.57 rad short of the reference turn.

Scaling that policy's residual does not help. With impact terminations off, the
qualification gates pass in 4, 2, 0, 0 and 0 of 64 attempts at residual gains 0,
0.25, 0.5, 0.75 and 1. Zero residual mostly fails `turn_in_flight` (median 244 deg,
57/64) and `limb_impact_bounded` (median 150 N, 55/64). The reference's 273 deg
flight turn leaves little margin under training randomization. Without randomization,
zero residual qualifies in 64 of 64 attempts; with any two of the three sources it
qualifies in 10 or fewer.

Training therefore starts with nominal physics. Foot friction, base COM offset and
encoder bias are sampled at every reset; their ranges stay at nominal for the first
500 PPO updates, then widen linearly to the full ranges by update 1,500. The
`Curriculum/randomization` metric shows the level (0 to 1). Foot friction ranges
over 0.8-1.5, a clean dry floor by the silicone estimate in `common/constants.py`,
not the shared 0.5-1.5 that includes dusty and wet floors. With zero residual and
full randomization, 3/63 HOME attempts land below 0.8 friction; over 0.8-1.5,
107/192 land. Replay keeps the full ranges, sampled once at startup. A 2,000-update run spends its last 500 updates at
full randomization.

The run with those settings (run 2, commit `9d1578e`) learned to stand still. By
update 450 no HOME attempt turned more than 0.31 rad, and its action std grew from
0.60 to 1.23. PPO scores a policy under its own sampling noise. With zero residual
and nominal physics, a flip from HOME returns 76.0 without noise and 71.7 at action
std 0.15 (1/64 crashes). At std 0.6 it returns -49.5 (64/64 crash), and standing
still returns 47.2. Three changes follow:

- PPO starts at action std 0.15 with no entropy bonus, so the noise cannot grow.
- Up to 1.10 s, an attempt ends when its forward turn is 1.5 rad from the
  reference. Standing still lags 2.6 rad by then; real flips lag 1.05 rad at most.
  Standing now ends at 0.985 s and returns -33.3.
- `flip_landing` pays from 1.30 s (the reference touchdown) to the episode end,
  while at least three feet are down and the body is within 30 deg of upright. The
  pay is exp(-(turn - 2 pi)^2 / 0.25), weighted 25: at most 55, about the
  termination penalty. Standing earns nothing.

With all three, zero residual from HOME returns 127.1 under nominal physics
(`flip_landing` 51.1, 64/64 land) and 124.6 at std 0.15. With full randomization
it returns -32.4, and 28/64 land. A crash on the shell still ends the attempt
first: with impact terminations off, the run 1 policy crashes, rolls back upright
and earns `flip_landing` in 60/64 attempts.

Run 3 (commit `473ac7e`) flips and lands. Under full randomization, update 1750
lands 146/192 HOME attempts, against 107/192 for zero residual, and ends upright
in 184/192. However, it settles at 6.06 rad, not 6.28; updates 1900 and 1999 settle
at 6.02 and 5.96 rad. At sigma 0.25 the landing term still paid 82-66% for that
shortfall, so `flip_landing` now uses exp(-(turn - 2 pi)^2 / 0.05) and 15 deg of
tilt. Under nominal physics, zero residual keeps 32.5 of its 33.6, and update 1750
keeps 13.9 of 29.0.

Run 4 (commit `0abfbea`) completes the turn: update 1900 settles at 6.26 rad and
lands 64/64 under nominal physics. It lands only 103/192 attempts with full
randomization, though, and update 1999 fails on self-contact in 43/64 attempts. The
policy is lopsided. At 1.10 s its right claw is 0.41 rad short of the left on J0 and
0.50 rad short on J1, and the reference matches the two within 0.01 rad. It lands
rolled 6 deg; its right claw takes 22 N against 4 N on the left, and the left
middle calf takes 55 N. The reference commands mirror within 0.043 rad, and a
zero-residual flip mirrors onto itself within 0.09 rad except during the 15 ms of
touchdown impact. So PPO now uses the mirror augmentation and mirror loss.

Run 5 (commit `1284d86`) is symmetric: the two claws match within 0.08 rad, and
the middle calf no longer touches. Under full randomization, update 1500 lands
190/192 HOME attempts; updates 1750-1999 land 128-145, losing the rest to limb
hits above 100 N. However, every landing exceeds the 50 N limb limit: 65 N
median at 1500 under nominal physics, against 35 N for zero residual. The policy
jumps higher (apex 0.32 m, not 0.27 m) and touches down at 6.4 rad, not 6.12 rad.
So the claws land alone, and the rear feet follow 30-45 ms later, where the
reference lands all six within 10 ms. At weight 20, `limb_impact` charged that
landing 7.9, so `limb_impact` now uses weight 60. Re-priced over the same
randomized probe attempts, a 45-55 N landing costs 9.8, a 65-75 N landing 28.5,
and a 75-100 N landing 50, no more than the termination penalty alone.

These settings follow a 64-environment probe of the 0.8-RSI, 50 N-termination
policy at iteration 1500 (Windows AMD CPU, native backend, training
randomization, every attempt from HOME). 46 of 64 attempts ended at 1.27 s with
limb force above 50 N, at 5.7 rad of turn, about 25 ms before and 0.4 rad short
of the reference touchdown (1.295 s, 6.12 rad). With the 100 N termination the
same policy completes 56 of 64 attempts, with a median peak limb force of 66 N;
17 of 64 complete under 50 N. The landing, not the airborne turn, is the gap.

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
qualified frontflip or a motion ready for physical hardware.

A zero-residual native replay of the complete three-second initial both-pad
reference matched its recorded joints within **2.93e-6 rad** and base height
within **1.36e-6 m**. Randomization and failure terminations were disabled for
that comparison so the entire failed trajectory could be checked.

This CPU smoke test builds and trains 16 environments on the qualified reference
for five updates. It proves stepping and PPO integration, not frontflip learning:

```powershell
.venv/Scripts/python.exe scripts/train.py --task jumper.frontflip --backend native --device cpu --num_envs 16 --max-iterations 5 --headless --no-tensorboard
```

The task appears in `scripts/train.py --list`; registry/parity and replay
contracts pass. Full-suite result before the gate change: **1,200 passed,
18 failed, 27 skipped**, with one Windows SIGTERM test deselected because it
terminates the test process. The 18 failures match the pre-change Windows
baseline (Linux input ABI, encoding/line endings, symlink privileges and socket
reuse). After the gate change (2026-10-25): **1,199 passed, 19 failed, 27 skipped**.
The same 19 tests fail on the commit before it. Colab/T4 execution and a successful
frontflip policy remain unverified.
