# Claw stand / push-ups

`jumper.claw_stand` tracks an **8.9 s hand-keyframed reference**: HOME, plant,
37.24 degree nose-up stand, hold 1.5 s, two lower/raise reps, return to HOME,
then hold HOME for 0.5 s. No controls file is needed: this is a timed gesture,
with no operator command axes, like `gesture_paw`.

[Reference MP4](claw_stand_reference.mp4) is **17.82 s, two cycles, kinematic
replay with no policy or physics integration**. It shows the intended motion;
it is not evidence of training convergence. No music is included.

## Edit and regenerate

[`tools/make_claw_stand_clip.py`](../../../tools/make_claw_stand_clip.py) owns
`POSES` (radians, each leg's J0..J4 or J0..J2) and `KEYFRAMES` (seconds, pose name).
Repeated poses hold; transitions use smoothstep. HOME comes from the robot's
calibration. Source leg numbering is LF/LM/LR/RR/RM/RF; mapping to the entity
is by joint name. Every frame is checked against the XML limits with 0.08 rad
margin; the actual minimum is 0.10 rad. Do not clamp a bad pose into range.

From the editable repo environment:

```bash
python tools/make_claw_stand_clip.py \
  --plot tasks/jumper/claw_stand/claw_stand_trajectories.png \
  --video tasks/jumper/claw_stand/claw_stand_reference.mp4 --repeats 2
```

The all-six-foot `solve_resting_pose`/`resting_floor` approach is reused from
`tools/import_wbc_gestures.py`. This tool iterates the lowest collision-mesh
points instead of the pad-centre sites, whose offset changes as a claw rotates.
Height and pitch are solved from contacts and COM, never prescribed to the clip.
The raised hold rests on both claws and rear feet; middle pads clear by 1.5 mm.
Transitions remain supported. The generator refuses floor penetration,
inter-leg collisions, insufficient COM margin or excessive static torque.

Both command and `meas_*` channels describe the same synthetic posture, as in
the gesture importer; `meas_*` does **not** imply a physical recording here.
All per-sample NPZ channels are float32. The source hash records the generator.
The shared conversion cache is content-hashed and ignored locally in
`media/.gitignore`; keep the preview outside `media/`, which reserves video
for the face screen.

[All trajectories and limits](claw_stand_trajectories.png),
[full samples](media/claw_stand.csv), and [measurements](media/claw_stand.json)
are generated together. After changing the table, regenerate and remeasure
before updating the following tuning numbers.

## Tuning

Measured at 50 Hz on Windows CPU / MuJoCo 3.11.0:

| Choice | Measurement and rule |
|---|---|
| `EPISODE_S = 4.44` | Half the 8.9 s clip, rounded down to a 20 ms tick |
| `joint_pos_std = 0.115` | Bow rule: 0.44 × position RMS amplitude 0.26209 rad |
| `joint_vel_std = 0.389` | Bow rule: 2/3 × velocity RMS 0.58308 rad/s |
| `anchor_height_error = 0.045` | Base 106.47–135.00 mm: 28.53 mm band plus margin |
| `anchor_tilt_error = 0.33` | Max tilt 37.242 degrees; 1−cos(max)=0.204, with margin |
| `support_foot_error = 0.025` | Claws + rear tips, task-local termination override; contact gaps ≤0.220 mm |
| `torque_headroom_weight = -2.0` | Preserves the servo's 1.2 N·m continuous-rating penalty |
| `joint_limit_weight = -10.0` | Reference stays inside all 0.9 soft limits |

The three termination thresholds are **errors relative to the reference**.
Specifically, tilt uses `abs(cos(reference tilt) - cos(actual tilt))`; the
default 0.3 would not terminate a perfectly tracked tilted pose. No shared
tracking code was changed. Peak joint speed is 2.507 rad/s.

Static gravity/normal-force optimization finds a feasible solution below the
1.7464 N·m servo plateau at every frame; the 1.5 s hold requires about
0.481 N·m at its most loaded joint. See `VALIDATION.md` for limits of this
measurement. Rewards and thresholds are starting points, not a trained result.

## Train in free Colab (T4)

Open [the notebook on this branch](https://colab.research.google.com/github/Claytonjc-git/jumper/blob/feat/claw-stand/notebooks/jumper_colab.ipynb)
and select a T4 GPU runtime. In **Advanced source**, set the first two values
below. Edit the **Basic settings cell source** for `TASK` because the dropdown
does not yet list custom tasks. Use a fresh runtime when changing source commits.

```python
REPOSITORY_URL = "https://github.com/Claytonjc-git/jumper.git"
SOURCE_REVISION = "feat/claw-stand"
WORKFLOW_MODE = "New training"
TASK = "jumper.claw_stand"
MODEL = "jumper"
NUM_ENVS = 1024
ITERATIONS = 3000
RUN_SMOKE = True
RUN_DEMO = False
SIM_COMMAND = "Sampled commands"
RUN_EVALUATION = False
SIM_SECONDS = 20.0
RECORD_JOINTS = True
```

1024 environments / 3000 iterations is a suggested first inspection budget,
not a measured T4 convergence point. Reduce to 256 if memory is insufficient.
Back up checkpoints to Drive/download ZIP, then use **Continue training**
for more iterations toward 10,000 if the curves and replay are improving.
Keep the notebook's default dependency versions. Fixed velocity commands
and its velocity evaluation do not apply to this motion-tracking task.

Watch `Metrics/motion/error_joint_pos`, `error_joint_vel`, `error_anchor_rot`
and body errors decrease while joint/body tracking rewards increase. Episode
length should approach 222 control steps (4.44 s); support/anchor failures should
decline. Inspect the sampler's most-failed time bins: the stand must improve too.
Watch `Episode_Metrics/peak_torque` and `Episode_Reward/torque_headroom` together;
persistent torque saturation is not successful convergence even with good poses.

Replay complete cycles: smooth claw planting, visibly nose-up hold, two clear
lower/raise reps, floor contact, no jitter/sliding/falls, smooth HOME return.
Use the 20 s notebook replay for a >15 s policy video. A short local smoke
checkpoint proves only that the environment builds and steps.

Local smoke command and complete verification results: [VALIDATION.md](VALIDATION.md).
