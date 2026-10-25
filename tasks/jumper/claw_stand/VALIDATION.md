# Validation record

Measured 2026-10-24 on Windows / AMD CPU, Python 3.13.14,
MuJoCo 3.11.0, Torch 2.14.1+cpu, Warp 1.18.0 (CPU device).
Editable `.[dev]` installation; `pip check` passes. The vendored `mjlab`
and `rsl_rl` resolve into this checkout. No NVIDIA training was performed.

## Reference

`python tools/make_claw_stand_clip.py` checks all 446 samples, including
smoothstep transitions, then writes the source NPZ, JSON measurements and
complete per-sample CSV. Optional `--plot` and `--video` outputs are committed.

- Time endpoints: 0–8.9 s at 50 Hz; converter holds 446 frames (8.92 s).
- 1.5 s high hold and two push-up reps, followed by 0.5 s at HOME.
- All 22 joints remain inside mechanical limits: minimum margin 0.10 rad.
  Independent check also found minimum soft-limit margin 0.015 rad.
- Base height 106.47–135.00 mm; maximum tilt 37.242 degrees nose-up.
- All collision meshes remain above the floor within floating-point noise.
  Both claws and rear pads are within 0.220 mm of it through the entire clip;
  all four touch during the high hold. Middle pads clear by 1.500 mm then.
- Strict contact support (mesh gaps below 1 micrometre) keeps projected COM
  at least 17.66 mm inside the support polygon on every sample. No frame has
  an airborne robot. The generator requires at least 5 mm of COM margin.
- No inter-leg penetration exceeding 0.1 mm; an independent raw-asset check
  also found no body-versus-limb penetration exceeding that tolerance.
- End-to-start horizontal drift is approximately 0.13 micrometres.
- Shared `ensure_motion_npz(0.02, MEDIA)` accepts the clip unchanged. Its
  legacy median middle/rear site checks remain active; no checks are bypassed.

The preview was rendered with the existing `ReplayVideo` recorder in a native
one-environment scene. Reference poses were written by joint name and forwarded
kinematically; no policy and no physics integration were used. Visual inspection
covered HOME, planting, raised hold, both lowering/raising repetitions and return.
The MP4 has 891 H.264 frames, 960×720 at 50 fps, duration 17.82 s (two cycles).
It deliberately lives outside `media/`, whose video slot means face animation.

## Servo plausibility

The generator solves a static linear program at every pose, using the model's
2.543 kg mass and Jacobians at the lowest vertices of strictly touching foot
meshes. Nonnegative vertical normal forces must balance all six floating-base
gravity degrees of freedom. The objective minimizes the largest absolute
joint torque. Every sample is feasible below the 1.7464 N·m servo plateau.

The worst minimum peak torque is **1.348 N·m**, at 8.36 s in the HOME return;
the 1.5 s high hold requires **0.481 N·m**. The worst transient exceeds the
1.2 N·m continuous rating, which is why the torque-headroom reward remains.
An independent audit allowing up to 0.25 mm of silicone contact closure lowered
the worst static minimum peak to 0.611 N·m; the generator uses the stricter case.

These are optimized static force-sharing bounds. They exclude inertia, actual
servo tracking and thermal history. Kinematic replay and five PPO iterations
do not establish dynamic convergence or successful real-robot execution.

## Runtime checks

```bash
python scripts/train.py --list
python scripts/train.py --task jumper.claw_stand --dry-run
python -m pytest tests/test_registry.py -q
python scripts/train.py --task jumper.claw_stand --backend native --device cpu \
  --num_envs 16 --max-iterations 5 --headless --no-tensorboard
python -m ruff check tools/make_claw_stand_clip.py tasks/jumper/claw_stand tasks/__init__.py
```

The new task is listed; dry resolution and all **21 registry tests pass**.
After the user-authorized one-line addition to the parity inventory, the
combined registry/parity check passes all **25 tests**.
The CPU smoke run exits successfully after five learning iterations / 1920
environment transitions. Lint passes. The smoke policy is untrained: its final
mean episode length was 11.08 steps, versus the 222-step training horizon, and
it hit the servo plateau. Do not use that checkpoint as a completed motion.

## Full repository suite

The exact requested `python -m pytest tests/ -q` was attempted. On this Windows
host, the existing `test_sigterm_runs_the_finally_blocks` sends SIGTERM to the
pytest process and terminates it before a final summary. The diagnostic command
was:

```bash
python -m pytest tests/ -q \
  --deselect=tests/test_tensorboard.py::test_sigterm_runs_the_finally_blocks
```

The first diagnostic run returned **1196 passed, 19 failed, 27 skipped,
1 deselected** in 129.30 s. Its task-inventory failure was fixed by adding
`"jumper.claw_stand"` to `NOT_LOCOMOTION` in `tests/test_task_parity.py`, with
the user's explicit permission for this one-line scope extension. All four
parity tests now pass. The final full diagnostic run returned **1197 passed,
18 failed, 27 skipped, 1 deselected** in 125.02 s, with no remaining task-inventory
failure.

The remaining 18 failures concern unchanged baseline code on this host:

| Failure class | Count |
|---|---:|
| Linux evdev struct assumes 24 bytes; Windows gives 16 | 1 |
| Unicode fixture written with Windows default cp1252 | 1 |
| Symlink creation requires Windows privilege | 3 |
| CRLF changes a scene fixture's byte-length error path | 1 |
| Cargo.lock parser assumes LF line endings | 1 |
| Windows socket-reuse behavior differs | 1 |
| Translation SHA checks see CRLF checkout bytes | 10 |

For all ten translation failures, the recorded stamp matches the exact HEAD
blob and the LF-normalized local source. Affected baseline tests and modules
were checked unchanged from HEAD. Skips mostly require exported policies;
others require music, Cargo, symlink privilege or off-thread GL support.
Logs are kept locally in the ignored `outputs/claw_stand_pytest_final.log`,
`outputs/claw_stand_pytest.log` and `outputs/claw_stand_baseline_checks.log`.
No shared tracking files were changed.
