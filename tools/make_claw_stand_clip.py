#!/usr/bin/env python3
"""Build and check a hand-keyframed, grounded claw stand (no trained policy).

Run from the editable repository: python tools/make_claw_stand_clip.py
The table is radians, addressed by leg/joint name, never the robot's column order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import import_wbc_dances as dances
import import_wbc_gestures as gestures
import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from tasks.jumper.common.constants import FEET as FOOT_BODIES
from tasks.jumper.common.constants import EFFORT_LIMIT, HOME
from tasks.jumper.common.dance.motion import LEG_JOINTS, _rpy_to_quat, load_source

# Each row is J0..J4 for a claw, J0..J2 for a walking leg. HOME is the shared
# calibrated hardware pose. These are poses, not PD offsets; keep fingers closed.
POSES = {
    "home": {leg: tuple(HOME[n] for n in names) for leg, names in LEG_JOINTS.items()},
    "plant": {
        0: (-0.062702, -0.933696, -0.401332, -1.862726, 0.0),
        5: (0.104195, -0.934421, 0.404873, 1.873303, 0.0),
        1: (-0.223967, 0.510450, -1.774573),
        4: (0.224699, -0.514651, 1.783355),
        2: (0.356977, 0.533758, -1.823168),
        3: (-0.357727, -0.538979, 1.834105),
    },
    "low": {
        0: (-0.230310, -0.514230, -0.451698, -1.733380, 0.0),
        5: (0.279836, -0.515079, 0.468874, 1.736305, 0.0),
        1: (-0.029016, 0.394363, -1.714941),
        4: (0.029423, -0.402781, 1.724866),
        2: (0.507053, 0.920412, -1.991976),
        3: (-0.508198, -0.931423, 2.003892),
    },
    "high": {
        0: (-0.492721, 0.072324, -1.366595, -0.685094, 0.0),
        5: (0.568792, 0.058618, 1.429664, 0.648434, 0.0),
        1: (0.357112, 0.223381, -1.488279),
        4: (-0.355371, -0.238313, 1.500476),
        2: (0.681087, 1.324663, -1.960739),
        3: (-0.681611, -1.339615, 1.970178),
    },
}

# Seconds include the lead-in, lead-out, and final HOME hold. Equal poses hold.
KEYFRAMES = (
    (0.00, "home"), (0.30, "home"), (1.20, "plant"),
    (2.40, "high"), (3.90, "high"),  # 1.5 s show-off hold
    (4.55, "low"), (5.20, "high"),  # rep 1
    (5.85, "low"), (6.50, "high"),  # rep 2
    (7.50, "plant"), (8.40, "home"), (8.90, "home"),
)
DT = 0.02
LIMIT_MARGIN = 0.08
REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks/jumper/claw_stand"
SUPPORT = (0, 1, 4, 5)  # gestures.FEET order: LF RF LM RM LR RR


class ContactRobot(gestures._Robot):
    """The existing six-foot floor solver, using mesh contacts instead of pad centres.

    A rotated claw's contact point can be 10 mm below its centre-site proxy.
    Iterate the six mesh support points and the same COM-containing resting plane.
    No body height or pitch is prescribed to the clip.
    """

    def __init__(self):
        super().__init__()
        self.home_z = 0.0  # mesh contact points, unlike the sites, sit on z=0
        self.pad_geoms = [self.model.geom(f"{body}_meshcol").id for body in FOOT_BODIES]
        self.vertices = []
        for gid in self.pad_geoms:
            mid = self.model.geom_dataid[gid]
            start, count = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            vertices = self.model.mesh_vert[start:start + count].astype(float)
            self.vertices.append(vertices[ConvexHull(vertices).vertices])

    def pad_vertices(self):
        return [v @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]
                for g, v in zip(self.pad_geoms, self.vertices, strict=True)]

    def feet_and_com(self, q):
        _, com = super().feet_and_com(q)
        contacts = np.zeros((len(q), 6, 3))
        up = np.array([0.0, 0.0, 1.0])
        for k, row in enumerate(q):
            self.data.qpos[:] = 0.0
            self.data.qpos[3] = 1.0
            self.data.qpos[self.q_adr] = row
            mujoco.mj_kinematics(self.model, self.data)
            vertices = self.pad_vertices()
            for _ in range(100):
                points = np.array([v[np.argmin(v @ up)] for v in vertices])
                normal, _ = gestures.resting_floor(points, com[k], up)
                if np.linalg.norm(normal - up) < 1e-8:
                    break
                up = normal
            else:
                raise ValueError(f"frame {k}: mesh contact plane did not converge")
            contacts[k] = points
            up = normal
        return contacts, com


def sample(robot):
    """Smoothstep each named joint between the table's poses, endpoints included."""
    time = np.arange(round(KEYFRAMES[-1][0] / DT) + 1) * DT
    rows = []
    times = np.array([t for t, _ in KEYFRAMES])
    assert np.all(np.diff(times) > 0), "keyframe times must increase"
    for _, label in KEYFRAMES:
        pose = POSES[label]
        assert set(pose) == set(LEG_JOINTS), f"{label}: missing or extra leg"
        by_name = {}
        for leg, names in LEG_JOINTS.items():
            assert len(pose[leg]) == len(names), f"{label}: wrong joint count for L{leg}"
            by_name.update(zip(names, pose[leg], strict=True))
        rows.append([by_name[n] for n in robot.names])
    rows = np.array(rows)
    assert np.isfinite(rows).all(), "nonfinite keyframe"
    assert np.all(rows >= robot.range[:, 0] + LIMIT_MARGIN), "keyframe below margin"
    assert np.all(rows <= robot.range[:, 1] - LIMIT_MARGIN), "keyframe above margin"
    index = np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2)
    u = gestures._smoothstep((time - times[index]) / (times[index + 1] - times[index]))
    q = rows[index] + (rows[index + 1] - rows[index]) * u[:, None]
    return time, q


def apply(robot, row, pose):
    d = robot.data
    d.qpos[:3] = pose[:3]
    d.qpos[3:7] = _rpy_to_quat(pose[None, 3:])[0]
    d.qpos[robot.q_adr] = row
    d.qvel[:] = 0
    mujoco.mj_forward(robot.model, d)


def static_peak_torque(robot, vertices):
    """Best static force sharing using touching pads and vertical normal forces.

    Balances all six floating-base gravity DOFs, minimizes the worst joint torque.
    This is a static feasibility bound; it excludes inertia and servo dynamics.
    """
    m, d = robot.model, robot.data
    columns = []
    for gid, v in zip(robot.pad_geoms, vertices, strict=True):
        point = v[np.argmin(v[:, 2])]
        if point[2] < 1e-6:
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jac, None, point, int(m.geom_bodyid[gid]))
            columns.append(jac[2])
    a = np.array(columns).T
    dofs = np.array([m.joint(n).dofadr[0] for n in robot.names])
    joint = a[dofs]
    n = len(columns)
    ub = np.vstack([np.column_stack([-joint, -np.ones(len(dofs))]),
                    np.column_stack([joint, -np.ones(len(dofs))])])
    solution = linprog(
        np.r_[np.zeros(n), 1.0], A_ub=ub,
        b_ub=np.r_[-d.qfrc_bias[dofs], d.qfrc_bias[dofs]],
        A_eq=np.column_stack([a[:6], np.zeros(6)]), b_eq=d.qfrc_bias[:6],
        bounds=[(0, None)] * (n + 1), method="highs",
    )
    assert solution.success, f"no static force balance: {solution.message}"
    peak = float(solution.x[-1])
    assert peak < EFFORT_LIMIT, f"static torque {peak:.3f} exceeds servo plateau"
    return peak


def check(robot, time, q, pose):
    """Fail on collisions, loss of geometrical support, or discontinuous motion."""
    m, d = robot.model, robot.data
    floor_min, self_min, support_gap, margin = np.inf, 0.0, 0.0, np.inf
    other_mesh_min = np.inf
    collision_vertices = []
    for gid in range(m.ngeom):
        if not m.geom(gid).name.endswith("_meshcol"):
            continue
        mid = m.geom_dataid[gid]
        start, count = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[start:start + count].astype(float)
        collision_vertices.append((gid, v[ConvexHull(v).vertices]))
    site_z = []
    torque = []
    for k, (row, base) in enumerate(zip(q, pose, strict=True)):
        apply(robot, row, base)
        vertices = robot.pad_vertices()
        bottoms = np.array([v[:, 2].min() for v in vertices])
        floor_min = min(floor_min, float(bottoms.min()))
        support_gap = max(support_gap, float(bottoms[list(SUPPORT)].max()))
        down = bottoms < 1e-6  # strictly touching; no assumed silicone closure
        points = np.array([v[np.argmin(v[:, 2])] for v in vertices])[down, :2]
        hull = ConvexHull(points)
        distance = -(hull.equations[:, :2] @ d.subtree_com[1, :2] + hull.equations[:, 2])
        margin = min(margin, float(distance.min()))
        assert distance.min() > 0.005, f"frame {k}: COM too near/outside support polygon"
        torque.append(static_peak_torque(robot, vertices))
        for gid, verts in collision_vertices:
            lowest = float((verts @ d.geom_xmat[gid].reshape(3, 3)[2]).min()
                           + d.geom_xpos[gid, 2])
            other_mesh_min = min(other_mesh_min, lowest)
            assert lowest > -0.0001, f"frame {k}: {m.geom(gid).name} through floor"
        site_z.append(d.site_xpos[robot.feet, 2].copy())
        for c in d.contact:
            a, b = (m.geom(int(g)).name for g in c.geom)
            # Raw asset has no per-leg masks; same-leg mesh overlaps are bolted
            # assemblies, and are excluded by the actual training entity.
            la, lb = a.split('_')[0], b.split('_')[0]
            if la in gestures.FEET and lb in gestures.FEET and la != lb:
                self_min = min(self_min, float(c.dist))
                assert c.dist > -0.0001, f"frame {k}: {a} hits {b}: {c.dist}"
    assert floor_min > -0.0001, f"foot mesh below floor: {floor_min}"
    assert support_gap < 0.0005, f"claw/rear foot gap exceeds 0.5 mm: {support_gap}"
    vel = np.gradient(q, DT, axis=0)
    amp = float(np.sqrt(np.mean(np.var(q, axis=0))))
    rms_vel = float(np.sqrt(np.mean(vel ** 2)))
    tilt = np.arccos(np.cos(pose[:, 3]) * np.cos(pose[:, 4]))
    report = {
        "duration_s": float(time[-1]), "frames": len(q),
        "position_rms_rad": amp, "velocity_rms_rad_s": rms_vel,
        "peak_joint_speed_rad_s": float(np.abs(vel).max()),
        "height_min_m": float(pose[:, 2].min()), "height_max_m": float(pose[:, 2].max()),
        "max_tilt_deg": float(np.degrees(tilt.max())),
        "minimum_joint_limit_margin_rad": float(np.minimum(
            q - robot.range[:, 0], robot.range[:, 1] - q).min()),
        "minimum_foot_mesh_z_m": floor_min, "maximum_support_gap_m": support_gap,
        "minimum_all_collision_mesh_z_m": other_mesh_min,
        "minimum_com_support_margin_m": margin, "worst_inter_leg_contact_m": self_min,
        "peak_static_minimax_torque_Nm": max(torque),
        "peak_static_torque_time_s": float(time[np.argmax(torque)]),
        "high_hold_static_minimax_torque_Nm": max(
            value for t, value in zip(time, torque, strict=True) if 2.4 <= t <= 3.9),
        "site_height_min_max_m": np.stack([np.min(site_z, axis=0),
                                            np.max(site_z, axis=0)]).tolist(),
    }
    for i, name in enumerate(robot.names):
        print(f"{name:14s} {q[:, i].min():8.4f} .. {q[:, i].max():8.4f} rad; "
              f"peak speed {np.abs(vel[:, i]).max():.3f} rad/s")
    for i, axis in enumerate(dances.AXES):
        print(f"body_{axis}: {pose[:, i].min():.6f} .. {pose[:, i].max():.6f}")
    print(json.dumps(report, indent=2))
    return report


def render_reference(path, time, q, pose, robot, repeats):
    """Drive the existing native scene/MP4 recorder with reference poses, no policy."""
    import torch

    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    from mjrl.viewer.video import ReplayVideo

    use_backend(resolve(backend="native", device="cpu", num_envs=1, training=False))
    from mjlab.envs import ManagerBasedRlEnv
    from tasks.jumper.claw_stand.env_cfg import env_cfg

    cfg = env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        entity = env.scene.entities["robot"]
        by_name = dict(zip(robot.names, q.T, strict=True))
        positions = torch.tensor(np.stack([by_name[n] for n in entity.joint_names], axis=1),
                                 dtype=torch.float32)
        roots = torch.tensor(np.column_stack([pose[:, :3], _rpy_to_quat(pose[:, 3:])]),
                             dtype=torch.float32)

        def set_frame(k):
            entity.write_root_link_pose_to_sim(roots[k:k + 1])
            entity.write_root_link_velocity_to_sim(torch.zeros((1, 6)))
            entity.write_joint_state_to_sim(positions[k:k + 1], torch.zeros_like(positions[:1]))
            env.sim.forward()

        set_frame(0)
        with ReplayVideo(env, path, fps=50, width=960, height=720,
                         camera_distance=0.85, azimuth=145, elevation=-17,
                         follow=False, lookat_height=0.10) as video:
            for _ in range(repeats):
                for k in range(1, len(time)):
                    set_frame(k)
                    video.update()
        print(f"Reference only, no physics integration/policy: {path} ({video.frame_count} frames)")
    finally:
        env.close()


def plot_trajectories(path, time, q, pose, robot):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 4, figsize=(16, 17), layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i < len(robot.names):
            ax.plot(time, q[:, i])
            ax.axhline(robot.range[i, 0], color="red", linestyle=":")
            ax.axhline(robot.range[i, 1], color="red", linestyle=":")
            ax.set_title(robot.names[i] + " [rad]")
        else:
            j = i - len(robot.names)
            ax.plot(time, pose[:, j])
            ax.set_title(f"Solved body_{dances.AXES[j]} [{'m' if j < 3 else 'rad'}]")
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.2)
    fig.suptitle("Kinematic claw stand reference; dotted red = mechanical limits")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TASK / "media/claw_stand.npz")
    parser.add_argument("--video", type=Path, help="kinematic reference preview, outside media/")
    parser.add_argument("--plot", type=Path, help="plot all joints, hard limits and solved base pose")
    parser.add_argument("--repeats", type=int, default=1, help="preview repetitions (2 exceeds 15 s)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.video and args.video.resolve().parent == args.output.resolve().parent:
        parser.error("put the reference video outside media/; that slot is for a face animation")
    robot = ContactRobot()
    time, q = sample(robot)
    pose, _ = gestures.solve_resting_pose(robot, q)
    report = check(robot, time, q, pose)
    out = {"time": time.astype(np.float32), "dt": np.array(DT),
           "phase": np.ones(len(q), dtype=np.float32)}
    for prefix in ("", "meas_"):
        for i, axis in enumerate(dances.AXES):
            out[f"{prefix}body_{axis}"] = pose[:, i].astype(np.float32)
        by_name = dict(zip(robot.names, q.T, strict=True))
        for leg, names in LEG_JOINTS.items():
            for j, name in enumerate(names):
                out[f"{prefix}L{leg}_j{j}"] = by_name[name].astype(np.float32)
    out["source"] = np.array("hand-keyframed:tools/make_claw_stand_clip.py; kinematic, not measured")
    out["source_sha256"] = np.array(hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    loaded = load_source(args.output)
    by_name = dict(zip(robot.names, q.T, strict=True))
    np.testing.assert_allclose(loaded.joint_pos, np.stack([by_name[n] for n in HOME], axis=1),
                               atol=1e-7)
    for key, value in out.items():
        if value.ndim and len(value) == len(q):
            assert value.dtype == np.float32, f"{key}: per-sample channel is not float32"
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(args.output.with_suffix(".csv"), np.column_stack([time, pose, q]),
               delimiter=",", header=",".join(["time"] + [f"body_{a}" for a in dances.AXES]
                                            + robot.names), comments="")
    print(f"Wrote {args.output}")
    if args.plot:
        plot_trajectories(args.plot, time, q, pose, robot)
    if args.video:
        render_reference(args.video, time, q, pose, robot, args.repeats)


if __name__ == "__main__":
    main()
