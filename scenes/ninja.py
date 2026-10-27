"""The robot dressed as a ninja, over any other scene -- for video.

Pick the world with `--ninja-base` (default `beach`); this scene keeps that
world exactly and adds a costume in its `decorate`: a dark suit in place of the
red and grey, a red headband with two tails, glowing eye slits on the display
and a katana strapped to the back.

**Visual only.** The suit is `geom_rgba` on the visual geoms, and every added
geom is `contype=0, conaffinity=0, density=0` on a body that carries an explicit
`<inertial>`, so mass, contacts and the policy's observation are unchanged.

The headband, sword and harness are fitted to the shell mesh itself, sampled in
the `base_link` frame at `qpos0` by ray casts, so they follow the shell if its
mesh changes.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from pathlib import Path

import numpy as np

from .registry import Scene, load

BASE_ENV = "MJRL_NINJA_BASE"
ROBOT_XML = Path(__file__).resolve().parents[1] / "assets" / "jumper" / "jumper.xml"
SHELL_GEOMS = ("upper_shell_link_visual", "display_module_link_visual")

# Old visual colour -> suit colour.
SUIT = {
    (0.85, 0.08, 0.05): (0.07, 0.08, 0.11, 1.0),     # red shell and links: indigo-black
    (0.68, 0.7, 0.72): (0.20, 0.21, 0.24, 1.0),      # grey forearms and thighs: charcoal
    (0.816, 0.82, 0.804): (0.34, 0.35, 0.38, 1.0),   # pads and tips: gunmetal
}
RED = (0.80, 0.04, 0.04, 1.0)
EYE = (0.85, 0.95, 1.0, 1.0)
STEEL = (0.10, 0.10, 0.11, 1.0)
GOLD = (0.78, 0.60, 0.22, 1.0)
WRAP = (0.45, 0.05, 0.06, 1.0)
LEATHER = (0.28, 0.16, 0.09, 1.0)

# Headband height on the shell: over the display in front, dipping at the back.
BAND_Z_MID, BAND_Z_DIP = 0.051, 0.007
BAND_HALF_HEIGHT = 0.0045
BAND_SEGMENTS = 72
GAP = 0.0008                 # how far cloth and straps sit off the shell

# The display plate leans back this far from vertical.
DISPLAY_TILT = np.radians(34.5)

# Katana across the dome, in plan: hilt back-left, tip front-right.
SAYA_RADIUS = 0.0045
HILT_XY, TIP_XY = (-0.058, 0.048), (0.060, -0.050)
STRAPS_AT = (0.25, 0.80)     # along the scabbard, tip = 0; straps run across in y
STRAP_HALF_WIDTH = 0.0042
RIVET_Z = 0.059              # straps end here, just above the band


def cli_args(group):
    """`--ninja-base`; see `scenes.load_cli_args`."""
    group.add_argument(
        "--ninja-base", default=os.environ.get(BASE_ENV, "beach"), metavar="NAME",
        help="the registered scene the ninja performs in (default beach)",
    )
    return ("ninja_base",)


def scene(ninja_base: str = "beach") -> Scene:
    if ninja_base == "ninja":
        raise ValueError("--ninja-base must name another scene, not ninja itself")
    base = load(ninja_base)

    def decorate(spec):
        if base.decorate is not None:
            base.decorate(spec)
        dress(spec)

    return dataclasses.replace(base, decorate=decorate)


def dress(spec) -> None:
    """Recolour the robot and add the costume. No-op on a world with no robot."""
    torso = next((b for b in spec.bodies if b.name.split("/")[-1] == "base_link"), None)
    if torso is None:
        return
    for geom in spec.geoms:
        if geom.name.endswith("_visual"):
            suit = SUIT.get(tuple(round(float(c), 3) for c in geom.rgba[:3]))
            if suit is not None:
                geom.rgba = suit
    # Geoms have no emission of their own; a glowing eye needs a material.
    spec.add_material(name="ninja_eye", rgba=EYE, emission=1.0)
    _headband(torso)
    _eyes(torso)
    _katana(torso)


@functools.lru_cache(maxsize=1)
def _shell():
    """The robot at `qpos0` on its own, and its shell and display geoms."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, tuple(m.geom(n).id for n in SHELL_GEOMS)


def _hit(pnt, vec):
    """Where a ray first meets the shell, and the normal facing it; None on a miss."""
    import mujoco

    m, d, geoms = _shell()
    pnt = np.asarray(pnt, float)
    vec = np.asarray(vec, float) / np.linalg.norm(vec)
    best = None
    for g in geoms:
        n = np.zeros(3)
        dist = mujoco.mj_rayMesh(m, d, g, pnt, vec, n)
        if dist >= 0 and (best is None or dist < best[0]):
            best = (dist, -n if n @ vec > 0 else n)
    return None if best is None else (pnt + best[0] * vec, best[1])


def _top(xy):
    """The shell's top surface over `xy`, or None off the shell."""
    return _hit([*xy, 0.3], [0.0, 0.0, -1.0])


def _add(body, name, gtype, size, rgba, *, pos=None, mat=None, fromto=None):
    import mujoco

    g = body.add_geom()
    g.name = f"ninja_{name}"
    g.type = gtype
    g.size = list(size) + [0.0] * (3 - len(size))
    g.rgba = rgba
    g.contype = g.conaffinity = 0
    g.density = 0.0
    g.group = 2
    if fromto is not None:
        g.fromto = list(fromto)
    else:
        g.pos = list(pos)
        if mat is not None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).flatten())
            g.quat = quat
    return g


def _strip(body, name, p0, p1, normal, half_width, half_thick, rgba):
    """A flat box from p0 to p1 whose face points along `normal`."""
    import mujoco

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.asarray(normal, float)
    n = n - t * (n @ t)
    n /= np.linalg.norm(n)
    return _add(body, name, mujoco.mjtGeom.mjGEOM_BOX,
                (np.linalg.norm(p1 - p0) / 2 + 0.0008, half_thick, half_width), rgba,
                pos=(p0 + p1) / 2, mat=np.column_stack([t, n, np.cross(t, n)]))


def _smooth(a, k=2):
    return np.mean([np.roll(a, s, axis=0) for s in range(-k, k + 1)], axis=0)


def _headband(torso):
    """A band lying on the shell all the way round, knotted at the back."""
    m, d, geoms = _shell()
    centre = d.geom_xpos[geoms[0]][:2]
    points, normals = [], []
    for th in 2 * np.pi * np.arange(BAND_SEGMENTS) / BAND_SEGMENTS:
        u = np.array([np.cos(th), np.sin(th), 0.0])
        z = BAND_Z_MID + BAND_Z_DIP * np.cos(th)
        p, n = _hit(np.array([*centre, z]) + 0.3 * u, -u)
        points.append(p)
        normals.append(n)
    normals = _smooth(np.array(normals))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points = _smooth(np.array(points)) + (GAP + 0.0012) * normals
    for i in range(BAND_SEGMENTS):
        j = (i + 1) % BAND_SEGMENTS
        _strip(torso, f"band{i}", points[i], points[j], normals[i] + normals[j],
               BAND_HALF_HEIGHT, 0.0012, RED)

    import mujoco

    back = BAND_SEGMENTS // 2
    knot = points[back] + 0.004 * normals[back]
    _add(torso, "knot", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.005, 0.008, 0.006), RED, pos=knot)
    for side in (-1, 1):
        mid = knot + [-0.040, side * 0.014, -0.012]
        end = mid + [-0.040, side * 0.012, -0.022]
        up = [0.0, 0.0, 1.0]
        _strip(torso, f"tail{side}a", knot, mid, up, 0.0060, 0.0010, RED)
        _strip(torso, f"tail{side}b", mid, end, up, 0.0050, 0.0010, RED)


def _eyes(torso):
    """Two slanted slits on the display, inner ends low."""
    normal = np.array([np.cos(DISPLAY_TILT), 0.0, np.sin(DISPLAY_TILT)])
    along = np.array([-np.sin(DISPLAY_TILT), 0.0, np.cos(DISPLAY_TILT)])
    centre = np.array([0.0838, 0.0, 0.045]) + 0.0012 * normal
    for side in (-1, 1):
        mid = centre + [0.0, side * 0.019, 0.0]
        inner = mid - [0.0, side * 0.010, 0.0] - 0.0018 * along
        outer = mid + [0.0, side * 0.010, 0.0] + 0.0018 * along
        eye = _strip(torso, f"eye{side}", inner, outer, normal, 0.0022, 0.0006, EYE)
        eye.material = "ninja_eye"


def _katana(torso):
    """A scabbard resting on the dome, held by two straps riveted to the shell."""
    import mujoco

    tip_xy, hilt_xy = np.array(TIP_XY), np.array(HILT_XY)
    # Lowest straight line that clears the dome everywhere along the scabbard.
    s = np.linspace(0.0, 1.0, 41)
    tops = np.array([_top(tip_xy + f * (hilt_xy - tip_xy))[0][2] for f in s])
    rest = tops + SAYA_RADIUS + GAP
    a, b = np.polyfit(s, rest, 1)
    a += np.max(rest - (a * s + b))
    tip = np.array([*tip_xy, b])
    hilt = np.array([*hilt_xy, a + b])
    axis = (hilt - tip) / np.linalg.norm(hilt - tip)
    pommel = hilt + 0.046 * axis
    _add(torso, "saya", mujoco.mjtGeom.mjGEOM_CAPSULE, (SAYA_RADIUS,), STEEL,
         fromto=np.concatenate([tip, hilt - 0.002 * axis]))
    _add(torso, "tsuba", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.0095, 0.0012), GOLD,
         fromto=np.concatenate([hilt - 0.0012 * axis, hilt + 0.0012 * axis]))
    _add(torso, "tsuka", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.0034,), WRAP,
         fromto=np.concatenate([hilt, pommel]))
    _add(torso, "kashira", mujoco.mjtGeom.mjGEOM_SPHERE, (0.0042,), GOLD, pos=pommel)

    for k, f in enumerate(STRAPS_AT):
        _strap(torso, f"strap{k}", tip + f * (hilt - tip), axis)


def _strap(torso, name, crossing, axis):
    """One strap across the dome over the scabbard, riveted above the band each side."""
    import mujoco

    over = SAYA_RADIUS + GAP + 0.001
    sword = axis[:2] / np.linalg.norm(axis[:2])
    for side in (-1, 1):
        path = []
        for h in np.arange(0.0, 0.2, 0.002):
            xy = crossing[:2] + [0.0, side * h]
            hit = _top(xy)
            if hit is None or hit[0][2] < RIVET_Z:
                break
            z = hit[0][2] + GAP + 0.001
            rel = xy - crossing[:2]
            off = abs(sword[0] * rel[1] - sword[1] * rel[0])
            if off < over:
                z = max(z, crossing[2] + np.sqrt(over**2 - off**2))
            path.append(np.array([*xy, z]))
        for i in range(len(path) - 1):
            normal = np.cross([1.0, 0.0, 0.0], path[i + 1] - path[i])
            _strip(torso, f"{name}{side}_{i}", path[i], path[i + 1], normal,
                   STRAP_HALF_WIDTH, 0.0007, LEATHER)
        _add(torso, f"{name}{side}_rivet", mujoco.mjtGeom.mjGEOM_SPHERE, (0.0022,), GOLD,
             pos=path[-1])
