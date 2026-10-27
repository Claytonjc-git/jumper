"""The robot dressed as a ninja, over any other scene -- for video.

Pick the world with `--ninja-base` (default `beach`); this scene keeps that
world exactly and adds a costume in its `decorate`: a dark suit in place of the
red and grey, a headband with two tails, glowing eye slits on the display and a
katana strapped to the back. `--ninja-style` picks the look: `classic` (indigo
suit, red headband) or `kintsugi` (black lacquer with gold-filled cracks).

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
STYLE_ENV = "MJRL_NINJA_STYLE"
ROBOT_XML = Path(__file__).resolve().parents[1] / "assets" / "jumper" / "jumper.xml"
SHELL_GEOMS = ("upper_shell_link_visual", "display_module_link_visual")

# The robot's own visual colours, as keys of a look's `suit`.
RED_PART, GREY_PART, PAD_PART = (0.85, 0.08, 0.05), (0.68, 0.7, 0.72), (0.816, 0.82, 0.804)
GOLD = (0.78, 0.60, 0.22, 1.0)

# A colour is an rgba, or the name of a material `dress` adds (drawn untinted).
# `suit` maps an old visual colour, or a visual geom's name, to its new colour.
LOOKS = {
    "classic": {
        "suit": {
            RED_PART: (0.07, 0.08, 0.11, 1.0),     # indigo-black
            GREY_PART: (0.20, 0.21, 0.24, 1.0),    # charcoal
            PAD_PART: (0.34, 0.35, 0.38, 1.0),     # gunmetal
        },
        "band": (0.80, 0.04, 0.04, 1.0),
        "eye": (0.85, 0.95, 1.0, 1.0),
        "saya": (0.10, 0.10, 0.11, 1.0),
        "wrap": (0.45, 0.05, 0.06, 1.0),
        "strap": (0.28, 0.16, 0.09, 1.0),
        "trim": GOLD,
    },
    "kintsugi": {
        "suit": {
            RED_PART: "ninja_kintsugi",
            GREY_PART: "ninja_kintsugi",
            "base_link_visual": "ninja_kintsugi",
            PAD_PART: "ninja_gold",
        },
        "band": (0.86, 0.66, 0.26, 1.0),
        "eye": (1.0, 0.80, 0.38, 1.0),
        "saya": "ninja_kintsugi",
        "wrap": (0.03, 0.03, 0.035, 1.0),
        "strap": (0.05, 0.05, 0.06, 1.0),
        "trim": "ninja_gold",
    },
}

# Kintsugi texture: black lacquer, gold veins. Fixed seed, so every video matches.
LACQUER, VEIN = (9, 9, 11), (226, 176, 72)
KINTSUGI_SIZE, KINTSUGI_SEED = 512, 7

# Headband height on the shell: over the display in front, dipping at the back.
BAND_Z_MID, BAND_Z_DIP = 0.051, 0.007
BAND_HALF_HEIGHT = 0.0045
BAND_SEGMENTS = 72
TAIL_HALF_WIDTH = 0.005
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
    """`--ninja-base` and `--ninja-style`; see `scenes.load_cli_args`."""
    group.add_argument(
        "--ninja-base", default=os.environ.get(BASE_ENV, "beach"), metavar="NAME",
        help="the registered scene the ninja performs in (default beach)",
    )
    group.add_argument(
        "--ninja-style", default=os.environ.get(STYLE_ENV, "classic"), choices=sorted(LOOKS),
        help="the costume: classic (indigo and red) or kintsugi (black and gold)",
    )
    return ("ninja_base", "ninja_style")


def scene(ninja_base: str = "beach", ninja_style: str = "classic") -> Scene:
    if ninja_base == "ninja":
        raise ValueError("--ninja-base must name another scene, not ninja itself")
    look = LOOKS[ninja_style]
    base = load(ninja_base)

    def decorate(spec):
        if base.decorate is not None:
            base.decorate(spec)
        dress(spec, look)

    return dataclasses.replace(base, decorate=decorate)


def dress(spec, look=LOOKS["classic"]) -> None:
    """Recolour the robot and add the costume. No-op on a world with no robot."""
    import mujoco

    torso = next((b for b in spec.bodies if b.name.split("/")[-1] == "base_link"), None)
    if torso is None:
        return
    # Geoms have no emission of their own; a glowing eye needs a material.
    spec.add_material(name="ninja_eye", rgba=look["eye"], emission=1.0)
    spec.add_material(name="ninja_gold", rgba=GOLD, specular=0.9, shininess=0.8,
                      reflectance=0.1)
    if "ninja_kintsugi" in (*look["suit"].values(), *look.values()):
        # MuJoCo projects a cube texture onto a mesh with no texture coordinates.
        faces = _kintsugi_faces()
        tex = spec.add_texture(name="ninja_kintsugi", type=mujoco.mjtTexture.mjTEXTURE_CUBE)
        tex.width, tex.height, tex.nchannel = faces.shape[1], faces.shape[0], 3
        tex.data = faces.tobytes()
        lacquer = spec.add_material(name="ninja_kintsugi", specular=0.7, shininess=0.9,
                                    reflectance=0.04, emission=0.15)
        lacquer.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ninja_kintsugi"

    suit = look["suit"]
    for geom in spec.geoms:
        if geom.name.endswith("_visual"):
            colour = suit.get(geom.name.split("/")[-1],
                              suit.get(tuple(round(float(c), 3) for c in geom.rgba[:3])))
            if colour is not None:
                _paint(geom, colour)
    _headband(spec, torso, look)
    _eyes(torso)
    _katana(spec, torso, look)


def _paint(geom, colour) -> None:
    if isinstance(colour, str):
        geom.material = colour
        geom.rgba = (0.5, 0.5, 0.5, 1.0)    # MuJoCo's default, which lets the material show
    else:
        geom.rgba = colour


@functools.lru_cache(maxsize=1)
def _kintsugi_faces() -> np.ndarray:
    """Six faces of branching gold cracks on black, stacked for a cube texture."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(KINTSUGI_SEED)
    big = 2 * KINTSUGI_SIZE    # drawn at twice the size, then reduced to smooth the edges
    faces = []
    for _ in range(6):
        img = Image.new("RGB", (big, big), LACQUER)
        draw = ImageDraw.Draw(img)
        # A few long cracks, each entering from an edge and heading across.
        cracks = []
        for _ in range(2):
            start = rng.uniform(0, big, 2)
            start[rng.integers(2)] = rng.choice([0.0, big - 1.0])
            inward = np.arctan2(*(big / 2 - start)[::-1]) + rng.uniform(-0.6, 0.6)
            cracks.append((start, inward, 20.0, 140))
        while cracks:
            p, heading, width, steps = cracks.pop()
            shade = rng.uniform(0.85, 1.1)
            vein = tuple(int(min(255, c * shade)) for c in VEIN)
            for _ in range(steps):
                heading += rng.normal(0.0, 0.25)
                q = p + rng.uniform(8, 20) * np.array([np.cos(heading), np.sin(heading)])
                w = max(1, round(width))
                draw.line([tuple(p), tuple(q)], fill=vein, width=w)
                draw.ellipse([*(q - w / 2), *(q + w / 2)], fill=vein)
                p, width = q, width * 0.994
                if not (0 <= p[0] < big and 0 <= p[1] < big):
                    break
                if width > 6 and rng.random() < 0.02:
                    turn = rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
                    cracks.append((p.copy(), heading + turn, width * 0.6, steps // 3))
        faces.append(np.asarray(img.resize((KINTSUGI_SIZE, KINTSUGI_SIZE), Image.LANCZOS)))
    return np.ascontiguousarray(np.vstack(faces))


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


def _add(body, name, gtype, size, rgba, *, pos=None, mat=None, fromto=None, mesh=None):
    import mujoco

    g = body.add_geom()
    g.name = f"ninja_{name}"
    g.type = gtype
    g.size = list(size) + [0.0] * (3 - len(size))
    _paint(g, rgba)
    g.contype = g.conaffinity = 0
    g.density = 0.0
    g.group = 2
    if mesh is not None:
        g.meshname = mesh
    elif fromto is not None:
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


def _ribbon(spec, body, name, points, normals, half_width, half_thick, colour, closed=False):
    """A flat band through `points`, its face along `normals`, as one seamless mesh."""
    import mujoco

    p = np.asarray(points, float)
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0) if closed else np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.asarray(normals, float)
    n -= t * np.sum(n * t, axis=1, keepdims=True)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    w = np.cross(t, n)

    def corner(i, a, b):
        return p[i] + a * half_width * w[i] + b * half_thick * n[i]

    # Each side gets its own vertices, so the edges stay sharp.
    square = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    verts, faces, outward = [], [], []
    for k in range(4):
        (a0, b0), (a1, b1) = square[k], square[(k + 1) % 4]
        base = len(verts)
        for i in range(len(p)):
            verts += [corner(i, a0, b0), corner(i, a1, b1)]
        for i in range(len(p) if closed else len(p) - 1):
            j = (i + 1) % len(p)
            faces += [(base + 2 * i, base + 2 * j, base + 2 * i + 1),
                      (base + 2 * i + 1, base + 2 * j, base + 2 * j + 1)]
            out = (a0 + a1) * w[i] + (b0 + b1) * n[i]
            outward += [out, out]
    if not closed:
        for i, out in ((0, -t[0]), (len(p) - 1, t[-1])):
            base = len(verts)
            verts += [corner(i, a, b) for a, b in square]
            faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            outward += [out, out]
    v, f = np.array(verts), np.array(faces)
    # Wind every triangle counter-clockwise seen from outside.
    facing = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = np.sum(facing * np.array(outward), axis=1) < 0
    f[flip] = f[flip][:, ::-1]

    mesh = spec.add_mesh(name=f"ninja_{name}")
    mesh.uservert = v.flatten().tolist()
    mesh.userface = f.flatten().tolist()
    return _add(body, name, mujoco.mjtGeom.mjGEOM_MESH, (), colour, mesh=f"ninja_{name}")


def _smooth(a, k=2):
    return np.mean([np.roll(a, s, axis=0) for s in range(-k, k + 1)], axis=0)


def _headband(spec, torso, look):
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
    _ribbon(spec, torso, "band", points, normals, BAND_HALF_HEIGHT, 0.0012, look["band"],
            closed=True)

    import mujoco

    back = BAND_SEGMENTS // 2
    knot = points[back] + 0.004 * normals[back]
    _add(torso, "knot", mujoco.mjtGeom.mjGEOM_ELLIPSOID, (0.005, 0.010, 0.006), look["band"],
         pos=knot)
    # The tails leave the knot side by side, edges apart, and fan out as they drop.
    s = np.linspace(0.0, 1.0, 16)[:, None]
    for side in (-1, 1):
        start = knot + [-0.002, side * (TAIL_HALF_WIDTH + 0.0008), 0.0]
        bend = start + [-0.042, side * 0.008, -0.010]
        end = bend + [-0.040, side * 0.016, -0.024]
        path = (1 - s) ** 2 * start + 2 * s * (1 - s) * bend + s**2 * end
        _ribbon(spec, torso, f"tail{side}", path, np.tile([0.0, 0.0, 1.0], (len(path), 1)),
                TAIL_HALF_WIDTH, 0.0010, look["band"])


def _eyes(torso):
    """Two slanted slits on the display, inner ends low."""
    normal = np.array([np.cos(DISPLAY_TILT), 0.0, np.sin(DISPLAY_TILT)])
    along = np.array([-np.sin(DISPLAY_TILT), 0.0, np.cos(DISPLAY_TILT)])
    centre = np.array([0.0838, 0.0, 0.045]) + 0.0012 * normal
    for side in (-1, 1):
        mid = centre + [0.0, side * 0.019, 0.0]
        inner = mid - [0.0, side * 0.010, 0.0] - 0.0018 * along
        outer = mid + [0.0, side * 0.010, 0.0] + 0.0018 * along
        _strip(torso, f"eye{side}", inner, outer, normal, 0.0022, 0.0006, "ninja_eye")


def _katana(spec, torso, look):
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
    _add(torso, "saya", mujoco.mjtGeom.mjGEOM_CAPSULE, (SAYA_RADIUS,), look["saya"],
         fromto=np.concatenate([tip, hilt - 0.002 * axis]))
    _add(torso, "tsuba", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.0095, 0.0012), look["trim"],
         fromto=np.concatenate([hilt - 0.0012 * axis, hilt + 0.0012 * axis]))
    _add(torso, "tsuka", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.0034,), look["wrap"],
         fromto=np.concatenate([hilt, pommel]))
    _add(torso, "kashira", mujoco.mjtGeom.mjGEOM_SPHERE, (0.0042,), look["trim"],
         pos=pommel)

    for k, f in enumerate(STRAPS_AT):
        _strap(spec, torso, look, f"strap{k}", tip + f * (hilt - tip), axis)


def _strap(spec, torso, look, name, crossing, axis):
    """One strap across the dome over the scabbard, riveted above the band each side."""
    import mujoco

    over = SAYA_RADIUS + GAP + 0.001
    sword = axis[:2] / np.linalg.norm(axis[:2])
    paths = []
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
        paths.append(path)
        _add(torso, f"{name}{side}_rivet", mujoco.mjtGeom.mjGEOM_SPHERE, (0.0022,), look["trim"],
             pos=path[-1])
    path = np.array(paths[0][::-1] + paths[1][1:])
    normals = np.cross([1.0, 0.0, 0.0], np.gradient(path, axis=0))
    _ribbon(spec, torso, name, path, normals, STRAP_HALF_WIDTH, 0.0007, look["strap"])
