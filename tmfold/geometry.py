"""Place the membrane on a 3D structure using the DeepTMHMM topology, and score their agreement.

Method
------
1. Each TM helix / beta strand gives an axis (first principal component of its CA atoms),
   oriented from its cytoplasmic end to its non-cytoplasmic end using the flanking labels.
   Their length-weighted mean is the starting membrane normal.
2. A coarse-then-fine grid search over normals within 30 degrees of the start (and its opposite),
   mid-plane offsets and half-thicknesses maximises:  fraction of TM residues inside the slab
   - 0.5 x fraction of loop residues inside the slab + 0.5 x fraction of loops on their predicted side
   - 0.1 x angle (radians) from the mean segment axis.
3. The normal points from the inside (cytoplasm) to the outside, so z < 0 is cytoplasmic.

This gives a membrane-oriented model (like OPM/PPM, but driven by the sequence-based topology)
plus quantitative structure-topology agreement scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .topology import INSIDE_LABELS, MEMBRANE_LABELS, OUTSIDE_LABELS, segments

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


@dataclass
class Membrane:
    normal: np.ndarray            # unit vector, inside -> outside
    center: np.ndarray            # point on the mid-plane
    half_thickness: float         # Å
    tm_in_slab: float             # fraction of TM residues within the slab
    loop_side_agreement: float    # fraction of inside/outside residues on their predicted side
    loop_in_slab: float
    z: dict[int, float] = field(default_factory=dict)   # residue -> depth along the normal (Å)

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and shift that put the mid-plane at z=0 with the normal along +z."""
        return rotation_to(self.normal, np.array([0.0, 0.0, 1.0])), self.center


def rotation_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(a, p)
        u /= np.linalg.norm(u)
        return 2 * np.outer(u, u) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def segment_axis(xyz: np.ndarray) -> np.ndarray:
    """Principal axis of a set of CA coordinates, pointing from the first to the last residue."""
    centred = xyz - xyz.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    if np.dot(xyz[-1] - xyz[0], axis) < 0:
        axis = -axis
    return axis


def _side(label: str) -> int:
    return -1 if label in INSIDE_LABELS else (1 if label in OUTSIDE_LABELS else 0)


def _cap_normals(center: np.ndarray, max_deg: float = 30, step_deg: float = 5, az_step: float = 30) -> np.ndarray:
    rot = rotation_to(np.array([0.0, 0, 1]), center)
    out = [center]
    for t in np.arange(step_deg, max_deg + 1e-6, step_deg):
        tr = np.radians(t)
        for ph in np.radians(np.arange(0, 360, az_step)):
            out.append(rot @ np.array([np.sin(tr) * np.cos(ph), np.sin(tr) * np.sin(ph), np.cos(tr)]))
    return np.array(out)


def fit_membrane(labels: str, ca: dict[int, np.ndarray]) -> Membrane | None:
    """Fit a membrane slab to CA coordinates (keyed by 1-based query position)."""
    segs = [(l, s, e) for l, s, e in segments(labels) if l in MEMBRANE_LABELS]
    axes, weights = [], []
    for lab, s, e in segs:
        pos = [p for p in range(s, e + 1) if p in ca]
        if len(pos) < 4:
            continue
        axis = segment_axis(np.array([ca[p] for p in pos]))
        before = _side(labels[s - 2]) if s > 1 else 0
        after = _side(labels[e]) if e < len(labels) else 0
        if before == -1 or after == 1:
            sign = 1          # runs inside -> outside
        elif before == 1 or after == -1:
            sign = -1
        else:
            sign = 0
        axes.append((axis, sign))
        weights.append(len(pos))
    if not axes:
        return None
    if any(sg for _, sg in axes):
        n0 = sum(w * sg * a for (a, sg), w in zip(axes, weights))
    else:  # no orientation info: align everything to the first axis
        ref = axes[0][0]
        n0 = sum(w * (a if np.dot(a, ref) >= 0 else -a) for (a, _), w in zip(axes, weights))
    if np.linalg.norm(n0) < 1e-6:  # e.g. a beta barrel whose strand signs cancel
        n0 = axes[0][0]
    n0 = n0 / np.linalg.norm(n0)

    positions = np.array(sorted(ca))
    X = np.array([ca[p] for p in positions])
    lab = np.array([labels[p - 1] for p in positions])
    is_tm = np.isin(lab, list(MEMBRANE_LABELS))
    side = np.array([_side(l) for l in lab])
    is_loop = side != 0
    c0 = X[is_tm].mean(0)

    offsets = np.arange(-8, 8.01, 1.0)
    halves = np.arange(12, 20.01, 1.0)

    def search(normals, best=None):
        for n in normals:
            z0 = (X - c0) @ n
            z = z0[None, None, :] - offsets[:, None, None]                    # (O, 1, N)
            in_slab = np.abs(z) <= halves[None, :, None]                      # (O, H, N)
            tm_in = in_slab[..., is_tm].mean(-1)
            loop_in = in_slab[..., is_loop].mean(-1) if is_loop.any() else np.zeros_like(tm_in)
            side_ok = (np.sign(z[..., is_loop]) == side[is_loop]).mean(-1) if is_loop.any() \
                else np.zeros((len(offsets), 1))
            # mild prior towards the mean segment axis (0.1 per radian): resolves plateaus and stops a
            # few loop residues from dragging the membrane off the helix bundle axis
            prior = -0.1 * float(np.arccos(np.clip(abs(np.dot(n, n0)), -1, 1)))
            score = tm_in - 0.5 * loop_in + 0.5 * side_ok + prior
            k = np.unravel_index(np.argmax(score), score.shape)
            if best is None or score[k] > best[0] + 1e-9:
                best = (score[k], n, offsets[k[0]], halves[k[1]], tm_in[k], loop_in[k], side_ok[k[0], 0])
        return best

    best = search(np.vstack([_cap_normals(n0), _cap_normals(-n0)]))                 # coarse: 5° steps to 30°
    best = search(_cap_normals(best[1], max_deg=5, step_deg=1, az_step=45), best)    # fine: 1° steps to 5°
    _, n, off, half, tm_in, loop_in, side_ok = best
    center = c0 + off * n
    zs = (X - center) @ n
    return Membrane(n, center, float(half), float(tm_in), float(side_ok), float(loop_in),
                    {int(p): float(v) for p, v in zip(positions, zs)})


def segment_table(labels: str, sequence: str, ca: dict[int, np.ndarray], mem: Membrane | None,
                  plddt: dict[int, float] | None) -> list[dict]:
    """Per-segment metrics: length, hydrophobicity, pLDDT, tilt and slab occupancy, with QC flags."""
    rows = []
    for lab, s, e in segments(labels):
        row = {"label": lab, "start": s, "end": e, "length": e - s + 1,
               "hydrophobicity_KD": round(float(np.mean([KYTE_DOOLITTLE.get(c, 0) for c in sequence[s - 1:e]])), 2)}
        if plddt:
            vals = [plddt[p] for p in range(s, e + 1) if p in plddt]
            row["mean_pLDDT"] = round(float(np.mean(vals)), 1) if vals else None
        pos = [p for p in range(s, e + 1) if p in ca]
        if mem is not None and pos:
            row["in_slab"] = round(float(np.mean([abs(mem.z[p]) <= mem.half_thickness for p in pos])), 2)
            row["mean_depth"] = round(float(np.mean([mem.z[p] for p in pos])), 1)
            if lab in MEMBRANE_LABELS and len(pos) >= 4:
                axis = segment_axis(np.array([ca[p] for p in pos]))
                row["tilt_deg"] = round(float(np.degrees(np.arccos(min(1, abs(np.dot(axis, mem.normal)))))), 1)
        flags = []
        if lab in MEMBRANE_LABELS:
            if row.get("mean_pLDDT") is not None and row["mean_pLDDT"] < 70:
                flags.append("low confidence")
            if row.get("in_slab", 1) < 0.6:
                flags.append("outside membrane")
            if lab == "M" and row["length"] < 14:
                flags.append("short helix")
            if row.get("tilt_deg", 0) > 45:
                flags.append("highly tilted")
        elif lab in INSIDE_LABELS | OUTSIDE_LABELS and mem is not None and "mean_depth" in row:
            if np.sign(row["mean_depth"]) != _side(lab) and abs(row["mean_depth"]) > mem.half_thickness:
                flags.append("wrong side")
        row["flags"] = ", ".join(flags)
        rows.append(row)
    return rows


def positive_inside(labels: str, sequence: str, flank: int = 15) -> dict:
    """von Heijne positive-inside check: K+R in loop residues flanking TM helices, by side."""
    counts = {"inside": 0, "outside": 0}
    tm_pos = [i for i, l in enumerate(labels) if l in MEMBRANE_LABELS]
    if not tm_pos:
        return {"KR_inside": 0, "KR_outside": 0, "positive_inside_ok": None}
    near = np.zeros(len(labels), bool)
    for i in tm_pos:
        near[max(0, i - flank):i + flank + 1] = True
    for i, (l, aa) in enumerate(zip(labels, sequence)):
        if near[i] and aa in "KR":
            if l in INSIDE_LABELS:
                counts["inside"] += 1
            elif l in OUTSIDE_LABELS:
                counts["outside"] += 1
    return {"KR_inside": counts["inside"], "KR_outside": counts["outside"],
            "positive_inside_ok": counts["inside"] >= counts["outside"]}


def membrane_dummy_atoms(mem: Membrane, ca_xyz: np.ndarray, spacing: float = 4.0, margin: float = 6.0,
                         oriented: bool = True) -> list[str]:
    """HETATM DUM atoms on both membrane planes (OPM convention), for PyMOL/ChimeraX/py3Dmol."""
    rot, shift = mem.frame()
    local = (ca_xyz - shift) @ rot.T
    r = np.linalg.norm(local[:, :2], axis=1).max() + margin if len(local) else 30
    grid = np.arange(-r, r + 1e-6, spacing)
    lines, serial = [], 1
    for sign, name in ((-1, "N"), (1, "O")):  # N = cytoplasmic plane, O = outer plane
        for x in grid:
            for y in grid:
                if x * x + y * y > r * r:
                    continue
                p = np.array([x, y, sign * mem.half_thickness])
                if not oriented:
                    p = rot.T @ p + shift
                lines.append(f"HETATM{serial % 100000:5d}  {name:<3s} DUM M{serial % 10000:4d}    "
                             f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           {name}")
                serial += 1
    return lines
