"""Visualisation: interactive 3D viewer (py3Dmol), residue profile and PAE plots (matplotlib)."""
from __future__ import annotations

import numpy as np

from .geometry import Membrane, rotation_to
from .structures import Structure
from .topology import LABELS, Topology, segments

TOPO_COLORS = {"S": "#e6a23c", "I": "#4c78a8", "O": "#9ecae9", "M": "#d62728", "B": "#2ca02c", "P": "#b279a2"}
PLDDT_BINS = [(90, "#0053D6", "very high (>90)"), (70, "#65CBF3", "confident (70-90)"),
              (50, "#FFDB13", "low (50-70)"), (0, "#FF7D45", "very low (<50)")]


def plddt_color(v: float) -> str:
    return next(c for t, c, _ in PLDDT_BINS if v >= t)


def _runs(values: dict[int, str]) -> list[tuple[str, int, int]]:
    out, cur = [], None
    for p in sorted(values):
        c = values[p]
        if cur and cur[0] == c and p == cur[2] + 1:
            cur[2] = p
        else:
            if cur:
                out.append(tuple(cur))
            cur = [c, p, p]
    if cur:
        out.append(tuple(cur))
    return out


def _plane_pdb(ca_xyz: np.ndarray, half: float, spacing: float = 4.0, margin: float = 6.0) -> str:
    """Pseudo-atoms on the two membrane planes (y = -half cytoplasmic 'N', y = +half outer 'O')."""
    r = float(np.linalg.norm(ca_xyz[:, [0, 2]], axis=1).max()) + margin if len(ca_xyz) else 30.0
    grid = np.arange(-r, r + 1e-6, spacing)
    lines, n = [], 1
    for y, el in ((-half, "N"), (half, "O")):
        for x in grid:
            for z in grid:
                if x * x + z * z <= r * r:
                    lines.append(f"HETATM{n % 100000:5d}  {el:<3s} DUM M{n % 10000:4d}    "
                                 f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {el}")
                    n += 1
    return "\n".join(lines) + "\nEND\n"


def display_frame(st: Structure, mem: Membrane | None) -> tuple[Structure, Membrane | None]:
    """Rotate so the membrane normal points up the screen (y axis), mid-plane at y=0."""
    if mem is None:
        return st, None
    rot = rotation_to(mem.normal, np.array([0.0, 1.0, 0.0]))
    st2 = st.transformed(rot, mem.center)
    mem2 = Membrane(np.array([0.0, 1.0, 0.0]), np.zeros(3), mem.half_thickness, mem.tm_in_slab,
                    mem.loop_side_agreement, mem.loop_in_slab, mem.z)
    return st2, mem2


def viewer(st: Structure, topo: Topology, mem: Membrane | None, color_by: str = "topology",
           width: int = 800, height: int = 550, show_membrane: bool = True):
    """Interactive 3D view. color_by: topology | plddt | hydrophobicity."""
    import py3Dmol

    from .geometry import KYTE_DOOLITTLE

    st_d, mem_d = display_frame(st, mem)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(st_d.to_pdb(), "pdb")
    view.setStyle({"model": 0}, {"cartoon": {"color": "#bbbbbb"}})
    if color_by == "plddt" and st.plddt():
        runs = _runs({p: plddt_color(v) for p, v in st.plddt().items()})
    elif color_by == "hydrophobicity":
        cmap = lambda v: "#c0392b" if v > 1.5 else ("#f5b7b1" if v > 0 else ("#aed6f1" if v > -1.5 else "#2471a3"))
        runs = _runs({r.number: cmap(KYTE_DOOLITTLE.get(r.one, 0)) for r in st.residues})
    else:
        runs = [(TOPO_COLORS.get(l, "#888"), s, e) for l, s, e in segments(topo.labels)]
    for color, s, e in runs:
        view.setStyle({"model": 0, "resi": f"{s}-{e}"}, {"cartoon": {"color": color}})
    if mem_d is not None and show_membrane:
        _, ca_xyz = st_d.ca()
        view.addModel(_plane_pdb(ca_xyz, mem_d.half_thickness), "pdb")
        view.setStyle({"model": 1, "elem": "N"}, {"sphere": {"radius": 0.6, "color": "#4c78a8", "opacity": 0.55}})
        view.setStyle({"model": 1, "elem": "O"}, {"sphere": {"radius": 0.6, "color": "#e15759", "opacity": 0.55}})
        top = ca_xyz[:, 1].max() if len(ca_xyz) else 30
        bottom = ca_xyz[:, 1].min() if len(ca_xyz) else -30
        view.addLabel("outside", {"position": {"x": 0, "y": float(top) + 4, "z": 0}, "backgroundColor": "#e15759",
                                  "fontColor": "white", "fontSize": 12})
        view.addLabel("cytoplasm", {"position": {"x": 0, "y": float(bottom) - 4, "z": 0}, "backgroundColor": "#4c78a8",
                                    "fontColor": "white", "fontSize": 12})
    view.zoomTo({"model": 0})
    return view


def save_viewer_html(view, path) -> None:
    try:
        html = view.write_html() if hasattr(view, "write_html") else view._make_html()
    except TypeError:
        html = view._make_html()
    if html is None:  # some py3Dmol versions write to a file object instead of returning text
        import io
        buf = io.StringIO()
        view.write_html(buf)
        html = buf.getvalue()
    with open(path, "w") as fh:
        fh.write(html if "<html" in html else f"<!doctype html><html><head><meta charset='utf-8'></head><body>{html}</body></html>")


def profile_plot(topo: Topology, st: Structure | None, mem: Membrane | None, uniprot_features=None,
                 title: str = "", path=None):
    """Residue-level profile: topology tracks, pLDDT and depth along the membrane normal."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    L = len(topo.labels)
    has_up = bool(uniprot_features)
    nrows = 1 + (1 if has_up else 0) + (1 if st and st.plddt() else 0) + (1 if mem else 0)
    ratios = [0.5] + ([0.5] if has_up else []) + [2] * (nrows - 1 - (1 if has_up else 0))
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 1.2 + 1.3 * nrows), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_1d(axes)
    i = 0
    ax = axes[i]
    for lab, s, e in segments(topo.labels):
        ax.barh(0, e - s + 1, left=s - 0.5, color=TOPO_COLORS.get(lab, "#888"), height=0.8)
    ax.set_yticks([0], ["DeepTMHMM"])
    ax.set_title(title or f"{topo.id}  ({topo.type})", loc="left")
    i += 1
    if has_up:
        ax = axes[i]
        colors = {"Transmembrane": "#d62728", "Intramembrane": "#ff9896"}
        for f in uniprot_features:
            c = colors.get(f["type"])
            if c is None:
                desc = f.get("description", "").lower()
                c = "#4c78a8" if "cytoplasm" in desc else "#9ecae9"
            ax.barh(0, f["end"] - f["start"] + 1, left=f["start"] - 0.5, color=c, height=0.8)
        ax.set_yticks([0], ["UniProt"])
        i += 1
    if st and st.plddt():
        ax = axes[i]
        pl = st.plddt()
        xs = sorted(pl)
        for t, c, _ in PLDDT_BINS:
            ax.axhspan(t, {90: 100, 70: 90, 50: 70, 0: 50}[t], color=c, alpha=0.15, lw=0)
        ax.plot(xs, [pl[x] for x in xs], color="black", lw=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel("pLDDT")
        i += 1
    if mem:
        ax = axes[i]
        xs = sorted(mem.z)
        ax.axhspan(-mem.half_thickness, mem.half_thickness, color="#f2e394", alpha=0.6, lw=0, label="membrane")
        ax.plot(xs, [mem.z[x] for x in xs], color="#555", lw=1)
        tm = [x for x in xs if topo.labels[x - 1] in "MB"]
        ax.scatter(tm, [mem.z[x] for x in tm], s=6, color="#d62728", zorder=3)
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_ylabel("depth (Å)\n← cyto | out →")
    axes[-1].set_xlabel("residue")
    axes[-1].set_xlim(0.5, L + 0.5)
    fig.legend(handles=[Patch(color=c, label=LABELS[k]) for k, c in TOPO_COLORS.items()],
               loc="upper right", ncol=6, frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if path:
        fig.savefig(path, dpi=200)
    return fig


def pae_plot(st: Structure, topo: Topology, path=None):
    import matplotlib.pyplot as plt

    if st.pae is None:
        return None
    nums = st.meta.get("pae_numbers") or [r.number for r in st.residues]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    extent = (nums[0] - 0.5, nums[-1] + 0.5, nums[-1] + 0.5, nums[0] - 0.5)
    im = ax.imshow(st.pae, cmap="Greens_r", vmin=0, vmax=30, extent=extent)
    for lab, s, e in segments(topo.labels):
        if lab in "MB":
            for v in (s, e):
                ax.axhline(v, color="#d62728", lw=0.4, alpha=0.6)
                ax.axvline(v, color="#d62728", lw=0.4, alpha=0.6)
    fig.colorbar(im, ax=ax, label="expected position error (Å)")
    ax.set_xlabel("aligned residue")
    ax.set_ylabel("scored residue")
    ax.set_title(f"PAE — {topo.id} (TM segment edges in red)", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig
