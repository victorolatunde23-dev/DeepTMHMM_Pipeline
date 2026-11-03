"""File outputs: membrane-oriented models, PyMOL/ChimeraX scripts and an HTML report."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from .geometry import Membrane, membrane_dummy_atoms
from .structures import Structure
from .topology import Topology, segments
from .viz import TOPO_COLORS

TOPO_CODE = {"I": 1, "O": 2, "M": 3, "B": 4, "S": 5, "P": 6}


def write_models(st: Structure, topo: Topology, mem: Membrane | None, outdir: Path) -> dict[str, Path]:
    """Membrane-oriented PDB (B = pLDDT, with DUM membrane atoms) and a topology-coded PDB."""
    files = {}
    if mem is not None:
        rot, shift = mem.frame()
        st_o = st.transformed(rot, shift)
        _, ca = st_o.ca()
        dum = membrane_dummy_atoms(mem, st.ca()[1], oriented=True)
        remark = [f"REMARK   1 tmfold membrane: normal along +z (cytoplasm at z<0), half-thickness "
                  f"{mem.half_thickness:.1f} A, TM in slab {mem.tm_in_slab:.2f}, loop side agreement "
                  f"{mem.loop_side_agreement:.2f}"]
    else:
        st_o, dum, remark = st, [], ["REMARK   1 tmfold: no membrane fitted (no TM segments in structure)"]
    p = outdir / "model_membrane_oriented.pdb"
    p.write_text("\n".join(remark) + "\n" + st_o.to_pdb(extra_hetatm=dum))
    files["oriented_pdb"] = p
    codes = {i + 1: float(TOPO_CODE.get(l, 0)) for i, l in enumerate(topo.labels)}
    p = outdir / "model_topology_bfactor.pdb"
    p.write_text("REMARK   1 B-factor = DeepTMHMM label: 1 inside 2 outside 3 TM helix 4 beta strand 5 signal 6 periplasm\n"
                 + st_o.to_pdb(b_override=codes, extra_hetatm=dum))
    files["topology_pdb"] = p
    files.update(write_scripts(topo, outdir))
    return files


def write_scripts(topo: Topology, outdir: Path) -> dict[str, Path]:
    pml = ["load model_membrane_oriented.pdb, model", "hide everything", "show cartoon, model and polymer",
           "show spheres, resn DUM", "set sphere_scale, 0.4, resn DUM", "set sphere_transparency, 0.5",
           "color skyblue, resn DUM and name N", "color salmon, resn DUM and name O"]
    cxc = ["open model_membrane_oriented.pdb", "cartoon", "hide atoms", "show :DUM atoms", "style :DUM sphere",
           "size :DUM atomRadius 0.6", "transparency :DUM 50"]
    for lab, s, e in segments(topo.labels):
        c = TOPO_COLORS.get(lab, "#888888")
        pml.append(f"set_color c_{lab}, [{int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}]")
        pml.append(f"color c_{lab}, model and polymer and resi {s}-{e}")
        cxc.append(f"color :{s}-{e} {c} target c")
    pml += ["orient model", "turn x, -90"]
    cxc += ["view", "turn x -90"]
    f1, f2 = outdir / "color_topology.pml", outdir / "color_topology.cxc"
    f1.write_text("\n".join(pml) + "\n")
    f2.write_text("\n".join(cxc) + "\n")
    return {"pymol": f1, "chimerax": f2}


def _img(path: Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def write_report(summary: pd.DataFrame, results: dict, outdir: Path) -> Path:
    """Self-contained HTML report: summary table plus a section per protein."""
    cols = [c for c in ["id", "description", "type", "length", "n_TM_helices", "n_beta_strands", "structure_source",
                        "mean_pLDDT_TM", "tm_in_slab", "loop_side_agreement", "agreement", "positive_inside_ok",
                        "uniprot_TM_overlap", "flags"] if c in summary]
    table = summary[cols].to_html(index=False, escape=True, na_rep="", float_format=lambda x: f"{x:.2f}")
    parts = []
    for pid, r in results.items():
        files = r.get("files", {})
        seg = r.get("segments")
        seg_html = pd.DataFrame(seg).to_html(index=False, na_rep="") if seg else ""
        viewer = files.get("viewer")
        link = f'<p><a href="{html.escape(str(Path(viewer).relative_to(outdir)))}">Open interactive 3D view</a></p>' \
            if viewer else ""
        parts.append(f"<section><h2 id='{html.escape(pid)}'>{html.escape(pid)}</h2>"
                     f"<p>{html.escape(r.get('description', ''))}</p>{link}"
                     f"{_img(files.get('profile'))}{_img(files.get('pae'))}{seg_html}</section>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>tmfold report</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;max-width:1200px}} table{{border-collapse:collapse;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 6px}} th{{background:#f4f4f4}} section{{margin-top:32px}}
</style></head><body><h1>tmfold report</h1>
<p>DeepTMHMM topology mapped onto 3D structures. Agreement scores: <b>tm_in_slab</b> = fraction of predicted
membrane residues inside the fitted membrane; <b>loop_side_agreement</b> = fraction of loop residues on their predicted side.</p>
{table}{''.join(parts)}</body></html>"""
    p = outdir / "report.html"
    p.write_text(page)
    return p
