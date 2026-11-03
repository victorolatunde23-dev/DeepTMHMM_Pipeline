"""End-to-end pipeline: topology -> structure -> membrane placement -> QC -> outputs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import structures as S
from .export import write_models, write_report
from .geometry import fit_membrane, positive_inside, segment_table
from .sequences import Protein, accessions_for_sequence, uniprot_membrane_features
from .topology import MEMBRANE_LABELS, Topology, merged_outputs, run_deeptmhmm


@dataclass
class Config:
    structure_source: str = "auto"   # auto (user > AFDB > predict) | afdb | predict | none
    predictor: str = "esmfold"       # esmfold (GPU, falls back to ESM Atlas API) | esmatlas | colabfold | none
    include_non_tm: bool = False     # also fetch structures for proteins with no TM segments
    compare_uniprot: bool = True     # compare with UniProt transmembrane annotations
    deeptmhmm_batch: int = 200
    make_viewer_html: bool = True
    make_plots: bool = True
    user_structures: dict = field(default_factory=dict)
    # user_structures[protein_id] = {"kind": "af3" | "file" | "pdb", "value": path or PDB ID, "chain": "A"}


# --------------------------------------------------------------------------- structure cache

def _save_cache(st: S.Structure, d: Path) -> None:
    (d / "structure.pdb").write_text(st.to_pdb())
    meta = {k: v for k, v in st.meta.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    meta.update(source=st.source, b_is_plddt=st.b_is_plddt)
    (d / "structure_meta.json").write_text(json.dumps(meta, indent=1))
    if st.pae is not None:
        np.save(d / "pae.npy", st.pae)


def _load_cache(d: Path) -> S.Structure | None:
    if not (d / "structure.pdb").exists() or not (d / "structure_meta.json").exists():
        return None
    meta = json.loads((d / "structure_meta.json").read_text())
    residues, _ = S._read_pdb_text((d / "structure.pdb").read_text(), "A")
    st = S.Structure(residues, meta.pop("source"), "A", meta.pop("b_is_plddt"), meta=meta)
    if (d / "pae.npy").exists():
        st.pae = np.load(d / "pae.npy")
    return st


def get_structure(p: Protein, cfg: Config, d: Path) -> S.Structure | None:
    cached = _load_cache(d)
    if cached is not None:
        return cached
    st = None
    user = cfg.user_structures.get(p.id)
    try:
        if user:
            kind, value, chain = user["kind"], user["value"], user.get("chain")
            st = (S.load_af3(value, d, chain) if kind == "af3" else
                  S.fetch_rcsb(value, d, chain) if kind == "pdb" else
                  S.read_structure(value, chain, source="user file", b_is_plddt=user.get("b_is_plddt", False)))
        if st is None and cfg.structure_source in ("auto", "afdb"):
            accs = [p.accession] if p.accession else accessions_for_sequence(p.sequence)
            for acc in accs[:5]:
                st = S.fetch_afdb(acc, d, p.sequence)
                if st is not None:
                    break
        if st is None and cfg.structure_source in ("auto", "predict") and cfg.predictor != "none":
            if cfg.predictor == "colabfold":
                st = S.predict_colabfold(p.id, p.sequence, d)
            elif cfg.predictor == "esmfold" and S.gpu_available():
                st = S.predict_esmfold_local(p.id, p.sequence, d)
            else:
                st = S.predict_esmfold_api(p.id, p.sequence, d)
    except Exception as e:
        print(f"  ! {p.id}: structure step failed: {e}")
        st = None
    if st is None:
        return None
    st = S.renumber_to_query(st, p.sequence)
    if st.meta.get("coverage", 1) < 0.5:
        print(f"  ! {p.id}: structure covers only {st.meta['coverage']:.0%} of the sequence")
    _save_cache(st, d)
    return st


# --------------------------------------------------------------------------- analysis

def _uniprot_overlap(topo: Topology, feats: list[dict]) -> float | None:
    up = set()
    for f in feats:
        if f["type"] in ("Transmembrane", "Intramembrane"):
            up.update(range(f["start"], f["end"] + 1))
    pred = {i + 1 for i, l in enumerate(topo.labels) if l in MEMBRANE_LABELS}
    if not up and not pred:
        return None
    return round(len(up & pred) / len(up | pred), 3)


def _agreement(mem) -> str:
    if mem is None:
        return ""
    if mem.tm_in_slab >= 0.8 and mem.loop_side_agreement >= 0.8:
        return "consistent"
    if mem.tm_in_slab >= 0.6 and mem.loop_side_agreement >= 0.6:
        return "partly consistent"
    return "inconsistent"


def analyse(p: Protein, topo: Topology, cfg: Config, outdir: Path) -> tuple[dict, dict]:
    d = outdir / "proteins" / p.id
    d.mkdir(parents=True, exist_ok=True)
    segs = topo.segments()
    sp = next((s for s in segs if s[0] == "S"), None)
    first = topo.labels.lstrip("S")[:1]
    row = {"id": p.id, "accession": p.accession or "", "description": p.description, "type": topo.type,
           "length": len(p.sequence),
           "n_TM_helices": sum(1 for s in segs if s[0] == "M"),
           "n_beta_strands": sum(1 for s in segs if s[0] == "B"),
           "signal_peptide": f"{sp[1]}-{sp[2]}" if sp else "",
           "N_terminus": {"I": "inside", "O": "outside", "P": "inside"}.get(first, ""),
           "TM_regions": ";".join(f"{a}-{b}" for l, a, b in segs if l in MEMBRANE_LABELS)}
    row.update(positive_inside(topo.labels, p.sequence))
    res = {"topology": topo, "description": p.description, "files": {}}

    feats = []
    if cfg.compare_uniprot and p.accession:
        try:
            feats = uniprot_membrane_features(p.accession)
        except Exception:
            feats = []
        row["uniprot_TM_count"] = sum(1 for f in feats if f["type"] == "Transmembrane")
        row["uniprot_TM_overlap"] = _uniprot_overlap(topo, feats)

    st = mem = None
    if cfg.structure_source != "none" and (topo.is_membrane or cfg.include_non_tm):
        st = get_structure(p, cfg, d)
    if st is not None:
        ca_nums, ca_xyz = st.ca()
        ca = dict(zip(ca_nums.tolist(), ca_xyz))
        mem = fit_membrane(topo.labels, ca) if topo.is_membrane else None
        pl = st.plddt()
        seg_rows = segment_table(topo.labels, p.sequence, ca, mem, pl)
        res["segments"] = seg_rows
        pd.DataFrame(seg_rows).to_csv(d / "segments.csv", index=False)
        tm_pl = [pl[i] for i in pl if topo.labels[i - 1] in MEMBRANE_LABELS] if pl else []
        row.update({
            "structure_source": st.source,
            "afdb_version": st.meta.get("afdb_version", ""),
            "structure_coverage": round(st.meta.get("coverage", 1), 3),
            "mean_pLDDT": round(float(np.mean(list(pl.values()))), 1) if pl else None,
            "mean_pLDDT_TM": round(float(np.mean(tm_pl)), 1) if tm_pl else None,
        })
        if mem is not None:
            row.update({"tm_in_slab": round(mem.tm_in_slab, 3), "loop_side_agreement": round(mem.loop_side_agreement, 3),
                        "membrane_thickness_A": round(2 * mem.half_thickness, 1), "agreement": _agreement(mem)})
        row["flags"] = "; ".join(f"{r['label']}{r['start']}-{r['end']}: {r['flags']}" for r in seg_rows if r["flags"])
        res["files"].update(write_models(st, topo, mem, d))
        if cfg.make_viewer_html:
            try:
                from .viz import save_viewer_html, viewer
                res["files"]["viewer"] = d / "viewer.html"
                save_viewer_html(viewer(st, topo, mem), res["files"]["viewer"])
            except Exception as e:
                res["files"].pop("viewer", None)
                print(f"  ! {p.id}: viewer export skipped ({e})")
    if cfg.make_plots:
        import matplotlib.pyplot as plt

        from .viz import pae_plot, profile_plot
        fig = profile_plot(topo, st, mem, feats, title=f"{p.id}  {p.description[:80]}", path=d / "profile.png")
        plt.close(fig)
        res["files"]["profile"] = d / "profile.png"
        if st is not None and st.pae is not None:
            fig = pae_plot(st, topo, path=d / "pae.png")
            plt.close(fig)
            res["files"]["pae"] = d / "pae.png"
    res.update(structure=st, membrane=mem, uniprot_features=feats)
    return row, res


def run(proteins: list[Protein], outdir: str | Path, cfg: Config | None = None,
        topologies: dict[str, Topology] | None = None) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline. Returns (summary table, per-protein results)."""
    cfg = cfg or Config()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if topologies is None:
        topologies = run_deeptmhmm(proteins, outdir, cfg.deeptmhmm_batch)
        merged_outputs(outdir, outdir / "deeptmhmm_merged")
    rows, results = [], {}
    for i, p in enumerate(proteins, 1):
        topo = topologies.get(p.id)
        if topo is None:
            print(f"  ! {p.id}: no DeepTMHMM result")
            continue
        print(f"[{i}/{len(proteins)}] {p.id} ({topo.type})")
        row, res = analyse(p, topo, cfg, outdir)
        rows.append(row)
        results[p.id] = res
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    write_report(summary, results, outdir)
    return summary, results
