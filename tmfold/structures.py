"""Structure acquisition and handling.

Sources, in the order the pipeline tries them by default:
  * user files: an AlphaFold 3 / AlphaFold Server result (zip or folder), a PDB/mmCIF file, or a PDB ID
  * AlphaFold Protein Structure Database (current release, via its API), matched by exact sequence
  * on-the-fly prediction: ColabFold (AlphaFold2), ESMFold on the Colab GPU, or the ESM Atlas API

Every structure is converted to a single protein chain renumbered to the query sequence, so
residue N in the structure is residue N of the DeepTMHMM topology string.
"""
from __future__ import annotations

import glob
import io
import json
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .align import map_positions
from .sequences import http_get

AFDB = "https://alphafold.ebi.ac.uk"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}


@dataclass
class Atom:
    name: str
    element: str
    xyz: np.ndarray
    b: float


@dataclass
class Residue:
    number: int
    name: str
    atoms: list[Atom]

    @property
    def one(self) -> str:
        return THREE_TO_ONE.get(self.name, "X")

    def atom(self, name: str) -> Atom | None:
        return next((a for a in self.atoms if a.name == name), None)


@dataclass
class Structure:
    residues: list[Residue]
    source: str
    chain: str = "A"
    b_is_plddt: bool = True          # predicted models store pLDDT (0-100) in the B-factor column
    pae: np.ndarray | None = None    # residue x residue predicted aligned error (Å)
    meta: dict = field(default_factory=dict)

    @property
    def sequence(self) -> str:
        return "".join(r.one for r in self.residues)

    def ca(self) -> tuple[np.ndarray, np.ndarray]:
        """(residue numbers, CA coordinates) for residues that have a CA atom."""
        nums, xyz = [], []
        for r in self.residues:
            a = r.atom("CA")
            if a is not None:
                nums.append(r.number)
                xyz.append(a.xyz)
        return np.array(nums, int), np.array(xyz, float).reshape(-1, 3)

    def per_residue_b(self) -> dict[int, float]:
        out = {}
        for r in self.residues:
            a = r.atom("CA") or (r.atoms[0] if r.atoms else None)
            if a is not None:
                out[r.number] = a.b
        return out

    def plddt(self) -> dict[int, float] | None:
        return self.per_residue_b() if self.b_is_plddt else None

    def transformed(self, rot: np.ndarray, shift: np.ndarray) -> "Structure":
        """Copy with coordinates mapped x -> rot @ (x - shift)."""
        res = [Residue(r.number, r.name, [Atom(a.name, a.element, rot @ (a.xyz - shift), a.b) for a in r.atoms])
               for r in self.residues]
        return Structure(res, self.source, self.chain, self.b_is_plddt, self.pae, dict(self.meta))

    def to_pdb(self, b_override: dict[int, float] | None = None, extra_hetatm: list[str] | None = None) -> str:
        lines, serial = [], 1
        for r in self.residues:
            for a in r.atoms:
                b = b_override.get(r.number, 0.0) if b_override is not None else a.b
                name = a.name if len(a.name) == 4 else f" {a.name:<3}"
                x, y, z = a.xyz
                lines.append(f"ATOM  {serial % 100000:5d} {name:4s} {r.name:>3s} {self.chain[:1]}{r.number % 10000:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{b:6.2f}          {a.element:>2s}")
                serial += 1
        lines.append("TER")
        lines += extra_hetatm or []
        lines.append("END")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- reading

def _read_pdb_text(text: str, chain: str | None) -> tuple[list[Residue], str]:
    """Minimal PDB ATOM parser (first model only) used when gemmi is unavailable."""
    chains: dict[str, dict[tuple, Residue]] = {}
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        if resname not in THREE_TO_ONE:
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        ch, num, icode = line[21], int(line[22:26]), line[26]
        el = line[76:78].strip() or line[12:16].strip()[0]
        atom = Atom(line[12:16].strip(), el, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                    float(line[60:66] or 0))
        res = chains.setdefault(ch, {}).setdefault((num, icode), Residue(num, resname, []))
        res.atoms.append(atom)
    if not chains:
        raise ValueError("no protein atoms found")
    ch = chain if chain in chains else max(chains, key=lambda c: len(chains[c]))
    return list(chains[ch].values()), ch


def _read_with_gemmi(path: Path, chain: str | None) -> tuple[list[Residue], str]:
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    candidates = {}
    for ch in model:
        residues = []
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if not (info and info.is_amino_acid()) and r.name not in THREE_TO_ONE:
                continue
            atoms = [Atom(a.name, a.element.name.upper(), np.array(a.pos.tolist()), a.b_iso)
                     for a in r if a.altloc in ("\0", "A")]
            if atoms:
                residues.append(Residue(r.seqid.num, r.name if r.name in THREE_TO_ONE else "UNK", atoms))
        if residues:
            candidates[ch.name] = residues
    if not candidates:
        raise ValueError(f"no protein chain in {path}")
    name = chain if chain in candidates else max(candidates, key=lambda c: len(candidates[c]))
    return candidates[name], name


def read_structure(path: str | Path, chain: str | None = None, source: str = "user",
                   b_is_plddt: bool = True) -> Structure:
    path = Path(path)
    try:
        residues, ch = _read_with_gemmi(path, chain)
    except ImportError:
        if path.suffix.lower() not in (".pdb", ".ent"):
            raise ImportError("Reading mmCIF needs gemmi: pip install gemmi")
        residues, ch = _read_pdb_text(path.read_text(), chain)
    return Structure(residues, source, ch, b_is_plddt, meta={"file": str(path)})


def renumber_to_query(st: Structure, query: str) -> Structure:
    """Keep residues that align to the query sequence and renumber them to query positions."""
    mapping = map_positions(st.sequence, query)          # structure index -> query position (1-based)
    keep, idx_kept = [], []
    for i, r in enumerate(st.residues):
        q = mapping.get(i)
        if q is not None:
            keep.append(Residue(q, r.name, r.atoms))
            idx_kept.append(i)
    pae = st.pae[np.ix_(idx_kept, idx_kept)] if st.pae is not None and len(st.pae) == len(st.residues) else None
    identical = sum(1 for r in keep if r.one == query[r.number - 1])
    meta = dict(st.meta, coverage=len(keep) / len(query), identity=identical / max(len(keep), 1))
    out = Structure(keep, st.source, "A", st.b_is_plddt, pae, meta)
    if pae is not None:
        out.meta["pae_numbers"] = [r.number for r in keep]
    return out


def _download(url: str, dest: Path) -> Path:
    r = http_get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _parse_pae_json(obj) -> np.ndarray | None:
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    for key in ("predicted_aligned_error", "pae"):
        if key in obj:
            return np.array(obj[key], float)
    return None


# --------------------------------------------------------------------------- AlphaFold DB

def afdb_entries(accession: str) -> list[dict]:
    r = http_get(f"{AFDB}/api/prediction/{accession}")
    if r.status_code == 404 or not r.ok:
        return []
    data = r.json()
    return data if isinstance(data, list) else [data]


def fetch_afdb(accession: str, dest: str | Path, sequence: str | None = None) -> Structure | None:
    """Download the current AlphaFold DB model for an accession (prefers an exact sequence match)."""
    entries = afdb_entries(accession)
    if not entries:
        return None
    seq_of = lambda e: e.get("sequence") or e.get("uniprotSequence") or ""
    entry = next((e for e in entries if sequence and seq_of(e) == sequence), entries[0])
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    url = entry.get("pdbUrl") or entry.get("cifUrl")
    if not url:
        return None
    f = _download(url, dest / f"afdb_{accession}{Path(url).suffix}")
    st = read_structure(f, source="AlphaFold DB")
    st.meta.update({
        "accession": accession,
        "model_id": entry.get("modelEntityId") or entry.get("entryId"),
        "afdb_version": entry.get("latestVersion"),
        "model_created": entry.get("modelCreatedDate"),
        "exact_sequence_match": bool(sequence) and seq_of(entry) == sequence,
        "fragment": (entry.get("sequenceStart") or entry.get("uniprotStart") or 1) != 1
                    or len(seq_of(entry)) != len(sequence or seq_of(entry)),
    })
    if entry.get("paeDocUrl"):
        try:
            st.pae = _parse_pae_json(http_get(entry["paeDocUrl"]).json())
        except Exception:
            pass
    return st


# --------------------------------------------------------------------------- experimental (RCSB)

def fetch_rcsb(pdb_id: str, dest: str | Path, chain: str | None = None) -> Structure:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    try:
        f = _download(f"https://files.rcsb.org/download/{pdb_id}.pdb", dest / f"{pdb_id}.pdb")
    except Exception:  # very large entries are mmCIF only
        f = _download(f"https://files.rcsb.org/download/{pdb_id}.cif", dest / f"{pdb_id}.cif")
    st = read_structure(f, chain=chain, source=f"PDB {pdb_id}", b_is_plddt=False)
    st.meta["pdb_id"] = pdb_id
    return st


# --------------------------------------------------------------------------- AlphaFold 3 / AlphaFold Server

def load_af3(path: str | Path, dest: str | Path, chain: str | None = None, model: int = 0) -> Structure:
    """Load an AlphaFold 3 (AlphaFold Server or local AF3) result from a zip file or folder."""
    path, dest = Path(path), Path(dest)
    if path.suffix == ".zip":
        out = dest / f"af3_{path.stem}"
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        path = out
    cifs = sorted(glob.glob(str(path / "**" / f"*model_{model}.cif"), recursive=True)) or \
        sorted(glob.glob(str(path / "**" / "*model.cif"), recursive=True)) or \
        sorted(glob.glob(str(path / "**" / "*.cif"), recursive=True))
    if not cifs:
        raise FileNotFoundError(f"no mmCIF model found in {path}")
    st = read_structure(cifs[0], chain=chain, source="AlphaFold 3")
    conf = sorted(glob.glob(str(path / "**" / f"*full_data_{model}.json"), recursive=True)) or \
        sorted(glob.glob(str(path / "**" / "*confidences.json"), recursive=True))
    if conf:
        data = json.loads(Path(conf[0]).read_text())
        pae = _parse_pae_json(data)
        chains = data.get("token_chain_ids")
        if pae is not None and chains is not None:
            sel = [i for i, c in enumerate(chains) if c == st.chain]
            pae = pae[np.ix_(sel, sel)]
        if pae is not None and len(pae) == len(st.residues):
            st.pae = pae
    st.meta["model_file"] = cifs[0]
    return st


# --------------------------------------------------------------------------- ColabFold (AlphaFold2)

def predict_colabfold(pid: str, sequence: str, dest: str | Path, num_models: int = 1) -> Structure:
    """Predict with ColabFold (AlphaFold2 + MMseqs2 MSA server). Requires colabfold installed and a GPU."""
    dest = Path(dest) / f"colabfold_{pid}"
    dest.mkdir(parents=True, exist_ok=True)
    fasta = dest / "query.fasta"
    fasta.write_text(f">{pid}\n{sequence}\n")
    if not glob.glob(str(dest / "*rank_001*.pdb")):
        subprocess.run(["colabfold_batch", "--num-models", str(num_models), str(fasta), str(dest)], check=True)
    pdb = sorted(glob.glob(str(dest / "*rank_001*.pdb")))[0]
    st = read_structure(pdb, source="ColabFold (AlphaFold2)")
    scores = sorted(glob.glob(str(dest / "*scores_rank_001*.json")))
    if scores:
        st.pae = _parse_pae_json(json.loads(Path(scores[0]).read_text()))
    return st


# --------------------------------------------------------------------------- ESMFold

_ESM = {}


def predict_esmfold_local(pid: str, sequence: str, dest: str | Path) -> Structure:
    """ESMFold through Hugging Face transformers on the local GPU (fits a Colab T4 up to ~800 aa)."""
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein as OFProtein, to_pdb

    if "model" not in _ESM:
        _ESM["tok"] = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        if torch.cuda.is_available():
            model = model.cuda()
            model.esm = model.esm.half()
            torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(64)
        _ESM["model"] = model.eval()
    model, tok = _ESM["model"], _ESM["tok"]
    ids = tok([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(ids)
    pos = atom14_to_atom37(out["positions"][-1], out)
    o = {k: v.to("cpu").numpy() for k, v in out.items() if hasattr(v, "to")}
    plddt = o["plddt"][0]
    plddt = plddt * 100 if plddt.max() <= 1.0 else plddt
    prot = OFProtein(aatype=o["aatype"][0], atom_positions=pos.cpu().numpy()[0], atom_mask=o["atom37_atom_exists"][0],
                     residue_index=o["residue_index"][0] + 1, b_factors=plddt,
                     chain_index=o["chain_index"][0] if "chain_index" in o else None)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"esmfold_{pid}.pdb"
    f.write_text(to_pdb(prot))
    st = read_structure(f, source="ESMFold")
    if "predicted_aligned_error" in o:
        st.pae = o["predicted_aligned_error"][0]
    return st


def predict_esmfold_api(pid: str, sequence: str, dest: str | Path) -> Structure:
    """ESMFold via the public ESM Atlas API (no GPU needed; sequences up to 400 aa)."""
    import requests

    if len(sequence) > 400:
        raise ValueError("ESM Atlas API accepts sequences up to 400 residues")
    r = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence, timeout=300)
    r.raise_for_status()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"esmatlas_{pid}.pdb"
    f.write_text(r.text)
    st = read_structure(f, source="ESMFold (ESM Atlas API)")
    b = [a.b for res in st.residues for a in res.atoms]
    if b and max(b) <= 1.0:  # API reports pLDDT on a 0-1 scale
        for res in st.residues:
            for a in res.atoms:
                a.b *= 100
    return st


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
