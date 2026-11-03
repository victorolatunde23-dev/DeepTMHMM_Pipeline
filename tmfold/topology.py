"""DeepTMHMM topology prediction (via BioLib) and parsing of its outputs."""
from __future__ import annotations

import itertools
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .sequences import Protein, to_fasta

LABELS = {"S": "signal", "I": "inside", "O": "outside", "M": "TM_helix", "B": "beta_strand", "P": "periplasm"}
MEMBRANE_LABELS = {"M", "B"}
INSIDE_LABELS = {"I", "P"}
OUTSIDE_LABELS = {"O"}


@dataclass
class Topology:
    id: str
    type: str          # TM, SP+TM, SP, BETA, GLOB
    sequence: str
    labels: str        # one label per residue

    @property
    def is_membrane(self) -> bool:
        return any(c in MEMBRANE_LABELS for c in self.labels)

    def segments(self) -> list[tuple[str, int, int]]:
        return segments(self.labels)


def segments(labels: str) -> list[tuple[str, int, int]]:
    """Run-length segments as (label, start, end), 1-based inclusive."""
    pos, out = 1, []
    for lab, grp in itertools.groupby(labels):
        n = len(list(grp))
        out.append((lab, pos, pos + n - 1))
        pos += n
    return out


def parse_3line(path: str | Path) -> dict[str, Topology]:
    lines = [l.rstrip("\n") for l in Path(path).read_text().splitlines() if l.strip()]
    out = {}
    for h, seq, lab in zip(lines[0::3], lines[1::3], lines[2::3]):
        pid, _, ptype = h[1:].partition("|")
        t = Topology(pid.strip(), ptype.strip(), seq.strip(), lab.strip())
        out[t.id] = t
    return out


def run_deeptmhmm(proteins: list[Protein], workdir: str | Path, batch_size: int = 200,
                  max_retries: int = 3) -> dict[str, Topology]:
    """Run DeepTMHMM on BioLib in resumable batches. Returns topologies keyed by protein ID."""
    import biolib  # pybiolib

    workdir = Path(workdir) / "deeptmhmm"
    workdir.mkdir(parents=True, exist_ok=True)
    app = None
    results: dict[str, Topology] = {}
    batches = [proteins[i:i + batch_size] for i in range(0, len(proteins), batch_size)]
    for idx, batch in enumerate(batches):
        bdir = workdir / f"batch_{idx:03d}"
        out = bdir / "biolib_results"
        three = out / "predicted_topologies.3line"
        if not three.exists():
            bdir.mkdir(exist_ok=True)
            (bdir / "input.fasta").write_text(to_fasta(batch))
            app = app or biolib.load("DTU/DeepTMHMM")
            for attempt in range(1, max_retries + 1):
                cwd = os.getcwd()
                try:
                    os.chdir(bdir)
                    print(f"[DeepTMHMM] batch {idx + 1}/{len(batches)}: {len(batch)} sequences (attempt {attempt})")
                    job = app.cli(args="--fasta input.fasta")
                    if hasattr(job, "save_files"):
                        job.save_files("biolib_results")
                except Exception as e:  # network / queue errors
                    print(f"[DeepTMHMM] batch {idx + 1} error: {e}")
                finally:
                    os.chdir(cwd)
                if three.exists():
                    break
                shutil.rmtree(out, ignore_errors=True)
                time.sleep(30 * attempt)
        if three.exists():
            results.update(parse_3line(three))
        else:
            print(f"[DeepTMHMM] batch {idx + 1} FAILED after {max_retries} attempts")
    return results


def merged_outputs(workdir: str | Path, dest: str | Path) -> None:
    """Concatenate raw DeepTMHMM outputs from all batches into `dest`."""
    workdir, dest = Path(workdir) / "deeptmhmm", Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for fname in ("predicted_topologies.3line", "TMRs.gff3"):
        parts = [p.read_text().rstrip("\n") for p in sorted(workdir.glob(f"batch_*/biolib_results/{fname}"))]
        if parts:
            (dest / fname).write_text("\n".join(parts) + "\n")
