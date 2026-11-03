"""Command line: tmfold --uniprot P02945 --out results/"""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Config, run
from .sequences import fetch_uniprot, parse_fasta, validate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tmfold", description="DeepTMHMM topology mapped onto 3D structures")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="FASTA file")
    src.add_argument("--uniprot", help="UniProt accessions, comma separated")
    src.add_argument("--query", help="UniProt search query")
    src.add_argument("--proteome", help="UniProt proteome ID")
    ap.add_argument("--out", default="tmfold_results")
    ap.add_argument("--max-sequences", type=int, default=0)
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--structure-source", default="auto", choices=["auto", "afdb", "predict", "none"])
    ap.add_argument("--predictor", default="esmfold", choices=["esmfold", "esmatlas", "colabfold", "none"])
    ap.add_argument("--include-non-tm", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    a = ap.parse_args(argv)

    if a.fasta:
        proteins = parse_fasta(Path(a.fasta).read_text())
    elif a.uniprot:
        proteins = fetch_uniprot("accessions", a.uniprot)
    elif a.query:
        proteins = fetch_uniprot("query", a.query, a.max_sequences, a.reviewed_only)
    else:
        proteins = fetch_uniprot("proteome", a.proteome, a.max_sequences, a.reviewed_only)
    proteins = validate(proteins)
    cfg = Config(structure_source=a.structure_source, predictor=a.predictor, include_non_tm=a.include_non_tm,
                 make_viewer_html=not a.no_viewer)
    summary, _ = run(proteins, a.out, cfg)
    print(summary.to_string(index=False, max_colwidth=40))
    print(f"\nReport: {Path(a.out) / 'report.html'}")


if __name__ == "__main__":
    main()
