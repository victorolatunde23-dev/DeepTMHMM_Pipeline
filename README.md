# DeepTMHMM Pipeline & TMFold

[![tests](https://github.com/victorolatunde23-dev/DeepTMHMM_Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/victorolatunde23-dev/DeepTMHMM_Pipeline/actions/workflows/tests.yml)

Two Google Colab notebooks for transmembrane protein analysis with [DeepTMHMM](https://services.healthtech.dtu.dk/services/DeepTMHMM-1.0/):

| Notebook | What it does | Open |
|---|---|---|
| **TMFold** | DeepTMHMM topology mapped onto 3D structures (AlphaFold DB, AlphaFold 3, PDB, ColabFold, ESMFold), with membrane placement and quality control | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/DeepTMHMM_Pipeline/blob/main/TMFold_Colab.ipynb) |
| **DeepTMHMM pipeline** | Topology prediction only, for many sequences, with a summary table and plots | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/DeepTMHMM_Pipeline/blob/main/DeepTMHMM_pipeline.ipynb) |

## TMFold

TMFold is inspired by [MembraneFold](https://biolib.com/KU/MembraneFold/) and extends the idea:

| | MembraneFold | TMFold |
|---|---|---|
| Proteins per run | 1 | Many: accession list, UniProt query or whole proteome |
| Structure sources | AlphaFold DB, OmegaFold, PDB upload | Current AlphaFold DB (v6), AlphaFold 3 / AlphaFold Server results, PDB ID, ColabFold (AlphaFold2), ESMFold |
| Raw sequences | Exact UniProt match → AlphaFold DB | Exact match via UniParc MD5 checksum → AlphaFold DB, else prediction |
| Membrane placement | – | Fitted membrane slab; membrane-oriented model with OPM-style dummy atoms |
| Quality control | pLDDT colouring | Structure–topology agreement scores, per-segment pLDDT, tilt, hydrophobicity and flags, positive-inside rule, UniProt annotation overlap, PAE map |
| Outputs | Plots, PDB, FASTA | Summary CSV, HTML report, oriented PDBs, PyMOL and ChimeraX scripts, interactive 3D viewers |
| Resumable | – | Yes: DeepTMHMM batches and structures are cached |
| Use outside Colab | Web app | Python package and `tmfold` command |

### How it works

1. **Topology**: DeepTMHMM runs on BioLib in resumable batches.
2. **Structure**, first available of:
   - Your own file: an AlphaFold 3 / AlphaFold Server zip, a PDB or mmCIF file, or a PDB ID with chain
   - The AlphaFold DB model, preferring an exact sequence match
   - A prediction with ESMFold (Colab GPU, or the ESM Atlas API for ≤400 aa without a GPU) or ColabFold

   Every structure is aligned and renumbered to the query sequence, so expression tags, missing loops and point mutations in experimental structures are handled.
3. **Membrane placement**: TM helix and β-strand axes, oriented by the flanking inside/outside labels, give a starting membrane normal. A grid search then refines the normal, mid-plane and thickness so that membrane residues sit in the slab and loops sit on their predicted side.
4. **Quality control**:
   - `tm_in_slab`: fraction of predicted membrane residues inside the fitted membrane
   - `loop_side_agreement`: fraction of loop residues on their predicted side
   - An overall `agreement` call: consistent, partly consistent or inconsistent
   - Per-segment flags: low confidence, outside membrane, short helix, highly tilted, wrong side
   - The K+R positive-inside rule
   - Residue overlap with UniProt transmembrane annotations

### AlphaFold 3

AlphaFold 3 model weights are not freely downloadable. To use AlphaFold 3:
1. Run your sequence on [AlphaFold Server](https://alphafoldserver.com).
2. Download the result zip.
3. Upload it in the notebook. Name the job after the protein ID so the zip is matched automatically.

Output from a local AlphaFold 3 install (a folder of `*.cif` and `*confidences.json` files) also works.

### Outputs

```
tmfold_run/
├── summary.csv                  one row per protein
├── report.html                  self-contained report
├── deeptmhmm_merged/            raw DeepTMHMM outputs
└── proteins/<id>/
    ├── model_membrane_oriented.pdb   membrane normal along +z, cytoplasm at z<0, B = pLDDT, DUM membrane atoms
    ├── model_topology_bfactor.pdb    B = topology code (1 in, 2 out, 3 TM helix, 4 β-strand, 5 signal, 6 periplasm)
    ├── color_topology.pml / .cxc     PyMOL / ChimeraX colouring scripts
    ├── viewer.html                   interactive 3D view
    ├── segments.csv                  per-segment metrics and flags
    ├── profile.png                   topology, UniProt, pLDDT and membrane-depth tracks
    └── pae.png                       predicted aligned error map
```

### Command line

```bash
pip install "tmfold @ git+https://github.com/victorolatunde23-dev/DeepTMHMM_Pipeline"
tmfold --uniprot P02945,P02724 --out results
tmfold --query "organism_id:83333 AND keyword:KW-0812 AND reviewed:true" --max-sequences 100 --out ecoli_tm
tmfold --fasta my_proteins.fasta --structure-source predict --predictor esmatlas
```

For ESMFold on your own GPU, install with `pip install "tmfold[esmfold] @ git+..."`.

### Development

```bash
pip install -e ".[test]"
pytest tests
```

The tests run offline. They cover parsing, alignment, membrane fitting on randomly rotated synthetic proteins, mocked AlphaFold DB and UniProt responses, and an end-to-end run.

## Citations

Please cite the tools you use:

- **DeepTMHMM**: Hallgren J *et al.* (2022). DeepTMHMM predicts alpha and beta transmembrane proteins using deep neural networks. *bioRxiv* doi:10.1101/2022.04.08.487609. Free for academic use; commercial use requires a licence (licensing@biolib.com).
- **AlphaFold DB**: Varadi M *et al.* (2024). *Nucleic Acids Research* 52:D368–D375.
- **AlphaFold 3**: Abramson J *et al.* (2024). *Nature* 630:493–500.
- **ColabFold**: Mirdita M *et al.* (2022). *Nature Methods* 19:679–682.
- **ESMFold**: Lin Z *et al.* (2023). *Science* 379:1123–1130.
- **MembraneFold** (inspiration): Gutierrez S *et al.* (2022). *bioRxiv* doi:10.1101/2022.12.06.518085.
