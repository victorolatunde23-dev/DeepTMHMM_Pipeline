# DeepTMHMM Pipeline for Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorolatunde23-dev/DeepTMHMM_Pipeline/blob/main/DeepTMHMM_pipeline.ipynb)

A Google Colab notebook that predicts transmembrane protein topology with [DeepTMHMM](https://services.healthtech.dtu.dk/services/DeepTMHMM-1.0/), using the official BioLib Python client (`pybiolib`). Predictions run on BioLib's cloud, so no GPU is needed.

## Features

- **Direct UniProt download**: fetch sequences by accession list, by any UniProt search query, or by whole proteome. Large downloads are paged automatically, and protein metadata is downloaded alongside.
- **Other inputs**: upload a FASTA file or paste sequences.
- **Batched, resumable runs** with automatic retries.
- **Results table**: predicted type (TM, SP+TM, SP, BETA, GLOB), helix and β-strand counts, signal peptide, N-terminus side and TM region positions, merged with UniProt metadata.
- **Topology plot** of the predicted regions.
- **Export**: CSVs, merged raw DeepTMHMM outputs and the plot as a zip, with an optional copy to Google Drive.

## Usage

1. Click the **Open in Colab** badge above.
2. In the **Settings** and **UniProt download** forms, choose your input.
3. Select **Runtime → Run all**.

### UniProt input examples

| `UNIPROT_SOURCE` | `UNIPROT_INPUT` example |
|---|---|
| `accessions` | `P02945, P0A917, OMPX_ECOLI` |
| `query` | `organism_id:9606 AND keyword:KW-0812 AND reviewed:true` |
| `proteome` | `UP000000625` |

An optional BioLib API token can be added as the Colab secret `BIOLIB_TOKEN`.

## Output files

| File | Contents |
|---|---|
| `deeptmhmm_summary.csv` | One row per protein |
| `deeptmhmm_regions.csv` | One row per predicted region |
| `merged_predicted_topologies.3line` | Raw DeepTMHMM topology output |
| `merged_TMRs.gff3` | Raw DeepTMHMM GFF3 output |
| `uniprot_download.fasta`, `uniprot_metadata.tsv` | Downloaded UniProt data |
| `topology_plot.png` | Topology figure |

## Citation

If you use DeepTMHMM, please cite:

> Hallgren J, Tsirigos KD, Pedersen MD, Almagro Armenteros JJ, Marcatili P, Nielsen H, Krogh A, Winther O (2022). DeepTMHMM predicts alpha and beta transmembrane proteins using deep neural networks. *bioRxiv*. https://doi.org/10.1101/2022.04.08.487609

DeepTMHMM is free for academic use. Commercial use requires a licence (licensing@biolib.com).
