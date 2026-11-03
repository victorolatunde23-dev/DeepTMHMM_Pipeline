"""Sequence input: FASTA parsing, UniProt download and sequence-to-accession lookup."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import requests

UNIPROT = "https://rest.uniprot.org"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")

_session = requests.Session()
_session.headers["User-Agent"] = "tmfold (https://github.com/victorolatunde23-dev/DeepTMHMM_Pipeline)"


def http_get(url: str, params: dict | None = None, tries: int = 4, timeout: int = 60, **kw) -> requests.Response:
    """GET with retries on rate limits, server errors and dropped connections."""
    last = None
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=timeout, **kw)
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = RuntimeError(f"HTTP {r.status_code}")
            time.sleep(2 ** attempt * 2)
            continue
        return r
    raise RuntimeError(f"GET {url} failed after {tries} attempts: {last}")


@dataclass
class Protein:
    id: str
    sequence: str
    description: str = ""
    accession: str | None = None


def safe_id(raw: str, fallback: str = "seq") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)[:60] or fallback


def parse_fasta(text: str) -> list[Protein]:
    """Parse FASTA text. UniProt headers (sp|ACC|NAME ...) use the accession as ID."""
    proteins, header, chunks = [], None, []

    def flush():
        if header is None:
            return
        seq = re.sub(r"[\s*\-]", "", "".join(chunks)).upper()
        token = header.split()[0] if header.split() else ""
        parts = token.split("|")
        acc = parts[1] if len(parts) >= 3 and parts[0] in ("sp", "tr") else None
        proteins.append(Protein(id=acc or token, sequence=seq, description=header, accession=acc))

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line)
    flush()
    return proteins


def validate(proteins: list[Protein]) -> list[Protein]:
    """Drop invalid sequences and make IDs safe and unique (DeepTMHMM uses the first header token)."""
    out, seen = [], set()
    for i, p in enumerate(proteins):
        bad = set(p.sequence) - VALID_AA
        if not p.sequence or bad:
            print(f"  ! skipping {p.id}: empty or invalid characters {sorted(bad)}")
            continue
        pid = safe_id(p.id, f"seq{i}")
        while pid in seen:
            pid += "_dup"
        seen.add(pid)
        out.append(Protein(pid, p.sequence, p.description, p.accession))
    return out


def to_fasta(proteins: list[Protein]) -> str:
    return "".join(f">{p.id}\n{p.sequence}\n" for p in proteins)


# --------------------------------------------------------------------------- UniProt

def _next_link(r: requests.Response) -> str | None:
    m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
    return m.group(1) if m else None


def uniprot_search(query: str, fmt: str = "fasta", fields: str | None = None, limit: int = 0,
                   include_isoforms: bool = False, page_size: int = 500) -> str:
    """Paginated UniProtKB search; returns concatenated FASTA or TSV text."""
    params = {"query": query, "format": fmt, "size": page_size}
    if fields:
        params["fields"] = fields
    if fmt == "fasta" and include_isoforms:
        params["includeIsoform"] = "true"
    url, chunks, n = f"{UNIPROT}/uniprotkb/search", [], 0
    while url:
        r = http_get(url, params)
        r.raise_for_status()
        params = None  # the next link carries the parameters
        text = r.text
        if fmt == "tsv" and chunks:
            text = text.split("\n", 1)[1] if "\n" in text else ""
        chunks.append(text)
        n += text.count(">") if fmt == "fasta" else text.count("\n") - (1 if len(chunks) == 1 else 0)
        if limit and n >= limit:
            break
        url = _next_link(r)
    return "".join(chunks)


def _accession_query(ids: list[str]) -> str:
    # accession:X matches primary accessions; id:X matches entry names such as OMPX_ECOLI
    return " OR ".join(f"id:{x}" if "_" in x else f"accession:{x}" for x in ids)


def fetch_uniprot(source: str, value: str, max_sequences: int = 0, reviewed_only: bool = False,
                  include_isoforms: bool = False) -> list[Protein]:
    """Download sequences by `accessions`, a UniProt `query`, or a `proteome` ID."""
    value = value.strip()
    if source == "accessions":
        ids = [i for i in re.split(r"[,;\s]+", value) if i]
        proteins = []
        for i in range(0, len(ids), 100):
            proteins += parse_fasta(uniprot_search(_accession_query(ids[i:i + 100]),
                                                   include_isoforms=include_isoforms))
        got = {p.accession for p in proteins} | {p.description.split()[0].split("|")[-1] for p in proteins}
        missing = [x for x in ids if x not in got]
        if missing:
            print("  ! not found in UniProtKB:", ", ".join(missing))
    else:
        q = f"proteome:{value}" if source == "proteome" else value
        if reviewed_only and "reviewed:" not in q:
            q = f"({q}) AND reviewed:true"
        proteins = parse_fasta(uniprot_search(q, limit=max_sequences, include_isoforms=include_isoforms))
    if max_sequences and len(proteins) > max_sequences:
        proteins = proteins[:max_sequences]
    return proteins


def uniprot_membrane_features(accession: str) -> list[dict]:
    """UniProt Transmembrane / Intramembrane / Topological domain annotations for one entry."""
    r = http_get(f"{UNIPROT}/uniprotkb/{accession}.json",
                 params={"fields": "ft_transmem,ft_intramem,ft_topo_dom"})
    if not r.ok:
        return []
    out = []
    for f in r.json().get("features", []):
        loc = f.get("location", {})
        start, end = loc.get("start", {}).get("value"), loc.get("end", {}).get("value")
        if start is None or end is None:
            continue
        out.append({"type": f.get("type"), "start": int(start), "end": int(end),
                    "description": f.get("description", "")})
    return out


def accessions_for_sequence(sequence: str) -> list[str]:
    """Find UniProtKB accessions whose sequence is identical, via the UniParc MD5 checksum."""
    md5 = hashlib.md5(sequence.upper().encode()).hexdigest().upper()
    r = http_get(f"{UNIPROT}/uniparc/search", params={"query": f"checksum:{md5}", "fields": "upi,accession",
                                                       "format": "json", "size": 5})
    if not r.ok:
        return []
    accs: list[str] = []
    for res in r.json().get("results", []):
        accs += res.get("uniProtKBAccessions", []) or []
        for x in res.get("uniParcCrossReferences", []) or []:
            if x.get("database", "").startswith("UniProtKB") and x.get("active", True):
                accs.append(x.get("id"))
    # primary accessions first; isoform IDs (P12345-2) last
    seen, ordered = set(), []
    for a in sorted(filter(None, accs), key=lambda a: ("-" in a, a)):
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered
