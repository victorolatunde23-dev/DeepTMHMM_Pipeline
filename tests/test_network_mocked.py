"""AlphaFold DB / UniProt / UniParc client code tested against mocked HTTP responses."""
import hashlib
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from tmfold import sequences, structures

SEQ = "MKLLVAG"
PDB = "".join(
    f"ATOM  {i + 1:5d}  CA  {r} A{i + 1:4d}    {i * 1.5:8.3f}{0:8.3f}{0:8.3f}  1.00{80 + i:6.2f}           C\n"
    for i, r in enumerate(["MET", "LYS", "LEU", "LEU", "VAL", "ALA", "GLY"])) + "END\n"


class Resp:
    def __init__(self, status=200, json_data=None, text="", content=b"", headers=None):
        self.status_code, self._json, self.text = status, json_data, text
        self.content = content or text.encode()
        self.headers = headers or {}
        self.ok = status < 400

    def json(self):
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


def fake_get(url, params=None, **kw):
    if url.endswith("/api/prediction/P12345"):
        return Resp(json_data=[{"modelEntityId": "AF-P12345-F1", "latestVersion": 6, "sequence": SEQ,
                                "sequenceStart": 1, "pdbUrl": "https://afdb/AF-P12345-F1-model_v6.pdb",
                                "paeDocUrl": "https://afdb/pae.json"}])
    if url.endswith("/api/prediction/P00000"):
        return Resp(status=404)
    if url.endswith(".pdb"):
        return Resp(text=PDB)
    if url.endswith("pae.json"):
        return Resp(json_data=[{"predicted_aligned_error": np.ones((7, 7)).tolist()}])
    if url.endswith("/uniparc/search"):
        assert params["query"] == "checksum:" + hashlib.md5(SEQ.encode()).hexdigest().upper()
        return Resp(json_data={"results": [{"uniProtKBAccessions": ["P12345-2", "P12345"]}]})
    if url.endswith("/uniprotkb/P12345.json"):
        return Resp(json_data={"features": [
            {"type": "Transmembrane", "location": {"start": {"value": 2}, "end": {"value": 6}}, "description": "Helical"}]})
    if "/uniprotkb/search" in url:
        if "page2" not in url:
            return Resp(text=">sp|P12345|A_HUMAN A\nMKL\n", headers={"Link": '<https://x/uniprotkb/search?page2>; rel="next"'})
        return Resp(text=">sp|Q99999|B_HUMAN B\nMKV\n")
    raise AssertionError(url)


@mock.patch.object(structures, "http_get", side_effect=fake_get)
@mock.patch.object(sequences, "http_get", side_effect=fake_get)
class TestMockedNetwork(unittest.TestCase):
    def test_afdb_current_fields(self, *_):
        with tempfile.TemporaryDirectory() as d:
            st = structures.fetch_afdb("P12345", d, SEQ)
        self.assertEqual(st.sequence, SEQ)
        self.assertEqual(st.meta["afdb_version"], 6)
        self.assertEqual(st.meta["model_id"], "AF-P12345-F1")
        self.assertTrue(st.meta["exact_sequence_match"])
        self.assertFalse(st.meta["fragment"])
        self.assertEqual(st.pae.shape, (7, 7))
        self.assertEqual(st.plddt()[1], 80.0)

    def test_afdb_missing(self, *_):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(structures.fetch_afdb("P00000", d, SEQ))

    def test_sequence_to_accession(self, *_):
        self.assertEqual(sequences.accessions_for_sequence(SEQ), ["P12345", "P12345-2"])

    def test_uniprot_features(self, *_):
        f = sequences.uniprot_membrane_features("P12345")
        self.assertEqual(f, [{"type": "Transmembrane", "start": 2, "end": 6, "description": "Helical"}])

    def test_uniprot_pagination(self, *_):
        prots = sequences.fetch_uniprot("query", "keyword:KW-0812")
        self.assertEqual([p.id for p in prots], ["P12345", "Q99999"])


if __name__ == "__main__":
    unittest.main()
