"""Offline tests (no network): parsing, alignment, membrane fitting and an end-to-end run."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tmfold.align import map_positions
from tmfold.geometry import fit_membrane, positive_inside, rotation_to, segment_table
from tmfold.pipeline import Config, run
from tmfold.sequences import Protein, parse_fasta, validate
from tmfold.structures import Atom, Residue, Structure, _read_pdb_text, renumber_to_query
from tmfold.topology import Topology, parse_3line, segments

LAYOUT = [("I", 10), ("M", 20), ("O", 8), ("M", 20), ("I", 8), ("M", 20), ("O", 8), ("M", 20), ("I", 10)]
LABELS = "".join(l * n for l, n in LAYOUT)
SEQ = "".join(("K" if l == "I" else "D" if l == "O" else "L") * n for l, n in LAYOUT)


def synthetic_ca(labels=LABELS, seed=0):
    """Four ideal helices crossing z in [-14, 14]; loops sit beyond the membrane on their side."""
    rng = np.random.default_rng(seed)
    pts, helix, centres = [], 0, [(0, 0), (10, 0), (10, 10), (0, 10)]
    for lab, s, e in segments(labels):
        n = e - s + 1
        if lab == "M":
            cx, cy = centres[helix % 4]
            up = labels[s - 2] == "I" if s > 1 else True
            for k in range(n):
                z = -14.25 + 1.5 * k
                a = np.radians(100 * k)
                pts.append([cx + 2.3 * np.cos(a), cy + 2.3 * np.sin(a), z if up else -z])
            helix += 1
        else:
            sign = -1 if lab == "I" else 1
            for k in range(n):
                pts.append([5 + 4 * np.cos(k), 5 + 4 * np.sin(k), sign * (18 + 2 * np.sin(k / 2))])
    xyz = np.array(pts)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    shift = rng.normal(size=3) * 20
    return xyz @ q.T + shift, q @ np.array([0, 0, 1.0])


def structure_from(xyz, seq):
    three = {"K": "LYS", "D": "ASP", "L": "LEU", "A": "ALA", "G": "GLY", "H": "HIS"}
    res = [Residue(i + 1, three[a], [Atom("CA", "C", p, 90.0)]) for i, (a, p) in enumerate(zip(seq, xyz))]
    return Structure(res, "test")


class TestParsing(unittest.TestCase):
    def test_fasta_uniprot_header(self):
        p = parse_fasta(">sp|P02945|BACR_HALSA Bacteriorhodopsin\nMLEL\nLPTA\n>my seq 2\nmk*\n")
        self.assertEqual((p[0].id, p[0].accession, p[0].sequence), ("P02945", "P02945", "MLELLPTA"))
        self.assertEqual((p[1].id, p[1].sequence), ("my", "MK"))

    def test_validate(self):
        v = validate([Protein("a b", "MKL"), Protein("a b", "MKL"), Protein("bad", "MK1")])
        self.assertEqual([x.id for x in v], ["a_b", "a_b_dup"])

    def test_3line_and_segments(self):
        with tempfile.NamedTemporaryFile("w", suffix=".3line", delete=False) as f:
            f.write(">P1 | SP+TM\nMKKLLAAAVVVLLLAAAGGG\nSSSSOOOMMMMMMIIIIIII\n")
        t = parse_3line(f.name)["P1"]
        self.assertEqual(t.type, "SP+TM")
        self.assertEqual(t.segments()[:2], [("S", 1, 4), ("O", 5, 7)])
        self.assertTrue(t.is_membrane)


class TestAlign(unittest.TestCase):
    def test_identical_and_substring(self):
        self.assertEqual(map_positions("MKL", "MKL"), {0: 1, 1: 2, 2: 3})
        self.assertEqual(map_positions("KLV", "MKLVA"), {0: 2, 1: 3, 2: 4})

    def test_tag_mutation_and_missing_loop(self):
        query = "MSTNKLLVAIGGTRDEQWYPLSAHHGRKLMNPQ"
        construct = "HHHHHHGS" + query[:10] + "A" + query[11:20] + query[25:]   # tag, mutation, 5-residue gap
        m = map_positions(construct, query)
        self.assertEqual(m[8], 1)                  # first native residue after the tag
        self.assertEqual(m[8 + 10], 11)            # mutated position keeps its number
        self.assertEqual(m[8 + 20], 26)            # residues after the missing loop
        self.assertNotIn(0, m)                     # tag not mapped


class TestGeometry(unittest.TestCase):
    def test_rotation_to(self):
        for a in (np.array([1.0, 2, 3]), np.array([0, 0, -1.0]), np.array([0, 0, 1.0])):
            r = rotation_to(a, np.array([0, 0, 1.0]))
            np.testing.assert_allclose(r @ (a / np.linalg.norm(a)), [0, 0, 1], atol=1e-8)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-8)

    def test_membrane_recovered_after_random_rotation(self):
        for seed in range(5):
            xyz, true_n = synthetic_ca(seed=seed)
            mem = fit_membrane(LABELS, {i + 1: p for i, p in enumerate(xyz)})
            self.assertGreater(np.dot(mem.normal, true_n), np.cos(np.radians(3)), f"seed {seed}")  # cytoplasm -> outside
            self.assertGreater(mem.tm_in_slab, 0.9)
            self.assertGreater(mem.loop_side_agreement, 0.95)

    def test_flipped_topology_flips_normal(self):
        xyz, true_n = synthetic_ca()
        flipped = LABELS.translate(str.maketrans("IO", "OI"))
        mem = fit_membrane(flipped, {i + 1: p for i, p in enumerate(xyz)})
        self.assertLess(np.dot(mem.normal, true_n), -0.97)

    def test_segment_table_flags(self):
        xyz, _ = synthetic_ca()
        ca = {i + 1: p for i, p in enumerate(xyz)}
        mem = fit_membrane(LABELS, ca)
        plddt = {i: (40.0 if 11 <= i <= 30 else 95.0) for i in ca}
        rows = segment_table(LABELS, SEQ, ca, mem, plddt)
        first_tm = next(r for r in rows if r["label"] == "M")
        self.assertIn("low confidence", first_tm["flags"])
        self.assertLess(first_tm["tilt_deg"], 3)

    def test_positive_inside(self):
        r = positive_inside(LABELS, SEQ)
        self.assertTrue(r["positive_inside_ok"])
        self.assertGreater(r["KR_inside"], 0)
        self.assertEqual(r["KR_outside"], 0)

    def test_no_tm(self):
        self.assertIsNone(fit_membrane("IIIIIIII", {i: np.zeros(3) + i for i in range(1, 9)}))


class TestStructures(unittest.TestCase):
    def test_pdb_roundtrip_and_renumber(self):
        xyz, _ = synthetic_ca()
        tagged = "HHH" + SEQ
        xyz_t = np.vstack([np.zeros((3, 3)), xyz])
        st = structure_from(xyz_t, tagged)
        residues, chain = _read_pdb_text(st.to_pdb(), None)
        self.assertEqual(len(residues), len(tagged))
        st2 = renumber_to_query(Structure(residues, "t"), SEQ)
        self.assertEqual(st2.residues[0].number, 1)
        self.assertAlmostEqual(st2.meta["coverage"], 1.0)
        np.testing.assert_allclose(st2.residues[0].atoms[0].xyz, xyz[0], atol=1e-3)


class TestEndToEnd(unittest.TestCase):
    def test_run_with_user_structure(self):
        xyz, _ = synthetic_ca()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdb = tmp / "model.pdb"
            pdb.write_text(structure_from(xyz, SEQ).to_pdb())
            cfg = Config(compare_uniprot=False, make_viewer_html=False,
                         user_structures={"synth": {"kind": "file", "value": str(pdb), "b_is_plddt": True}})
            topo = {"synth": Topology("synth", "TM", SEQ, LABELS)}
            summary, res = run([Protein("synth", SEQ)], tmp / "out", cfg, topologies=topo)
            row = summary.iloc[0]
            self.assertEqual(row["n_TM_helices"], 4)
            self.assertEqual(row["agreement"], "consistent")
            d = tmp / "out" / "proteins" / "synth"
            for f in ("model_membrane_oriented.pdb", "model_topology_bfactor.pdb", "segments.csv", "profile.png",
                      "color_topology.pml", "color_topology.cxc"):
                self.assertTrue((d / f).exists(), f)
            self.assertTrue((tmp / "out" / "report.html").exists())
            # oriented model: TM residues near z = 0, cytoplasmic loops at negative z
            oriented, _ = _read_pdb_text((d / "model_membrane_oriented.pdb").read_text(), "A")
            z = {r.number: r.atoms[0].xyz[2] for r in oriented}
            self.assertLess(np.mean([z[i] for i in range(1, 11)]), -10)
            self.assertLess(abs(np.mean([z[i] for i in range(11, 31)])), 5)
            # cache is reused on a second run
            summary2, _ = run([Protein("synth", SEQ)], tmp / "out", cfg, topologies=topo)
            self.assertEqual(summary2.iloc[0]["tm_in_slab"], row["tm_in_slab"])


if __name__ == "__main__":
    unittest.main()
