"""Fixture integrity tests: counts and shapes of the published data files.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_json(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


def load_jsonl(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestFixtures(unittest.TestCase):
    def test_phase1_test_set(self):
        rows = load_jsonl("data/test_set.jsonl")
        self.assertEqual(len(rows), 64)
        for r in rows:
            self.assertEqual(set(r), {"id", "category", "question", "reference", "must"})

    def test_phase2_train(self):
        rows = load_json("data/xml_train.json")
        self.assertEqual(len(rows), 700)
        for r in rows:
            self.assertEqual(set(r), {"q", "a"})

    def test_phase2_test(self):
        rows = load_json("data/xml_test.json")
        self.assertEqual(len(rows), 60)
        for r in rows:
            self.assertEqual(set(r), {"q", "a", "gold"})
            self.assertIn(r["gold"]["kind"], ("voucher", "ledger"))

    def test_model_samples(self):
        samples = load_json("data/model_generated_samples.json")
        self.assertEqual(len(samples), 2)
        for s in samples:
            self.assertIn("model_output", s)
            self.assertTrue(s["model_output"].startswith("<ENVELOPE>"))
            self.assertIn("provenance", s)

    def test_no_scraped_docs_published(self):
        # chunks.jsonl (scraped Tally help text) must never be published.
        names = []
        for dirpath, _, files in os.walk(ROOT):
            if ".git" in dirpath:
                continue
            names.extend(files)
        self.assertNotIn("chunks.jsonl", names)


if __name__ == "__main__":
    unittest.main()
