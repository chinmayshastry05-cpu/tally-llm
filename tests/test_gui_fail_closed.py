"""Fail-closed tests: generation must refuse while the adapter is missing,
and must never return template XML. Standard library only.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml.generate import (  # noqa: E402
    AdapterMissingError,
    adapter_status,
    find_adapter,
    generate_voucher,
)


class TestFailClosed(unittest.TestCase):
    def test_no_adapter_present_in_clean_checkout(self):
        self.assertIsNone(find_adapter())
        self.assertFalse(adapter_status()["generation_enabled"])

    def test_generate_refuses_without_adapter(self):
        with self.assertRaises(AdapterMissingError):
            generate_voucher("Record a receipt of Rs 5,000 from Iyer Foods into HDFC Bank.")

    def test_refusal_mentions_missing_adapter(self):
        try:
            generate_voucher("anything")
        except AdapterMissingError as e:
            self.assertIn("adapter", str(e).lower())
            self.assertNotIn("<ENVELOPE>", str(e), "refusal must not smuggle out XML")


if __name__ == "__main__":
    unittest.main()
