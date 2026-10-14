"""Tests for tally_xml.validate.check — standard library only.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml.validate import check  # noqa: E402


def load_json(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


class TestValidate(unittest.TestCase):
    def test_surviving_model_samples_pass_all_checks(self):
        samples = load_json("data/model_generated_samples.json")
        self.assertEqual(len(samples), 2, "exactly two surviving model samples")
        for s in samples:
            with self.subTest(instruction=s["instruction"][:40]):
                self.assertEqual(
                    check(s["model_output"]),
                    {"wellformed": 1, "schema": 1, "balanced": 1, "fields": 0},
                    "samples have no gold attached, so fields must be 0",
                )

    def test_gold_answers_pass_all_checks(self):
        tests = load_json("data/xml_test.json")
        self.assertEqual(len(tests), 60)
        for t in tests:
            with self.subTest(q=t["q"][:40]):
                self.assertEqual(check(t["a"], t["gold"]),
                                 {"wellformed": 1, "schema": 1, "balanced": 1, "fields": 1})

    def test_garbage_fails_all_checks(self):
        self.assertEqual(check("not xml at all"),
                         {"wellformed": 0, "schema": 0, "balanced": 0, "fields": 0})
        self.assertEqual(check(""),
                         {"wellformed": 0, "schema": 0, "balanced": 0, "fields": 0})

    def test_unbalanced_voucher_fails_balanced(self):
        xml = ("<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
               "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
               "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\"><DATE>20251214</DATE>"
               "<VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Bank</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-1000</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Customer</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>999</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "</VOUCHER></TALLYMESSAGE></DATA></BODY></ENVELOPE>")
        r = check(xml)
        self.assertEqual(r["wellformed"], 1)
        self.assertEqual(r["schema"], 1)
        self.assertEqual(r["balanced"], 0)

    def test_wrong_voucher_type_fails_schema(self):
        xml = ("<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
               "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
               "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\"><DATE>20251214</DATE>"
               "<VOUCHERTYPENAME>Journal</VOUCHERTYPENAME>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Bank</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-1000</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Customer</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>1000</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "</VOUCHER></TALLYMESSAGE></DATA></BODY></ENVELOPE>")
        self.assertEqual(check(xml)["schema"], 0)


if __name__ == "__main__":
    unittest.main()
