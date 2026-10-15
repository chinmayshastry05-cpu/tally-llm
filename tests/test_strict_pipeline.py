"""Fresh tests for the strict validator + deterministic pipeline.

DEVELOPER-WRITTEN, NOT ACCOUNTANT-REVIEWED, NOT PRODUCTION PROOF.
These tests are independent of the training templates: ledger names,
amounts, and dates are invented here (Acme Traders, City Bank, Zenith
Supplies, etc.), not copied from data/xml_train.json or data/xml_test.json.

Run from the repo root:  python3 -m unittest discover -s tests -v

Acceptance gate (this file): every invalid-date / missing-field case is
rejected, and no wrong accepted transaction appears in this set.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml.strict import validate_intent, validate_strict  # noqa: E402
from tally_xml.pipeline import run_pipeline  # noqa: E402
from tally_xml.inference import inference_status, run_adapter_inference  # noqa: E402
from tally_xml.generate import AdapterMissingError  # noqa: E402

# Caller-provided known ledgers for these tests. Deliberately NOT the
# training-data ledgers. The validator invents no ledger list.
KNOWN = {"City Bank", "Harbor Bank", "Acme Traders", "Zenith Supplies",
         "Maple Stores", "Northwind Traders", "Smith & Sons"}


def xml_voucher(date="20250721", vchtype="Receipt", vtype_name="Receipt",
                action="Create", entries=None, header_id="Vouchers"):
    """Build a voucher XML string for strict-validation tests."""
    if entries is None:
        entries = [
            ("City Bank", "Yes", "-5000"),
            ("Acme Traders", "No", "5000"),
        ]
    inner = "".join(
        f"<ALLLEDGERENTRIES.LIST><LEDGERNAME>{name}</LEDGERNAME>"
        f"<ISDEEMEDPOSITIVE>{deemed}</ISDEEMEDPOSITIVE>"
        f"<AMOUNT>{amount}</AMOUNT></ALLLEDGERENTRIES.LIST>"
        for name, deemed, amount in entries
    )
    vch = f' VCHTYPE="{vchtype}"' if vchtype is not None else ""
    act = f' ACTION="{action}"' if action is not None else ""
    date_el = f"<DATE>{date}</DATE>" if date is not None else ""
    vtn = f"<VOUCHERTYPENAME>{vtype_name}</VOUCHERTYPENAME>" if vtype_name is not None else ""
    return (
        "<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
        f"<TYPE>Data</TYPE><ID>{header_id}</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
        f"<VOUCHER{vch}{act}>{date_el}{vtn}{inner}</VOUCHER>"
        "</TALLYMESSAGE></DATA></BODY></ENVELOPE>"
    )


def valid_intent():
    return {
        "voucher_type": "Receipt",
        "date": "2025-07-21",
        "entries": [
            {"ledger": "City Bank", "side": "Dr", "amount": "5000.00"},
            {"ledger": "Acme Traders", "side": "Cr", "amount": "5000.00"},
        ],
    }


class TestStrictXmlDates(unittest.TestCase):
    def test_valid_receipt_accepted(self):
        ok, errors = validate_strict(xml_voucher(), KNOWN)
        self.assertTrue(ok, errors)

    def test_valid_payment_accepted(self):
        ok, errors = validate_strict(xml_voucher(vchtype="Payment", vtype_name="Payment"), KNOWN)
        self.assertTrue(ok, errors)

    def test_valid_journal_accepted(self):
        ok, errors = validate_strict(xml_voucher(vchtype="Journal", vtype_name="Journal"), KNOWN)
        self.assertTrue(ok, errors)

    def test_leap_year_date_accepted(self):
        ok, errors = validate_strict(xml_voucher(date="20240229"), KNOWN)
        self.assertTrue(ok, errors)

    def test_impossible_date_20261399_rejected(self):
        # THE HOLE: old check() accepted this (8-digit regex). Strict must reject.
        ok, errors = validate_strict(xml_voucher(date="20261399"), KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("real calendar date" in e for e in errors), errors)

    def test_feb30_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="20250230"), KNOWN)
        self.assertFalse(ok)

    def test_feb29_nonleap_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="20250229"), KNOWN)
        self.assertFalse(ok)

    def test_zero_date_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="00000000"), KNOWN)
        self.assertFalse(ok)

    def test_month13_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="20251301"), KNOWN)
        self.assertFalse(ok)

    def test_day32_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="20251232"), KNOWN)
        self.assertFalse(ok)

    def test_year_out_of_range_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="18991231"), KNOWN)
        self.assertFalse(ok)
        ok, _ = validate_strict(xml_voucher(date="21010101"), KNOWN)
        self.assertFalse(ok)

    def test_short_date_rejected(self):
        ok, _ = validate_strict(xml_voucher(date="2025072"), KNOWN)
        self.assertFalse(ok)


class TestStrictXmlFields(unittest.TestCase):
    def test_missing_date_rejected(self):
        ok, _ = validate_strict(xml_voucher(date=None), KNOWN)
        self.assertFalse(ok)

    def test_missing_voucher_type_name_rejected(self):
        ok, _ = validate_strict(xml_voucher(vtype_name=None), KNOWN)
        self.assertFalse(ok)

    def test_missing_vchtype_rejected(self):
        ok, _ = validate_strict(xml_voucher(vchtype=None), KNOWN)
        self.assertFalse(ok)

    def test_missing_action_rejected(self):
        ok, _ = validate_strict(xml_voucher(action=None), KNOWN)
        self.assertFalse(ok)

    def test_missing_ledgername_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-5000"), ("", "No", "5000")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_missing_amount_rejected(self):
        xml = ("<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
               "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
               "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\"><DATE>20250721</DATE>"
               "<VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE></ALLLEDGERENTRIES.LIST>"
               "</VOUCHER></TALLYMESSAGE></DATA></BODY></ENVELOPE>")
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_missing_isdeemedpositive_rejected(self):
        xml = ("<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
               "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
               "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\"><DATE>20250721</DATE>"
               "<VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
               "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
               "<AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
               "</VOUCHER></TALLYMESSAGE></DATA></BODY></ENVELOPE>")
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_unknown_ledger_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-5000"),
                                   ("Mystery Vendor Ltd", "No", "5000")])
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("not in the caller-provided known ledgers" in e for e in errors), errors)

    def test_known_ledgers_none_fails_closed(self):
        ok, errors = validate_strict(xml_voucher(), None)
        self.assertFalse(ok)
        self.assertTrue(any("known_ledgers not provided" in e for e in errors), errors)

    def test_wrong_voucher_type_rejected(self):
        # Sales is not in the narrow pipeline's default allowlist.
        ok, _ = validate_strict(xml_voucher(vchtype="Sales", vtype_name="Sales"), KNOWN)
        self.assertFalse(ok)

    def test_vchtype_mismatch_rejected(self):
        ok, _ = validate_strict(xml_voucher(vchtype="Receipt", vtype_name="Payment"), KNOWN)
        self.assertFalse(ok)

    def test_unbalanced_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-5000"),
                                   ("Acme Traders", "No", "4999")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_single_entry_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-5000")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)


class TestStrictXmlMoney(unittest.TestCase):
    def test_zero_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "0"),
                                   ("Acme Traders", "No", "0")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_comma_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-1,000"),
                                   ("Acme Traders", "No", "1,000")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_currency_symbol_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-Rs 5000"),
                                   ("Acme Traders", "No", "Rs 5000")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_scientific_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-1e3"),
                                   ("Acme Traders", "No", "1e3")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_nan_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-NaN"),
                                   ("Acme Traders", "No", "NaN")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_inf_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-Inf"),
                                   ("Acme Traders", "No", "Inf")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_sign_mismatch_rejected(self):
        # Dr (Yes) with a positive amount is a sign error.
        xml = xml_voucher(entries=[("City Bank", "Yes", "5000"),
                                   ("Acme Traders", "No", "5000")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_three_decimal_amount_rejected(self):
        xml = xml_voucher(entries=[("City Bank", "Yes", "-5000.123"),
                                   ("Acme Traders", "No", "5000.123")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_empty_ledgername_rejected(self):
        xml = xml_voucher(entries=[("   ", "Yes", "-5000"),
                                   ("Acme Traders", "No", "5000")])
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_malformed_xml_rejected(self):
        ok, _ = validate_strict("<ENVELOPE><HEADER>unclosed", KNOWN)
        self.assertFalse(ok)

    def test_empty_input_rejected(self):
        ok, _ = validate_strict("", KNOWN)
        self.assertFalse(ok)

    def test_non_xml_input_rejected(self):
        ok, _ = validate_strict("not xml at all", KNOWN)
        self.assertFalse(ok)


class TestIntentValidation(unittest.TestCase):
    def test_valid_intent_accepted(self):
        ok, errors = validate_intent(valid_intent(), KNOWN)
        self.assertTrue(ok, errors)

    def test_intent_extra_key_rejected(self):
        intent = valid_intent()
        intent["note"] = "extra field the model invented"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_missing_key_rejected(self):
        intent = valid_intent()
        del intent["date"]
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_amount_as_number_rejected(self):
        # Amount must be a STRING (Decimal precision). JSON float is rejected.
        intent = valid_intent()
        intent["entries"][0]["amount"] = 5000.00
        ok, errors = validate_intent(intent, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("must be a string" in e for e in errors), errors)

    def test_intent_negative_amount_rejected(self):
        intent = valid_intent()
        intent["entries"][0]["amount"] = "-5000.00"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_zero_amount_rejected(self):
        intent = valid_intent()
        intent["entries"][0]["amount"] = "0.00"
        intent["entries"][1]["amount"] = "0.00"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_comma_amount_rejected(self):
        intent = valid_intent()
        intent["entries"][0]["amount"] = "5,000.00"
        intent["entries"][1]["amount"] = "5,000.00"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_unbalanced_rejected(self):
        intent = valid_intent()
        intent["entries"][1]["amount"] = "4999.00"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_unknown_ledger_rejected(self):
        intent = valid_intent()
        intent["entries"][1]["ledger"] = "Ghost Supplier Inc"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_bad_side_rejected(self):
        intent = valid_intent()
        intent["entries"][0]["side"] = "Debit"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_bad_date_rejected(self):
        intent = valid_intent()
        intent["date"] = "2025-13-99"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_bad_voucher_type_rejected(self):
        intent = valid_intent()
        intent["voucher_type"] = "Sales"
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_single_entry_rejected(self):
        intent = valid_intent()
        intent["entries"] = intent["entries"][:1]
        ok, _ = validate_intent(intent, KNOWN)
        self.assertFalse(ok)

    def test_intent_non_dict_rejected(self):
        ok, _ = validate_intent("not an object", KNOWN)
        self.assertFalse(ok)
        ok, _ = validate_intent(None, KNOWN)
        self.assertFalse(ok)

    def test_intent_known_ledgers_none_fails_closed(self):
        ok, _ = validate_intent(valid_intent(), None)
        self.assertFalse(ok)


class TestPipeline(unittest.TestCase):
    def test_pipeline_receipt_end_to_end(self):
        result = run_pipeline(valid_intent(), KNOWN)
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("<ENVELOPE>", result["xml"])
        self.assertIn("20250721", result["xml"])
        # Built XML must pass the strict gate (round-trip).
        ok, errors = validate_strict(result["xml"], KNOWN)
        self.assertTrue(ok, errors)

    def test_pipeline_payment_end_to_end(self):
        intent = {
            "voucher_type": "Payment",
            "date": "2025-08-02",
            "entries": [
                {"ledger": "Zenith Supplies", "side": "Dr", "amount": "1200.50"},
                {"ledger": "Harbor Bank", "side": "Cr", "amount": "1200.50"},
            ],
        }
        result = run_pipeline(intent, KNOWN)
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("20250802", result["xml"])
        self.assertIn("Payment", result["xml"])

    def test_pipeline_journal_end_to_end(self):
        intent = {
            "voucher_type": "Journal",
            "date": "2024-02-29",  # leap-year date, valid
            "entries": [
                {"ledger": "Maple Stores", "side": "Dr", "amount": "750.00"},
                {"ledger": "Northwind Traders", "side": "Cr", "amount": "750.00"},
            ],
        }
        result = run_pipeline(intent, KNOWN)
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("20240229", result["xml"])

    def test_pipeline_rejects_invalid_intent(self):
        intent = valid_intent()
        intent["date"] = "20261399"
        result = run_pipeline(intent, KNOWN)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["xml"])
        self.assertTrue(result["errors"])

    def test_pipeline_escapes_ampersand_ledger(self):
        # "Smith & Sons" contains '&' — the builder must escape it to
        # "&amp;" in XML; strict validation must pass on the round-trip.
        intent = {
            "voucher_type": "Receipt",
            "date": "2025-07-21",
            "entries": [
                {"ledger": "City Bank", "side": "Dr", "amount": "100.00"},
                {"ledger": "Smith & Sons", "side": "Cr", "amount": "100.00"},
            ],
        }
        result = run_pipeline(intent, KNOWN)
        self.assertTrue(result["ok"], result["errors"])
        self.assertIn("Smith &amp; Sons", result["xml"])
        self.assertNotIn("Smith & Sons", result["xml"].replace("Smith &amp; Sons", ""))
        ok, errors = validate_strict(result["xml"], KNOWN)
        self.assertTrue(ok, errors)

    def test_pipeline_injection_like_ledger_rejected(self):
        # Ledger text that looks like XML tags. Not in known ledgers, so
        # rejected by the ledger-presence gate — never embedded raw.
        intent = {
            "voucher_type": "Receipt",
            "date": "2025-07-21",
            "entries": [
                {"ledger": "City Bank", "side": "Dr", "amount": "100.00"},
                {"ledger": "<LEDGERNAME>Fake</LEDGERNAME>", "side": "Cr", "amount": "100.00"},
            ],
        }
        result = run_pipeline(intent, KNOWN)
        self.assertFalse(result["ok"])
        self.assertNotIn("<LEDGERNAME>Fake</LEDGERNAME>", result["xml"] or "")

    def test_pipeline_sql_like_ledger_rejected(self):
        intent = {
            "voucher_type": "Receipt",
            "date": "2025-07-21",
            "entries": [
                {"ledger": "City Bank", "side": "Dr", "amount": "100.00"},
                {"ledger": "'; DROP TABLE vouchers; --", "side": "Cr", "amount": "100.00"},
            ],
        }
        result = run_pipeline(intent, KNOWN)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["xml"])

    def test_pipeline_non_dict_input_rejected(self):
        result = run_pipeline("just a string", KNOWN)
        self.assertFalse(result["ok"])
        self.assertIsNone(result["xml"])


class TestInferenceFailClosed(unittest.TestCase):
    def test_inference_fails_closed_no_adapter(self):
        # No adapter exists in this repo, so inference must refuse.
        with self.assertRaises(AdapterMissingError):
            run_adapter_inference("Record a receipt.", KNOWN)

    def test_inference_status_reports_blockers(self):
        status = inference_status()
        self.assertFalse(status["inference_ready"])
        self.assertFalse(status["generation_enabled"])
        self.assertTrue(status["blockers"], "must name concrete blockers")
        self.assertFalse(status["adapter_present"])


if __name__ == "__main__":
    unittest.main()
