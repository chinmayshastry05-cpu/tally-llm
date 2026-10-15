"""Deterministic receipt/payment/journal pipeline (production path).

Flow:
    JSON intent (model output, validated) -> deterministic XML builder
    -> strict XML re-validation -> XML string or errors.

The LLM NEVER does accounting arithmetic and NEVER produces free-form XML
on this path. The LLM's only job is to emit a structured JSON intent
(voucher_type, date, entries). Deterministic code:

- validates the intent (``strict.validate_intent``),
- applies Dr/Cr signs and builds the XML with :mod:`xml.etree` (which
  escapes ledger text — no string-concatenated XML),
- re-validates the built XML with ``strict.validate_strict`` (round-trip
  gate). If the built XML fails, the pipeline fails closed.

``run_pipeline`` never raises for invalid input — it returns
``{"ok": bool, "xml": str|None, "errors": [...]}``. Callers must check
``ok`` before using ``xml``.

Fixture JSON intents exercise this pipeline deterministically. Fixture
runs are NOT model tests: they prove the deterministic code works, not
that any model can emit valid intents.
"""
import xml.etree.ElementTree as ET
from decimal import Decimal

from .strict import validate_intent, validate_strict


def _tally_date(iso_date):
    """Convert ISO YYYY-MM-DD to Tally YYYYMMDD. Caller validated it."""
    return iso_date.replace("-", "")


def build_voucher_xml(intent, known_ledgers):
    """Build TallyPrime XML from a validated JSON intent.

    Raises ValueError with the validation errors if the intent is invalid
    or if the built XML fails the strict round-trip gate. On success
    returns the XML string.
    """
    ok, errors = validate_intent(intent, known_ledgers)
    if not ok:
        raise ValueError(f"invalid intent: {'; '.join(errors)}")

    envelope = ET.Element("ENVELOPE")
    header = ET.SubElement(envelope, "HEADER")
    ET.SubElement(header, "VERSION").text = "1"
    ET.SubElement(header, "TALLYREQUEST").text = "Import"
    ET.SubElement(header, "TYPE").text = "Data"
    ET.SubElement(header, "ID").text = "Vouchers"
    body = ET.SubElement(envelope, "BODY")
    ET.SubElement(body, "DESC")
    data = ET.SubElement(body, "DATA")
    message = ET.SubElement(data, "TALLYMESSAGE")
    voucher = ET.SubElement(message, "VOUCHER")
    voucher.set("VCHTYPE", intent["voucher_type"])
    voucher.set("ACTION", "Create")
    voucher.set("OBJVIEW", "Accounting Voucher View")
    ET.SubElement(voucher, "DATE").text = _tally_date(intent["date"])
    ET.SubElement(voucher, "VOUCHERTYPENAME").text = intent["voucher_type"]
    for entry in intent["entries"]:
        item = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(item, "LEDGERNAME").text = entry["ledger"]
        if entry["side"] == "Dr":
            ET.SubElement(item, "ISDEEMEDPOSITIVE").text = "Yes"
            ET.SubElement(item, "AMOUNT").text = f"-{entry['amount']}"
        else:
            ET.SubElement(item, "ISDEEMEDPOSITIVE").text = "No"
            ET.SubElement(item, "AMOUNT").text = entry["amount"]

    xml_text = ET.tostring(envelope, encoding="unicode")
    # Round-trip gate: the built XML must pass the strict validator.
    ok_xml, xml_errors = validate_strict(xml_text, known_ledgers)
    if not ok_xml:
        raise ValueError(f"built XML failed strict validation (builder bug): {'; '.join(xml_errors)}")
    return xml_text


def run_pipeline(intent, known_ledgers):
    """Run the full pipeline. Returns {"ok", "xml", "errors"}.

    ``intent`` may be a dict (parsed JSON) or any other value — non-dict
    input is reported as an error, never raised.
    """
    if not isinstance(intent, dict):
        return {"ok": False, "xml": None, "errors": ["intent must be a JSON object"]}
    ok, errors = validate_intent(intent, known_ledgers)
    if not ok:
        return {"ok": False, "xml": None, "errors": errors}
    try:
        xml_text = build_voucher_xml(intent, known_ledgers)
    except ValueError as e:
        return {"ok": False, "xml": None, "errors": [str(e)]}
    return {"ok": True, "xml": xml_text, "errors": []}


# Backwards-compatible alias: the pipeline's money check uses Decimal.
def decimal_sum(amounts):
    """Exact Decimal sum (helper for callers/tests)."""
    total = Decimal("0")
    for a in amounts:
        total += a
    return total
