"""Strict validator for TallyPrime XML import messages (production gate).

This is the production gate for the narrow receipt/payment/journal
pipeline. It is STRICTER than :func:`tally_xml.validate.check` (the
structural subset validator used by the GUI/samples):

- Real calendar dates (``datetime.date``), not an 8-digit regex. 20261399,
  20250230, 20250229 (non-leap), 00000000 are all rejected.
- Required fields: DATE, VOCHERTYPENAME, every entry's LEDGERNAME /
  ISDEEMEDPOSITIVE / AMOUNT, VCHTYPE, ACTION.
- Allowed voucher types: caller-configurable, default Receipt/Payment/
  Journal (the narrow pipeline's scope). Anything else is rejected.
- Money as :class:`decimal.Decimal` (never float): finite, non-zero, at
  most 2 decimal places, strict numeric format. Commas, currency symbols,
  scientific notation, whitespace, NaN/Inf, empty strings are rejected.
- Balance: exact Decimal sum == 0 (no float tolerance).
- Ledger presence: every LEDGERNAME must be in the caller-provided
  ``known_ledgers`` set. ``known_ledgers`` is REQUIRED for voucher
  messages; ``None`` fails closed. No ledger list is invented here.
- Other impossible values: year outside 1900-2100, empty/whitespace-only
  names, missing ACTION, sign/ISDEEMEDPOSITIVE mismatch.

Returns ``(ok, errors)`` where ``errors`` is a list of human-readable
strings. Never raises for invalid input — invalid input returns
``(False, [...])``. This is the fail-closed gate: callers must check
``ok`` before using any output.

For JSON intent validation (the model-output side), see
:func:`validate_intent` in this module.
"""
import datetime
import re
from decimal import Decimal, InvalidOperation
import xml.etree.ElementTree as ET

# Narrow pipeline scope. Callers widen explicitly via the parameter.
DEFAULT_ALLOWED_VOUCHER_TYPES = frozenset({"Receipt", "Payment", "Journal"})

_MIN_YEAR = 1900
_MAX_YEAR = 2100

# Strict money format: digits, optional decimal with 1-2 places. No sign
# here — the sign is applied by the builder from Dr/Cr. No commas, no
# currency symbols, no scientific notation, no whitespace, no empty string.
_MONEY_RE = re.compile(r"\d+(\.\d{1,2})?")

# Strict ISO date for JSON intent: YYYY-MM-DD.
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Strict Tally date in XML: YYYYMMDD, 8 digits.
_TALLY_DATE_RE = re.compile(r"\d{8}")


def _parse_decimal(text):
    """Parse a strict decimal money string. Returns (Decimal, error|None).

    Rejects: non-string, empty/whitespace, commas, currency symbols,
    scientific notation, more than 2 decimal places, zero, NaN/Inf,
    non-numeric text. Leading '+' is rejected (ambiguous). Leading '-'
    is rejected here — the builder applies the sign from Dr/Cr.
    """
    if not isinstance(text, str):
        return None, f"amount must be a string, got {type(text).__name__}"
    if text != text.strip() or not text:
        return None, "amount must be a non-empty string without surrounding whitespace"
    if not _MONEY_RE.fullmatch(text):
        return None, f"amount {text!r} is not a strict decimal (digits, optional 1-2 decimal places)"
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None, f"amount {text!r} is not a valid decimal"
    if not value.is_finite():
        return None, f"amount {text!r} is not finite"
    if value <= 0:
        return None, f"amount {text!r} must be positive (sign comes from Dr/Cr side)"
    return value, None


def _check_real_date(yyyymmdd, errors):
    """Validate an 8-digit Tally date as a real calendar date."""
    if not _TALLY_DATE_RE.fullmatch(yyyymmdd or ""):
        errors.append(f"DATE {yyyymmdd!r} is not an 8-digit YYYYMMDD string")
        return None
    year = int(yyyymmdd[0:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])
    if not (_MIN_YEAR <= year <= _MAX_YEAR):
        errors.append(f"DATE {yyyymmdd!r} year {year} outside {_MIN_YEAR}-{_MAX_YEAR}")
        return None
    try:
        return datetime.date(year, month, day)
    except ValueError:
        errors.append(f"DATE {yyyymmdd!r} is not a real calendar date")
        return None


def _check_ledger_name(name, known_ledgers, errors, where):
    """Validate a ledger name: non-empty, no control chars, in known set."""
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{where}: LEDGERNAME is missing or empty")
        return
    if name != name.strip():
        errors.append(f"{where}: LEDGERNAME {name!r} has surrounding whitespace")
        return
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        errors.append(f"{where}: LEDGERNAME contains control characters")
        return
    if known_ledgers is None:
        errors.append(f"{where}: known_ledgers not provided — cannot verify ledger presence (fail-closed)")
        return
    if name not in known_ledgers:
        errors.append(f"{where}: LEDGERNAME {name!r} is not in the caller-provided known ledgers")


def validate_strict(xml_text, known_ledgers=None, allowed_voucher_types=None):
    """Strictly validate a TallyPrime XML import message.

    Args:
        xml_text: the XML string to validate.
        known_ledgers: set/list of allowed ledger names (REQUIRED for
            voucher messages; ``None`` fails closed). Not used for LEDGER
            creation messages (a new ledger is being created).
        allowed_voucher_types: set of allowed VCHTYPE values. Defaults to
            Receipt/Payment/Journal.

    Returns:
        (ok, errors): ok is True only if every check passes.
    """
    errors = []
    allowed = frozenset(allowed_voucher_types) if allowed_voucher_types is not None else DEFAULT_ALLOWED_VOUCHER_TYPES
    if not isinstance(xml_text, str) or not xml_text.strip():
        return False, ["input is empty or not a string"]
    text = xml_text.strip()
    # Isolate the ENVELOPE block (same convention as validate.check).
    m = re.search(r"<ENVELOPE>.*</ENVELOPE>", text, re.S)
    if m:
        text = m.group(0)
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        return False, [f"XML is not well-formed: {e}"]

    def g(path):
        return (root.findtext(path) or "").strip()

    if root.tag != "ENVELOPE":
        errors.append("root element must be ENVELOPE")
        return False, errors
    if g("HEADER/TALLYREQUEST") != "Import":
        errors.append("HEADER/TALLYREQUEST must be 'Import'")
    if g("HEADER/TYPE") != "Data":
        errors.append("HEADER/TYPE must be 'Data'")
    tm = root.find("BODY/DATA/TALLYMESSAGE")
    if tm is None:
        errors.append("BODY/DATA/TALLYMESSAGE is missing")
        return False, errors

    voucher = tm.find("VOUCHER")
    ledger = tm.find("LEDGER")
    if ledger is not None:
        return _validate_ledger_strict(tm, ledger, g, errors)
    if voucher is None:
        errors.append("TALLYMESSAGE must contain VOUCHER or LEDGER")
        return False, errors
    return _validate_voucher_strict(voucher, g, known_ledgers, allowed, errors)


def _validate_ledger_strict(tm, ledger, g, errors):
    """Strict checks for a LEDGER creation message."""
    if g("HEADER/ID") != "All Masters":
        errors.append("HEADER/ID must be 'All Masters' for a LEDGER message")
    name = ledger.get("NAME")
    if not name or not name.strip():
        errors.append("LEDGER NAME attribute is missing or empty")
    parent = ledger.findtext("PARENT")
    if not parent or not parent.strip():
        errors.append("LEDGER PARENT is missing or empty")
    name_list = ledger.findtext("NAME.LIST/NAME")
    if name and name_list != name:
        errors.append("LEDGER NAME.LIST/NAME must equal the NAME attribute")
    return (not errors), errors


def _validate_voucher_strict(voucher, g, known_ledgers, allowed, errors):
    """Strict checks for a VOUCHER message."""
    if g("HEADER/ID") != "Vouchers":
        errors.append("HEADER/ID must be 'Vouchers' for a VOUCHER message")
    vchtype = voucher.get("VCHTYPE")
    if not vchtype:
        errors.append("VOUCHER VCHTYPE attribute is missing")
    elif vchtype not in allowed:
        errors.append(f"VOUCHER VCHTYPE {vchtype!r} is not in allowed types {sorted(allowed)}")
    vtype_name = voucher.findtext("VOUCHERTYPENAME")
    if not vtype_name or not vtype_name.strip():
        errors.append("VOUCHERTYPENAME is missing or empty")
    elif vchtype and vtype_name.strip() != vchtype:
        errors.append(f"VOUCHERTYPENAME {vtype_name.strip()!r} must equal VCHTYPE {vchtype!r}")
    action = voucher.get("ACTION")
    if action != "Create":
        errors.append(f"VOUCHER ACTION must be 'Create', got {action!r}")

    date_text = (voucher.findtext("DATE") or "").strip()
    if not date_text:
        errors.append("VOUCHER DATE is missing or empty")
    else:
        _check_real_date(date_text, errors)

    entries = voucher.findall("ALLLEDGERENTRIES.LIST")
    if len(entries) < 2:
        errors.append(f"VOUCHER must have at least 2 ledger entries, got {len(entries)}")

    amounts = []
    has_dr = False
    has_cr = False
    for i, entry in enumerate(entries):
        where = f"entry {i + 1}"
        ledger_name = (entry.findtext("LEDGERNAME") or "")
        _check_ledger_name(ledger_name, known_ledgers, errors, where)
        deemed = (entry.findtext("ISDEEMEDPOSITIVE") or "").strip()
        if deemed not in ("Yes", "No"):
            errors.append(f"{where}: ISDEEMEDPOSITIVE must be 'Yes' or 'No', got {deemed!r}")
            continue
        amount_text = (entry.findtext("AMOUNT") or "")
        # XML AMOUNT is signed (Dr negative, Cr positive). Parse the signed
        # form strictly: optional leading '-', then the strict decimal.
        signed = amount_text.strip()
        if not signed or signed != amount_text.strip():
            errors.append(f"{where}: AMOUNT is missing or has surrounding whitespace")
            continue
        body = signed[1:] if signed.startswith("-") else signed
        if signed.startswith("+") or not _MONEY_RE.fullmatch(body or ""):
            errors.append(f"{where}: AMOUNT {amount_text!r} is not a strict signed decimal")
            continue
        try:
            value = Decimal(signed)
        except InvalidOperation:
            errors.append(f"{where}: AMOUNT {amount_text!r} is not a valid decimal")
            continue
        if not value.is_finite():
            errors.append(f"{where}: AMOUNT {amount_text!r} is not finite")
            continue
        if value == 0:
            errors.append(f"{where}: AMOUNT must be non-zero")
            continue
        # Sign consistency: Dr (deemed Yes) is negative; Cr (deemed No) is positive.
        if deemed == "Yes" and not (value < 0):
            errors.append(f"{where}: ISDEEMEDPOSITIVE 'Yes' (Dr) requires a negative AMOUNT, got {amount_text!r}")
            continue
        if deemed == "No" and not (value > 0):
            errors.append(f"{where}: ISDEEMEDPOSITIVE 'No' (Cr) requires a positive AMOUNT, got {amount_text!r}")
            continue
        amounts.append(value)
        if value < 0:
            has_dr = True
        else:
            has_cr = True

    if entries and not (has_dr and has_cr):
        errors.append("VOUCHER must have at least one Dr entry and one Cr entry")
    if amounts and sum(amounts, Decimal("0")) != 0:
        errors.append(f"VOUCHER is unbalanced: Decimal sum is {sum(amounts, Decimal('0'))}, must be exactly 0")
    return (not errors), errors


# ---------------------------------------------------------------------------
# JSON intent validation (model-output side)
# ---------------------------------------------------------------------------

_INTENT_REQUIRED = {"voucher_type", "date", "entries"}
_ENTRY_REQUIRED = {"ledger", "side", "amount"}
_INTENT_ALLOWED_TYPES = frozenset({"Receipt", "Payment", "Journal"})


def validate_intent(intent, known_ledgers):
    """Validate a structured JSON intent (model output) for the pipeline.

    Intent schema (exact — extra keys are rejected):
        {
          "voucher_type": "Receipt" | "Payment" | "Journal",
          "date": "YYYY-MM-DD" (real calendar date),
          "entries": [
            {"ledger": "<name>", "side": "Dr" | "Cr", "amount": "<decimal string>"},
            ...
          ]
        }

    Amount is a STRING (never a JSON number) so Decimal precision is exact.
    ``known_ledgers`` is REQUIRED; ``None`` fails closed.

    Returns (ok, errors). Never raises for invalid input.
    """
    errors = []
    if not isinstance(intent, dict):
        return False, ["intent must be a JSON object"]
    extra = set(intent.keys()) - _INTENT_REQUIRED
    if extra:
        errors.append(f"intent has unexpected keys: {sorted(extra)}")
    missing = _INTENT_REQUIRED - set(intent.keys())
    if missing:
        errors.append(f"intent is missing required keys: {sorted(missing)}")
        return False, errors

    vtype = intent.get("voucher_type")
    if vtype not in _INTENT_ALLOWED_TYPES:
        errors.append(f"voucher_type must be one of {sorted(_INTENT_ALLOWED_TYPES)}, got {vtype!r}")

    date_text = intent.get("date")
    if not isinstance(date_text, str) or not _ISO_DATE_RE.fullmatch(date_text or ""):
        errors.append(f"date must be an ISO YYYY-MM-DD string, got {date_text!r}")
    else:
        year, month, day = int(date_text[0:4]), int(date_text[5:7]), int(date_text[8:10])
        if not (_MIN_YEAR <= year <= _MAX_YEAR):
            errors.append(f"date year {year} outside {_MIN_YEAR}-{_MAX_YEAR}")
        else:
            try:
                datetime.date(year, month, day)
            except ValueError:
                errors.append(f"date {date_text!r} is not a real calendar date")

    entries = intent.get("entries")
    if not isinstance(entries, list):
        errors.append("entries must be a list")
        return False, errors
    if len(entries) < 2:
        errors.append(f"entries must have at least 2 items, got {len(entries)}")

    dr_total = Decimal("0")
    cr_total = Decimal("0")
    has_dr = False
    has_cr = False
    for i, entry in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        extra_e = set(entry.keys()) - _ENTRY_REQUIRED
        if extra_e:
            errors.append(f"{where} has unexpected keys: {sorted(extra_e)}")
        missing_e = _ENTRY_REQUIRED - set(entry.keys())
        if missing_e:
            errors.append(f"{where} is missing required keys: {sorted(missing_e)}")
            continue
        _check_ledger_name(entry.get("ledger"), known_ledgers, errors, where)
        side = entry.get("side")
        if side not in ("Dr", "Cr"):
            errors.append(f"{where}: side must be 'Dr' or 'Cr', got {side!r}")
            continue
        value, err = _parse_decimal(entry.get("amount"))
        if err is not None:
            errors.append(f"{where}: {err}")
            continue
        if side == "Dr":
            dr_total += value
            has_dr = True
        else:
            cr_total += value
            has_cr = True
    if entries and not (has_dr and has_cr):
        errors.append("entries must include at least one Dr and one Cr entry")
    if has_dr and has_cr and dr_total != cr_total:
        errors.append(f"entries are unbalanced: Dr total {dr_total} != Cr total {cr_total}")
    return (not errors), errors
