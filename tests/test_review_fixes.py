"""Review-fix tests (independent review, 2026-10-15).

1. Strict validator: reject surrounding prose/trailing text, multiple
   envelopes/messages/vouchers, voucher+ledger combos, duplicate required
   nodes, unknown payload fields/attributes; amount whitespace check fixed.
2. GUI: Validate + Samples use the strict gate with caller-supplied known
   ledgers; legacy checks are research only, never the verdict/download gate.
3. Inference: adapter_dir override threaded everywhere; base resolved from
   adapter config + allowlisted; base loaded explicitly then adapter once;
   local-files-only, no trust_remote_code, known ledgers in prompt, bounds,
   Qwen3 thinking disabled.
4. Docs: "production gate/path" renamed to prototype acceptance gate/path.

Inference load-path tests use mocks and are WIRING tests ONLY — they prove
the loading code calls the right functions with the right arguments. They
are NOT inference proof: no real adapter, deps, or GPU exists here.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml.strict import validate_strict  # noqa: E402
from tally_xml.gui import Handler, sample_strict_result  # noqa: E402
from tally_xml import generate as generate_mod  # noqa: E402
from tally_xml import inference as inference_mod  # noqa: E402
from tally_xml.generate import AdapterMissingError  # noqa: E402

KNOWN = {"City Bank", "Acme Traders", "Harbor Bank"}

VOUCHER_INNER = (
    "<DATE>20250721</DATE><VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
    "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
    "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
    "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
    "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
)

VALID = (
    "<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
    "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
    f"<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">{VOUCHER_INNER}</VOUCHER>"
    "</TALLYMESSAGE></DATA></BODY></ENVELOPE>"
)

LEDGER_MSG = (
    "<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
    "<TYPE>Data</TYPE><ID>All Masters</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
    "<LEDGER NAME=\"Acme Traders\"><PARENT>Sundry Debtors</PARENT>"
    "<NAME.LIST><NAME>Acme Traders</NAME></NAME.LIST></LEDGER>"
    "</TALLYMESSAGE></DATA></BODY></ENVELOPE>"
)


def _replace_voucher(xml, new_voucher):
    """Swap the VOUCHER element of the valid message for a crafted one."""
    start = xml.index("<VOUCHER")
    end = xml.index("</VOUCHER>") + len("</VOUCHER>")
    return xml[:start] + new_voucher + xml[end:]


class TestStrictAdversarial(unittest.TestCase):
    """Nothing is silently ignored: extra content is rejected, not stripped."""

    def test_valid_still_passes(self):
        ok, errors = validate_strict(VALID, KNOWN)
        self.assertTrue(ok, errors)

    def test_valid_ledger_still_passes(self):
        ok, errors = validate_strict(LEDGER_MSG, KNOWN)
        self.assertTrue(ok, errors)

    def test_trailing_prose_rejected(self):
        ok, errors = validate_strict(VALID + "\nPlease import this, thanks!", KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("trailing" in e for e in errors), errors)

    def test_leading_prose_rejected(self):
        ok, errors = validate_strict("Here is the voucher:\n" + VALID, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("surrounding" in e or "start with" in e for e in errors), errors)

    def test_two_envelopes_rejected(self):
        ok, _ = validate_strict(VALID + VALID, KNOWN)
        self.assertFalse(ok)

    def test_trailing_comment_rejected(self):
        ok, _ = validate_strict(VALID + "<!-- done -->", KNOWN)
        self.assertFalse(ok)

    def test_xml_declaration_prefix_allowed(self):
        ok, errors = validate_strict('<?xml version="1.0"?>' + VALID, KNOWN)
        self.assertTrue(ok, errors)

    def test_two_tallymessages_rejected(self):
        xml = VALID.replace(
            "</TALLYMESSAGE></DATA>",
            "</TALLYMESSAGE><TALLYMESSAGE><VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            + VOUCHER_INNER + "</VOUCHER></TALLYMESSAGE></DATA>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("TALLYMESSAGE" in e for e in errors), errors)

    def test_voucher_and_ledger_combo_rejected(self):
        xml = VALID.replace(
            "</VOUCHER></TALLYMESSAGE>",
            "</VOUCHER><LEDGER NAME=\"Acme Traders\"><PARENT>Sundry Debtors</PARENT>"
            "<NAME.LIST><NAME>Acme Traders</NAME></NAME.LIST></LEDGER></TALLYMESSAGE>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("exactly one of VOUCHER or LEDGER" in e for e in errors), errors)

    def test_two_vouchers_rejected(self):
        # Second voucher is VALID — multiplicity alone must reject.
        xml = VALID.replace(
            "</VOUCHER></TALLYMESSAGE>",
            "</VOUCHER><VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            + VOUCHER_INNER + "</VOUCHER></TALLYMESSAGE>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("exactly one of VOUCHER or LEDGER" in e for e in errors), errors)

    def test_valid_first_invalid_second_voucher_rejected(self):
        # The old code checked only the first voucher and ignored the rest.
        bad_inner = VOUCHER_INNER.replace("20250721", "20261399")
        xml = VALID.replace(
            "</VOUCHER></TALLYMESSAGE>",
            "</VOUCHER><VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            + bad_inner + "</VOUCHER></TALLYMESSAGE>")
        ok, _ = validate_strict(xml, KNOWN)
        self.assertFalse(ok)

    def test_duplicate_date_rejected(self):
        bad = (
            "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            "<DATE>20250721</DATE><DATE>20250721</DATE>"
            "<VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "</VOUCHER>"
        )
        ok, errors = validate_strict(_replace_voucher(VALID, bad), KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("duplicate <DATE>" in e for e in errors), errors)

    def test_duplicate_ledgername_in_entry_rejected(self):
        bad = (
            "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            "<DATE>20250721</DATE><VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
            "<LEDGERNAME>City Bank</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "</VOUCHER>"
        )
        ok, errors = validate_strict(_replace_voucher(VALID, bad), KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("duplicate <LEDGERNAME>" in e for e in errors), errors)

    def test_unknown_voucher_child_rejected(self):
        bad = (
            "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            "<DATE>20250721</DATE><VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
            "<NARRATION>hidden note</NARRATION>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "</VOUCHER>"
        )
        ok, errors = validate_strict(_replace_voucher(VALID, bad), KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("unknown children" in e for e in errors), errors)

    def test_unknown_entry_child_rejected(self):
        bad = (
            "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
            "<DATE>20250721</DATE><VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT>"
            "<COSTCENTRE>HQ</COSTCENTRE></ALLLEDGERENTRIES.LIST>"
            "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
            "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
            "</VOUCHER>"
        )
        ok, errors = validate_strict(_replace_voucher(VALID, bad), KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("unknown children" in e for e in errors), errors)

    def test_unknown_header_child_rejected(self):
        xml = VALID.replace("<TYPE>Data</TYPE>", "<TYPE>Data</TYPE><FOO>bar</FOO>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("unknown children" in e for e in errors), errors)

    def test_unknown_voucher_attribute_rejected(self):
        xml = VALID.replace(
            '<VOUCHER VCHTYPE="Receipt" ACTION="Create">',
            '<VOUCHER VCHTYPE="Receipt" ACTION="Create" FOO="bar">')
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("unknown attributes" in e for e in errors), errors)

    def test_duplicate_header_rejected(self):
        xml = VALID.replace(
            "</HEADER>",
            "</HEADER><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
            "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("duplicate <HEADER>" in e for e in errors), errors)

    def test_unknown_body_child_rejected(self):
        xml = VALID.replace("<DATA>", "<FOO>x</FOO><DATA>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("unknown children" in e for e in errors), errors)

    def test_amount_whitespace_rejected(self):
        # The old check compared stripped-to-stripped and never fired.
        bad = VOUCHER_INNER.replace("<AMOUNT>-5000</AMOUNT>", "<AMOUNT> -5000 </AMOUNT>")
        xml = VALID.replace(VOUCHER_INNER, bad)
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("surrounding whitespace" in e for e in errors), errors)

    def test_ledger_unknown_child_rejected(self):
        xml = LEDGER_MSG.replace(
            "<PARENT>Sundry Debtors</PARENT>",
            "<PARENT>Sundry Debtors</PARENT><FOO>x</FOO>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("unknown children" in e for e in errors), errors)

    def test_ledger_duplicate_parent_rejected(self):
        xml = LEDGER_MSG.replace(
            "<PARENT>Sundry Debtors</PARENT>",
            "<PARENT>Sundry Debtors</PARENT><PARENT>Sundry Debtors</PARENT>")
        ok, errors = validate_strict(xml, KNOWN)
        self.assertFalse(ok)
        self.assertTrue(any("duplicate <PARENT>" in e for e in errors), errors)


def _get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(port, path, data):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class TestGuiStrictGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_valid_with_caller_ledgers_passes_strict(self):
        code, body = _post(self.port, "/api/validate", json.dumps({
            "xml": VALID, "known_ledgers": sorted(KNOWN)}).encode())
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertTrue(payload["strict_ok"], payload["strict_errors"])
        self.assertIn("legacy", payload)
        self.assertIn("research only", payload["legacy_label"])

    def test_impossible_date_strict_fails_while_legacy_passes(self):
        # Legacy check() accepts 20261399 (8-digit regex). Strict must be
        # the verdict: strict_ok False even though legacy schema/balanced = 1.
        xml = VALID.replace("20250721", "20261399")
        code, body = _post(self.port, "/api/validate", json.dumps({
            "xml": xml, "known_ledgers": sorted(KNOWN)}).encode())
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertFalse(payload["strict_ok"])
        self.assertTrue(any("real calendar date" in e for e in payload["strict_errors"]))
        self.assertEqual(payload["legacy"]["schema"], 1)
        self.assertEqual(payload["legacy"]["balanced"], 1)

    def test_missing_known_ledgers_fails_closed(self):
        code, body = _post(self.port, "/api/validate", json.dumps({"xml": VALID}).encode())
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertFalse(payload["strict_ok"])
        self.assertTrue(any("known_ledgers not provided" in e for e in payload["strict_errors"]))

    def test_empty_known_ledgers_fails_closed(self):
        code, body = _post(self.port, "/api/validate", json.dumps({
            "xml": VALID, "known_ledgers": []}).encode())
        self.assertEqual(code, 200)
        self.assertFalse(json.loads(body)["strict_ok"])

    def test_unknown_ledger_strict_fails(self):
        code, body = _post(self.port, "/api/validate", json.dumps({
            "xml": VALID, "known_ledgers": ["City Bank"]}).encode())
        self.assertEqual(code, 200)
        payload = json.loads(body)
        self.assertFalse(payload["strict_ok"])
        self.assertTrue(any("not in the caller-provided known ledgers" in e for e in payload["strict_errors"]))

    def test_samples_strict_gated_on_fixture_context(self):
        code, body = _get(self.port, "/api/samples")
        self.assertEqual(code, 200)
        samples = json.loads(body)
        self.assertEqual(len(samples), 2)
        for s in samples:
            self.assertIn("strict", s)
            self.assertTrue(s["strict"]["ok"], s["strict"]["errors"])
            self.assertTrue(s["strict"]["fixture_ledgers"])
            self.assertIn("NOT real company masters", s["fixture_context"])
            self.assertIn("validation", s)  # legacy still present, research only

    def test_sample_download_allowed_when_strict_passes(self):
        for n in (0, 1):
            code, body = _get(self.port, f"/api/download/sample?n={n}")
            self.assertEqual(code, 200, body[:200])
            self.assertIn(b"<ENVELOPE>", body)

    def test_sample_gate_blocks_mutated_sample(self):
        mutated = {"model_output": VALID.replace("20250721", "20261399")}
        result = sample_strict_result(mutated)
        self.assertFalse(result["ok"])

    def test_page_has_ledgers_input_and_strict_labels(self):
        code, body = _get(self.port, "/")
        self.assertEqual(code, 200)
        page = body.decode()
        self.assertIn('id="ledgers"', page)
        self.assertIn("Strict acceptance gate", page)
        self.assertIn("research only", page)
        self.assertIn("not</strong> a complete Tally schema validator", page)
        self.assertIn("not</strong> enable generation", page)


def _make_adapter_dir(base="Qwen/Qwen3-0.6B"):
    """Temp dir with complete fake adapter + tokenizer files (no real weights)."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "adapter_config.json"), "w") as f:
        json.dump({"base_model_name_or_path": base}, f)
    with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
        f.write(b"fake-weights-not-real")
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump({}, f)
    return d


_VALID_INTENT = {
    "voucher_type": "Receipt",
    "date": "2025-07-21",
    "entries": [
        {"ledger": "City Bank", "side": "Dr", "amount": "100.00"},
        {"ledger": "Acme Traders", "side": "Cr", "amount": "100.00"},
    ],
}


def _mock_ml_modules(decode_text=None):
    """MagicMock ML modules. WIRING ONLY — not inference proof."""
    transformers = MagicMock()
    torch_mod = MagicMock()
    peft = MagicMock()
    tok = MagicMock()
    tok.apply_chat_template.return_value = "prompt text"
    tok.return_value = {"input_ids": MagicMock()}
    tok.decode.return_value = decode_text if decode_text is not None else json.dumps(_VALID_INTENT)
    transformers.AutoTokenizer.from_pretrained.return_value = tok
    transformers.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    peft.PeftModel.from_pretrained.return_value = MagicMock()
    return {
        "transformers": transformers,
        "torch": torch_mod,
        "peft": peft,
        "safetensors": MagicMock(),
    }, tok, transformers, peft


class TestInferenceWiring(unittest.TestCase):
    """WIRING tests (mocked deps). NOT inference proof."""

    def test_find_adapter_respects_override(self):
        complete = _make_adapter_dir()
        empty = tempfile.mkdtemp()
        self.assertEqual(generate_mod.find_adapter(complete), complete)
        self.assertIsNone(generate_mod.find_adapter(empty))
        # Global dir has no adapter in this checkout.
        self.assertIsNone(generate_mod.find_adapter())

    def test_inference_status_respects_override(self):
        complete = _make_adapter_dir()
        status = inference_mod.inference_status(complete)
        self.assertTrue(status["adapter_present"])
        self.assertEqual(status["adapter_dir"], os.path.abspath(complete))
        self.assertEqual(status["base_model"], "Qwen/Qwen3-0.6B")
        self.assertFalse(status["generation_enabled"])
        empty = tempfile.mkdtemp()
        self.assertFalse(inference_mod.inference_status(empty)["adapter_present"])

    def test_resolve_base_model_allowlist(self):
        ok, err = inference_mod.resolve_base_model(_make_adapter_dir("Qwen/Qwen3-0.6B"))
        self.assertEqual(ok, "Qwen/Qwen3-0.6B")
        self.assertIsNone(err)
        ok, err = inference_mod.resolve_base_model(_make_adapter_dir("Qwen/Qwen3-0.6B-Instruct"))
        self.assertIsNone(ok)
        self.assertIn("not the allowlisted", err)
        ok, err = inference_mod.resolve_base_model(_make_adapter_dir("meta-llama/Llama-3-8B"))
        self.assertIsNone(ok)
        # Local dir named Qwen3-0.6B is allowed.
        local_base = tempfile.mkdtemp(prefix="x-")
        local_qwen = os.path.join(local_base, "Qwen3-0.6B")
        os.makedirs(local_qwen)
        ok, err = inference_mod.resolve_base_model(_make_adapter_dir(local_qwen))
        self.assertEqual(ok, local_qwen)
        self.assertIsNone(err)
        # Missing base key -> error.
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({}, f)
        ok, err = inference_mod.resolve_base_model(d)
        self.assertIsNone(ok)
        self.assertIn("no base_model_name_or_path", err)

    def test_override_beats_global_for_missing_adapter(self):
        # Global (patched) has a complete adapter; override is empty ->
        # must raise AdapterMissingError (override checked, not global).
        complete = _make_adapter_dir()
        empty = tempfile.mkdtemp()
        with patch.object(inference_mod, "ADAPTER_DIR", complete):
            with self.assertRaises(AdapterMissingError):
                inference_mod.run_adapter_inference("Record a receipt.", KNOWN, adapter_dir=empty)
            # Default (no override) uses the patched global -> gets past
            # the adapter check and fails at the dep check instead.
            with self.assertRaises(RuntimeError) as cm:
                inference_mod.run_adapter_inference("Record a receipt.", KNOWN)
            self.assertIn("ML dependencies missing", str(cm.exception))

    def test_load_path_wiring_with_mocks(self):
        """WIRING ONLY: base loaded explicitly, adapter applied once, local-only,
        no remote code, thinking disabled, ledgers in prompt."""
        d = _make_adapter_dir()
        mocks, tok, transformers, peft = _mock_ml_modules()
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            result = inference_mod.run_adapter_inference(
                "Record a receipt of 100 from Acme Traders into City Bank.",
                KNOWN, adapter_dir=d)
        self.assertEqual(result, _VALID_INTENT)
        # Base loaded explicitly from its resolved path, NOT the adapter dir.
        base_call = transformers.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(base_call[0][0], "Qwen/Qwen3-0.6B")
        self.assertTrue(base_call[1]["local_files_only"])
        self.assertFalse(base_call[1]["trust_remote_code"])
        # Tokenizer: local-only, no remote code.
        tok_call = transformers.AutoTokenizer.from_pretrained.call_args
        self.assertEqual(tok_call[0][0], d)
        self.assertTrue(tok_call[1]["local_files_only"])
        self.assertFalse(tok_call[1]["trust_remote_code"])
        # Adapter applied ONCE: (base, adapter_dir).
        peft_call = peft.PeftModel.from_pretrained
        self.assertEqual(peft_call.call_count, 1)
        self.assertEqual(peft_call.call_args[0][1], d)
        self.assertTrue(peft_call.call_args[1]["local_files_only"])
        self.assertFalse(peft_call.call_args[1]["trust_remote_code"])
        # Thinking disabled; known ledgers in the system prompt.
        chat_call = tok.apply_chat_template.call_args
        self.assertFalse(chat_call[1]["enable_thinking"])
        system_text = chat_call[0][0][0]["content"]
        self.assertIn("City Bank", system_text)
        self.assertIn("Acme Traders", system_text)

    def test_bounds_enforced(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules()
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            with self.assertRaises(ValueError):
                inference_mod.run_adapter_inference("x" * 2001, KNOWN, adapter_dir=d)
            with self.assertRaises(ValueError):
                inference_mod.run_adapter_inference("ok", KNOWN, adapter_dir=d, max_new_tokens=513)
            with self.assertRaises(ValueError):
                inference_mod.run_adapter_inference("ok", KNOWN, adapter_dir=d, max_new_tokens=0)
            with self.assertRaises(ValueError):
                inference_mod.run_adapter_inference("ok", None, adapter_dir=d)

    def test_think_block_stripped_defensively(self):
        d = _make_adapter_dir()
        raw = "<think>let me think about this</think> " + json.dumps(_VALID_INTENT)
        mocks, _, _, _ = _mock_ml_modules(decode_text=raw)
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            result = inference_mod.run_adapter_inference("Record a receipt.", KNOWN, adapter_dir=d)
        self.assertEqual(result, _VALID_INTENT)


if __name__ == "__main__":
    unittest.main()
