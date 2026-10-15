"""Dual-adapter wiring tests (mocked ML deps) + weight-structure checks.

WIRING tests only — mocked deps prove the loading/parsing code calls
the right functions with the right arguments. They are NOT inference
proof: no real adapter, deps, or GPU exists here.

The real-adapter test is gated by TALLY_LOCAL_ADAPTERS_DIR (set to the
dir holding tally-lora-json-adapter/ + tally-lora-phase2-adapter/) and
verifies size + sha256 + LoRA structure of the delivered weights
without loading them (stdlib-only header parse).

Run from the repo root:  python3 -m unittest discover -s tests -v
Run with real files:   TALLY_LOCAL_ADAPTERS_DIR=/path/to/adapters \
                         python3 -m unittest tests.test_dual_adapter -v
"""
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml import inference as inference_mod  # noqa: E402
from tally_xml import weights as weights_mod  # noqa: E402
from tally_xml.generate import AdapterMissingError  # noqa: E402

KNOWN = {"City Bank", "Acme Traders", "Harbor Bank"}

VALID_XML = (
    "<ENVELOPE><HEADER><VERSION>1</VERSION><TALLYREQUEST>Import</TALLYREQUEST>"
    "<TYPE>Data</TYPE><ID>Vouchers</ID></HEADER><BODY><DESC></DESC><DATA><TALLYMESSAGE>"
    "<VOUCHER VCHTYPE=\"Receipt\" ACTION=\"Create\">"
    "<DATE>20250721</DATE><VOUCHERTYPENAME>Receipt</VOUCHERTYPENAME>"
    "<ALLLEDGERENTRIES.LIST><LEDGERNAME>City Bank</LEDGERNAME>"
    "<ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE><AMOUNT>-5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
    "<ALLLEDGERENTRIES.LIST><LEDGERNAME>Acme Traders</LEDGERNAME>"
    "<ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE><AMOUNT>5000</AMOUNT></ALLLEDGERENTRIES.LIST>"
    "</VOUCHER></TALLYMESSAGE></DATA></BODY></ENVELOPE>"
)

_VALID_INTENT = {
    "voucher_type": "Receipt",
    "date": "2025-07-21",
    "entries": [
        {"ledger": "City Bank", "side": "Dr", "amount": "100.00"},
        {"ledger": "Acme Traders", "side": "Cr", "amount": "100.00"},
    ],
}


def _make_adapter_dir(with_tokenizer=True):
    """Temp dir with complete fake adapter (+ tokenizer files unless told otherwise)."""
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "adapter_config.json"), "w") as f:
        json.dump({"base_model_name_or_path": "Qwen/Qwen3-0.6B"}, f)
    with open(os.path.join(d, "adapter_model.safetensors"), "wb") as f:
        f.write(b"fake-weights-not-real")
    if with_tokenizer:
        with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(d, "tokenizer.json"), "w") as f:
            json.dump({}, f)
    return d


def _make_tokenizer_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump({}, f)
    return d


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


class TestDualAdapterWiring(unittest.TestCase):
    """WIRING tests (mocked deps). NOT inference proof."""

    def test_request_json_accepts_valid_intent(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules()
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_json_adapter_request("Record a receipt.", KNOWN, adapter_dir=d)
        self.assertTrue(res["ok"])
        self.assertFalse(res["refused"])
        self.assertEqual(res["intent"], _VALID_INTENT)

    def test_request_json_accepts_refuse(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules(decode_text="REFUSE: not a voucher request")
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_json_adapter_request("What is the capital of France?",
                                                         KNOWN, adapter_dir=d)
        self.assertTrue(res["ok"])
        self.assertTrue(res["refused"])
        self.assertEqual(res["reason"], "not a voucher request")
        self.assertIsNone(res["intent"])

    def test_request_json_refuse_without_reason_is_error(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules(decode_text="REFUSE:")
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_json_adapter_request("What is the capital of France?",
                                                         KNOWN, adapter_dir=d)
        self.assertFalse(res["ok"])
        self.assertFalse(res["refused"])

    def test_request_json_invalid_json_is_error_not_raise(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules(decode_text="this is not json")
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_json_adapter_request("Record a receipt.", KNOWN, adapter_dir=d)
        self.assertFalse(res["ok"])
        self.assertIn("not valid JSON", res["errors"][0])

    def test_request_xml_valid_passes_gate(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules(decode_text=VALID_XML)
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_xml_adapter_inference("Record a receipt.", KNOWN, adapter_dir=d)
        self.assertTrue(res["ok"])
        self.assertEqual(res["xml"], VALID_XML)

    def test_request_xml_prose_rejected_by_gate(self):
        d = _make_adapter_dir()
        mocks, _, _, _ = _mock_ml_modules(
            decode_text="Here is your voucher: " + VALID_XML)
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_xml_adapter_inference("Record a receipt.", KNOWN, adapter_dir=d)
        self.assertFalse(res["ok"])
        self.assertIsNone(res["xml"])

    def test_request_xml_unbalanced_rejected_by_gate(self):
        d = _make_adapter_dir()
        bad = VALID_XML.replace("<AMOUNT>5000</AMOUNT>", "<AMOUNT>5001</AMOUNT>")
        mocks, _, _, _ = _mock_ml_modules(decode_text=bad)
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            res = inference_mod.run_xml_adapter_inference("Record a receipt.", KNOWN, adapter_dir=d)
        self.assertFalse(res["ok"])

    def test_tokenizer_dir_override(self):
        d = _make_adapter_dir(with_tokenizer=False)
        tdir = _make_tokenizer_dir()
        mocks, _, transformers, _ = _mock_ml_modules()
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            # Without the override the tokenizer check fails closed.
            with self.assertRaises(RuntimeError):
                inference_mod.run_adapter_inference("Record a receipt.", KNOWN, adapter_dir=d)
            # With the override the tokenizer loads from the given dir.
            inference_mod.run_adapter_inference(
                "Record a receipt.", KNOWN, adapter_dir=d, tokenizer_dir=tdir)
        tok_call = transformers.AutoTokenizer.from_pretrained.call_args
        self.assertEqual(tok_call[0][0], tdir)
        # Status also respects the override.
        self.assertFalse(inference_mod.inference_status(d)["tokenizer_present"])
        self.assertTrue(inference_mod.inference_status(d, tokenizer_dir=tdir)["tokenizer_present"])

    def test_session_loads_model_once_for_many_prompts(self):
        d = _make_adapter_dir()
        mocks, _, transformers, peft = _mock_ml_modules()
        with patch.dict(sys.modules, mocks), \
                patch.object(inference_mod, "check_dependencies",
                             return_value={"available": {}, "missing": []}):
            with inference_mod.AdapterSession(d) as session:
                session.generate_raw("Record a receipt.", KNOWN)
                session.generate_raw("Record a payment.", KNOWN)
        self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_count, 1)
        self.assertEqual(peft.PeftModel.from_pretrained.call_count, 1)
        self.assertEqual(transformers.AutoTokenizer.from_pretrained.call_count, 1)

    def test_session_missing_adapter_fails_closed(self):
        empty = tempfile.mkdtemp()
        with self.assertRaises(AdapterMissingError):
            inference_mod.AdapterSession(empty)

    def test_dual_adapter_status(self):
        d1 = _make_adapter_dir()
        d2 = _make_adapter_dir()
        st = inference_mod.dual_adapter_status(d1, d2)
        self.assertTrue(st["json"]["adapter_present"])
        self.assertTrue(st["xml"]["adapter_present"])
        self.assertFalse(st["generation_enabled"])
        empty = tempfile.mkdtemp()
        st2 = inference_mod.dual_adapter_status(empty, d2)
        self.assertFalse(st2["json"]["adapter_present"])
        self.assertFalse(st2["both_ready"])


class TestSafetensorsHeader(unittest.TestCase):
    """Stdlib-only header parser: synthetic file + real-file (env-gated)."""

    def _write_safetensors(self, directory, tensors):
        """Write a minimal safetensors file (header + zero data)."""
        header = {}
        offset = 0
        for name, (dtype, shape) in tensors.items():
            nbytes = 1
            for dim in shape:
                nbytes *= dim
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes * 2]}
            offset += nbytes * 2
        header_bytes = json.dumps(header).encode("utf-8")
        path = os.path.join(directory, "adapter_model.safetensors")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * offset)
        return path

    def test_header_parse_and_structure_ok(self):
        d = tempfile.mkdtemp()
        prefix = "base_model.model.model.layers.0"
        tensors = {
            f"{prefix}.self_attn.q_proj.lora_A.weight": ("F16", [32, 1024]),
            f"{prefix}.self_attn.q_proj.lora_B.weight": ("F16", [2048, 32]),
            f"{prefix}.mlp.gate_proj.lora_A.weight": ("F16", [32, 1024]),
            f"{prefix}.mlp.gate_proj.lora_B.weight": ("F16", [3072, 32]),
        }
        self._write_safetensors(d, tensors)
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": 32, "target_modules": ["q_proj", "gate_proj"]}, f)
        parsed, err = weights_mod.read_safetensors_header(
            os.path.join(d, "adapter_model.safetensors"))
        self.assertIsNone(err)
        self.assertEqual(len(parsed), 4)
        res = weights_mod.check_adapter_weights(d)
        self.assertTrue(res["ok"], res["errors"])
        self.assertEqual(res["tensors"], 4)
        self.assertEqual(res["layers"], 1)
        self.assertEqual(res["rank"], 32)
        self.assertEqual(res["modules"], ["gate_proj", "q_proj"])

    def test_header_rank_mismatch_flagged(self):
        d = tempfile.mkdtemp()
        prefix = "base_model.model.model.layers.0"
        tensors = {
            f"{prefix}.self_attn.q_proj.lora_A.weight": ("F16", [16, 1024]),
            f"{prefix}.self_attn.q_proj.lora_B.weight": ("F16", [2048, 32]),
        }
        self._write_safetensors(d, tensors)
        with open(os.path.join(d, "adapter_config.json"), "w") as f:
            json.dump({"r": 32, "target_modules": ["q_proj"]}, f)
        res = weights_mod.check_adapter_weights(d)
        self.assertFalse(res["ok"])
        self.assertTrue(any("rank" in e for e in res["errors"]))

    def test_header_garbage_rejected(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "adapter_model.safetensors")
        with open(path, "wb") as f:
            f.write(b"not a safetensors file")
        _, err = weights_mod.read_safetensors_header(path)
        self.assertIsNotNone(err)

    @unittest.skipUnless(os.environ.get("TALLY_LOCAL_ADAPTERS_DIR"),
                         "TALLY_LOCAL_ADAPTERS_DIR not set — real-file check skipped")
    def test_real_adapter_files_size_sha_structure(self):
        """Delivered weights: exact size + sha256 (per Instinct mail) + LoRA structure."""
        base = os.environ["TALLY_LOCAL_ADAPTERS_DIR"]
        expected = {
            "tally-lora-phase2-adapter": (
                "df7a06085ab20f8095641304776a9a6dd55c33bdbff077be7b6ba26375966922"),
            "tally-lora-json-adapter": (
                "4bfb4d8258206249483609e3d746e23bb5600b13a7fa79a88a3884bee72ed60c"),
        }
        for dirname, sha in expected.items():
            d = os.path.join(base, dirname)
            ok, errors = weights_mod.verify_adapter_file(d, sha, 80792456)
            self.assertTrue(ok, f"{dirname}: {errors}")
            res = weights_mod.check_adapter_weights(d)
            self.assertTrue(res["ok"], f"{dirname}: {res['errors']}")
            self.assertEqual(res["rank"], 32)
            self.assertEqual(res["layers"], 28)
            self.assertEqual(res["modules"],
                             ["down_proj", "gate_proj", "k_proj", "o_proj",
                              "q_proj", "up_proj", "v_proj"])
            # Every layer x module x A/B must be present.
            self.assertEqual(res["tensors"], 2 * len(res["modules"]) * res["layers"])


if __name__ == "__main__":
    unittest.main()
