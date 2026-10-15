"""Adapter-status honesty tests: config-only is NOT an adapter, and a
complete adapter still does not enable generation while inference is unwired.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tally_xml.generate as gen  # noqa: E402
from tally_xml.generate import (  # noqa: E402
    AdapterMissingError,
    adapter_status,
    find_adapter,
    generate_voucher,
)


class TestAdapterStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = gen.ADAPTER_DIR
        gen.ADAPTER_DIR = self.tmp

    def tearDown(self):
        gen.ADAPTER_DIR = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, content=b"x"):
        with open(os.path.join(self.tmp, name), "wb") as f:
            f.write(content)

    def test_missing_adapter(self):
        st = adapter_status()
        self.assertFalse(st["config_present"])
        self.assertFalse(st["weights_present"])
        self.assertFalse(st["adapter_present"])
        self.assertFalse(st["inference_wired"])
        self.assertFalse(st["generation_enabled"])
        self.assertIsNone(find_adapter())
        with self.assertRaises(AdapterMissingError):
            generate_voucher("anything")

    def test_config_only_is_not_an_adapter(self):
        self._write("adapter_config.json", b"{}")
        st = adapter_status()
        self.assertTrue(st["config_present"])
        self.assertFalse(st["weights_present"])
        self.assertFalse(st["adapter_present"], "config alone must not count as an adapter")
        self.assertFalse(st["generation_enabled"])
        self.assertIsNone(find_adapter())
        with self.assertRaises(AdapterMissingError) as cm:
            generate_voucher("anything")
        self.assertIn("safetensors", str(cm.exception).lower())

    def test_empty_weights_file_is_not_an_adapter(self):
        self._write("adapter_config.json", b"{}")
        self._write("adapter_model.safetensors", b"")
        self.assertFalse(adapter_status()["weights_present"])
        self.assertFalse(adapter_status()["adapter_present"])

    def test_config_plus_weights_still_disabled_while_inference_unwired(self):
        self._write("adapter_config.json", b"{}")
        self._write("adapter_model.safetensors", b"fake-weights-bytes")
        st = adapter_status()
        self.assertTrue(st["config_present"])
        self.assertTrue(st["weights_present"])
        self.assertTrue(st["adapter_present"])
        self.assertFalse(st["inference_wired"])
        self.assertFalse(
            st["generation_enabled"],
            "complete adapter must NOT enable generation while inference is unwired",
        )
        self.assertEqual(find_adapter(), self.tmp)
        with self.assertRaises(NotImplementedError) as cm:
            generate_voucher("anything")
        self.assertIn("not wired", str(cm.exception).lower())

    def test_status_message_never_claims_generation_enabled_while_unwired(self):
        for setup in (lambda: None, lambda: self._write("adapter_config.json", b"{}")):
            setup()
            st = adapter_status()
            if not st["generation_enabled"]:
                self.assertNotIn("generation enabled", st["message"].lower())


if __name__ == "__main__":
    unittest.main()
