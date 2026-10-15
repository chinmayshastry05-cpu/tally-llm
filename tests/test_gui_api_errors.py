"""GUI API error-handling tests: empty input returns a JSON error object
(never a crash), and malformed JSON gets a graceful 400.

Run from the repo root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml.gui import Handler  # noqa: E402


def _post(port, path, data, content_type="application/json"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method="POST",
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class TestGuiApiErrors(unittest.TestCase):
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

    def test_empty_input_returns_error_object(self):
        code, body = _post(self.port, "/api/validate", json.dumps({"xml": ""}).encode())
        self.assertEqual(code, 400)
        self.assertIn("error", json.loads(body))

    def test_missing_xml_key_returns_error_object(self):
        code, body = _post(self.port, "/api/validate", json.dumps({}).encode())
        self.assertEqual(code, 400)
        self.assertIn("error", json.loads(body))

    def test_malformed_json_graceful_400(self):
        code, body = _post(self.port, "/api/validate", b"{not json")
        self.assertEqual(code, 400)
        payload = json.loads(body)
        self.assertIn("error", payload)
        self.assertIn("JSON", payload["error"])

    def test_malformed_json_on_generate_graceful_400(self):
        code, body = _post(self.port, "/api/generate", b"[1,2,")
        self.assertEqual(code, 400)
        self.assertIn("error", json.loads(body))

    def test_non_object_json_graceful_400(self):
        code, body = _post(self.port, "/api/validate", json.dumps([1, 2]).encode())
        self.assertEqual(code, 400)
        self.assertIn("error", json.loads(body))

    def test_valid_xml_still_validates(self):
        xml = "<ENVELOPE><HEADER><TALLYREQUEST>Import</TALLYREQUEST></HEADER></ENVELOPE>"
        code, body = _post(self.port, "/api/validate", json.dumps({"xml": xml}).encode())
        self.assertEqual(code, 200)
        self.assertIn("wellformed", json.loads(body))

    def test_page_shows_server_errors_and_no_enable_generation_claim(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as r:
            page = r.read().decode()
        self.assertIn("if(v.error)", page, "validate() must render server error objects")
        self.assertIn("not</strong> enable generation", page)
        self.assertIn("not</strong> a complete Tally schema validator", page)


if __name__ == "__main__":
    unittest.main()
