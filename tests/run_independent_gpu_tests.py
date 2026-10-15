"""Run the 30 independent test cases against ONE adapter on GPU.

Usage (from the repo root, on Kaggle with a GPU):
    python3 tests/run_independent_gpu_tests.py --adapter json \
        --json-adapter-dir /path/to/tally-lora-json-adapter
    python3 tests/run_independent_gpu_tests.py --adapter xml \
        --xml-adapter-dir /path/to/tally-lora-phase2-adapter

What it does:
- Loads the adapter ONCE (AdapterSession) and runs all 30 cases from
  tests/independent_gpu_tests.json against it.
- Voucher cases: the adapter output must match the expected intent
  exactly (voucher_type, date, entries; amounts compared as Decimals).
  JSON adapter: output parsed as a JSON intent. XML adapter: output
  parsed from the strict-gated XML back into an intent for comparison.
- Refuse cases: the JSON adapter must emit REFUSE: <reason>. The XML
  adapter has no trained REFUSE contract: it passes iff it does NOT
  emit a voucher that passes the strict gate (REFUSE text, invalid XML,
  or gate-failing XML all pass).
- Prints per-case PASS/FAIL + a summary; exit code 0 iff all pass.
  Optionally writes a JSON report with --out.

Base model + tokenizer: the adapter config names the hub id
Qwen/Qwen3-0.6B. If the base is not cached, this script downloads it
once via huggingface_hub.snapshot_download (explicit, visible here —
the library itself stays local_files_only). The tokenizer is taken from
--tokenizer-dir, else the base-model snapshot, else the adapter dir.

Prompt template caveat: the runner uses the system prompts in
tally_xml/inference.py. If GPU results are poor, align the prompt
template with the one used during training before concluding the
adapter is at fault — the runner's template is a best effort, not the
training template.
"""
import argparse
import json
import os
import sys
import traceback
from decimal import Decimal, InvalidOperation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tally_xml import inference as inference_mod  # noqa: E402


def _intent_matches(got, expected):
    """Exact intent match: type, date, entries in order, Decimal amounts."""
    if not isinstance(got, dict):
        return False, "got is not an intent object"
    if got.get("voucher_type") != expected.get("voucher_type"):
        return False, f"voucher_type {got.get('voucher_type')!r} != {expected.get('voucher_type')!r}"
    if got.get("date") != expected.get("date"):
        return False, f"date {got.get('date')!r} != {expected.get('date')!r}"
    got_entries = got.get("entries")
    exp_entries = expected.get("entries")
    if not isinstance(got_entries, list) or len(got_entries) != len(exp_entries):
        return False, f"entries count {len(got_entries) if isinstance(got_entries, list) else '?'} != {len(exp_entries)}"
    for i, (g, e) in enumerate(zip(got_entries, exp_entries)):
        if g.get("ledger") != e.get("ledger"):
            return False, f"entries[{i}].ledger {g.get('ledger')!r} != {e.get('ledger')!r}"
        if g.get("side") != e.get("side"):
            return False, f"entries[{i}].side {g.get('side')!r} != {e.get('side')!r}"
        try:
            if Decimal(str(g.get("amount"))) != Decimal(str(e.get("amount"))):
                return False, f"entries[{i}].amount {g.get('amount')!r} != {e.get('amount')!r}"
        except InvalidOperation:
            return False, f"entries[{i}].amount {g.get('amount')!r} is not a decimal"
    return True, ""


def _xml_to_intent(xml_text):
    """Parse strict-gated voucher XML back into an intent dict for comparison."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    voucher = root.find("BODY/DATA/TALLYMESSAGE/VOUCHER")
    if voucher is None:
        return None, "no VOUCHER element in XML"
    tally_date = (voucher.findtext("DATE") or "").strip()
    if len(tally_date) != 8 or not tally_date.isdigit():
        return None, f"DATE {tally_date!r} is not YYYYMMDD"
    iso_date = f"{tally_date[0:4]}-{tally_date[4:6]}-{tally_date[6:8]}"
    entries = []
    for entry in voucher.findall("ALLLEDGERENTRIES.LIST"):
        ledger = (entry.findtext("LEDGERNAME") or "").strip()
        deemed = (entry.findtext("ISDEEMEDPOSITIVE") or "").strip()
        amount_text = (entry.findtext("AMOUNT") or "").strip()
        if deemed == "Yes":
            side = "Dr"
        elif deemed == "No":
            side = "Cr"
        else:
            return None, f"ISDEEMEDPOSITIVE {deemed!r} is not Yes/No"
        try:
            amount = abs(Decimal(amount_text))
        except InvalidOperation:
            return None, f"AMOUNT {amount_text!r} is not a decimal"
        entries.append({"ledger": ledger, "side": side, "amount": str(amount)})
    return {
        "voucher_type": voucher.get("VCHTYPE"),
        "date": iso_date,
        "entries": entries,
    }, ""


def _ensure_base_cached(adapter_dir):
    """Return the local base-model dir, downloading the hub snapshot if needed."""
    base_path, err = inference_mod.resolve_base_model(adapter_dir)
    if err is not None:
        raise RuntimeError(f"Cannot resolve base model: {err}")
    if os.path.isdir(base_path):
        return base_path
    # Hub id: download once into the HF cache so the library's
    # local_files_only loads succeed.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "Base model is the hub id "
            f"{base_path!r} but huggingface_hub is not installed; "
            "pre-download the base model or install huggingface_hub."
        )
    return snapshot_download(repo_id=base_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the 30 independent GPU tests against one adapter.")
    ap.add_argument("--adapter", required=True, choices=("json", "xml"),
                    help="which adapter to test")
    ap.add_argument("--json-adapter-dir", default=None)
    ap.add_argument("--xml-adapter-dir", default=None)
    ap.add_argument("--tokenizer-dir", default=None,
                    help="tokenizer dir; default: base-model snapshot, else adapter dir")
    ap.add_argument("--cases", default=os.path.join(ROOT, "tests", "independent_gpu_tests.json"))
    ap.add_argument("--out", default=None, help="optional JSON report path")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args(argv)

    if args.adapter == "json":
        adapter_dir = args.json_adapter_dir or os.environ.get(
            "TALLY_JSON_ADAPTER_DIR", inference_mod.JSON_ADAPTER_DIR)
    else:
        adapter_dir = args.xml_adapter_dir or os.environ.get(
            "TALLY_XML_ADAPTER_DIR", inference_mod.XML_ADAPTER_DIR)
    if not adapter_dir or not os.path.isdir(adapter_dir):
        print(f"FAIL: adapter dir {adapter_dir!r} does not exist", file=sys.stderr)
        return 2
    with open(args.cases) as f:
        suite = json.load(f)
    cases = suite["cases"]

    try:
        base_dir = _ensure_base_cached(adapter_dir)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        if inference_mod.check_tokenizer(base_dir):
            tokenizer_dir = base_dir
        else:
            tokenizer_dir = adapter_dir
    print(f"adapter={args.adapter} dir={adapter_dir}")
    print(f"base={base_dir} tokenizer={tokenizer_dir}")

    results = []
    with inference_mod.AdapterSession(adapter_dir, tokenizer_dir) as session:
        for case in cases:
            cid, kind = case["id"], case["kind"]
            known = set(case["known_ledgers"])
            try:
                if args.adapter == "json":
                    res = session.request_json(case["prompt"], known, args.max_new_tokens)
                    if kind == "voucher":
                        if res["refused"]:
                            results.append((cid, False, "adapter refused a voucher case: " + (res["reason"] or "")))
                        elif not res["ok"]:
                            results.append((cid, False, "; ".join(res["errors"])))
                        else:
                            ok, why = _intent_matches(res["intent"], case["expected"])
                            results.append((cid, ok, why))
                    else:  # refuse
                        if res["ok"] and res["refused"]:
                            results.append((cid, True, ""))
                        elif res["ok"]:
                            results.append((cid, False, f"adapter emitted a voucher for a refuse case: {res['intent']!r}"))
                        else:
                            results.append((cid, False, "adapter errored on a refuse case: " + "; ".join(res["errors"])))
                else:  # xml adapter
                    res = session.request_xml(case["prompt"], known, args.max_new_tokens)
                    if kind == "voucher":
                        if not res["ok"]:
                            results.append((cid, False, "strict gate rejected output: " + "; ".join(res["errors"])))
                        else:
                            intent, err = _xml_to_intent(res["xml"])
                            if intent is None:
                                results.append((cid, False, f"could not parse gated XML: {err}"))
                            else:
                                ok, why = _intent_matches(intent, case["expected"])
                                results.append((cid, ok, why))
                    else:  # refuse: pass iff NO valid voucher came out
                        if res["ok"]:
                            results.append((cid, False, f"adapter emitted a strict-passing voucher for a refuse case: {res['xml'][:200]!r}"))
                        else:
                            results.append((cid, True, ""))
            except Exception as e:  # noqa: BLE001 - per-case failure must not abort the run
                results.append((cid, False, f"exception: {e}\n{traceback.format_exc(limit=2)}"))

    passed = sum(1 for _, ok, _ in results if ok)
    for cid, ok, detail in results:
        line = f"{cid}: {'PASS' if ok else 'FAIL'}"
        if detail:
            line += f" — {detail.splitlines()[0]}"
        print(line)
    print(f"\n{passed}/{len(results)} passed (adapter={args.adapter})")
    if args.out:
        report = {
            "adapter": args.adapter,
            "adapter_dir": adapter_dir,
            "base_dir": base_dir,
            "tokenizer_dir": tokenizer_dir,
            "passed": passed,
            "total": len(results),
            "results": [{"id": cid, "ok": ok, "detail": detail} for cid, ok, detail in results],
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
