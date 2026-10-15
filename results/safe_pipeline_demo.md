# Safe pipeline demo — deterministic receipt/payment/journal flows

Branch: `tally-safe-pipeline`. Results file: `results/safe_pipeline_demo.json`.

## What this demo proves (and does not prove)

- PROVES: the deterministic pipeline (JSON intent → validation → ET-built
  XML → strict round-trip gate) works for receipt, payment, and journal
  flows with fixture intents. All three flows pass pipeline + strict
  round-trip validation.
- DOES NOT PROVE: that any model can emit valid JSON intents. The fixture
  intents are hand-written developer fixtures, NOT model outputs. Fixture
  mocks cannot count as a model test.
- ADAPTER-BACKED NL INFERENCE DID NOT RUN. Blockers: no complete adapter,
  missing ML dependencies (transformers, torch, peft, safetensors),
  tokenizer files missing. See `inference_status()`.

## Exact commands (run from the repo root)

```bash
# Full test suite (87 tests, incl. 62 fresh strict/pipeline tests)
python3 -m unittest discover -s tests -v

# Demo: three fixture flows + inference status (writes safe_pipeline_demo.json)
python3 /tmp/demo_safe_pipeline.py
# (demo script is ephemeral; the same flows are covered by
# tests/test_strict_pipeline.py::TestPipeline end-to-end tests)
```

## Demo results (2026-10-15)

| Flow | Pipeline | Strict round-trip | XML bytes |
|---|---|---|---|
| Receipt (2025-07-21, City Bank Dr 5000.00 / Acme Traders Cr 5000.00) | ok | ok | 633 |
| Payment (2025-08-02, Zenith Supplies Dr 1200.50 / Harbor Bank Cr 1200.50) | ok | ok | 638 |
| Journal (2024-02-29 leap date, Maple Stores Dr 750.00 / Northwind Traders Cr 750.00) | ok | ok | 639 |

Adapter-backed NL inference: NOT RUN (see blockers above).

## Tested vs untested

- TESTED: strict validator (dates, fields, ledgers, money, balance),
  intent validation, deterministic XML builder + round-trip gate,
  fail-closed inference (missing adapter/deps/tokenizer), escaping
  (`Smith & Sons` → `Smith &amp; Sons`), injection-like ledger rejection.
- UNTESTED: adapter-backed NL inference (no adapter, no ML deps, no GPU,
  insufficient RAM), live Tally Prime import (no Tally/Wine on this
  Linux machine — marked NOT TESTED), JSON-intent capability of any
  adapter (no JSON training data or eval exists; the Phase 2 adapter was
  trained to emit XML).
- DEVELOPER-WRITTEN, NOT ACCOUNTANT-REVIEWED, NOT PRODUCTION PROOF.
