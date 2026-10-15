# tally-llm

Small-model experiments for TallyPrime: answer Tally questions, and convert
plain-English accounting instructions into TallyPrime XML import messages.

**Status: research prototype. Not an operational accounting assistant.**
Nothing here has been tested in live Tally Prime, and the trained model
artifacts are missing (see below). The GUI runs, validates XML, and refuses
generation until an adapter exists — it does not claim to be a working
accounting assistant merely because the GUI runs.

**Branch `tally-safe-pipeline` (new):** strict validator + deterministic
JSON-intent → XML pipeline for receipt/payment/journal. The deterministic
pipeline is tested with fixture intents (not model outputs). Adapter-backed
NL inference is NOT wired to a working model and has NOT run — see
"Adapter / JSON mismatch" below. Live Tally import: NOT TESTED.

## What is in this repo

| Path | Contents |
|---|---|
| `notebooks/phase1_tally_train.ipynb` | Phase 1: synthetic Q&A generation → LoRA fine-tune → eval |
| `notebooks/phase2_tally_xml.ipynb` | Phase 2 (audited): plain-English → Tally XML, executed run |
| `data/test_set.jsonl` | Phase 1 eval set: 64 questions (synthetic) |
| `data/xml_train.json` | Phase 2 training: 700 synthetic instruction→XML pairs |
| `data/xml_test.json` | Phase 2 test: 60 synthetic pairs with gold records |
| `data/model_generated_samples.json` | The only 2 surviving fine-tuned model-generated XML samples |
| `tally_xml/validate.py` | Structural validator (subset — NOT a complete Tally schema validator): wellformed / schema / balanced / fields. Has a known hole (accepts impossible dates like 20261399) — see `strict.py` for the production gate. |
| `tally_xml/strict.py` | STRICT production validator + JSON intent validation: real calendar dates, Decimal money, caller-provided known ledgers, voucher-type allowlist, fail-closed. |
| `tally_xml/pipeline.py` | Deterministic JSON-intent → XML builder (ET-escaped) + strict round-trip gate. The LLM never does arithmetic or free-form XML here. |
| `tally_xml/inference.py` | Adapter inference wiring (fail-closed, UNVERIFIED): adapter/deps/tokenizer checks + JSON-intent parsing. Fails closed today (no adapter, no ML deps). |
| `tally_xml/gui.py` | Local GUI (stdlib only): validate, samples, fail-closed generate |
| `tally_xml/generate.py` | Generation entry point — refuses without a trained adapter |
| `results/` | Benchmark + eval write-ups with provenance and caveats |
| `tests/` | Validator, fail-closed, strict/pipeline, and fixture tests (stdlib `unittest`) |

## Honest caveats (read before using)

- **Phase 1 scores are keyword scores** on a 64-question synthetic set, not
  live accuracy: base .438, base+RAG .656, fine-tuned .484, fine-tuned+RAG
  .844, ChatGPT .891. See `results/phase1_benchmark.md`.
- **Phase 2 scores are on a synthetic 60-test set** in the same templated
  style as training data. Fine-tuned: 1.0 on all four checks. See
  `results/phase2_synthetic_eval.md`.
- **XML is structure-validated only**, never imported into a real Tally
  company. **NOT TESTED IN LIVE TALLY.**
- **Missing artifacts:** the Phase 2 LoRA adapter, the full fine-tuned
  outputs (`res_ft.json`), and the Phase 1 merged model were saved only to
  volatile Colab/Drive locations and are **not shipped**. The two samples in
  `data/model_generated_samples.json` are the only surviving model outputs.
  The 60 gold answers in `data/xml_test.json` are gold answers, not model
  outputs — never present them as such.
- **On-device (Cactus) execution was not tested.**
- `chunks.jsonl` (scraped Tally help text) is deliberately **excluded** —
  third-party scraped content. The Phase 1 notebook references it as its
  training-data source.

## Run it

Requirements: Python 3.10+ standard library only. No third-party
dependencies, no GPU, no network, no spend.

```bash
# GUI — validate XML, view the two model samples, (disabled) generation
python3 -m tally_xml.gui
# → open http://127.0.0.1:7860

# Tests
python3 -m unittest discover -s tests -v

# Validate an XML file from the shell
python3 -c "import sys,json; from tally_xml.validate import check; print(check(open(sys.argv[1]).read()))" sample.xml
```

## Adapter / JSON mismatch (read before trusting any "JSON" claim)

The Phase 2 adapter was trained to emit **XML**, not JSON intent:
`data/xml_train.json` (700 pairs) maps natural-language instruction to a
full TallyPrime XML ENVELOPE. No JSON-intent training data, JSON schema,
or JSON-output eval exists in this repo. A prompt change cannot prove
JSON capability — the adapter's weights are tuned for XML. Prompting it
for JSON may yield XML, malformed JSON, or JSON with wrong values.

Architecture:
- **Production path** (this branch): JSON intent ONLY → deterministic
  validation → deterministic XML builder. Currently DISABLED for
  model-backed use: no JSON-capable adapter exists. Fixture intents test
  the deterministic code; fixture mocks are NOT a model test.
- **Legacy/research path:** XML-emitting adapter → strict validator
  (`tally_xml.strict.validate_strict`). Also DISABLED (no adapter,
  inference unwired). If a retrained adapter appears, evaluate it as
  research only — never as the production path.

## Safe pipeline (branch `tally-safe-pipeline`)

Deterministic receipt/payment/journal flows with fixture intents:

```bash
python3 -m unittest discover -s tests -v
# 87 tests, incl. 62 fresh strict/pipeline tests (invalid dates, missing
# fields, unknown ledgers, malformed model output, bad money, unbalanced
# journals, escaping/injection-like ledger text). Developer-written, not
# accountant-reviewed, not production proof.
```

See `results/safe_pipeline_demo.md` for the demo (3 flows pass pipeline +
strict round-trip; adapter-backed inference did NOT run).

## Enabling generation (adapter setup)

Generation is **fail-closed** and stays disabled in this build. Restoring
adapter files alone does **not** enable generation, because model inference
is not wired in: even with a complete adapter,
`tally_xml.generate.generate_voucher()` refuses with `NotImplementedError`
until inference is actually implemented and verified. Never treat "adapter
files present" as "generation works".

Adapter states (see `adapter_status()`):

| State | `adapter_present` | `generation_enabled` |
|---|---|---|
| No adapter files | false | false |
| Only `adapter_config.json` (weights missing/empty) | false — not a usable adapter | false |
| Config + real weights, inference unwired | true | false — inference not implemented |

To complete the adapter files, place a fine-tuned LoRA adapter at:

```
models/adapter/adapter_config.json
models/adapter/adapter_model.safetensors
```

(or set `TALLY_ADAPTER_DIR`). The weights file must be non-empty; a config
file on its own does not count as an adapter. Model inference itself is not
wired into this build — see `results/phase2_synthetic_eval.md` for what the
executed run produced. Never import generated XML into production books
without independent review.
