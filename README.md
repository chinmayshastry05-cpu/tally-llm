# tally-llm

Small-model experiments for TallyPrime: answer Tally questions, and convert
plain-English accounting instructions into TallyPrime XML import messages.

**Status: research prototype. Not an operational accounting assistant.**
Nothing here has been tested in live Tally Prime, and the trained model
artifacts are missing (see below). The GUI runs, validates XML, and refuses
generation until an adapter exists — it does not claim to be a working
accounting assistant merely because the GUI runs.

## What is in this repo

| Path | Contents |
|---|---|
| `notebooks/phase1_tally_train.ipynb` | Phase 1: synthetic Q&A generation → LoRA fine-tune → eval |
| `notebooks/phase2_tally_xml.ipynb` | Phase 2 (audited): plain-English → Tally XML, executed run |
| `data/test_set.jsonl` | Phase 1 eval set: 64 questions (synthetic) |
| `data/xml_train.json` | Phase 2 training: 700 synthetic instruction→XML pairs |
| `data/xml_test.json` | Phase 2 test: 60 synthetic pairs with gold records |
| `data/model_generated_samples.json` | The only 2 surviving fine-tuned model-generated XML samples |
| `tally_xml/validate.py` | Structural validator: wellformed / schema / balanced / fields |
| `tally_xml/gui.py` | Local GUI (stdlib only): validate, samples, fail-closed generate |
| `tally_xml/generate.py` | Generation entry point — refuses without a trained adapter |
| `results/` | Benchmark + eval write-ups with provenance and caveats |
| `tests/` | Validator, fail-closed, and fixture tests (stdlib `unittest`) |

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

## Enabling generation (adapter setup)

Generation is **fail-closed**: with no adapter, the GUI and
`tally_xml.generate.generate_voucher()` refuse, and no template XML is
produced in its place. To enable it, place a fine-tuned LoRA adapter at:

```
models/adapter/adapter_config.json
models/adapter/adapter_model.safetensors
```

(or set `TALLY_ADAPTER_DIR`). Model inference itself is not wired into this
build — see `results/phase2_synthetic_eval.md` for what the executed run
produced. Never import generated XML into production books without
independent review.
