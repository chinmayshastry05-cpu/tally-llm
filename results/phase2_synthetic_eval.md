# Phase 2 — plain-English → TallyPrime XML (synthetic 60-test set)

Source: `notebooks/phase2_tally_xml.ipynb` (executed run) with
`data/xml_train.json` (700 synthetic examples), `data/xml_test.json`
(60 synthetic tests), `tally_xml/validate.py`.

## Executed results (from the notebook's `RESULT` output lines)

| Model | wellformed | schema | balanced | fields | per-voucher-type fields |
|---|---|---|---|---|---|
| Base, one-shot | 1.0 | 0.367 | 0.25 | 0.15 | Receipt .75, all others 0.0 |
| Fine-tuned (LoRA, Qwen3-0.6B) | 1.0 | 1.0 | 1.0 | 1.0 | All types 1.0 |

Per-type breakdown is the `fields` score per voucher type (Receipt, Journal,
Sales, Payment, Contra, Purchase, ledger).

## What these numbers are — and are not

- **Synthetic 60-test set only.** All 60 tests are synthetic examples in the
  same templated style as the 700 training examples. A perfect score here
  means the model learned the template, not that it handles real-world
  wording, edge cases, or arbitrary ledgers.
- **XML is structure-validated only.** `validate.py` checks well-formedness,
  envelope/schema shape, debit/credit balance, and field agreement with the
  gold record. It does not check accounting correctness beyond that.
- **NOT TESTED IN LIVE TALLY.** No Tally Prime instance was available on the
  build machine. No sample has been imported into any Tally company, test or
  otherwise. Do not import generated XML into production books without
  independent review.

## Missing artifacts (stated plainly — not shipped)

- **`res_ft.json`** — the fine-tuned model's outputs on all 60 tests. Saved
  only to the volatile Colab `/content/p2` directory; not embedded in the
  notebook, not recovered.
- **LoRA adapter / merged model** — same fate. The GUI therefore fails
  closed: generation is disabled until an adapter is placed under
  `models/adapter/`. No template output is substituted for model output.
- Because `res_ft.json` is missing, the 60 gold test answers in
  `data/xml_test.json` must **never** be presented as model-generated
  outputs. They are gold answers.

## Surviving model-generated samples

Only **two** fine-tuned model-generated XML samples survive, printed by the
notebook's final cell from `res_ft.json` before it was lost. They are in
`data/model_generated_samples.json`, individually labelled with this
provenance. Both pass wellformed/schema/balanced checks.
