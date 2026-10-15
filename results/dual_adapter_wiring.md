# Dual-adapter wiring (2026-10-15)

Wires BOTH trained LoRA adapters (base `Qwen/Qwen3-0.6B`) into the repo.
Work authorized by the owner in his own voice; GPU verification is a
separate step run by the coordinator (see below).

## What was wired

- `tally_xml/inference.py` — dual-adapter wiring:
  - JSON adapter (prototype acceptance path): `run_json_adapter_request` /
    `AdapterSession.request_json`. Emits a validated JSON intent; out-of-scope
    requests emit exactly `REFUSE: <reason>` (a successful refusal, not an error).
  - XML adapter (legacy/research path): `run_xml_adapter_inference` /
    `AdapterSession.request_xml`. Output is UNTRUSTED: it passes through
    `validate_strict` and is research only — never prototype acceptance output.
  - `AdapterSession`: loads base + LoRA + tokenizer ONCE and reuses it across
    prompts (a 30-case GPU run must not reload per case). Setup fails closed:
    missing adapter / deps / tokenizer / non-allowlisted base.
  - `tokenizer_dir` override: a LoRA adapter dir has no tokenizer files; the
    caller supplies the base-model tokenizer dir explicitly. Default stays the
    adapter dir (existing tests cover that layout).
  - `dual_adapter_status()` reports both adapters; `generation_enabled` stays
    False until a real invocation succeeds end-to-end.
  - Loading rules unchanged: base loaded explicitly, adapter applied ONCE,
    `local_files_only=True`, `trust_remote_code=False`, bounded input/output,
    Qwen3 thinking disabled, `<think>` blocks stripped.
- `tally_xml/weights.py` (new, stdlib-only): safetensors header parsing +
  LoRA structure checks against `adapter_config.json` (rank, target modules,
  A/B pairing) + size/sha256 verification. Never reads tensor data.
- `tests/test_dual_adapter.py` (new): mocked wiring tests (REFUSE, XML gate,
  tokenizer override, session reuse, dual status) + header-parser tests +
  a real-file check (gated by `TALLY_LOCAL_ADAPTERS_DIR`) asserting the
  delivered weights' size + sha256 + structure.
- `tests/independent_gpu_tests.json` (new): 30 hand-written independent
  cases (15 voucher + 15 refuse), NOT from the training generator.
- `tests/run_independent_gpu_tests.py` (new): GPU runner — loads the
  adapter once, runs all 30 cases, exact-match scoring, per-case
  PASS/FAIL + summary, exit 0 iff all pass.

## How to run the GPU verification (coordinator / Kaggle)

```sh
python3 tests/run_independent_gpu_tests.py --adapter json \
    --json-adapter-dir /path/to/tally-lora-json-adapter
python3 tests/run_independent_gpu_tests.py --adapter xml \
    --xml-adapter-dir /path/to/tally-lora-phase2-adapter
```

The runner downloads the base model once into the HF cache if missing
(explicit, in the runner — the library stays `local_files_only`).
Caveat: the runner uses the system prompts in `tally_xml/inference.py`;
if GPU results are poor, align the prompt template with the one used
during training before concluding the adapter is at fault.

## Verified locally (this machine)

- Full suite: `python3 -m unittest discover -s tests` — 142 tests, all pass.
- Real-file check (`TALLY_LOCAL_ADAPTERS_DIR`): both adapters exactly
  80,792,456 bytes, sha256 matches the delivery mail (JSON:
  `4bfb4d82…`, XML: `df7a0608…`), LoRA structure verified: 392 tensors,
  28 layers, rank 32, all 7 target modules (q/k/v/o/gate/up/down).
- All 30 test golds pass `validate_intent` against their known ledgers.
- Runner fails closed cleanly here (no `huggingface_hub`, no GPU): exit 2
  with a clear message, no traceback.

## NOT verified (honest gap list)

- NO real adapter invocation has succeeded anywhere. ML deps
  (transformers/torch/peft/safetensors) are not installed here, the base
  model + tokenizer are not present, and this machine has no GPU.
- GPU metrics (the mail's 1.0 scores) are the trainer's same-generator
  numbers, NOT independent verification. The 30 tests above are the
  independent set; they have not been run against either adapter yet.
- The XML adapter's REFUSE behaviour is unknown (no trained REFUSE
  contract): for refuse cases the runner passes iff it emits NO
  strict-passing voucher.
- `generate_voucher()` / GUI generation stays disabled
  (`INFERENCE_WIRED` False) until a real invocation succeeds.
- Prompt-template alignment with training is unverified (see caveat above).
- No Tally import was attempted; no real accounting data was used.
