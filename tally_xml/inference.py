"""Adapter inference wiring (fail-closed, UNVERIFIED).

This module wires real adapter inference: explicit adapter-file checks,
dependency checks, tokenizer checks, generation, JSON parsing, and
intent validation. Every step fails closed with a clear error.

STATUS: IMPLEMENTED BUT UNVERIFIED. No adapter is present
(``models/`` is empty), the ML dependencies (transformers/torch/peft/
safetensors) are not installed, and this machine has no GPU and ~0.5 GB
free RAM — not enough to load Qwen3-0.6B. ``run_adapter_inference``
therefore fails closed today. ``generation_enabled`` in
``tally_xml.generate`` stays False until inference is actually
implemented AND verified with a real adapter. Restoring adapter files
alone never enables generation.

Legacy/research path: an XML-emitting adapter (Phase 2 style) is NOT the
production path. If one is ever evaluated, its XML output is untrusted
input: pass it through ``strict.validate_strict`` and treat it as
research only. Never present adapter XML as production output without
the strict gate, and never claim a prompt change proves JSON capability
— the Phase 2 adapter was trained to emit XML (see
``data/xml_train.json``), not JSON intent.
"""
import importlib.util
import json
import os

from .generate import ADAPTER_DIR, find_adapter
from .strict import validate_intent

# ML deps are imported lazily INSIDE run_adapter_inference, after the
# dependency check, so importing this module stays stdlib-only.
_ML_DEPS = ("transformers", "torch", "peft", "safetensors")


def check_dependencies():
    """Report which ML dependencies are importable (no import side effects).

    Returns {"available": {name: bool}, "missing": [names]}.
    """
    available = {}
    for name in _ML_DEPS:
        available[name] = importlib.util.find_spec(name) is not None
    return {
        "available": available,
        "missing": [n for n, ok in available.items() if not ok],
    }


def check_tokenizer(adapter_dir=None):
    """True if tokenizer files are present in the adapter dir.

    Looks for tokenizer_config.json (required) plus at least one of
    tokenizer.json / vocab.json. A LoRA adapter alone has no tokenizer;
    the base-model tokenizer must be supplied alongside it.
    """
    d = adapter_dir or ADAPTER_DIR
    has_config = os.path.exists(os.path.join(d, "tokenizer_config.json"))
    has_vocab = (
        os.path.exists(os.path.join(d, "tokenizer.json"))
        or os.path.exists(os.path.join(d, "vocab.json"))
    )
    return bool(has_config and has_vocab)


def inference_status():
    """Machine-readable status: adapter, deps, tokenizer, and verdict."""
    adapter = find_adapter()
    deps = check_dependencies()
    tok = check_tokenizer()
    reasons = []
    if adapter is None:
        reasons.append("no complete adapter (adapter_config.json + non-empty adapter_model.safetensors)")
    if deps["missing"]:
        reasons.append(f"missing ML dependencies: {', '.join(deps['missing'])}")
    if not tok:
        reasons.append("tokenizer files missing from adapter dir")
    return {
        "adapter_present": adapter is not None,
        "adapter_dir": os.path.abspath(ADAPTER_DIR),
        "dependencies": deps,
        "tokenizer_present": tok,
        "inference_ready": not reasons,
        "blockers": reasons,
        "generation_enabled": False,  # stays False until verified end-to-end
    }


def run_adapter_inference(prompt, known_ledgers, adapter_dir=None, max_new_tokens=256):
    """Run the adapter on a prompt and return a validated JSON intent.

    Fail-closed at every step:
    1. No complete adapter -> AdapterMissingError (via find_adapter).
    2. Missing ML deps -> RuntimeError naming the missing deps.
    3. Missing tokenizer -> RuntimeError.
    4. Model load/generation failure -> the underlying exception propagates.
    5. Output is not valid JSON, or JSON fails intent validation ->
       ValueError with the parse/validation errors.

    The model is asked for a JSON intent ONLY (voucher_type/date/entries).
    Its output is never trusted as XML and never used for arithmetic —
    deterministic code validates and builds the XML (see pipeline.py).

    This function is IMPLEMENTED but UNVERIFIED: no adapter, deps, or
    tokenizer exists on this machine, so it fails closed at step 1/2/3.
    """
    from .generate import AdapterMissingError

    if find_adapter() is None:
        raise AdapterMissingError(
            "No complete adapter found. Generation refused (fail-closed)."
        )
    deps = check_dependencies()
    if deps["missing"]:
        raise RuntimeError(
            f"ML dependencies missing, cannot run inference: {', '.join(deps['missing'])}"
        )
    if not check_tokenizer(adapter_dir):
        raise RuntimeError("Tokenizer files missing from adapter dir; cannot run inference.")

    # Lazy imports: only reached when deps are confirmed present.
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    from peft import PeftModel  # noqa: F401

    d = adapter_dir or ADAPTER_DIR
    tokenizer = AutoTokenizer.from_pretrained(d)
    base = AutoModelForCausalLM.from_pretrained(d, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, d)
    model.eval()
    messages = [
        {
            "role": "system",
            "content": (
                "Emit ONLY a JSON object with keys voucher_type, date, entries. "
                "voucher_type is Receipt, Payment, or Journal. date is YYYY-MM-DD. "
                "entries is a list of {ledger, side, amount} where side is Dr or Cr "
                "and amount is a decimal string (e.g. \"5000.00\"). No other text."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        intent = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"adapter output is not valid JSON: {e}; raw output: {raw[:300]!r}")
    ok, errors = validate_intent(intent, known_ledgers)
    if not ok:
        raise ValueError(f"adapter output failed intent validation: {'; '.join(errors)}")
    return intent
