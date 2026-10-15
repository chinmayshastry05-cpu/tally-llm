"""Adapter inference wiring (fail-closed, UNVERIFIED).

This module wires real adapter inference: explicit adapter-file checks,
dependency checks, tokenizer checks, base-model resolution, adapter
loading, generation, JSON parsing, and intent validation. Every step
fails closed with a clear error.

Loading rules (security):
- The base model is resolved from the adapter's own
  ``adapter_config.json`` (``base_model_name_or_path``), NOT assumed.
  It must be the allowlisted ``Qwen/Qwen3-0.6B`` (hub id) or a local
  directory named ``Qwen3-0.6B``. Anything else is rejected.
- The base is loaded explicitly, then the adapter is applied ONCE via
  ``PeftModel.from_pretrained(base, adapter_dir)``. The adapter dir is
  NEVER passed to ``AutoModelForCausalLM.from_pretrained`` (that could
  auto-load/reapply adapters).
- ``local_files_only=True`` on every ``from_pretrained`` call (no
  implicit network download), ``trust_remote_code=False`` everywhere
  (no remote code execution).
- The ``adapter_dir`` override is threaded through EVERY check
  (adapter files, tokenizer, status) — never silently ignored.
- Input is bounded (max prompt chars), output is bounded (max new
  tokens), Qwen3 thinking is disabled for JSON extraction, and any
  ``<think>`` block is stripped defensively before JSON parsing.
- The caller's known ledgers are included in the extraction prompt so
  the model emits only those ledger names (still enforced by
  ``validate_intent`` — the prompt is a hint, the validator is the gate).

STATUS: IMPLEMENTED BUT UNVERIFIED. No adapter is present
(``models/`` is empty), the ML dependencies (transformers/torch/peft/
safetensors) are not installed, and this machine has no GPU and ~0.5 GB
free RAM — not enough to load Qwen3-0.6B. ``run_adapter_inference``
therefore fails closed today. ``generation_enabled`` stays False until
a real adapter invocation actually succeeds end-to-end. Restoring
adapter files alone never enables generation.

Legacy/research path: an XML-emitting adapter (Phase 2 style) is NOT
the prototype acceptance path. If one is ever evaluated, its XML output
is untrusted input: pass it through ``strict.validate_strict`` and
treat it as research only. Never present adapter XML as prototype
acceptance output without the strict gate, and never claim a prompt
change proves JSON capability — the Phase 2 adapter was trained to
emit XML (see ``data/xml_train.json``), not JSON intent.
"""
import importlib.util
import json
import os
import re

from .generate import ADAPTER_DIR, find_adapter, read_adapter_config
from .strict import validate_intent

# ML deps are imported lazily INSIDE run_adapter_inference, after the
# dependency check, so importing this module stays stdlib-only.
_ML_DEPS = ("transformers", "torch", "peft", "safetensors")

# Bounds for adapter-backed JSON extraction.
_MAX_PROMPT_CHARS = 2000
_MAX_NEW_TOKENS = 512

# Base-model allowlist: the intended Phase 2 base. Hub id, or a local
# directory with exactly this name.
_ALLOWED_BASE_HUB_ID = "Qwen/Qwen3-0.6B"
_ALLOWED_BASE_DIR_NAME = "Qwen3-0.6B"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


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


def resolve_base_model(adapter_dir=None):
    """Resolve the base model from the adapter's own config + allowlist.

    Returns (base_path, error|None). ``base_path`` is the hub id
    ``Qwen/Qwen3-0.6B`` or a local directory named ``Qwen3-0.6B``.
    Anything else — missing config, missing key, non-allowlisted base,
    local path that does not exist — is an error (fail-closed).
    """
    cfg, err = read_adapter_config(adapter_dir)
    if err is not None:
        return None, err
    base = cfg.get("base_model_name_or_path")
    if not base or not isinstance(base, str):
        return None, "adapter_config.json has no base_model_name_or_path"
    if base == _ALLOWED_BASE_HUB_ID:
        return base, None
    if os.path.isdir(base) and os.path.basename(os.path.normpath(base)) == _ALLOWED_BASE_DIR_NAME:
        return base, None
    return None, (
        f"base_model_name_or_path {base!r} is not the allowlisted "
        f"{_ALLOWED_BASE_HUB_ID} (hub id) or a local directory named "
        f"{_ALLOWED_BASE_DIR_NAME}"
    )


def inference_status(adapter_dir=None):
    """Machine-readable status: adapter, deps, tokenizer, base, verdict.

    ``adapter_dir`` overrides the default for EVERY check. ``generation_enabled``
    stays False until a real adapter invocation succeeds end-to-end.
    """
    d = adapter_dir or ADAPTER_DIR
    adapter = find_adapter(d)
    deps = check_dependencies()
    tok = check_tokenizer(d)
    reasons = []
    base_path, base_err = None, None
    if adapter is None:
        reasons.append("no complete adapter (adapter_config.json + non-empty adapter_model.safetensors)")
    else:
        base_path, base_err = resolve_base_model(d)
        if base_err is not None:
            reasons.append(f"base model not allowlisted/resolvable: {base_err}")
    if deps["missing"]:
        reasons.append(f"missing ML dependencies: {', '.join(deps['missing'])}")
    if not tok:
        reasons.append("tokenizer files missing from adapter dir")
    return {
        "adapter_present": adapter is not None,
        "adapter_dir": os.path.abspath(d),
        "base_model": base_path,
        "base_model_error": base_err,
        "dependencies": deps,
        "tokenizer_present": tok,
        "inference_ready": not reasons,
        "blockers": reasons,
        "generation_enabled": False,  # stays False until a real invocation succeeds
    }


def run_adapter_inference(prompt, known_ledgers, adapter_dir=None, max_new_tokens=256):
    """Run the adapter on a prompt and return a validated JSON intent.

    Fail-closed at every step:
    1. No complete adapter in the (possibly overridden) dir ->
       AdapterMissingError.
    2. Missing ML deps -> RuntimeError naming the missing deps.
    3. Missing tokenizer -> RuntimeError.
    4. Base model not resolvable/allowlisted -> RuntimeError.
    5. Bad prompt/bounds/known_ledgers -> ValueError.
    6. Model load/generation failure -> the underlying exception propagates.
    7. Output is not valid JSON, or JSON fails intent validation ->
       ValueError with the parse/validation errors.

    The model is asked for a JSON intent ONLY (voucher_type/date/entries).
    Its output is never trusted as XML and never used for arithmetic —
    deterministic code validates and builds the XML (see pipeline.py).

    This function is IMPLEMENTED but UNVERIFIED: no adapter, deps, or
    tokenizer exists on this machine, so it fails closed at step 1/2/3.
    """
    from .generate import AdapterMissingError

    d = adapter_dir or ADAPTER_DIR
    if find_adapter(d) is None:
        raise AdapterMissingError(
            f"No complete adapter found in {os.path.abspath(d)}. "
            "Generation refused (fail-closed)."
        )
    deps = check_dependencies()
    if deps["missing"]:
        raise RuntimeError(
            f"ML dependencies missing, cannot run inference: {', '.join(deps['missing'])}"
        )
    if not check_tokenizer(d):
        raise RuntimeError("Tokenizer files missing from adapter dir; cannot run inference.")
    base_path, base_err = resolve_base_model(d)
    if base_err is not None:
        raise RuntimeError(f"Cannot resolve base model: {base_err}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > _MAX_PROMPT_CHARS:
        raise ValueError(
            f"prompt too long ({len(prompt)} chars > {_MAX_PROMPT_CHARS}); input is bounded"
        )
    if not isinstance(max_new_tokens, int) or not (1 <= max_new_tokens <= _MAX_NEW_TOKENS):
        raise ValueError(f"max_new_tokens must be an integer in 1..{_MAX_NEW_TOKENS}")
    if known_ledgers is None:
        raise ValueError("known_ledgers is required (fail-closed)")

    # Lazy imports: only reached when deps are confirmed present.
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    from peft import PeftModel  # noqa: F401

    # Local files only (no implicit download), no remote code.
    # Base loaded EXPLICITLY from its own resolved path; the adapter is
    # applied ONCE. The adapter dir is never passed to
    # AutoModelForCausalLM.from_pretrained (that could auto-load/reapply
    # adapters).
    tokenizer = AutoTokenizer.from_pretrained(
        d, local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype="auto", local_files_only=True, trust_remote_code=False
    )
    model = PeftModel.from_pretrained(base, d, local_files_only=True, trust_remote_code=False)
    model.eval()
    ledgers_text = ", ".join(sorted(str(x) for x in known_ledgers))
    messages = [
        {
            "role": "system",
            "content": (
                "Emit ONLY a JSON object with keys voucher_type, date, entries. "
                "voucher_type is Receipt, Payment, or Journal. date is YYYY-MM-DD. "
                "entries is a list of {ledger, side, amount} where side is Dr or Cr "
                "and amount is a decimal string (e.g. \"5000.00\"). No other text. "
                f"Use ONLY ledger names from this known-ledgers list: {ledgers_text}. "
                "Do not invent ledger names. Do not think out loud; emit only the JSON object."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Defensive: strip any <think>...</think> block if the model emitted
    # thinking despite enable_thinking=False, before JSON parsing.
    raw = _THINK_RE.sub("", raw).strip()
    try:
        intent = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"adapter output is not valid JSON: {e}; raw output: {raw[:300]!r}")
    ok, errors = validate_intent(intent, known_ledgers)
    if not ok:
        raise ValueError(f"adapter output failed intent validation: {'; '.join(errors)}")
    return intent
