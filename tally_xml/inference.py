"""Dual-adapter inference wiring (fail-closed, UNVERIFIED).

Two trained LoRA adapters (same base, ``Qwen/Qwen3-0.6B``):

- JSON adapter (prototype acceptance path): emits a structured JSON intent
  (voucher_type/date/entries). Out-of-scope requests emit exactly
  ``REFUSE: <reason>``. Output is validated by ``validate_intent``;
  deterministic code builds the XML (see pipeline.py).
- XML adapter (legacy/research path): emits a TallyPrime XML import
  message. Its output is UNTRUSTED input: it is passed through
  ``strict.validate_strict`` and treated as research only. Never
  presented as prototype acceptance output, and never used to claim
  JSON capability.

Both paths share one loading core (``AdapterSession``): explicit
adapter-file checks, dependency checks, tokenizer checks, base-model
resolution from the adapter's own ``adapter_config.json`` + allowlist,
base loaded explicitly, adapter applied ONCE via
``PeftModel.from_pretrained(base, adapter_dir)``. The adapter dir is
NEVER passed to ``AutoModelForCausalLM.from_pretrained`` (that could
auto-load/reapply adapters). ``local_files_only=True`` on every
``from_pretrained`` call (no implicit network download),
``trust_remote_code=False`` everywhere (no remote code execution).
Input is bounded (max prompt chars), output is bounded (max new
tokens), Qwen3 thinking is disabled, and any ``<think>`` block is
stripped defensively before parsing.

Tokenizer: a LoRA adapter dir has no tokenizer files. By default the
tokenizer is loaded from the adapter dir (the layout the existing
tests cover); pass ``tokenizer_dir`` explicitly — e.g. the local
base-model directory — to supply one. ``AdapterSession`` refuses
when neither dir has tokenizer files.

STATUS: IMPLEMENTED BUT UNVERIFIED. No real adapter invocation has
succeeded anywhere: the ML dependencies (transformers/torch/peft/
safetensors) are not installed on this machine, the base model and
its tokenizer are not present, and this machine has no GPU.
``generation_enabled`` stays False and ``generate.INFERENCE_WIRED``
stays False until a real adapter invocation actually succeeds
end-to-end. Restoring adapter files alone never enables generation.
Local verification covers only adapter-file structure (see
``weights.check_adapter_weights``) and mocked wiring tests — NOT
inference. GPU verification is a separate step run by the caller
(see ``tests/run_independent_gpu_tests.py``).
"""
import importlib.util
import json
import os
import re

from .generate import ADAPTER_DIR, find_adapter, read_adapter_config
from .strict import validate_intent, validate_strict

# ML deps are imported lazily INSIDE AdapterSession, after the
# dependency check, so importing this module stays stdlib-only.
_ML_DEPS = ("transformers", "torch", "peft", "safetensors")

# Bounds for adapter-backed extraction.
_MAX_PROMPT_CHARS = 2000
_MAX_NEW_TOKENS = 512

# Base-model allowlist: the intended Phase 2 base. Hub id, or a local
# directory with exactly this name.
_ALLOWED_BASE_HUB_ID = "Qwen/Qwen3-0.6B"
_ALLOWED_BASE_DIR_NAME = "Qwen3-0.6B"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_REFUSE_RE = re.compile(r"^REFUSE\s*:\s*(.+)", re.S)

# Default adapter dirs (env-overridable). The JSON adapter falls back to
# the historical single-adapter dir; the XML adapter has its own.
JSON_ADAPTER_DIR = os.environ.get("TALLY_JSON_ADAPTER_DIR", ADAPTER_DIR)
XML_ADAPTER_DIR = os.environ.get("TALLY_XML_ADAPTER_DIR", os.path.join("models", "adapter-xml"))

_JSON_SYSTEM_TEMPLATE = (
    "Emit ONLY a JSON object with keys voucher_type, date, entries. "
    "voucher_type is Receipt, Payment, or Journal. date is YYYY-MM-DD. "
    "entries is a list of {{ledger, side, amount}} where side is Dr or Cr "
    "and amount is a decimal string (e.g. \"5000.00\"). No other text. "
    "Use ONLY ledger names from this known-ledgers list: {ledgers}. "
    "Do not invent ledger names. Do not think out loud; emit only the JSON object."
)
_JSON_REFUSE_SUFFIX = (
    "If the request is out of scope — not a Receipt, Payment, or Journal "
    "voucher, or missing/invalid required fields (ledger, amount, date) — "
    "output exactly: REFUSE: <reason>. Never emit a voucher for an "
    "out-of-scope request, and never emit anything except the JSON object "
    "or the REFUSE line."
)
_XML_SYSTEM_TEMPLATE = (
    "Emit ONLY a TallyPrime XML import message: exactly one <ENVELOPE> "
    "element containing one VOUCHER (Receipt, Payment, or Journal) with "
    "DATE (YYYYMMDD), VOUCHERTYPENAME, and at least 2 ALLLEDGERENTRIES.LIST "
    "entries (LEDGERNAME, ISDEEMEDPOSITIVE Yes for Dr / No for Cr, AMOUNT "
    "signed: negative for Dr, positive for Cr). Dr and Cr totals must "
    "balance exactly. No other text. Use ONLY ledger names from this "
    "known-ledgers list: {ledgers}. Do not invent ledger names. Do not "
    "think out loud; emit only the XML message."
)


def _ledgers_text(known_ledgers):
    return ", ".join(sorted(str(x) for x in known_ledgers))


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
    """True if tokenizer files are present in the given dir.

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


def resolve_tokenizer_dir(adapter_dir=None, tokenizer_dir=None):
    """Resolve where the tokenizer comes from.

    Explicit ``tokenizer_dir`` wins. Otherwise the adapter dir is used
    when it holds tokenizer files. Returns (dir|None, error|None).
    """
    if tokenizer_dir:
        return tokenizer_dir, None
    d = adapter_dir or ADAPTER_DIR
    if check_tokenizer(d):
        return d, None
    return None, (
        "tokenizer files missing from adapter dir; pass tokenizer_dir "
        "(e.g. the base-model directory) to supply one"
    )


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


def inference_status(adapter_dir=None, tokenizer_dir=None):
    """Machine-readable status: adapter, deps, tokenizer, base, verdict.

    ``adapter_dir`` overrides the default for EVERY check, and
    ``tokenizer_dir`` overrides where the tokenizer is looked for.
    ``generation_enabled`` stays False until a real adapter invocation
    succeeds end-to-end.
    """
    d = adapter_dir or ADAPTER_DIR
    adapter = find_adapter(d)
    deps = check_dependencies()
    tdir, terr = resolve_tokenizer_dir(d, tokenizer_dir)
    tok = tdir is not None and check_tokenizer(tdir)
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
        reasons.append(f"tokenizer files missing: {terr or 'no tokenizer dir resolved'}")
    return {
        "adapter_present": adapter is not None,
        "adapter_dir": os.path.abspath(d),
        "tokenizer_dir": os.path.abspath(tdir) if tdir else None,
        "base_model": base_path,
        "base_model_error": base_err,
        "dependencies": deps,
        "tokenizer_present": bool(tok),
        "inference_ready": not reasons,
        "blockers": reasons,
        "generation_enabled": False,  # stays False until a real invocation succeeds
    }


def dual_adapter_status(json_adapter_dir=None, xml_adapter_dir=None, tokenizer_dir=None):
    """Status for BOTH adapters (JSON primary, XML legacy/research).

    ``tokenizer_dir``, when given, applies to both (same base tokenizer).
    """
    js = inference_status(json_adapter_dir or JSON_ADAPTER_DIR, tokenizer_dir=tokenizer_dir)
    xs = inference_status(xml_adapter_dir or XML_ADAPTER_DIR, tokenizer_dir=tokenizer_dir)
    return {
        "json": js,
        "xml": xs,
        "both_ready": bool(js["inference_ready"] and xs["inference_ready"]),
        "generation_enabled": False,  # stays False until a real invocation succeeds
    }


class AdapterSession:
    """A loaded adapter (base + LoRA + tokenizer) reused across generations.

    Setup checks fail closed in __init__: missing adapter, missing ML
    deps, missing tokenizer, unresolvable/non-allowlisted base. Use as
    a context manager; the model is loaded ONCE per session, not once
    per prompt (a 30-case GPU run must not reload per case).
    """

    def __init__(self, adapter_dir=None, tokenizer_dir=None):
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
        tdir, terr = resolve_tokenizer_dir(d, tokenizer_dir)
        if tdir is None or not check_tokenizer(tdir):
            raise RuntimeError(f"Tokenizer files missing; cannot run inference. {terr or ''}".strip())
        base_path, base_err = resolve_base_model(d)
        if base_err is not None:
            raise RuntimeError(f"Cannot resolve base model: {base_err}")
        self.adapter_dir = d
        self.tokenizer_dir = tdir
        self.base_model = base_path

        # Lazy imports: only reached when deps are confirmed present.
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from peft import PeftModel  # noqa: F401

        # Local files only (no implicit download), no remote code.
        # Base loaded EXPLICITLY from its own resolved path; the adapter
        # is applied ONCE. The adapter dir is never passed to
        # AutoModelForCausalLM.from_pretrained (that could auto-load/
        # reapply adapters).
        self._tokenizer = AutoTokenizer.from_pretrained(
            tdir, local_files_only=True, trust_remote_code=False
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype="auto", local_files_only=True, trust_remote_code=False
        )
        self._model = PeftModel.from_pretrained(base, d, local_files_only=True, trust_remote_code=False)
        self._model.eval()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Release the model/tokenizer references (lets the caller free GPU/RAM)."""
        self._model = None
        self._tokenizer = None

    def _check_prompt_args(self, prompt, known_ledgers, max_new_tokens):
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

    def generate_raw(self, prompt, known_ledgers, max_new_tokens=256, system_content=None):
        """Generate raw model text for a prompt. Returns the stripped text.

        Raises ValueError for bad prompt/bounds/known_ledgers. Model
        output is returned unvalidated — callers must parse + validate it.
        """
        self._check_prompt_args(prompt, known_ledgers, max_new_tokens)
        if system_content is None:
            system_content = _JSON_SYSTEM_TEMPLATE.format(ledgers=_ledgers_text(known_ledgers))
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        raw = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Defensive: strip any <think>...</think> block if the model emitted
        # thinking despite enable_thinking=False, before parsing.
        return _THINK_RE.sub("", raw).strip()

    def request_json(self, prompt, known_ledgers, max_new_tokens=256):
        """Run the JSON adapter on a prompt. Returns a result dict.

        {"ok", "intent"|None, "refused", "reason"|None, "errors": [...]}.
        A ``REFUSE: <reason>`` output is a successful refusal (ok=True),
        not an error. Invalid JSON / failed validation is ok=False.
        Raises only for setup failures (missing adapter/deps/tokenizer/
        base) and bad arguments — never for model output problems.
        """
        self._check_prompt_args(prompt, known_ledgers, max_new_tokens)
        system = _JSON_SYSTEM_TEMPLATE.format(ledgers=_ledgers_text(known_ledgers))
        system = system + " " + _JSON_REFUSE_SUFFIX
        raw = self.generate_raw(prompt, known_ledgers, max_new_tokens, system)
        match = _REFUSE_RE.match(raw)
        if match:
            reason = match.group(1).strip()
            if not reason:
                return {"ok": False, "intent": None, "refused": False, "reason": None,
                        "errors": ["adapter output REFUSE without a reason"]}
            return {"ok": True, "intent": None, "refused": True, "reason": reason, "errors": []}
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"ok": False, "intent": None, "refused": False, "reason": None,
                    "errors": [f"adapter output is not valid JSON: {e}; raw output: {raw[:300]!r}"]}
        ok, errors = validate_intent(intent, known_ledgers)
        if not ok:
            return {"ok": False, "intent": None, "refused": False, "reason": None, "errors": errors}
        return {"ok": True, "intent": intent, "refused": False, "reason": None, "errors": []}

    def request_xml(self, prompt, known_ledgers, max_new_tokens=256):
        """Run the XML adapter on a prompt. Returns a result dict.

        {"ok", "xml"|None, "errors": [...]}. The output is UNTRUSTED:
        it passes through ``validate_strict``; anything failing the gate
        is ok=False. RESEARCH ONLY — never prototype acceptance output.
        Raises only for setup failures and bad arguments.
        """
        self._check_prompt_args(prompt, known_ledgers, max_new_tokens)
        system = _XML_SYSTEM_TEMPLATE.format(ledgers=_ledgers_text(known_ledgers))
        raw = self.generate_raw(prompt, known_ledgers, max_new_tokens, system)
        ok, errors = validate_strict(raw, known_ledgers)
        return {"ok": ok, "xml": raw if ok else None, "errors": errors}


def run_adapter_inference(prompt, known_ledgers, adapter_dir=None, max_new_tokens=256,
                          tokenizer_dir=None):
    """Run the (JSON) adapter on a prompt and return a validated JSON intent.

    Fail-closed at every step: missing adapter -> AdapterMissingError;
    missing ML deps / tokenizer / unresolvable base -> RuntimeError;
    bad prompt/bounds/known_ledgers -> ValueError; non-JSON output or
    failed intent validation -> ValueError. A ``REFUSE: ...`` output is
    NOT accepted here (it is not a JSON intent) — use
    :func:`run_json_adapter_request` for REFUSE-aware calls.

    The model is asked for a JSON intent ONLY (voucher_type/date/entries).
    Its output is never trusted as XML and never used for arithmetic —
    deterministic code validates and builds the XML (see pipeline.py).

    This function is IMPLEMENTED but UNVERIFIED: no real adapter
    invocation has succeeded on any machine yet.
    """
    with AdapterSession(adapter_dir, tokenizer_dir) as session:
        # system_content=None: generate_raw validates args first, then
        # formats the default JSON system prompt.
        raw = session.generate_raw(prompt, known_ledgers, max_new_tokens, None)
    try:
        intent = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"adapter output is not valid JSON: {e}; raw output: {raw[:300]!r}")
    ok, errors = validate_intent(intent, known_ledgers)
    if not ok:
        raise ValueError(f"adapter output failed intent validation: {'; '.join(errors)}")
    return intent


def run_json_adapter_request(prompt, known_ledgers, adapter_dir=None, max_new_tokens=256,
                             tokenizer_dir=None):
    """REFUSE-aware JSON adapter call. See ``AdapterSession.request_json``."""
    with AdapterSession(adapter_dir, tokenizer_dir) as session:
        return session.request_json(prompt, known_ledgers, max_new_tokens)


def run_xml_adapter_inference(prompt, known_ledgers, adapter_dir=None, max_new_tokens=256,
                              tokenizer_dir=None):
    """XML adapter call (legacy/research). See ``AdapterSession.request_xml``."""
    with AdapterSession(adapter_dir, tokenizer_dir) as session:
        return session.request_xml(prompt, known_ledgers, max_new_tokens)
