"""Stdlib-only adapter weight checks (no torch/safetensors dependency).

The safetensors container is simple: an 8-byte little-endian header
length, a JSON header mapping tensor name -> {dtype, shape, data_offsets},
then the raw tensor bytes. This module reads ONLY the header (never the
tensor data) and verifies the LoRA structure against the adapter's own
``adapter_config.json``:

- every tensor is ``lora_A`` or ``lora_B`` for a target module,
- A/B pairs are consistent (same layer + module, shared rank ``r``),
- the rank matches the config's ``r``,
- the target modules match the config's ``target_modules``.

This proves the weights file is a structurally complete LoRA adapter for
the declared base — it does NOT prove the weights produce good output.
Only a real GPU run does that (see ``tests/run_independent_gpu_tests.py``).
"""
import hashlib
import json
import os
import struct

WEIGHTS_FILE = "adapter_model.safetensors"
CONFIG_FILE = "adapter_config.json"

_MAX_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB: headers are small; anything bigger is corrupt


def read_safetensors_header(path):
    """Read a safetensors header. Returns ({name: {dtype, shape}}, error|None).

    Reads only the 8-byte length + JSON header. Never reads tensor data.
    """
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None, f"{path}: file too short for a safetensors header"
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
                return None, f"{path}: implausible safetensors header length {header_len}"
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                return None, f"{path}: truncated safetensors header"
            header = json.loads(header_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        return None, f"{path}: cannot read safetensors header: {e}"
    if not isinstance(header, dict):
        return None, f"{path}: safetensors header is not a JSON object"
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or "shape" not in meta or "dtype" not in meta:
            return None, f"{path}: tensor {name!r} has a malformed header entry"
        tensors[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"])}
    return tensors, None


def verify_adapter_file(adapter_dir, expected_sha256=None, expected_bytes=None):
    """Verify a weights file's size and sha256. Returns (ok, errors)."""
    errors = []
    path = os.path.join(adapter_dir, WEIGHTS_FILE)
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return False, [f"cannot stat {path}: {e}"]
    if expected_bytes is not None and size != expected_bytes:
        errors.append(f"{WEIGHTS_FILE} is {size} bytes, expected {expected_bytes}")
    if expected_sha256 is not None:
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as e:
            return False, [f"cannot read {path}: {e}"]
        if digest.hexdigest() != expected_sha256.lower():
            errors.append(f"{WEIGHTS_FILE} sha256 mismatch (file differs from expected)")
    return (not errors), errors


def check_adapter_weights(adapter_dir):
    """Verify LoRA weight structure against adapter_config.json.

    Returns {"ok": bool, "errors": [...], "tensors": int, "layers": int,
    "rank": int|None, "modules": [...]}. Never reads tensor data.
    """
    cfg_path = os.path.join(adapter_dir, CONFIG_FILE)
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "errors": [f"cannot read {cfg_path}: {e}"],
                "tensors": 0, "layers": 0, "rank": None, "modules": []}
    errors = []
    rank = cfg.get("r")
    if not isinstance(rank, int) or rank <= 0:
        errors.append(f"adapter_config.json has invalid rank r={rank!r}")
    target_modules = cfg.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        errors.append("adapter_config.json has no target_modules list")
        target_modules = []
    tensors, err = read_safetensors_header(os.path.join(adapter_dir, WEIGHTS_FILE))
    if err is not None:
        errors.append(err)
        return {"ok": False, "errors": errors, "tensors": 0,
                "layers": 0, "rank": rank, "modules": []}

    # Every tensor must be <prefix>.lora_A.<suffix> or .lora_B. — the
    # canonical PEFT naming is
    #   base_model.model.<layer>.self_attn.<module>.lora_A.weight
    #   base_model.model.<layer>.mlp.<module>.lora_A.weight
    # Collect (layer, module, kind) per tensor and check A/B pairing.
    pairs = {}
    layers = set()
    modules_seen = set()
    for name, meta in tensors.items():
        parts = name.split(".")
        if "lora_A" in parts:
            kind = "A"
        elif "lora_B" in parts:
            kind = "B"
        else:
            errors.append(f"tensor {name!r} is not a LoRA A/B tensor")
            continue
        # Layer index: the integer right after "layers" in the name.
        layer = None
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer = int(parts[idx + 1])
        if layer is None:
            errors.append(f"tensor {name!r} has no layer index")
            continue
        layers.add(layer)
        # Module: the target-module name appearing in the tensor name.
        module = next((m for m in target_modules if m in parts), None)
        if module is None:
            errors.append(f"tensor {name!r} is not for a declared target module")
            continue
        modules_seen.add(module)
        shape = meta["shape"]
        if len(shape) != 2:
            errors.append(f"tensor {name!r} has non-2D shape {shape}")
            continue
        if kind == "A" and shape[0] != rank:
            errors.append(f"tensor {name!r} lora_A first dim {shape[0]} != rank {rank}")
        if kind == "B" and shape[1] != rank:
            errors.append(f"tensor {name!r} lora_B second dim {shape[1]} != rank {rank}")
        key = (layer, module)
        if key in pairs and kind in pairs[key]:
            errors.append(f"duplicate {kind} tensor for layer {layer} module {module}")
        pairs.setdefault(key, {})[kind] = shape
    for (layer, module), kinds in sorted(pairs.items()):
        if set(kinds) != {"A", "B"}:
            errors.append(f"layer {layer} module {module} is missing a LoRA half: {sorted(kinds)}")
            continue
        # A is (r, in_features), B is (out_features, r): the shared
        # inner dims must agree (A rows == B cols == r already checked),
        # and the outer dims must be consistent: A cols == B rows is NOT
        # required (A maps in->r, B maps r->out), but both must be > 0.
        if pairs[(layer, module)]["A"][1] <= 0 or pairs[(layer, module)]["B"][0] <= 0:
            errors.append(f"layer {layer} module {module} has a zero outer dim")
    missing_modules = sorted(set(target_modules) - modules_seen)
    if missing_modules:
        errors.append(f"target modules with no tensors: {missing_modules}")
    return {
        "ok": not errors,
        "errors": errors,
        "tensors": len(tensors),
        "layers": len(layers),
        "rank": rank,
        "modules": sorted(modules_seen),
    }
