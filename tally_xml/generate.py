"""Fail-closed generation entry point.

The fine-tuned LoRA adapter and the full fine-tuned outputs (``res_ft.json``)
were saved only to the volatile Colab ``/content/p2`` directory during the
Phase 2 run and are MISSING. Until a trained adapter is placed under
``models/adapter/`` (or ``TALLY_ADAPTER_DIR``), generation is refused.

This module deliberately contains NO template/fallback generator: producing
template XML and presenting it as trained-model output would be dishonest.
Use :func:`tally_xml.validate.check` to validate XML, or inspect the two
surviving model-generated samples in ``data/model_generated_samples.json``.

Status honesty rules (do not weaken them):
- ``adapter_present`` means the adapter files are really there: config AND
  non-empty weights. A lone ``adapter_config.json`` is NOT an adapter.
- ``generation_enabled`` stays False until model inference is actually
  implemented and verified. Restoring adapter files alone never enables it.
"""
import os

ADAPTER_DIR = os.environ.get("TALLY_ADAPTER_DIR", os.path.join("models", "adapter"))
CONFIG_FILE = "adapter_config.json"
WEIGHTS_FILE = "adapter_model.safetensors"

# Model inference is NOT wired into this build. generate_voucher() raises
# NotImplementedError even with a complete adapter until this becomes True
# via a real, verified inference implementation — never by config alone.
INFERENCE_WIRED = False


class AdapterMissingError(RuntimeError):
    """Raised when generation is requested without a trained adapter."""


def config_present():
    """True if adapter_config.json exists under the adapter dir."""
    return os.path.exists(os.path.join(ADAPTER_DIR, CONFIG_FILE))


def weights_present():
    """True if real (non-empty) adapter weights exist under the adapter dir."""
    p = os.path.join(ADAPTER_DIR, WEIGHTS_FILE)
    return os.path.exists(p) and os.path.getsize(p) > 0


def find_adapter():
    """Return the adapter directory if a complete adapter is present, else None.

    Complete = adapter_config.json AND non-empty adapter_model.safetensors.
    A config-only directory is not an adapter.
    """
    if config_present() and weights_present():
        return ADAPTER_DIR
    return None


def adapter_status():
    """Machine-readable status for the GUI status panel.

    ``generation_enabled`` is False until inference is actually implemented
    and verified. The three states are distinct: missing adapter, config-only
    (weights missing), and complete adapter with unwired inference.
    """
    cfg = config_present()
    wts = weights_present()
    found = find_adapter()
    if found is not None and not INFERENCE_WIRED:
        message = (
            "Adapter files found (adapter_config.json + adapter_model.safetensors), "
            "but model inference is NOT wired into this build, so generation "
            "stays DISABLED. Restoring adapter files alone does not enable "
            "generation. See results/phase2_synthetic_eval.md."
        )
    elif cfg and not wts:
        message = (
            "Only adapter_config.json found — adapter_model.safetensors is "
            "missing or empty. This is NOT a usable adapter. Generation is "
            "DISABLED. Restore the real trained weights to complete the adapter; "
            "generation still stays disabled until inference is wired in."
        )
    elif not cfg:
        message = (
            "No trained adapter found. Generation is DISABLED (fail-closed). "
            "Place the fine-tuned LoRA adapter under models/adapter/ "
            "(adapter_config.json + adapter_model.safetensors) to complete the "
            "adapter files — generation still stays disabled until model "
            "inference is implemented and verified. "
            "No template or demo XML is generated in its place."
        )
    else:  # pragma: no cover - INFERENCE_WIRED True path, kept for the future
        message = "Trained adapter found and inference wired — generation enabled."
    return {
        "config_present": cfg,
        "weights_present": wts,
        "adapter_present": found is not None,
        "inference_wired": INFERENCE_WIRED,
        "adapter_dir": os.path.abspath(ADAPTER_DIR),
        "generation_enabled": found is not None and INFERENCE_WIRED,
        "message": message,
    }


def generate_voucher(description, **kwargs):
    """Generate TallyPrime XML for a plain-English instruction.

    Refuses while the adapter is incomplete. With a complete adapter it still
    refuses (NotImplementedError) because inference is not wired in.
    """
    if find_adapter() is None:
        if config_present():
            raise AdapterMissingError(
                "adapter_config.json found but adapter_model.safetensors is "
                "missing or empty — not a usable adapter. Generation refused."
            )
        raise AdapterMissingError(adapter_status()["message"])
    if not INFERENCE_WIRED:
        raise NotImplementedError(
            "Adapter present but model inference is not wired into this build. "
            "See results/phase2_synthetic_eval.md for what the executed run produced."
        )
    raise NotImplementedError("Inference wiring not implemented.")  # pragma: no cover
