"""Fail-closed generation entry point.

The fine-tuned LoRA adapter and the full fine-tuned outputs (``res_ft.json``)
were saved only to the volatile Colab ``/content/p2`` directory during the
Phase 2 run and are MISSING. Until a trained adapter is placed under
``models/adapter/`` (or ``TALLY_ADAPTER_DIR``), generation is refused.

This module deliberately contains NO template/fallback generator: producing
template XML and presenting it as trained-model output would be dishonest.
Use :func:`tally_xml.validate.check` to validate XML, or inspect the two
surviving model-generated samples in ``data/model_generated_samples.json``.
"""
import os

ADAPTER_DIR = os.environ.get("TALLY_ADAPTER_DIR", os.path.join("models", "adapter"))


class AdapterMissingError(RuntimeError):
    """Raised when generation is requested without a trained adapter."""


def find_adapter():
    """Return the adapter directory if a trained adapter is present, else None."""
    d = ADAPTER_DIR
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "adapter_config.json")):
        return d
    return None


def adapter_status():
    """Machine-readable status for the GUI status panel."""
    found = find_adapter()
    return {
        "adapter_present": found is not None,
        "adapter_dir": os.path.abspath(ADAPTER_DIR),
        "generation_enabled": found is not None,
        "message": (
            "Trained adapter found — generation enabled."
            if found
            else (
                "No trained adapter found. Generation is DISABLED (fail-closed). "
                "Place the fine-tuned LoRA adapter under models/adapter/ "
                "(adapter_config.json + adapter_model.safetensors) to enable it. "
                "No template or demo XML is generated in its place."
            )
        ),
    }


def generate_voucher(description, **kwargs):
    """Generate TallyPrime XML for a plain-English instruction.

    Always refuses while the adapter is missing. When an adapter is present
    this is the single place where model inference would be wired in.
    """
    if find_adapter() is None:
        raise AdapterMissingError(adapter_status()["message"])
    raise NotImplementedError(
        "Adapter present but model inference is not wired into this build. "
        "See results/phase2_synthetic_eval.md for what the executed run produced."
    )
