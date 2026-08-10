from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llapdiffusion.configs import config as base_config
from llapdiffusion.models.summarizer import DECODE_MODES


PREDICT_TYPES = ("v", "x0", "eps")
DEFAULT_PREDICT_TYPE = "v"


def normalize_predict_type(value: object) -> str:
    """Return the canonical diffusion prediction parameterization name."""
    raw = (
        DEFAULT_PREDICT_TYPE
        if value is None or (isinstance(value, str) and not value.strip())
        else value
    )
    name = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "epsilon": "eps",
        "noise": "eps",
        "x_0": "x0",
        "xstart": "x0",
        "x_start": "x0",
        "velocity": "v",
        "v_prediction": "v",
    }
    normalized = aliases.get(name, name)
    if normalized not in PREDICT_TYPES:
        choices = ", ".join(PREDICT_TYPES)
        raise ValueError(f"Unknown predict_type {value!r}; expected one of: {choices}.")
    return normalized


def summarizer_decode_suffix(config_obj: object) -> str:
    """Filename tag for the stage-2 decoding objective (B21 §6).

    The summarizer filename stamps only ``PRED`` and ``VAE_LATENT_CHANNELS``, so two
    summarizers trained under different objectives collide at one path -- and every loader is
    ``strict=True``, so the collision surfaces as a crash in a later stage rather than where it
    was caused. The legacy ``"mean"`` objective keeps the bare name so existing artifacts
    resolve unchanged; anything else is suffixed.
    """
    mode = str(getattr(config_obj, "SUM_DECODE_MODE", "mean")).strip().lower()
    if mode not in DECODE_MODES:
        raise ValueError(f"Unknown SUM_DECODE_MODE={mode!r}; expected one of {sorted(DECODE_MODES)}")
    return "" if mode == "mean" else f"_{mode}"


def artifact_tag_suffix(config_obj: object) -> str:
    """Free-form tag appended to BOTH stage-1 and stage-2 artifact names.

    For runs that must not claim the canonical name even though their architecture is the
    canonical one -- e.g. retraining the legacy objective in this tree as a matched control
    for a new one. The stage-1 trainer derives its own save path, so the tag has to live in
    the shared helper rather than in a caller's ``VAE_CKPT`` override, or the control run
    would silently overwrite the shipped artifact.
    """
    tag = str(getattr(config_obj, "ARTIFACT_TAG", "") or "").strip().strip("_")
    return f"_{tag}" if tag else ""


def summarizer_ckpt_path(config_obj: object) -> Path:
    """Resolve the stage-2 artifact path from a config, objective suffix included."""
    sum_dir = getattr(config_obj, "SUM_DIR", None) or base_config.SUM_DIR
    pred = getattr(config_obj, "PRED", base_config.PRED)
    channels = getattr(config_obj, "VAE_LATENT_CHANNELS", base_config.VAE_LATENT_CHANNELS)
    suffix = summarizer_decode_suffix(config_obj) + artifact_tag_suffix(config_obj)
    return Path(sum_dir) / f"{pred}-{channels}-summarizer{suffix}.pt"


def vae_entity_suffix(config_obj: object) -> str:
    """Filename tag for the stage-1 entity treatment.

    ``_entity`` marks the decoder-side entity embedding (pre-existing). ``_entenc`` adds the
    encoder-side one that stops the pooled latent being permutation-invariant (B20/B5); it is a
    different architecture, so it must not share a path with the artifacts already on disk.
    """
    suffix = "_entity" if bool(getattr(config_obj, "VAE_ENTITY_CONDITION", False)) else ""
    if bool(getattr(config_obj, "VAE_ENTITY_ENCODE", False)):
        suffix += "_entenc"
    return suffix


def vae_ckpt_path(config_obj: object, kind: str = "elbo") -> Path:
    """Resolve a stage-1 artifact path from a config, entity suffixes included."""
    vae_dir = getattr(config_obj, "VAE_DIR", None) or base_config.VAE_DIR
    pred = getattr(config_obj, "PRED", base_config.PRED)
    channels = getattr(config_obj, "VAE_LATENT_CHANNELS", base_config.VAE_LATENT_CHANNELS)
    target_suffix = str(getattr(config_obj, "TARGET_ARTIFACT_SUFFIX", "") or "")
    suffix = vae_entity_suffix(config_obj) + artifact_tag_suffix(config_obj)
    return Path(vae_dir) / f"pred-{pred}_ch-{channels}{suffix}{target_suffix}_{kind}.pt"


def refresh_artifact_paths(config_obj: object) -> object:
    """Recompute ``VAE_CKPT``/``SUM_CKPT`` from the config's current architecture flags.

    The two paths are derived once inside ``apply_dataset_preset``; flipping
    ``SUM_DECODE_MODE`` or ``VAE_ENTITY_ENCODE`` afterwards would otherwise leave them
    pointing at the previous architecture's file, which every ``strict=True`` loader would
    then reject (or, worse, load). Call this after changing either flag.
    """
    setattr(config_obj, "VAE_CKPT", str(vae_ckpt_path(config_obj)))
    setattr(config_obj, "SUM_CKPT", str(summarizer_ckpt_path(config_obj)))
    return config_obj


def clone_config(source: object = base_config) -> SimpleNamespace:
    data = {}
    for name in dir(source):
        if name.startswith("_"):
            continue
        value = getattr(source, name)
        if callable(value):
            continue
        data[name] = value
    return SimpleNamespace(**data)


def dataloader_kwargs(config_obj: object) -> dict:
    """DataLoader worker options for ``run_experiment``, resolved from the config.

    Every dataset family's ``run_experiment`` accepts these, but nothing used to pass them,
    so loaders ran single-threaded in the training process. The date-batching path uses a
    fixed ``_ListBatchSampler`` with no shuffling and seeds its only per-sample RNG from the
    sample identity, so workers can change neither the order nor the contents of a batch --
    which is what ``DiffusionSplitCache``'s sequential fingerprint alignment relies on.

    ``persistent_workers``/``prefetch_factor`` are omitted at 0 workers: PyTorch rejects both.
    """
    workers = int(getattr(config_obj, "DATALOADER_NUM_WORKERS", 0) or 0)
    if workers <= 0:
        return {"num_workers": 0}
    prefetch = getattr(config_obj, "DATALOADER_PREFETCH_FACTOR", None)
    kwargs = {
        "num_workers": workers,
        "persistent_workers": bool(getattr(config_obj, "DATALOADER_PERSISTENT_WORKERS", False)),
    }
    if prefetch is not None:
        kwargs["prefetch_factor"] = int(prefetch)
    return kwargs


def make_jsonable(obj: Any):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
