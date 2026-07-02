import json
import logging
import os
import torch

from typing import List, Dict, Any, Optional

from transformers import AutoModel, AutoConfig

from auto_llm.registry.estimator_registry import (
    CTX_LENGTH_KEYS,
    MODEL_PARAMS_CACHE_PATH,
    GPU_PARAMS_CACHE_PATH,
)

logger = logging.getLogger(__name__)


def cache_model_params(model_name: str):
    model_meta = {}

    model = AutoModel.from_pretrained(model_name)
    N = sum(p.numel() for p in model.parameters())

    model_config = AutoConfig.from_pretrained(model_name).to_dict()

    # Build list of dicts to search: top-level first, then known nested sub-configs
    # (e.g. multimodal models like Gemma 3 4b store text params under "text_config")
    search_dicts = [model_config]
    for sub_key in ("text_config", "language_config"):
        if sub_key in model_config and isinstance(model_config[sub_key], dict):
            search_dicts.append(model_config[sub_key])

    max_length = -1
    for d in search_dicts:
        for key in CTX_LENGTH_KEYS:
            if key in d:
                max_length = d[key]
                break
        if max_length != -1:
            break

    model_meta[model_name] = {"num_params": N, "max_length": max_length}

    return model_meta


def get_model_params(
    model_names: List[str] = None,
    model_params_cache_path: str = MODEL_PARAMS_CACHE_PATH,
) -> Dict[str, Any]:
    try:
        with open(model_params_cache_path, "r") as f:
            models_meta = json.load(f)
    except FileNotFoundError:
        models_meta = {}

    if not model_names:
        return models_meta

    for model_name in model_names:
        model_meta = models_meta.get(model_name)

        if not model_meta:
            model_meta = cache_model_params(model_name=model_name)
            models_meta.update(model_meta)

    parent = os.path.dirname(model_params_cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(model_params_cache_path, "w+") as f:
        json.dump(models_meta, f, indent=4)

    return models_meta


def get_gpu_params(
    gpu_params_cache_path: str = GPU_PARAMS_CACHE_PATH,
) -> Dict[str, Any]:
    parent = os.path.dirname(gpu_params_cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(gpu_params_cache_path):
        with open(gpu_params_cache_path, "w") as f:
            json.dump({}, f)
    with open(gpu_params_cache_path, "r") as f:
        gpu_params = json.load(f)

    return gpu_params


def resolve_gpu_name(
    gpu_name: Optional[str] = None,
    gpu_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve the GPU name to use for estimation.

    Uses a hybrid approach: if *gpu_name* is provided **and** matches a key in
    *gpu_params*, it is returned directly.  Otherwise the function falls back
    to auto-detection via ``torch.cuda.get_device_name(0)`` and matches the
    result against the *gpu_params* keys by case-insensitive substring.

    Args:
        gpu_name: Explicit GPU name supplied by the user/config.  When
            ``None`` (the default), auto-detection is used immediately.
        gpu_params: Dictionary of known GPU specifications keyed by name.
            Loaded from the GPU params cache when ``None``.

    Returns:
        The matching *gpu_params* key, or ``None`` when no match is found or
        no CUDA device is available.
    """
    if gpu_params is None:
        try:
            gpu_params = get_gpu_params()
        except Exception as exc:
            logger.warning("Could not load GPU params cache: %s", exc)
            return None

    # --- try explicit name first ---
    if gpu_name:
        # exact key match
        if gpu_name in gpu_params:
            return gpu_name
        # case-insensitive match
        for key in gpu_params:
            if key.lower() == gpu_name.lower():
                return key
        logger.warning(
            "Explicit gpu_name '%s' not found in gpu_params — "
            "falling back to auto-detection.",
            gpu_name,
        )

    # --- auto-detection via torch.cuda ---
    try:

        if not torch.cuda.is_available():
            logger.warning("No CUDA device available — cannot auto-detect GPU.")
            return None

        device_name = torch.cuda.get_device_name(0)
        logger.info("Auto-detected GPU device: %s", device_name)

        # case-insensitive substring match against known keys
        device_lower = device_name.lower()
        for key in gpu_params:
            if key.lower() in device_lower or device_lower in key.lower():
                logger.info("Matched GPU device to params key: %s", key)
                return key

        logger.warning(
            "Auto-detected GPU '%s' does not match any key in gpu_params.",
            device_name,
        )
        return None
    except Exception as exc:
        logger.warning("GPU auto-detection failed: %s", exc)
        return None
