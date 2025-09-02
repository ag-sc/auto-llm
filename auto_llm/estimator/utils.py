import json
from typing import List, Dict, Any

from transformers import AutoModel, AutoConfig

CTX_LENGTH_KEYS = [
    "max_position_embeddings",
    "n_positions",
    "seq_length",
    "max_seq_len",
]


MODEL_PARAMS_CACHE_PATH = "/vol/auto_llm/stats/model_params.json"
GPU_PARAMS_CACHE_PATH = "/vol/auto_llm/stats/gpu_params.json"


def cache_model_params(model_name: str):
    model_meta = {}

    model = AutoModel.from_pretrained(model_name)
    N = sum(p.numel() for p in model.parameters())

    model_config = AutoConfig.from_pretrained(model_name).to_dict()
    for key in CTX_LENGTH_KEYS:
        if key in list(model_config.keys()):
            max_length = model_config[key]
            break
    else:
        max_length = -1

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

    with open(model_params_cache_path, "w+") as f:
        json.dump(models_meta, f, indent=4)

    return models_meta


def get_gpu_params(
    gpu_params_cache_path: str = GPU_PARAMS_CACHE_PATH,
) -> Dict[str, Any]:
    with open(gpu_params_cache_path, "r") as f:
        gpu_params = json.load(f)

    return gpu_params
