# possible keys where the max context length of a model is set in its HF configuration
CTX_LENGTH_KEYS = [
    "max_position_embeddings",
    "n_positions",
    "seq_length",
    "max_seq_len",
]


MODEL_PARAMS_CACHE_PATH = "/vol/auto_llm/stats/model_params.json"
GPU_PARAMS_CACHE_PATH = "/vol/auto_llm/stats/gpu_params.json"
