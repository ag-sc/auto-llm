SAMPLE_MODEL_CONFIG = {
    "gemma-2-2b": {
        "model_provider": "HuggingFace",
        "model": "google/gemma-2-2b",
        "device": "cuda:0",
        "max_context_length": 1024,
    },
    "gemma-2-2b-sft": {
        "model_provider": "HuggingFace",
        "model": "google/gemma-2-2b",
        "adapter": "/vol/auto_llm/sft_models/pico_ad_gemma-2-2b_non-conv_zero-shot",
        "device": "cuda:0",
        "max_context_length": 1024,
    },
}

SAMPLE_GENERATION_CONFIG = {"max_new_tokens": 100, "do_sample": True, "top_p": 1}
