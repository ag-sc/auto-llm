import argparse
import shutil


import lm_eval.models.mistral3
import lm_eval.models.huggingface
from auto_llm.evaluator.utils import hf_apply_chat_template, mistral3_create_tokenizer, run_lm_eval_harness

# to get STDOUT in wandb. See: https://github.com/wandb/wandb/issues/2182#issuecomment-1447879531
shutil._USE_CP_SENDFILE = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    if not args.config_path:
        raise Exception("config path should be provided!")

    # Monkey-patching Mistral 3 tokenizer
    lm_eval.models.mistral3.Mistral3LM._create_tokenizer = mistral3_create_tokenizer

    # Monkey-patching `apply_chat_template` for Qwen3.5 models
    lm_eval.models.huggingface.HFLM.apply_chat_template = hf_apply_chat_template

    run_lm_eval_harness(config_path=args.config_path)
