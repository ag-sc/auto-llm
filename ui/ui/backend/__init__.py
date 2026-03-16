from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

CONFIGS_DIR = f"{ROOT_DIR}/.cache"
OUTPUT_DIR = f"{CONFIGS_DIR}/sft_models/"
