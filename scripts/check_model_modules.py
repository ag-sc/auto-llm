#!/usr/bin/env python3
"""Check module names in a model to validate PEFT target_modules.

Usage:
  python scripts/check_model_modules.py --model MODEL_NAME [--hf-token TOKEN] [--output OUTPUT]

Examples:
  python scripts/check_model_modules.py --model google/gemma-2-2b --output /tmp/gemma_modules.txt

This script tries to load the model in low-memory mode and prints any module names
that match common projection names used by LoRA/PEFT (q_proj, k_proj, v_proj, o_proj,
q_attn, gate_proj, down_proj, etc.). It also dumps the first N module names.

Notes:
- The script uses `transformers.AutoModelForCausalLM.from_pretrained` with
  low_cpu_mem_usage=True and device_map='auto'.
- Running this on a machine without GPUs or internet/HF access may fail.
"""

import argparse
import os
import sys
from typing import List

try:
    from transformers import AutoModelForCausalLM, AutoConfig
except Exception as e:
    print("Error importing transformers:", e)
    print("Make sure you have transformers installed in your environment.")
    sys.exit(2)

CANDIDATES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_attn",
    "gate_proj",
    "down_proj",
    "attn",
    "proj",
]


def find_matches(named_modules: List[str], candidates: List[str]):
    matches = []
    for name in named_modules:
        for c in candidates:
            if c in name:
                matches.append((name, c))
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path (HF repo id)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token if needed")
    parser.add_argument("--output", default=None, help="Path to write output (optional)")
    parser.add_argument("--first-n", type=int, default=400, help="How many first modules to print (default=400)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    model_name = args.model
    print(f"Checking model: {model_name}")

    try:
        # Only load config first (fast) and print keys; then try to load model in low memory.
        cfg = AutoConfig.from_pretrained(model_name)
        print("Loaded config. Keys (sample):", list(cfg.to_diffable_dict().keys())[:50])
    except Exception as e:
        print("Failed to load config for model:", e)

    try:
        print("Attempting to load model (low_cpu_mem_usage=True, device_map='auto')...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_auth_token=os.getenv("HF_TOKEN", None),
        )
    except Exception as e:
        print("Failed to load model. Error:", e)
        print("If you are on a headless machine without GPUs or without HF access, consider running this where the model can be downloaded or use a small local checkpoint for testing.")
        sys.exit(3)

    named = [name for name, _ in model.named_modules()]
    print(f"Total named modules: {len(named)}")

    matches = find_matches(named, CANDIDATES)

    out_lines = []
    out_lines.append(f"Model: {model_name}\n")
    out_lines.append(f"Total named modules: {len(named)}\n")

    if matches:
        out_lines.append("Matched candidate module names:\n")
        for m, c in matches:
            out_lines.append(f"  {m}  <-- matched by '{c}'\n")
    else:
        out_lines.append("No obvious candidate module names found among the first modules.\n")

    out_lines.append("\nFirst modules (up to first-n):\n")
    for idx, name in enumerate(named[: args.first_n]):
        out_lines.append(f"{idx:04d}: {name}\n")

    output = "".join(out_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wrote results to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
