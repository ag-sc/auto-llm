#!/usr/bin/env python3
import pathlib
import re

ROOT = pathlib.Path("config_files/trainer_configs/open-medical-llm-benchmark/energy/openmedicalLLM_mixed")
if not ROOT.exists():
    raise SystemExit(f"Folder not found: {ROOT}")

# Keys and desired lines (2-space indent inside trainer_args)
desired = [
    ('logging_steps', '10'),
    ('eval_strategy', '"steps"'),
    ('eval_steps', '50'),
    ('save_strategy', '"steps"'),
    ('save_steps', '50'),
    ('load_best_model_at_end', 'true'),
    ('metric_for_best_model', '"eval_loss"'),
    ('greater_is_better', 'false'),
    ('save_total_limit', '2'),
    ('save_only_model', 'true'),
    ('early_stopping_patience', '5'),
    ('early_stopping_threshold', '1e-4'),
]

files = list(ROOT.glob('*.yaml'))
modified = []
for path in files:
    text = path.read_text()
    if 'trainer_args:' not in text:
        continue
    # find start of trainer_args
    m = re.search(r"^trainer_args:\n", text, flags=re.M)
    if not m:
        continue
    start = m.end()
    # find next top-level key (line that starts at column 0 and ends with ':' ) after start
    m2 = re.search(r"^\w.*:\n", text[start:], flags=re.M)
    if m2:
        end = start + m2.start()
    else:
        end = len(text)
    block = text[start:end]
    orig_block = block
    # Ensure each desired key present or updated
    for key, val in desired:
        # regex to find key: possibly with quotes and spaces
        pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*.*$", flags=re.M)
        line = f"  {key}: {val}"
        if pat.search(block):
            block = pat.sub(line, block)
        else:
            # append before the end of block (but before any trailing blank lines)
            # insert at end with a newline
            if not block.endswith('\n'):
                block = block + '\n'
            block = block + line + '\n'
    if block != orig_block:
        new_text = text[:start] + block + text[end:]
        path.write_text(new_text)
        modified.append(str(path))

print(f"Processed {len(files)} files, modified {len(modified)} files")
for p in modified:
    print(p)
