import os

# resolve relative to the repo root, so it works regardless of the current working directory
LM_EVAL_HARNESS_CUSTOM_TASKS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tasks")
)
