import yaml

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.tasks.task import BaseTask

description = """
# Sequence To Structured Output Task

Input: Sequence
Output: Structured Output

Example tasks: NER task, PICO
"""

with open("config_files/trainer_configs/sample_configs/sample_sequence_to_structured_output_trainer_run.yaml", "r") as f:
    config = yaml.safe_load(f.read())

sample_trainer_run_config = TrainerRunConfig.model_validate(config)

SequenceToStructuredOutputTask = BaseTask(
    name="sequence_to_structured_output",
    description=description,
    sample_trainer_run_config=sample_trainer_run_config,
)
