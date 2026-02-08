import yaml

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.tasks.task import BaseTask

description = """
# Sequence To Sequence Task

Input: Sequence
Output: Sequence

Example tasks: Question Answering (generative), Summarization, Machine Translation
"""

with open("config_files/sample_configs/sample_sequence_to_sequence_trainer_run.yaml", "r") as f:
    config = yaml.safe_load(f.read())

sample_trainer_run_config = TrainerRunConfig.model_validate(config)

SequenceToSequenceTask = BaseTask(
    name="sequence_to_sequence",
    description=description,
    sample_trainer_run_config=sample_trainer_run_config,
)
