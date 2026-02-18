import yaml

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.tasks.task import TaskRegistry, Task

description = """
# Sequence To Sequence Task

Input: Sequence
Output: Sequence

Example tasks: Question Answering (generative), Summarization, Machine Translation
"""

with open("config_files/trainer_configs/sample_configs/sample_sequence_to_sequence_trainer_run.yaml", "r") as f:
    config = yaml.safe_load(f.read())

sample_trainer_run_config = TrainerRunConfig.model_validate(config)

@TaskRegistry.register("sequence_to_sequence")
class SequenceToSequenceTask(Task):
    name: str = "sequence_to_sequence"
    description: str = description
    sample_trainer_run_config: TrainerRunConfig = sample_trainer_run_config