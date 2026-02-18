import yaml

from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.tasks.task import TaskRegistry, Task

description = """
# Sequence To Label Task

Input: Sequence
Output: Label

Example tasks: Question Answering (multiple-choice), Sentiment Analysis, Sentence Classification

"""


with open("config_files/trainer_configs/sample_configs/sample_sequence_to_label_trainer_run.yaml", "r") as f:
    config = yaml.safe_load(f.read())

sample_trainer_run_config = TrainerRunConfig.model_validate(config)

@TaskRegistry.register("sequence_to_label")
class SequenceToLabelTask(Task):
    name: str = "sequence_to_label"
    description: str = description
    sample_trainer_run_config: TrainerRunConfig = sample_trainer_run_config
