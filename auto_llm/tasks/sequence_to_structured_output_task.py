from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.tasks.task import BaseTask

description = ...

sample_trainer_run_config = TrainerRunConfig(
    auto_llm_trainer_args=...,
    trainer_args=...,
    trainer_data_builder_config=...,
    peft_config=...,
)

SequenceToStructuredOutputTask = BaseTask(
    name="sequence_to_structured_output",
    description=description,
    metrics_list=["accuracy"],
    models_list=["google/gemma-2-2b"],
    sample_trainer_run_config=sample_trainer_run_config,
)
