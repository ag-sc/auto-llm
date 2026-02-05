from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.tasks.task import BaseTask

description = """
# Sequence To Sequence Task

Input: Sequence
Output: Sequence

Example tasks: Question Answering (generative), Summarization, Machine Translation

# Sequence To Label Task

Input: Sequence
Output: Label

Example tasks: Question Answering (multiple-choice), Sentiment Analysis, Sentence Classification

# Sequence To Structured Output Task

Input: Sequence
Output: Structured Output

Example tasks: NER task, PICO


Reference: https://huggingface.co/docs/transformers/tasks

"""


sample_trainer_run_config = TrainerRunConfig(
    auto_llm_trainer_args=...,
    trainer_args=...,
    trainer_data_builder_config=...,
    peft_config=...,
)

SequenceToLabelTask = BaseTask(
    name="sequence_to_label",
    description=description,
    metrics_list=["accuracy"],
    models_list=["google/gemma-2-2b"],
    sample_trainer_run_config=sample_trainer_run_config,
)
