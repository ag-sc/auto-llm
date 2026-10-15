from typing import Optional
from pydantic import BaseModel, Field

from auto_llm.dto.global_config import ModelConfig, PromptConfig, DataConfig, TrackerConfig
from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.dto.evaluator_run_config import EvaluatorConfig


class AutoLLMRunConfig(BaseModel):
    tracker_config: Optional[TrackerConfig] = None
    model: ModelConfig = Field(alias="model_config") # Pydantic reserved so we have to use alias
    prompt_config: PromptConfig
    data_config: DataConfig

    trainer: Optional[TrainerRunConfig] = None
    evaluator: Optional[EvaluatorConfig] = None





"""

from typing import Optional
from pydantic import BaseModel, Field, model_validator

from auto_llm.dto.global_config import ModelConfig, PromptConfig, DataConfig, TrackerConfig




class AutoLLMRunConfig(BaseModel):

    tracker_config: TrackerConfig = Field(
        description="Configuration for experiment trackers like W&B",
        title="Tracker Config",
        default=None,
    )

    data_config: DataConfig = Field(
        description="Global configurations for the run",
        title="Global Configs",
    )

    model_config: ModelConfig = Field(
        description="Model specific configurations",
        title="Model Config",
    )

    prompt_config: PromptConfig = Field(
        description="Prompt specific configurations",
        title="Prompt Config",
    )

    trainer: Optional[TrainerRunConfig] = ...
    evaluator: Optional[EvaluatorConfig] = None


    @model_validator(mode="after")
    def propagate_shared_fields(self) -> "AutoLLMRunConfig":
        model = self.model_config
        prompt = self.prompt_config
        data = self.data_config
        tracker = self.tracker_config

        # Trainer
        if self.trainer is not None:
            t = self.trainer
            if t.auto_llm_trainer_args is not None:
                t.auto_llm_trainer_args.model_name = model.model_name
                t.auto_llm_trainer_args.attn_implementation = model.attn_implementation
                t.auto_llm_trainer_args.truncation = model.truncation

            if t.trainer_args is not None:
                t.trainer_args.max_length = model.max_length
                t.trainer_args.report_to = "wandb" if tracker else None

            if t.trainer_data_builder_config is not None:
                t.trainer_data_builder_config.instruction_template = prompt.instruction_template
                t.trainer_data_builder_config.input_template = prompt.input_template
                t.trainer_data_builder_config.output_template = prompt.output_template
                t.trainer_data_builder_config.limit = data.limit
                t.trainer_data_builder_config.num_few_shot_examples = prompt.num_few_shot_examples

        # Evaluator
        if self.evaluator is not None:
            e = self.evaluator
            e.model_args = {
                "pretrained": model.model_name,
                "truncation": model.truncation,
                "max_length": model.max_length,
            }
            e.num_fewshot = prompt.num_few_shot_examples
            e.system_instruction = prompt.instruction_template
            e.limit = data.limit
            if tracker:
                e.wandb_args = {
                    "project": tracker.wandb_project,
                    "name": tracker.wandb_run_name,
                    "id": tracker.wandb_run_id,
                    "group": tracker.wandb_run_group,
                    "job_type": "evaluation",
                }

        return self
"""

# TODO
# - try out validators, if not make duplicate fields optional in their classes
# - check out hydra hierarchy


# DID
# made the duplicate fields in trainer optional
