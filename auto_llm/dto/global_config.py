from typing import  Optional
from pydantic import BaseModel, Field

class DataConfig(BaseModel):

    # TODO questions:
    # - dataset_path doesnt appear in LM-eval-harness config, also the tasks seem to be predefined by the dataset
    # - output_template doesnt appear in "

    #dataset_type: ...  # ?

    limit: Optional[int] = Field(
        title="Limit for dataset samples",
        description="Number of samples to consider",
        default=None,
    )


class ModelConfig(BaseModel):
    model_name: str = Field(
        description="Name of the model to further train",
        title="Model Name",
        examples=["google/gemma-2-2b-it"],
    )
    attn_implementation: str = Field(
        description="Name of the model to further train",
        title="Model Name",
        examples=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        default="flash_attention_2",
    )

    max_length: int = Field(
        description="Maximum sequence length. This lets the tokenizer decide how long to pad and/or to truncate the input sequences",
        title="Maximum Length",
        default=2048,
    )

    truncation: bool = Field(
        description="Sets truncation of the input sequences",
        title="Truncation",
        default=True,
    )


class PromptConfig(BaseModel):
    # = system_instruction for LM-eval     = description in task config
    instruction_template: str = Field(
        title="Instruction Template",
        description="The template for the **instruction**. This is the text that goes to the ``system`` field in case of an instruction tuned model. You can also add placeholders to be replaced with input or examples.<br><br>Possible placeholders: ``{{input}}``, ``{{examples}}``",
    )

    input_template: str = Field()

    output_template: str = Field()

    # = num_fewshot for LM-Eval
    num_few_shot_examples: Optional[int] = Field(
        title="Number of Few-Shot Examples",
        description="The number of few-shot examples to include in the formatted prompt for in-context learning.",
        default=None,
    )


class TrackerConfig(BaseModel):
    wandb_project: Optional[str] = Field(
        description="The W&B project name to which this training run belongs. This is used to group multiple training runs together in W&B for better organization and comparison.",
        title="W&B Project",
        default=None,
    )
    wandb_run_name: Optional[str] = Field(
        description="The W&B run name for this training run. This is used to identify the training run in W&B.",
        title="W&B Run Name",
        default=None,
    )
    wandb_run_id: Optional[str] = Field(
        description="The W&B run ID associated with this training run. This is used to fetch the training metrics and plots from W&B after the run is completed.",
        title="W&B Run ID",
        default=None,
    )
    wandb_run_group: Optional[str] = Field(
        description="The W&B run group name to which this training run belongs. This is used to group multiple training runs together in W&B for better organization and comparison.",
        title="W&B Run Group",
        default=None,
    )
    wandb_run_type: Optional[str] = Field(
        description="The W&B run job type.",
        title="W&B Run Group",
        default="training",
    )
