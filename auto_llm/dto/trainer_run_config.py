from typing import Literal, List, Optional, Union

from peft import TaskType
from pydantic import BaseModel, Field
from transformers import SchedulerType, IntervalStrategy
from transformers.trainer_utils import SaveStrategy

from auto_llm.dto.builder_config import TrainerDataBuilderConfig


class LoraConfig(BaseModel):
    r: int = Field(
        description="Lora attention dimension (the rank).", title="r", default=16
    )
    lora_alpha: int = Field(
        description="The alpha parameter for Lora scaling.",
        title="lora_alpha",
        default=32,
    )
    lora_dropout: float = Field(
        description="The dropout probability for Lora layers.",
        title="lora_dropout",
        default=0.05,
    )
    target_modules: Optional[Union[List[str], str]] = Field(
        description="The names of the modules to apply the adapter to.",
        title="target_modules",
        default="all-linear",
    )
    task_type: Optional[TaskType] = Field(
        description="The task type for which the model is being fine-tuned (e.g., causal language modeling, sequence classification). Possible values `peft.TaskType`",
        title="Task Type",
        default="CAUSAL_LM",
    )


class AutoLlmTrainerArgs(BaseModel):
    trainer_type: Literal["sft"] = Field(
        description="Sets the Trainer Type. Supported types: ['sft']",
        title="Trainer Type",
        default="sft",
    )
    model_name: str = Field(
        description="Name of the model to further train",
        title="Model Name",
        examples=["google/gemma-2-2b-it"],
    )
    attn_implementation: str = Field(
        description="Name of the model to further train",
        title="Model Name",
        examples=["eager", "sdpa", "flash_attention_2", "flash_attention_3"],
        default="eager",
    )
    truncation: bool = Field(
        description="Sets truncation of the input sequences",
        title="Truncation",
        default=True,
    )
    completion_only_loss: bool = Field(
        description="Sets the loss computation of the Trainer. If set to True, input tokens are ignored for loss computation. This is useful while instruction tuning the model. If set to False, all tokens in the sequences are considered for the loss computation. This is the default causal language modeling objective.",
        title="Completion Only Loss",
        default=True,
    )


class TrainerArgs(BaseModel):
    # The following parameters are a subset of transformers.training_args.TrainingArguments
    # model and tokenizer related
    max_length: int = Field(
        description="Maximum sequence length. This lets the tokenizer decide how long to pad and/or to truncate the input sequences",
        title="Maximum Length",
        default=1024,
    )
    bf16: bool = Field(
        description="Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training.",
        title="bf16",
        default=True,
    )
    fp16: bool = Field(
        description="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.",
        title="fp16",
        default=False,
    )

    # bsz related
    per_device_train_batch_size: int = Field(
        description="Batch size per device accelerator core/CPU for training.",
        title="Per Device Train Batch Size",
        default=4,
    )
    per_device_eval_batch_size: int = Field(
        description="Batch size per device accelerator core/CPU for evaluation.",
        title="Per Device Evaluation Batch Size",
        default=4,
    )
    gradient_accumulation_steps: int = Field(
        description="Batch size per device accelerator core/CPU for evaluation.",
        title="Per Device Evaluation Batch Size",
        default=4,
    )
    auto_find_batch_size: bool = Field(
        description="Batch size per device accelerator core/CPU for evaluation.",
        title="Per Device Evaluation Batch Size",
        default=True,
    )

    # training related
    output_dir: str = Field(
        description="The output directory where the model checkpoints will be written.",
        title="Output Directory",
    )
    resume_from_checkpoint: Optional[str] = Field(
        description="The path to a folder with a valid checkpoint for your model. If not set, starts training from the pre-trained model.",
        title="Resume From Checkpoint",
        default=None,
    )
    gradient_checkpointing: bool = Field(
        description="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
        title="Gradient Checkpointing",
        default=True,
    )
    num_train_epochs: int = Field(
        description="Total number of training epochs to perform",
        title="Number of Training Epochs",
        default=5,
    )
    learning_rate: float = Field(
        description="The initial learning rate for the optimizer.",
        title="Learning Rate",
        default=5e-5,
    )
    weight_decay: float = Field(
        description="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the optimizer.",
        title="Weight Decay",
        default=0,
    )
    lr_scheduler_type: SchedulerType = Field(
        description="The scheduler type to use.",
        title="Learning Rate Scheduler Type",
        default=SchedulerType.LINEAR,
    )
    warmup_steps: int = Field(
        description="Number of steps used for a linear warmup from 0 to `learning_rate`.",
        title="Warmup Steps",
        default=10,
    )

    # tracking/logging related
    report_to: Optional[str] = Field(
        description="The tracker to use.",
        title="Report To",
        default="wandb",
    )
    run_name: str = Field(
        description="The run name in the tracker.",
        title="Run Name",
    )
    logging_steps: int | float = Field(
        description="The frequency of tracking and logging.",
        title="Logging Steps",
        default=0.1,
    )

    # eval/save related
    eval_strategy: IntervalStrategy = Field(
        description="The frequency of performing evaluation on the eval-split.",
        title="Evaluation Strategy",
        default=IntervalStrategy.STEPS,
    )
    label_names: List[str] = Field(
        description="Name of the column containing labels. Need to be set explicitly while training PEFT models.",
        title="Label Names",
        default=["labels"],
    )
    bf16_full_eval: bool = Field(
        description="Whether to use full bf16 evaluation instead of 32-bit.",
        title="bf16 Evaluation",
        default=False,
    )
    fp16_full_eval: bool = Field(
        description="Whether to use full fp16 evaluation instead of 32-bit.",
        title="fp16 Evaluation",
        default=False,
    )

    save_strategy: SaveStrategy = Field(
        description="The frequency of saving model checkpoints.",
        title="Save Strategy",
        default=IntervalStrategy.STEPS,
    )
    save_steps: int | float = Field(
        description="Number of updates steps before two checkpoint saves if save_strategy=steps",
        title="Save Strategy",
        default=0.1,
    )
    save_total_limit: Optional[int] = Field(
        description="If a value is passed, will limit the total amount of checkpoints.",
        title="Save Total Limit",
        default=None,
    )


class TrainerRunConfig(BaseModel):
    auto_llm_trainer_args: AutoLlmTrainerArgs = Field(
        description="Args specific to AutoLLM Trainer",
        title="AutoLLM Trainer Args",
    )
    trainer_args: TrainerArgs = Field(
        description="Args inherited from HuggingFace TR Trainer",
        title="Trainer Args",
    )
    trainer_data_builder_config: TrainerDataBuilderConfig = Field(
        description="Configuration for building trainer data",
        title="Trainer Data Builder Config",
    )
    peft_config: Optional[LoraConfig] = Field(
        description="Configuration for PEFT technique",
        title="PEFT Config",
        default=None,
    )
