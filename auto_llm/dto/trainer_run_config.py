from typing import Literal, List

from pydantic import BaseModel
from transformers import SchedulerType, IntervalStrategy
from transformers.trainer_utils import SaveStrategy

from auto_llm.dto.builder_config import TrainerDataBuilderConfig


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None


class TrainerArgs(BaseModel):
    # The following parameters are a subset of transformers.training_args.TrainingArguments
    # model and tokenizer related
    max_length: int
    bf16: bool = None
    fp16: bool = None

    # bsz related
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int = None
    auto_find_batch_size: bool = None

    # training related
    output_dir: str
    resume_from_checkpoint: str = None
    gradient_checkpointing: bool = None
    num_train_epochs: int
    learning_rate: float = 5e-5
    weight_decay: float = 0
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR
    warmup_steps: int = None

    # tracking/logging related
    report_to: str
    run_name: str
    logging_steps: int | float

    # eval/save related
    eval_strategy: IntervalStrategy
    bf16_full_eval: bool = False
    fp16_full_eval: bool = False

    save_strategy: SaveStrategy
    save_steps: int | float = None
    save_total_limit: int = None


class TrainerRunConfig(BaseModel):
    trainer_type: Literal["sft"]
    model_name: str
    truncation: bool
    peft_config: LoraConfig = None
    trainer_data_builder_config: TrainerDataBuilderConfig
    completion_only_loss: bool
    trainer_args: TrainerArgs
