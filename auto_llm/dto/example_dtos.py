from auto_llm.dto.builder_config import TrainerDataBuilderConfig, SftDatasetType
from auto_llm.dto.trainer_run_config import (
    TrainerRunConfig,
    AutoLlmTrainerArgs,
    TrainerArgs,
    LoraConfig,
)


# conv.     few shot       peft
# conv.     few shot       full weights
# conv.     zero-shot       peft
# conv.     zero-shot       full weights

EXAMPLE_TRAINER_RUN_CONFIG = TrainerRunConfig(
    auto_llm_trainer_args=AutoLlmTrainerArgs(
        trainer_type="sft",
        model_name="google/gemma-2-2b-it",
        truncation=True,
        completion_only_loss=True,
    ),
    trainer_args=TrainerArgs(
        max_length=1024,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=True,
        auto_find_batch_size=False,
        output_dir="",
        resume_from_checkpoint=None,
        gradient_checkpointing=True,
        num_train_epochs=10,
        learning_rate=5e-5,
        weight_decay=0,
        lr_scheduler_type="linear",
        warmup_steps=0,
        report_to="wandb",
        run_name="some name",
        logging_steps=0.1,
        eval_strategy="steps",
        bf16_full_eval=False,
        fp16_full_eval=False,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=None,
    ),
    trainer_data_builder_config=TrainerDataBuilderConfig(
        dataset_dir="/vol/auto_llm/processed_datasets/pico/AD",
        instruction_template="",
        input_template="",
        output_template="",
        dataset_type=SftDatasetType.CONVERSATIONAL,
        instruction_input_separator="\n",
        use_system_message=True,
        parse_output_as_json=True,
        num_few_shot_examples=None,
        few_shot_examples_split="validation",
    ),
    peft_config=LoraConfig(),
)
