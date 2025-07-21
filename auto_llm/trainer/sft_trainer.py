import os

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from auto_llm.builder.data_builder import DataBuilder
from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.pre_processor.sft_pre_procesor import SftPreProcessor

WANDB_TRAIN_PROJECT = "llm4kmu-train"


class SftTrainerWrapper:
    def __init__(self, config: TrainerRunConfig):
        self.config = config

    def run(self):
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name,
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name,
            token=os.getenv("HF_TOKEN"),
        )

        builder = DataBuilder(dataset_path=self.config.dataset_path)
        ds_dict = builder.build()

        pre_processor = SftPreProcessor(
            tokenizer=tokenizer, completion_only_loss=self.config.completion_only_loss
        )  # True

        # TRL SftTrainer relies on `return_assistant_tokens_mask` in `apply_chat_template` to get the assistant mask
        # tokens. However, this works only if there is *generation* keyword in the chat template. Hence,
        # manually pre-processing dataset if conversational and demands only completion loss.
        skip_prepare_dataset = False
        completion_only_loss = False
        if self.config.completion_only_loss:
            if pre_processor.is_dataset_conversational(dataset_dict=ds_dict):
                ds_dict = ds_dict.map(
                    function=pre_processor.pre_process,
                    fn_kwargs=dict(
                        max_length=self.config.trainer_args.max_length,
                        truncation=self.config.truncation,
                    ),
                    batched=True,
                )
                skip_prepare_dataset = True
            else:
                # for non-conversational dataset, use TRL's dataset prep.
                # TODO: decide if this is needed or custom pre-processor suffices
                completion_only_loss = True

        trainer_args = SFTConfig(
            **self.config.trainer_args.model_dump(),
            dataset_kwargs={"skip_prepare_dataset": skip_prepare_dataset},
            completion_only_loss=completion_only_loss
        )

        if self.config.trainer_args.report_to == "wandb":
            os.environ["WANDB_PROJECT"] = WANDB_TRAIN_PROJECT

        peft_config = None
        if self.config.peft_config:
            peft_config = LoraConfig(
                r=self.config.peft_config.r,
                lora_alpha=self.config.peft_config.lora_alpha,
                lora_dropout=self.config.peft_config.lora_dropout,
                target_modules=self.config.peft_config.target_modules,
            )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=trainer_args,
            peft_config=peft_config,
            train_dataset=ds_dict["train"],
            eval_dataset=ds_dict["val"],
        )

        trainer.train()

        trainer.save_model(self.config.trainer_args.output_dir)
        tokenizer.save_pretrained(self.config.trainer_args.output_dir)
