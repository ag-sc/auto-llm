import os

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from auto_llm.builder.trainer_data_builder.sft_data_builder import (
    ConversationalSftDataBuilder,
    PromptCompletionsSftDataBuilder,
)
from auto_llm.builder.trainer_data_builder.trainer_data_builder import (
    TrainerDataBuilder,
)
from auto_llm.dto.builder_config import SftDatasetType, DatasetSplit
from auto_llm.dto.trainer_run_config import TrainerRunConfig
from auto_llm.pre_processor.sft_pre_procesor import SftPreProcessor

WANDB_TRAIN_PROJECT = "llm4kmu-train"


class SftTrainerWrapper:
    """
    A wrapper class for the Hugging Face TRL SFTTrainer.

    This class handles the end-to-end training pipeline, including:
    - Loading the pre-trained model and tokenizer from Hugging Face Hub.
    - Building and pre-processing the dataset using the custom data builder.
    - Executing `SFTTrainer` with the prepared model, data, and configurations.
    - Saving the fine-tuned model and tokenizer to the specified output directory.
    """

    def __init__(self, config: TrainerRunConfig):
        self.config = config

    def run(self):
        # TODO: add attn implementation
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config.auto_llm_trainer_args.model_name,
            token=os.getenv("HF_TOKEN"),
            attn_implementation=self.config.auto_llm_trainer_args.attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.auto_llm_trainer_args.model_name,
            token=os.getenv("HF_TOKEN"),
        )

        tokenizer.pad_token = tokenizer.eos_token

        # While FT, pad to the right. See https://github.com/huggingface/transformers/issues/34842#issuecomment-2528550342.
        tokenizer.padding_side = "right"

        builder = self.get_trainer_data_builder(config=self.config)
        ds_dict = builder.build()

        print("Train Dataset")
        print(ds_dict["train"])
        for key, value in ds_dict["train"][0].items():
            print(f"{key}\n{value}")

        pre_processor = SftPreProcessor(
            tokenizer=tokenizer,
            completion_only_loss=self.config.auto_llm_trainer_args.completion_only_loss,
        )  # True

        # TRL SftTrainer relies on `return_assistant_tokens_mask` in `apply_chat_template` to get the assistant mask
        # tokens. However, this works only if there is *generation* keyword in the chat template. Hence,
        # manually pre-processing dataset if conversational and demands only completion loss.
        skip_prepare_dataset = False
        completion_only_loss = False
        if self.config.auto_llm_trainer_args.completion_only_loss:
            if pre_processor.is_dataset_conversational(dataset_dict=ds_dict):
                ds_dict = ds_dict.map(
                    function=pre_processor.pre_process,
                    fn_kwargs=dict(
                        max_length=self.config.trainer_args.max_length,
                        truncation=self.config.auto_llm_trainer_args.truncation,
                    ),
                    batched=True,
                )
                skip_prepare_dataset = True
            else:
                # for non-conversational dataset, use TRL's dataset prep.
                # TODO: decide if this is needed or custom pre-processor suffices
                completion_only_loss = True
                print("Doing non-sft preprocessor")
                ds_dict = ds_dict.map(
                    function=pre_processor.pre_process,
                    fn_kwargs=dict(
                        max_length=self.config.trainer_args.max_length,
                        truncation=self.config.auto_llm_trainer_args.truncation,
                    ),
                    batched=True,
                )
                skip_prepare_dataset = True

        trainer_args = SFTConfig(
            **self.config.trainer_args.model_dump(),
            dataset_kwargs={"skip_prepare_dataset": skip_prepare_dataset},
            completion_only_loss=completion_only_loss,
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
                task_type=self.config.peft_config.task_type,
            )

        # TODO: SFTTrainer not returning eval loss
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=trainer_args,
            peft_config=peft_config,
            train_dataset=ds_dict[DatasetSplit.TRAIN],
            eval_dataset=ds_dict[DatasetSplit.VALIDATION],
        )

        print(ds_dict[DatasetSplit.TRAIN][0])

        trainer.train()

        trainer.save_model(self.config.trainer_args.output_dir)
        tokenizer.save_pretrained(self.config.trainer_args.output_dir)

    @staticmethod
    def get_trainer_data_builder(config: TrainerRunConfig) -> TrainerDataBuilder:
        if (
            config.trainer_data_builder_config.dataset_type
            == SftDatasetType.CONVERSATIONAL
        ):
            builder = ConversationalSftDataBuilder(
                **config.trainer_data_builder_config.model_dump()
            )
        elif (
            config.trainer_data_builder_config.dataset_type
            == SftDatasetType.PROMPT_COMPLETIONS
        ):
            builder = PromptCompletionsSftDataBuilder(
                **config.trainer_data_builder_config.model_dump()
            )
        else:
            raise Exception(
                f"Invalid dataset_type: {config.trainer_data_builder_config.dataset_type}"
            )

        return builder
