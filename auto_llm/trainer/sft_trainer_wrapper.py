import os
from typing import Dict, Any

import torch
from accelerate import Accelerator, DistributedType
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
from auto_llm.registry.estimator_registry import CTX_LENGTH_KEYS
from auto_llm.registry.tracker_registry import WANDB_TRAIN_PROJECT
from auto_llm.trainer.trainer_wrapper import TrainerWrapper

accelerator = Accelerator()


class SftTrainerWrapper(TrainerWrapper):
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
        hf_model_config = AutoConfig.from_pretrained(
            self.config.auto_llm_trainer_args.model_name
        ).to_dict()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config.auto_llm_trainer_args.model_name,
            token=os.getenv("HF_TOKEN"),
            attn_implementation=self.config.auto_llm_trainer_args.attn_implementation,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,  # TODO: pass this as trainer arg?
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.auto_llm_trainer_args.model_name,
            token=os.getenv("HF_TOKEN"),
        )

        tokenizer.pad_token = tokenizer.eos_token

        max_length = self.get_max_length(
            max_length=self.config.trainer_args.max_length,
            hf_model_config=hf_model_config,
        )

        # While FT, pad to the right. See https://github.com/huggingface/transformers/issues/34842#issuecomment-2528550342.
        tokenizer.padding_side = "right"

        builder = self.get_trainer_data_builder(config=self.config)
        ds_dict = builder.build()

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
                self.logger.info("Using custom preprocessor for Conversational dataset")
                ds_dict = ds_dict.map(
                    function=pre_processor.pre_process,
                    fn_kwargs=dict(
                        max_length=max_length,
                        truncation=self.config.auto_llm_trainer_args.truncation,
                    ),
                    batched=True,
                )
                skip_prepare_dataset = True
            else:
                # for non-conversational dataset, use TRL's dataset prep.
                # TODO: decide if this is needed or custom pre-processor suffices
                completion_only_loss = True
                self.logger.info(
                    "Using custom preprocessor for Non-conversational dataset"
                )
                ds_dict = ds_dict.map(
                    function=pre_processor.pre_process,
                    fn_kwargs=dict(
                        max_length=self.config.trainer_args.max_length,
                        truncation=self.config.auto_llm_trainer_args.truncation,
                    ),
                    batched=True,
                )
                skip_prepare_dataset = True

        use_reentrant = None
        ddp_find_unused_parameters = None
        if accelerator.state.distributed_type == DistributedType.FSDP:
            use_reentrant = True
        elif accelerator.state.distributed_type == DistributedType.MULTI_GPU:  # for ddp
            use_reentrant = False
            ddp_find_unused_parameters = False

        trainer_args = SFTConfig(
            **self.config.trainer_args.model_dump(),
            dataset_kwargs={"skip_prepare_dataset": skip_prepare_dataset},
            completion_only_loss=completion_only_loss,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
            ddp_find_unused_parameters=ddp_find_unused_parameters,
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

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=trainer_args,
            peft_config=peft_config,
            train_dataset=ds_dict[DatasetSplit.TRAIN],
            eval_dataset=ds_dict[DatasetSplit.VALIDATION],
        )

        self.logger.info("Train Dataset")
        self.logger.info(f"# train samples: {len(ds_dict['train'])}")
        self.logger.info(ds_dict["train"])
        for key, value in ds_dict["train"][0].items():
            self.logger.info(f"{key}\n{value}")

        trainer.train()

        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        #
        # full_state_dict_config = FullStateDictConfig(
        #     offload_to_cpu=True, rank0_only=True
        # )
        # with FSDP.state_dict_type(
        #     model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        # ):
        #     state_dict = model.state_dict()
        #
        # model.save_pretrained(
        #     save_directory=self.config.trainer_args.output_dir,
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=state_dict,
        # )

        trainer.save_model(self.config.trainer_args.output_dir)
        tokenizer.save_pretrained(self.config.trainer_args.output_dir)

        self.logger.info(
            f"Model and Tokenizer saved in the path: {self.config.trainer_args.output_dir}"
        )

    @staticmethod
    def get_max_length(
        hf_model_config: Dict[str, Any],
        max_length: int = None,
    ):
        # Set max_length to the configured value, if it exists. Otherwise, find the model max context length.
        if not max_length:
            for key in CTX_LENGTH_KEYS:
                if key in list(hf_model_config.keys()):
                    max_length = hf_model_config[key]
                    break
            else:
                raise Exception(f"Max length can not be found in the model config!")

            # Model max length can be as large as 131072. This is unnecessary while SFT. Setting a minimum of 1024,
            # if max_length not configured by the user.
            max_length = min(1024, max_length)
        return max_length

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
