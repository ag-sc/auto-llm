import os
from typing import Dict, Any

import torch
from accelerate import Accelerator, DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training # Importato prepare_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
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
    def __init__(self, config: TrainerRunConfig):
        self.config = config

    def run(self):
        hf_model_config = AutoConfig.from_pretrained(
            self.config.auto_llm_trainer_args.model_name
        ).to_dict()

        # START QLORA CONFIG 
        bnb_config = None
        if getattr(self.config, "quantization_config", None) is not None:
            q = self.config.quantization_config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=q.load_in_4bit,
                load_in_8bit=getattr(q, "load_in_8bit", False),
                bnb_4bit_quant_type=q.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=q.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=getattr(torch, q.bnb_4bit_compute_dtype),
            )
            self.logger.info("Start QLoRa: applying BitsAndBytesConfig.")

        model_kwargs = {
            "pretrained_model_name_or_path": self.config.auto_llm_trainer_args.model_name,
            "token": os.getenv("HF_TOKEN"),
            "attn_implementation": self.config.auto_llm_trainer_args.attn_implementation,
            "low_cpu_mem_usage": True,
            "device_map": "auto", 
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        
        if self.config.quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.trainer_args.gradient_checkpointing,
            )
        elif (
            self.config.peft_config is not None
            and self.config.trainer_args.gradient_checkpointing
        ):
            model.enable_input_require_grads()
        # END QLORA CONFIG 

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.auto_llm_trainer_args.model_name,
            token=os.getenv("HF_TOKEN"),
        )

        tokenizer.pad_token = tokenizer.eos_token
        max_length = self.get_max_length(
            max_length=self.config.trainer_args.max_length,
            hf_model_config=hf_model_config,
        )
        tokenizer.padding_side = "right"

        builder = self.get_trainer_data_builder(config=self.config)
        ds_dict = builder.build()

        pre_processor = SftPreProcessor(
            tokenizer=tokenizer,
            completion_only_loss=self.config.auto_llm_trainer_args.completion_only_loss,
        )

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
                    desc="Pre-processing dataset",
                )
                skip_prepare_dataset = True
            else:
                completion_only_loss = True
                self.logger.info("Using custom preprocessor for Non-conversational dataset")
                ds_dict = ds_dict.map(
                    function=pre_processor.pre_process,
                    fn_kwargs=dict(
                        max_length=self.config.trainer_args.max_length,
                        truncation=self.config.auto_llm_trainer_args.truncation,
                    ),
                    batched=True,
                    desc="Pre-processing dataset",
                )
                skip_prepare_dataset = True

        # Gestione DistributedType
        use_reentrant = False 
        ddp_find_unused_parameters = None
        if accelerator.state.distributed_type == DistributedType.FSDP:
            use_reentrant = True
        elif accelerator.state.distributed_type == DistributedType.MULTI_GPU:
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

        trainer.train()
        trainer.save_model(self.config.trainer_args.output_dir)
        tokenizer.save_pretrained(self.config.trainer_args.output_dir)

        self.logger.info(f"Model and Tokenizer saved in: {self.config.trainer_args.output_dir}")

    @staticmethod
    def get_max_length(hf_model_config: Dict[str, Any], max_length: int = None):
        if not max_length:
            for key in CTX_LENGTH_KEYS:
                if key in list(hf_model_config.keys()):
                    max_length = hf_model_config[key]
                    break
            else:
                raise Exception(f"Max length can not be found in the model config!")
            max_length = min(1024, max_length)
        return max_length

    @staticmethod
    def get_trainer_data_builder(config: TrainerRunConfig) -> TrainerDataBuilder:
        if config.trainer_data_builder_config.dataset_type == SftDatasetType.CONVERSATIONAL:
            builder = ConversationalSftDataBuilder(**config.trainer_data_builder_config.model_dump())
        elif config.trainer_data_builder_config.dataset_type == SftDatasetType.PROMPT_COMPLETIONS:
            builder = PromptCompletionsSftDataBuilder(**config.trainer_data_builder_config.model_dump())
        else:
            raise Exception(f"Invalid dataset_type: {config.trainer_data_builder_config.dataset_type}")
        return builder