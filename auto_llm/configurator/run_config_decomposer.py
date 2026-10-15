from typing import Optional

from auto_llm.dto.run_config import AutoLLMRunConfig
from auto_llm.dto.trainer_run_config import TrainerRunConfig

from lm_eval.config.evaluate_config import EvaluatorConfig as LmEvalEvaluatorConfig

class RunConfigDecomposer:

    def __init__(self, run_config: AutoLLMRunConfig):
        self.run_config = run_config

    def to_trainer_run_config(self) -> Optional[TrainerRunConfig]:
    
        if self.run_config.trainer is None:
            return None

        t = self.run_config.trainer
        m = self.run_config.model
        p = self.run_config.prompt_config
        d = self.run_config.data_config
        trk = self.run_config.tracker_config

        # AutoLlmTrainerArgs 
        if t.auto_llm_trainer_args.model_name is None:
            t.auto_llm_trainer_args.model_name = m.model_name
        if t.auto_llm_trainer_args.attn_implementation is None:
            t.auto_llm_trainer_args.attn_implementation = m.attn_implementation
        if t.auto_llm_trainer_args.truncation is None:
            t.auto_llm_trainer_args.truncation = m.truncation

        # TrainerArgs
        if t.trainer_args.max_length is None:
            t.trainer_args.max_length = m.max_length
        if t.trainer_args.report_to is None and trk is not None:
            t.trainer_args.report_to = "wandb"

        # DataBuilder
        if t.trainer_data_builder_config.instruction_template is None:
            t.trainer_data_builder_config.instruction_template = p.instruction_template
        if t.trainer_data_builder_config.input_template is None:
            t.trainer_data_builder_config.input_template = p.input_template
        if t.trainer_data_builder_config.output_template is None:
            t.trainer_data_builder_config.output_template = p.output_template
        if t.trainer_data_builder_config.limit is None:
            t.trainer_data_builder_config.limit = d.limit
        if t.trainer_data_builder_config.num_few_shot_examples is None:
            t.trainer_data_builder_config.num_few_shot_examples = p.num_few_shot_examples

        return TrainerRunConfig(
            auto_llm_trainer_args=t.auto_llm_trainer_args,
            trainer_args=t.trainer_args,
            trainer_data_builder_config=t.trainer_data_builder_config,
            peft_config=t.peft_config,
            tracker_config=trk,
        )

    #TODO adapther_path here should be unneccesary
    def to_evaluator_config(self, adapter_path: Optional[str] = None) -> Optional[LmEvalEvaluatorConfig]:

        if self.run_config.evaluator is None:
            return None

        e = self.run_config.evaluator
        m = self.run_config.model
        p = self.run_config.prompt_config
        d = self.run_config.data_config
        trk = self.run_config.tracker_config

        if e.model_args.pretrained is None:
            e.model_args.pretrained = m.model_name
        if e.model_args.truncation is None:
            e.model_args.truncation = m.truncation
        if e.model_args.max_length is None:
            e.model_args.max_length = m.max_length
            
        # Translate PEFT adapter and precision
        if adapter_path:
            e.model_args.peft = adapter_path
        if self.run_config.trainer and self.run_config.trainer.trainer_args.bf16:
            e.model_args.dtype = "bfloat16"

        # EvaluatorConfig
        if e.tasks is None:
            # Assumes dataset_path ends with the task name (e.g., /data/german_ner -> german_ner)
            e.tasks = d.dataset_path.split("/")[-1] 
        if e.limit is None:
            e.limit = d.limit

        # Format W&B args if TrackerConfig exists
        wandb_args = {}
        if trk:
            wandb_args = {
                "project": trk.wandb_project,
                "name": trk.wandb_run_name,
                "id": trk.wandb_run_id,
                "group": trk.wandb_run_group,
                "job_type": "evaluation",
            }

        # Build and return the REAL lm-eval-harness dataclass
        return LmEvalEvaluatorConfig(
            model="hf",
            model_args=e.model_args.model_dump(exclude_none=True),
            tasks=e.tasks,
            batch_size=e.batch_size,
            max_batch_size=e.max_batch_size,
            limit=e.limit,
            device=e.model_args.device,
            output_path=e.output_path,
            apply_chat_template=e.apply_chat_template,
            write_out=e.write_out,
            log_samples=e.log_samples,
            wandb_args=wandb_args,
        )
