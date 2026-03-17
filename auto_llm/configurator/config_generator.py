import enum
import json
import os
from typing import List, Dict, Any, Optional
import uuid
import yaml
from pydantic import BaseModel

from auto_llm.dto.builder_config import TrainerDataBuilderConfig, SftDatasetType
from auto_llm.dto.trainer_run_config import (
    TrackerConfig,
    TrainerArgs,
    AutoLlmTrainerArgs,
    LoraConfig,
    TrainerRunConfig,
)
from auto_llm.registry.evaluator_registry import LM_EVAL_HARNESS_CUSTOM_TASKS_PATH
from auto_llm.registry.tracker_registry import WANDB_PROJECT


class Priority(str, enum.Enum):
    PRIORITY_ONE = "1"
    PRIORITY_TWO = "2"
    PRIORITY_THREE = "3"


class ConfigMode(str, enum.Enum):
    TRAINER_RUN_CFG = "trainer_run_cfg"
    EVALUATOR_RUN_CFG = "evaluator_run_cfg"


class ConfiguratorOutput(BaseModel):
    run_name: str
    run_id: Optional[str] = None
    config_path: str
    mode: ConfigMode
    config: Dict[str, Any]
    priority: Priority


class TrainEvalRunConfigurator:
    def __init__(
        self,
        model_names: List[str],
        task: str,
        dataset_path: str,
        output_path: str,
        configs_path: str,
        instruction_template: str,
        input_template: str,
        output_template: str,
    ) -> None:
        self.model_names = model_names
        self.task = task
        self.dataset_path = dataset_path
        self.dataset_name = self.dataset_path.split("/")[-1]

        self.output_path = output_path
        self.configs_path = configs_path
        os.makedirs(self.configs_path, exist_ok=True)

        self.instruction_template = instruction_template
        self.input_template = input_template
        self.output_template = output_template

        group_id = str(uuid.uuid4().hex)
        self.run_group = self.dataset_name + "_" + group_id
        self.save_config_group()

        self.trainer_run_configs_path = os.path.join(self.configs_path, "trainer_run_configs")
        self.evaluator_run_configs_path = os.path.join(self.configs_path, "evaluator_run_configs")

    def generate(self) -> List[ConfiguratorOutput]:
        # create sub-folders, if they do not exist
        if not os.path.exists(self.trainer_run_configs_path):
            os.makedirs(self.trainer_run_configs_path)

        if not os.path.exists(self.evaluator_run_configs_path):
            os.makedirs(self.evaluator_run_configs_path)

        trainer_config_outputs = []
        for model_name in self.model_names:
            config_outputs = self._generate_trainer_config_outputs(model_name=model_name)
            trainer_config_outputs.extend(config_outputs)

        evaluator_config_outputs = []
        # build evaluator configs for pre-trained models
        for model_name in self.model_names:
            config_outputs = self._generate_evaluator_config_outputs(model_name=model_name)
            evaluator_config_outputs.extend(config_outputs)

        # build evaluator configs for fine-tuned models
        for config_output in trainer_config_outputs:
            config_outputs = self._generate_evaluator_config_outputs(trainer_config_output=config_output)
            evaluator_config_outputs.extend(config_outputs)

        all_config_outputs = []
        all_config_outputs.extend(trainer_config_outputs)
        all_config_outputs.extend(evaluator_config_outputs)
        return all_config_outputs

    def _generate_trainer_config_outputs(self, model_name: str) -> List[ConfiguratorOutput]:
        config_outputs = []
        model_name_repr = model_name.split("/")[-1]
        # TODO: set dataset_type based on the type of model
        dataset_type = SftDatasetType.PROMPT_COMPLETIONS
        run_name = f"{self.dataset_name}_{model_name_repr}_{dataset_type}"
        model_output_dir = os.path.join(self.output_path, run_name)

        trainer_run_config_paths = self.get_trainer_run_config(
            dataset_type=dataset_type,
            model_name=model_name,
            model_output_dir=model_output_dir,
            run_name=run_name,
        )

        config_outputs.extend(trainer_run_config_paths)

        return config_outputs

    def get_trainer_run_config(self, dataset_type: str, model_name: str, model_output_dir: str, run_name: str) -> List[ConfiguratorOutput]:
        config_outputs = []
        auto_llm_trainer_args = self.build_auto_llm_trainer_args(model_name=model_name)

        # full weights FT
        curr_run_name = f"fft_{run_name}"
        config_output = self._build_trainer_config(
            run_name=curr_run_name,
            model_output_dir=os.path.join(self.output_path, curr_run_name),
            auto_llm_trainer_args=auto_llm_trainer_args,
            dataset_type=dataset_type,
            peft_config=None,
        )
        config_outputs.append(config_output)

        # PEFT fine tuning
        peft_config = self.build_peft_config()
        curr_run_name = f"lora_{run_name}"
        config_output = self._build_trainer_config(
            run_name=curr_run_name,
            model_output_dir=os.path.join(self.output_path, curr_run_name),
            auto_llm_trainer_args=auto_llm_trainer_args,
            dataset_type=dataset_type,
            peft_config=peft_config,
        )
        config_outputs.append(config_output)

        return config_outputs

    def _build_trainer_config(
        self,
        run_name: str,
        model_output_dir: str,
        auto_llm_trainer_args: AutoLlmTrainerArgs,
        dataset_type: str,
        peft_config: LoraConfig = None,
    ) -> ConfiguratorOutput:
        trainer_args = self.build_trainer_args(model_output_dir=model_output_dir)
        trainer_data_builder_config = self.build_trainer_data_builder_config(dataset_type=dataset_type)

        trainer_run_config = self.build_trainer_run_config(
            auto_llm_trainer_args=auto_llm_trainer_args,
            trainer_args=trainer_args,
            trainer_data_builder_config=trainer_data_builder_config,
            peft_config=peft_config,
            run_name=run_name,
        )
        config = trainer_run_config.model_dump(mode="json")
        config_path = self.save_config_yaml(
            config=config,
            configs_path=self.trainer_run_configs_path,
            config_name=f"{run_name}_trainer_run_config.yaml",
        )

        return ConfiguratorOutput(
            run_name=run_name,
            run_id=trainer_run_config.tracker_config.wandb_run_id,
            config_path=config_path,
            mode=ConfigMode.TRAINER_RUN_CFG,
            config=config,
            priority=Priority.PRIORITY_TWO,
        )

    def _generate_evaluator_config_outputs(self, model_name: str = None, trainer_config_output: ConfiguratorOutput = None):
        run_name = None
        # TODO: set dataset_type based on the type of model
        dataset_type = SftDatasetType.PROMPT_COMPLETIONS
        if model_name:
            model_name_repr = model_name.split("/")[-1]
            run_name = f"pre_{self.dataset_name}_{model_name_repr}_{dataset_type}"

        if trainer_config_output:
            model_name = trainer_config_output.config.get("auto_llm_trainer_args").get("model_name")
            run_name = trainer_config_output.run_name

        config_outputs = None
        if trainer_config_output:
            if "lora" in trainer_config_output.run_name:
                model_output_dir = trainer_config_output.config.get("trainer_args").get("output_dir")
                model_name = trainer_config_output.config.get("auto_llm_trainer_args").get("model_name")
                model_args = f"pretrained={model_name},peft={model_output_dir},attn_implementation=flash_attention_2"
                config_outputs = self.get_evaluator_run_config(
                    model_args=model_args,
                    run_name=run_name,
                    priority=Priority.PRIORITY_THREE,
                )
            elif "fft" in trainer_config_output.run_name:
                model_output_dir = trainer_config_output.config.get("trainer_args").get("output_dir")
                model_args = f"pretrained={model_output_dir},attn_implementation=flash_attention_2"
                config_outputs = self.get_evaluator_run_config(
                    model_args=model_args,
                    run_name=run_name,
                    priority=Priority.PRIORITY_THREE,
                )
        else:
            model_args = f"pretrained={model_name},attn_implementation=flash_attention_2"
            config_outputs = self.get_evaluator_run_config(
                model_args=model_args,
                run_name=run_name,
                priority=Priority.PRIORITY_ONE,
            )

        return config_outputs

    def get_evaluator_run_config(self, model_args: str, run_name: str, priority: Priority) -> List[ConfiguratorOutput]:
        config_outputs = []

        unique_run_id = str(uuid.uuid4().hex)

        wandb_project = f"project={WANDB_PROJECT}"
        wandb_run_name = f"name={run_name}"
        wandb_run_id = f"id={unique_run_id}"
        wandb_group = f"group={self.run_group}"
        wandb_job_type = f"job_type=evaluation"
        wandb_args = f"{wandb_project},{wandb_run_name},{wandb_run_id},{wandb_group},{wandb_job_type}"

        # TODO: evaluator can also take different parameters, including few-shots, etc. Handle this.
        task = f"{self.dataset_name}"
        eval_config = {
            "model": "hf",
            "tasks": task,
            "model_args": model_args,
            "wandb_args": wandb_args,
            "write_out": True,
            "log_samples": True,
            "output_path": "/vol/auto_llm/eval_results",
            "include_path": LM_EVAL_HARNESS_CUSTOM_TASKS_PATH,
        }
        config_path = self.save_config_yaml(
            config=eval_config,
            configs_path=self.evaluator_run_configs_path,
            config_name=f"{run_name}_eval_run_config.yaml",
        )

        config_outputs.append(
            ConfiguratorOutput(
                run_name=run_name,
                run_id=unique_run_id,
                config_path=config_path,
                mode=ConfigMode.EVALUATOR_RUN_CFG,
                config=eval_config,
                priority=priority,
            )
        )
        return config_outputs

    def build_auto_llm_trainer_args(self, model_name: str):
        auto_llm_trainer_args = AutoLlmTrainerArgs(model_name=model_name)
        return auto_llm_trainer_args

    def build_trainer_run_config(
        self,
        auto_llm_trainer_args: AutoLlmTrainerArgs,
        trainer_args: TrainerArgs,
        trainer_data_builder_config: TrainerDataBuilderConfig,
        peft_config: LoraConfig = None,
        run_name: str = None,
    ):
        unique_run_id = str(uuid.uuid4().hex)
        tracker_config = TrackerConfig(
            wandb_project=WANDB_PROJECT,
            wandb_run_name=run_name,
            wandb_run_id=unique_run_id,
            wandb_run_group=self.run_group,
        )

        trainer_run_config = TrainerRunConfig(
            auto_llm_trainer_args=auto_llm_trainer_args,
            trainer_args=trainer_args,
            trainer_data_builder_config=trainer_data_builder_config,
            peft_config=peft_config,
            tracker_config=tracker_config,
        )
        return trainer_run_config

    def build_trainer_args(self, model_output_dir: str):
        trainer_args = TrainerArgs(output_dir=model_output_dir)
        return trainer_args

    def build_trainer_data_builder_config(self, dataset_type: str):
        # TODO: set parse_output_as_json based on the type of task - structured output / otherwise
        trainer_data_builder_config = TrainerDataBuilderConfig(
            dataset_dir=self.dataset_path,
            dataset_type=dataset_type,
            instruction_template=self.instruction_template,
            input_template=self.input_template,
            output_template=self.output_template,
            instruction_input_separator="\n",
            # limit=100,  # for debug
        )
        return trainer_data_builder_config

    def build_peft_config(self):
        peft_config = LoraConfig()
        return peft_config

    def save_config_yaml(self, config: Dict[str, Any], configs_path: str, config_name: str) -> str:
        config_path = os.path.join(configs_path, config_name)
        with open(config_path, "w+") as f:
            yaml.dump(config, f)
        print(f"Saved configuration: {config_path}")
        return config_path

    def save_config_group(self):
        run_group_dict = {"run_group": self.run_group}
        run_group_path = os.path.join(self.configs_path, "run_group.json")
        with open(run_group_path, "w+") as f:
            json.dump(run_group_dict, f)
        print(f"Saved Run Group Dict: {run_group_path}")
