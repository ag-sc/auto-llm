import enum
import os
from typing import List, Dict, Any

import yaml
from pydantic import BaseModel

from auto_llm.dto.builder_config import TrainerDataBuilderConfig, SftDatasetType
from auto_llm.dto.trainer_run_config import (
    TrainerArgs,
    AutoLlmTrainerArgs,
    LoraConfig,
    TrainerRunConfig,
)


class Priority(enum.Enum):
    PRIORITY_ONE = 1
    PRIORITY_TWO = 2
    PRIORITY_THREE = 3


class ConfigMode(enum.Enum):
    TRAINER_RUN_CFG = "trainer_run_cfg"
    EVALUATOR_RUN_CFG = "evaluator_run_cfg"


class ConfiguratorOutput(BaseModel):
    run_name: str
    config_path: str
    mode: ConfigMode
    config: Dict[str, Any]
    priority: Priority


MODEL_NAMES = ["google/gemma-2-2b", "google/gemma-2-2b-it"]
TASKS = ["pico"]


class TrainEvalRunConfigurator:
    def __init__(
        self,
        model_names: List[str],
        task: str,
        dataset_name: str,
        dataset_dir: str,
        output_path: str,
        configs_path: str,
        instruction_template: str,
        input_template: str,
        output_template: str,
    ) -> None:
        self.model_names = model_names
        self.task = task
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.output_path = output_path
        self.configs_path = configs_path
        self.instruction_template = instruction_template
        self.input_template = input_template
        self.output_template = output_template

        self.trainer_run_configs_path = os.path.join(
            self.configs_path, "trainer_run_configs"
        )
        self.evaluator_run_configs_path = os.path.join(
            self.configs_path, "evaluator_run_configs"
        )

    def generate(self):
        # create sub-folders, if they do not exist
        if not os.path.exists(self.trainer_run_configs_path):
            os.makedirs(self.trainer_run_configs_path)

        if not os.path.exists(self.evaluator_run_configs_path):
            os.makedirs(self.evaluator_run_configs_path)

        trainer_config_outputs = []
        for model_name in self.model_names:
            config_outputs = self._generate_trainer_config_outputs(
                model_name=model_name
            )
            trainer_config_outputs.extend(config_outputs)

        evaluator_config_outputs = []
        # build evaluator configs for pre-trained models
        for model_name in self.model_names:
            config_outputs = self._generate_evaluator_config_outputs(
                model_name=model_name
            )
            evaluator_config_outputs.extend(config_outputs)

        # build evaluator configs for fine-tuned models
        for config_output in trainer_config_outputs:
            config_outputs = self._generate_evaluator_config_outputs(
                trainer_config_output=config_output
            )
            evaluator_config_outputs.extend(config_outputs)

        all_config_outputs = []
        all_config_outputs.extend(trainer_config_outputs)
        all_config_outputs.extend(evaluator_config_outputs)
        return all_config_outputs

    def _generate_trainer_config_outputs(
        self, model_name: str
    ) -> List[ConfiguratorOutput]:
        config_outputs = []
        model_name_repr = model_name.split("/")[-1]
        # TODO: set dataset_type based on the type of model
        dataset_type = SftDatasetType.PROMPT_COMPLETIONS
        run_name = f"{self.task}_{self.dataset_name}_{model_name_repr}_{dataset_type}"
        model_output_dir = os.path.join(self.output_path, run_name)

        trainer_run_config_paths = self.get_trainer_run_config(
            dataset_type=dataset_type,
            model_name=model_name,
            model_output_dir=model_output_dir,
            run_name=run_name,
        )

        config_outputs.extend(trainer_run_config_paths)

        return config_outputs

    def get_trainer_run_config(
        self, dataset_type: str, model_name: str, model_output_dir: str, run_name: str
    ) -> List[ConfiguratorOutput]:
        config_outputs = []
        auto_llm_trainer_args = self.build_auto_llm_trainer_args(model_name=model_name)

        # full weights FT
        config_output = self._build_trainer_config(
            run_name=f"fft-{run_name}",
            model_output_dir=model_output_dir,
            auto_llm_trainer_args=auto_llm_trainer_args,
            dataset_type=dataset_type,
            peft_config=None,
        )
        config_outputs.append(config_output)

        # PEFT fine tuning
        peft_config = self.build_peft_config()
        config_output = self._build_trainer_config(
            run_name=f"lora-{run_name}",
            model_output_dir=model_output_dir,
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
        trainer_args = self.build_trainer_args(
            model_output_dir=model_output_dir, run_name=run_name
        )
        trainer_data_builder_config = self.build_trainer_data_builder_config(
            dataset_type=dataset_type
        )

        trainer_run_config = self.build_trainer_run_config(
            auto_llm_trainer_args=auto_llm_trainer_args,
            trainer_args=trainer_args,
            trainer_data_builder_config=trainer_data_builder_config,
            peft_config=peft_config,
        )
        config = trainer_run_config.model_dump(mode="json")
        config_path = self.save_config_yaml(
            config=config,
            configs_path=self.trainer_run_configs_path,
            config_name=f"{run_name}_trainer_run_config.yaml",
        )

        return ConfiguratorOutput(
            run_name=run_name,
            config_path=config_path,
            mode=ConfigMode.TRAINER_RUN_CFG,
            config=config,
            priority=Priority.PRIORITY_TWO,
        )

    def _generate_evaluator_config_outputs(
        self, model_name: str = None, trainer_config_output: ConfiguratorOutput = None
    ):
        run_name = None
        # TODO: set dataset_type based on the type of model
        dataset_type = SftDatasetType.PROMPT_COMPLETIONS
        if model_name:
            model_name_repr = model_name.split("/")[-1]
            run_name = (
                f"pre_{self.task}_{self.dataset_name}_{model_name_repr}_{dataset_type}"
            )

        if trainer_config_output:
            model_name = trainer_config_output.config.get("auto_llm_trainer_args").get(
                "model_name"
            )
            run_name = trainer_config_output.run_name

        config_outputs = None
        if trainer_config_output:
            if "lora" in trainer_config_output.run_name:
                model_output_dir = trainer_config_output.config.get("trainer_args").get(
                    "output_dir"
                )
                model_name = trainer_config_output.config.get(
                    "auto_llm_trainer_args"
                ).get("model_name")
                model_args = f"pretrained={model_name},peft={model_output_dir}"
                config_outputs = self.get_evaluator_run_config(
                    model_args=model_args,
                    run_name=run_name,
                    priority=Priority.PRIORITY_THREE,
                )
            elif "fft" in trainer_config_output.run_name:
                model_output_dir = trainer_config_output.config.get("trainer_args").get(
                    "output_dir"
                )
                model_args = f"pretrained={model_output_dir}"
                config_outputs = self.get_evaluator_run_config(
                    model_args=model_args,
                    run_name=run_name,
                    priority=Priority.PRIORITY_THREE,
                )
        else:
            model_args = f"pretrained={model_name}"
            config_outputs = self.get_evaluator_run_config(
                model_args=model_args,
                run_name=run_name,
                priority=Priority.PRIORITY_ONE,
            )

        return config_outputs

    def get_evaluator_run_config(
        self, model_args: str, run_name: str, priority: Priority
    ) -> List[ConfiguratorOutput]:
        config_outputs = []

        # TODO: evaluator can also take different parameters, including few-shots, etc. Handle this.
        task = f"{self.dataset_name}_{self.task}"
        eval_config = {
            "model": "hf",
            "tasks": task,
            "model_args": model_args,
            "wandb_args": f"project=llm4kmu-eval,name={run_name}",
            "write_out": True,
            "log_samples": True,
            "output_path": "/vol/auto_llm/eval_results",
            "include_path": "config_files/evaluator_configs/tasks",
        }
        config_path = self.save_config_yaml(
            config=eval_config,
            configs_path=self.evaluator_run_configs_path,
            config_name=f"{run_name}_eval_run_config.yaml",
        )

        config_outputs.append(
            ConfiguratorOutput(
                run_name=run_name,
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
    ):
        trainer_run_config = TrainerRunConfig(
            auto_llm_trainer_args=auto_llm_trainer_args,
            trainer_args=trainer_args,
            trainer_data_builder_config=trainer_data_builder_config,
            peft_config=peft_config,
        )
        return trainer_run_config

    def build_trainer_args(self, model_output_dir: str, run_name: str):
        trainer_args = TrainerArgs(
            run_name=run_name,
            output_dir=model_output_dir,
        )
        return trainer_args

    def build_trainer_data_builder_config(self, dataset_type: str):
        # TODO: set parse_output_as_json based on the type of task - structured output / otherwise
        trainer_data_builder_config = TrainerDataBuilderConfig(
            dataset_dir=self.dataset_dir,
            dataset_type=dataset_type,
            instruction_template=self.instruction_template,
            input_template=self.input_template,
            output_template=self.output_template,
            parse_output_as_json=True,
        )
        return trainer_data_builder_config

    def build_peft_config(self):
        peft_config = LoraConfig()
        return peft_config

    def save_config_yaml(
        self, config: Dict[str, Any], configs_path: str, config_name: str
    ) -> str:
        config_path = os.path.join(configs_path, config_name)
        with open(config_path, "w+") as f:
            yaml.dump(config, f)
        print(f"Saved configuration: {config_path}")
        return config_path
