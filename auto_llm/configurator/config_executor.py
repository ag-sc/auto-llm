import subprocess
from pathlib import Path
from typing import List

from auto_llm.configurator.config_generator import (
    ConfiguratorOutput,
    Priority,
    ConfigMode,
)
from auto_llm.registry.configurator_registry import (
    EVALUATOR_RUN_SCRIPT,
    TRAINER_RUN_SCRIPT,
)


class ConfigExecutor:
    def __init__(self, configurator_outputs: List[ConfiguratorOutput]):
        self.configurator_outputs = configurator_outputs

    def execute(self):
        # sort config outputs based on priority
        # separate evaluator and trainer cfgs
        # run pre-trained eval configs -> prio 1
        # run trainer cfgs -> prio 2
        # how to decide ddp or fsdp?
        # run evaluator configs for ft models -> prio 3
        configs_path = Path(
            self.configurator_outputs[0].config_path
        ).parent.parent.absolute()

        configurator_outputs_dict = [
            output.model_dump(mode="json") for output in self.configurator_outputs
        ]
        configurator_outputs_dict.sort(key=self.func)

        self.configurator_outputs = [
            ConfiguratorOutput.model_validate(configurator_output_dict)
            for configurator_output_dict in configurator_outputs_dict
        ]

        cmds = {key: [] for key in Priority.__members__}
        for cfg_output in self.configurator_outputs:
            key = cfg_output.priority.name
            if cfg_output.mode == ConfigMode.TRAINER_RUN_CFG:
                cmds[key].append(self._get_trainer_run_suffix(cfg_output))
            elif cfg_output.mode == ConfigMode.EVALUATOR_RUN_CFG:
                cmds[key].append(self._get_evaluator_run_suffix(cfg_output))

        all_cmds = []
        all_job_ids = []
        for prio_str, cmds_list in cmds.items():
            all_cmds.append(f"\n\n# Priority {Priority[prio_str].value} runs below")

            job_ids_prio = []
            if prio_str == Priority.PRIORITY_ONE.name:
                dependencies = None
            else:
                job_ids = f":$".join(all_job_ids[-1])
                if prio_str == Priority.PRIORITY_TWO.name:
                    dependencies = f"--dependency=afterany:${job_ids}"
                elif prio_str == Priority.PRIORITY_THREE.name:
                    dependencies = f"--dependency=afterok:${job_ids}"
                else:
                    raise Exception(f"Unknown priority type {prio_str}")

            for idx, cmd in enumerate(cmds_list):
                job_id_var = f"p{Priority[prio_str].value}j{idx+1}"
                job_ids_prio.append(job_id_var)

                if dependencies:
                    cmd_w_prefix = (
                        f"{job_id_var}=$(sbatch --parsable {dependencies} {cmd})"
                    )
                else:
                    cmd_w_prefix = f"{job_id_var}=$(sbatch --parsable {cmd})"

                all_cmds.append(cmd_w_prefix)
            all_job_ids.append(job_ids_prio)

        print(all_cmds)
        run_script_path = f"{configs_path}/run.sh"
        with open(run_script_path, "w") as f:
            f.write("\n".join(all_cmds))

        print(f"Run scripts added here: {run_script_path}")

        self.start_runs(script_path=run_script_path)

    @staticmethod
    def func(e):
        return e["priority"]

    @staticmethod
    def _get_trainer_run_suffix(cfg_output: ConfiguratorOutput):
        venv_path = "venv"
        env_path = "../env.sh"
        parallelism = ...
        # TODO: how to decide upon which parallelism to use
        cmd = f"{TRAINER_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path} {parallelism}"
        return cmd

    @staticmethod
    def _get_evaluator_run_suffix(cfg_output: ConfiguratorOutput) -> str:
        venv_path = "venv"
        env_path = "../env.sh"
        cmd = f"{EVALUATOR_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path}"
        return cmd

    @staticmethod
    def start_runs(script_path: str):
        info = subprocess.check_output(["bash", script_path])
        info = info.decode("utf-8")
        print(info)


if __name__ == "__main__":
    configurator_outputs = [
        ConfiguratorOutput(
            run_name="sft",
            config_path="configs/evaluator_run_configs/fft-pico_eval_run_config.yaml",
            mode=ConfigMode.TRAINER_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_TWO,
        ),
        ConfiguratorOutput(
            run_name="baseline_pre_trained_eval",
            config_path="configs/evaluator_run_configs/pre-pico_eval_run_config.yaml",
            mode=ConfigMode.EVALUATOR_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_ONE,
        ),
        ConfiguratorOutput(
            run_name="sft",
            config_path="configs/trainer_run_configs/fft-pico_trainer_run_config.yaml",
            mode=ConfigMode.TRAINER_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_TWO,
        ),
        ConfiguratorOutput(
            run_name="sft_model_Eval",
            config_path="configs/evaluator_run_configs/fft-pico_eval_run_config.yaml",
            mode=ConfigMode.EVALUATOR_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_THREE,
        ),
        ConfiguratorOutput(
            run_name="sft",
            config_path="configs/trainer_run_configs/fft-pico_trainer_run_config.yaml",
            mode=ConfigMode.TRAINER_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_TWO,
        ),
        ConfiguratorOutput(
            run_name="sft",
            config_path="configs/trainer_run_configs/fft-pico_trainer_run_config.yaml",
            mode=ConfigMode.TRAINER_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_TWO,
        ),
        ConfiguratorOutput(
            run_name="baseline_pre_trained_eval",
            config_path="configs/evaluator_run_configs/pre-pico_eval_run_config.yaml",
            mode=ConfigMode.EVALUATOR_RUN_CFG,
            config={},
            priority=Priority.PRIORITY_ONE,
        ),
    ]

    executor = ConfigExecutor(configurator_outputs=configurator_outputs)
    executor.execute()
