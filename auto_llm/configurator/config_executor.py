import subprocess
from pathlib import Path
from typing import List

from auto_llm.configurator.config_generator import (
    ConfiguratorOutput,
    Priority,
    ConfigMode,
)
from auto_llm.registry.configurator_registry import SLURM_EVALUATOR_RUN_SCRIPT, NATIVE_EVALUATOR_RUN_SCRIPT, SLURM_TRAINER_RUN_SCRIPT, NATIVE_TRAINER_RUN_SCRIPT

# TODO: Implement python-based config executor


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class ConfigExecutor:
    """
    Slurm based Config Executor adding dependencies based on priority.
    Use this when there is no job already reserved with resources.
    """

    def __init__(self, configurator_outputs: List[ConfiguratorOutput]):
        self.configurator_outputs = configurator_outputs

    def execute(self):
        # sort config outputs based on priority
        # separate evaluator and trainer cfgs
        # run pre-trained eval configs -> prio 1
        # run trainer cfgs -> prio 2
        # how to decide ddp or fsdp?
        # run evaluator configs for ft models -> prio 3
        configs_path = Path(self.configurator_outputs[0].config_path).parent.parent.absolute()

        configurator_outputs_dict = [output.model_dump(mode="json") for output in self.configurator_outputs]
        configurator_outputs_dict.sort(key=self.func)

        self.configurator_outputs = [ConfiguratorOutput.model_validate(configurator_output_dict) for configurator_output_dict in configurator_outputs_dict]

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
                    cmd_w_prefix = f"{job_id_var}=$(sbatch --parsable {dependencies} {cmd})"
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
        parallelism = "ddp"
        # TODO: how to decide upon which parallelism to use
        cmd = f"{SLURM_TRAINER_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path} {parallelism}"
        return cmd

    @staticmethod
    def _get_evaluator_run_suffix(cfg_output: ConfiguratorOutput) -> str:
        venv_path = "venv"
        env_path = "../env.sh"
        cmd = f"{SLURM_EVALUATOR_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path}"
        return cmd

    @staticmethod
    def start_runs(script_path: str):
        info = subprocess.check_output(["bash", script_path])
        info = info.decode("utf-8")
        print(info)


class SequentialConfigExecutor:
    """
    Slurm based Config Executor without adding dependencies based on priority.
    Use this when there is already a job reserved with resources.
    """

    def __init__(self, configurator_outputs: List[ConfiguratorOutput], job_id_to_attach: int):
        self.configurator_outputs = configurator_outputs
        self.job_id_to_attach = job_id_to_attach

    def execute(self):
        # sort config outputs based on priority
        # separate evaluator and trainer cfgs
        # run pre-trained eval configs -> prio 1
        # run trainer cfgs -> prio 2
        # how to decide ddp or fsdp?
        # run evaluator configs for ft models -> prio 3
        configs_path = Path(self.configurator_outputs[0].config_path).parent.parent.absolute()

        configurator_outputs_dict = [output.model_dump(mode="json") for output in self.configurator_outputs]
        configurator_outputs_dict.sort(key=self.func)

        self.configurator_outputs = [ConfiguratorOutput.model_validate(configurator_output_dict) for configurator_output_dict in configurator_outputs_dict]

        cmds = {key: [] for key in Priority.__members__}
        for cfg_output in self.configurator_outputs:
            key = cfg_output.priority.name
            if cfg_output.mode == ConfigMode.TRAINER_RUN_CFG:
                cmds[key].append((cfg_output, self._get_trainer_run_suffix(cfg_output)))
            elif cfg_output.mode == ConfigMode.EVALUATOR_RUN_CFG:
                cmds[key].append((cfg_output, self._get_evaluator_run_suffix(cfg_output)))

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

            for idx, (cfg_output, cmd) in enumerate(cmds_list):
                all_cmds.append(
                    f'echo "[$(date \'+%F %T\')] Starting priority={Priority[prio_str].value} run_name={cfg_output.run_name} '
                    f'run_id={cfg_output.run_id} config={cfg_output.config_path}"'
                )
                cmd_w_prefix = f"srun --jobid $JOB_ID bash {cmd}"

                all_cmds.append(cmd_w_prefix)
            all_job_ids.append(job_ids_prio)

        all_cmds.append(f'\necho "[$(date \'+%F %T\')] All {len(self.configurator_outputs)} runs finished"')

        print(all_cmds)
        run_script_path = f"{configs_path}/run.sh"
        with open(run_script_path, "w") as f:
            # set -e: stop at the first failing run instead of continuing with the next priority
            text = f"""
#!/bin/bash
set -eo pipefail
ROOT_DIR={ROOT_DIR}
cd $ROOT_DIR
echo $PWD
trap 'echo "[$(date "+%F %T")] FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

JOB_ID={self.job_id_to_attach}
            """
            f.write(text.strip())
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
        parallelism = ""  #  "ddp"
        # TODO: how to decide upon which parallelism to use
        cmd = f"{SLURM_TRAINER_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path} {parallelism}"
        return cmd

    @staticmethod
    def _get_evaluator_run_suffix(cfg_output: ConfiguratorOutput) -> str:
        venv_path = "venv"
        env_path = "../env.sh"
        cmd = f"{SLURM_EVALUATOR_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path}"
        return cmd

    @staticmethod
    def start_runs(script_path: str):
        # Unlike the task-spooler executor, this run.sh runs the jobs in the foreground (srun) for hours,
        # so start it detached in its own session (no shared tmux session that a concurrent submit could kill).
        log_path = Path(script_path).with_name("run.log")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                ["bash", script_path], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            )

        # Catch immediate failures (e.g. invalid JOB_ID) so they reach the caller
        try:
            returncode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Pipeline started (pid {proc.pid}), log: {log_path}")
            return
        if returncode != 0:
            raise RuntimeError(f"Pipeline failed to start (exit {returncode}), see {log_path}")
        print(f"Pipeline finished, log: {log_path}")


class TaskSpoolerSequentialConfigExecutor:
    """
    Task-Spooler based Config Executor without adding dependencies based on priority.
    """

    def __init__(self, configurator_outputs: List[ConfiguratorOutput]):
        self.configurator_outputs = configurator_outputs

    def execute(self):
        # sort config outputs based on priority
        # separate evaluator and trainer cfgs
        # run pre-trained eval configs -> prio 1
        # run trainer cfgs -> prio 2
        # how to decide ddp or fsdp?
        # run evaluator configs for ft models -> prio 3
        configs_path = Path(self.configurator_outputs[0].config_path).parent.parent.absolute()

        configurator_outputs_dict = [output.model_dump(mode="json") for output in self.configurator_outputs]
        configurator_outputs_dict.sort(key=self.func)

        self.configurator_outputs = [ConfiguratorOutput.model_validate(configurator_output_dict) for configurator_output_dict in configurator_outputs_dict]

        cmds = {key: [] for key in Priority.__members__}
        for cfg_output in self.configurator_outputs:
            key = cfg_output.priority.name
            if cfg_output.mode == ConfigMode.TRAINER_RUN_CFG:
                cmds[key].append((cfg_output, self._get_trainer_run_suffix(cfg_output)))
            elif cfg_output.mode == ConfigMode.EVALUATOR_RUN_CFG:
                cmds[key].append((cfg_output, self._get_evaluator_run_suffix(cfg_output)))

        all_cmds = []
        all_job_ids = []
        for prio_str, cmds_list in cmds.items():
            all_cmds.append(f"\n\n# Priority {Priority[prio_str].value} runs below")

            job_ids_prio = []
            if prio_str == Priority.PRIORITY_ONE.name:
                dependencies = None
            else:
                job_ids = f",$".join(all_job_ids[-1])
                if prio_str == Priority.PRIORITY_TWO.name:
                    dependencies = f"-D ${job_ids}"  # the job will be run after the job of given IDs ends
                elif prio_str == Priority.PRIORITY_THREE.name:
                    dependencies = f"-W ${job_ids}"  # the job will be run after the job of given IDs ends well (exit code 0)
                else:
                    raise Exception(f"Unknown priority type {prio_str}")

            for idx, (cfg_output, cmd) in enumerate(cmds_list):
                job_id_var = f"p{Priority[prio_str].value}j{idx+1}"
                job_ids_prio.append(job_id_var)

                if prio_str == Priority.PRIORITY_THREE.name:
                    # add a short delay before evaluating trained models. This helps the model weights
                    # to be saved before the evaluation starts. The sleep must be part of the queued job:
                    # run.sh only enqueues, so a sleep there would not delay the job itself.
                    cmd = f'-c "sleep 60s && bash {cmd}"'

                # TODO: how to set the num of GPUs? Now setting 1 always.
                if dependencies:
                    cmd_w_prefix = f"{job_id_var}=$(ts -G 1 {dependencies} bash {cmd})"
                else:
                    cmd_w_prefix = f"{job_id_var}=$(ts -G 1 bash {cmd})"

                all_cmds.append(cmd_w_prefix)
                # One key=value line per job in run.log, mapping the ts job id to its run
                depends_on = f"${job_ids}" if dependencies else "-"
                all_cmds.append(
                    f'echo "ts_job=${job_id_var} priority={Priority[prio_str].value} run_name={cfg_output.run_name} '
                    f'run_id={cfg_output.run_id} depends_on={depends_on} config={cfg_output.config_path}"'
                )
            all_job_ids.append(job_ids_prio)

        all_cmds.append(f'\necho "[$(date \'+%F %T\')] All {len(self.configurator_outputs)} jobs queued"')

        print(all_cmds)
        run_script_path = f"{configs_path}/run.sh"
        with open(run_script_path, "w") as f:
            # set -e: stop queuing if any ts call fails, so later jobs don't get broken dependencies
            text = f"""
#!/bin/bash
set -eo pipefail
ROOT_DIR={ROOT_DIR}
cd $ROOT_DIR
echo $PWD
trap 'echo "[$(date "+%F %T")] FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR
echo "[$(date '+%F %T')] Queuing {len(self.configurator_outputs)} jobs on task-spooler (TS_SOCKET=${{TS_SOCKET:-default}})"
"""
            f.write(text.strip())
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
        parallelism = ""  #  "ddp"
        # TODO: how to decide upon which parallelism to use
        cmd = f"{NATIVE_TRAINER_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path} {parallelism}"
        return cmd

    @staticmethod
    def _get_evaluator_run_suffix(cfg_output: ConfiguratorOutput) -> str:
        venv_path = "venv"
        env_path = "../env.sh"
        cmd = f"{NATIVE_EVALUATOR_RUN_SCRIPT} {cfg_output.config_path} {venv_path} {env_path}"
        return cmd

    @staticmethod
    def start_runs(script_path: str):
        # run.sh only enqueues jobs in task-spooler and exits quickly; the ts daemon runs them.
        # Running it directly (no shared tmux session) avoids concurrent submissions killing each other
        # and lets failures propagate to the caller.
        log_path = Path(script_path).with_name("run.log")
        with open(log_path, "w") as log:
            result = subprocess.run(["bash", script_path], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Submitting jobs failed (exit {result.returncode}), see {log_path}")
        print(f"Jobs queued in task-spooler, log: {log_path}")
