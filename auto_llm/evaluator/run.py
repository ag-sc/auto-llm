import argparse
import logging
import shutil
import os

import yaml
from lm_eval.__main__ import cli_evaluate

from auto_llm.evaluator.utils import parse_lm_eval_config, evaluate_and_capture, aggregate_eval_scores

from auto_llm.profiler.energy_profiler import EnergyProfiler
from auto_llm.profiler.wandb_energy_logger import WandbEnergyLogger
from auto_llm.profiler.utils import parse_wandb_args
from auto_llm.estimator.estimation_pipeline import EstimationPipeline
from auto_llm.estimator.emission_comparator import EmissionComparator
import datasets.config


# to get STDOUT in wandb. See: https://github.com/wandb/wandb/issues/2182#issuecomment-1447879531
shutil._USE_CP_SENDFILE = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")
    args = parser.parse_args()
    if not args.config_path:
        raise Exception("config path should be provided!")

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Pop energy_profiling and profiler options before forwarding config
    # to lm-eval-harness (it doesn't recognise these keys).
    energy_profiling = config.pop("energy_profiling", False)
    gpu_name = config.pop("gpu_name", None)
    tracking_mode = config.pop("tracking_mode", "machine")
    force_cpu_power = config.pop("force_cpu_power", None)
    force_ram_power = config.pop("force_ram_power", None)
    auto_pareto_cfg = config.pop("auto_pareto", None) or {}
    
    # Allow custom dataset code (e.g. bigbio/pubmed_qa) before any task loading
    if config.get("trust_remote_code"):
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
        try:
            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        except (ImportError, AttributeError):
            pass

    # Read (but don't pop) wandb_args — lm-eval-harness still needs them.
    wandb_args = parse_wandb_args(config.get("wandb_args"))

    lm_eval_args = parse_lm_eval_config(config)

    # start LM eval harness
    if energy_profiling:
        output_dir = config.get("output_path", ".")
        wandb_project = wandb_args.get("project", "auto-llm")
        wandb_name = wandb_args.get("name", "eval")
        log_to_wandb = bool(wandb_args)

        # Pre-run: persist energy estimate (best-effort)
        pipeline = EstimationPipeline(
            output_dir=output_dir,
            gpu_name=gpu_name,
            is_eval=True,
            config_path=args.config_path,
        )
        estimate = pipeline.run()

        with EnergyProfiler(
            output_dir=output_dir,
            project_name=wandb_project,
            experiment_name=wandb_name,
            tracking_mode=tracking_mode,
            force_cpu_power=force_cpu_power,
            force_ram_power=force_ram_power,
        ) as profiler:
            eval_results = evaluate_and_capture(lm_eval_args)

        # Post-run: compare estimated vs actual (best-effort)
        comparator = EmissionComparator(
            estimated_emissions=estimate or {},
            actual_emissions=profiler.final_emissions_data,
        )
        comparison = comparator.compare()
        if comparison:
            EmissionComparator.save_comparison(comparison, output_dir)

        # Log all energy metrics to a single wandb run
        if log_to_wandb:
            wandb_logger = WandbEnergyLogger(
                project=wandb_project,
                name=wandb_name,
            )
            wandb_logger.log(pipeline.get_wandb_metrics())
            wandb_logger.log(profiler.get_wandb_metrics())
            wandb_logger.log(comparator.get_wandb_metrics())
            if eval_results:
                wandb_logger.log(aggregate_eval_scores(eval_results))
            wandb_logger.flush()

            # Optional in-process refresh of the project Pareto workspace.
            # Opt-in via ``auto_pareto.enabled: true`` in the eval YAML.
            # Best-effort: any failure is logged and the eval job still
            # exits 0. See README.md §"Auto-refresh option".
            if auto_pareto_cfg.get("enabled"):
                try:
                    from auto_llm.evaluator.plots.wandb_pareto_plot import (
                        refresh_pareto_workspace,
                    )

                    pareto_entity = (
                        auto_pareto_cfg.get("entity")
                        or wandb_args.get("entity")
                        or os.environ.get("WANDB_ENTITY")
                    )
                    if not pareto_entity:
                        raise RuntimeError(
                            "auto_pareto.entity not set and WANDB_ENTITY env var "
                            "is empty; cannot refresh Pareto workspace."
                        )

                    url = refresh_pareto_workspace(
                        entity=pareto_entity,
                        project=auto_pareto_cfg.get("project") or wandb_project,
                        energy_key=auto_pareto_cfg.get(
                            "energy_key", "emissions/energy_consumed_kWh"
                        ),
                        score_scale=float(auto_pareto_cfg.get("score_scale", 100.0)),
                        tag=auto_pareto_cfg.get("tag", "energy-profiling") or None,
                        workspace_name=auto_pareto_cfg.get(
                            "workspace_name", "Pareto Frontier"
                        ),
                        dry_run=bool(auto_pareto_cfg.get("dry_run", False)),
                        skip_backfill=bool(auto_pareto_cfg.get("skip_backfill", False)),
                        skip_panel=bool(auto_pareto_cfg.get("skip_panel", False)),
                        skip_preset=bool(auto_pareto_cfg.get("skip_preset", False)),
                    )
                    if url:
                        logging.getLogger(__name__).info(
                            "Pareto workspace refreshed: %s", url
                        )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "auto_pareto refresh failed (best-effort, eval job will "
                        "succeed): %s",
                        exc,
                    )

            # NOTE: When ``auto_pareto.enabled`` is left at its default
            # (false), the ``pareto/<label>/*`` fields are NOT written here.
            # Pareto optimality is a cross-run property — a run only knows
            # whether it is on the frontier relative to every other run in
            # the wandb project. After a sweep / batch of eval jobs has
            # finished, refresh the frontier flags and workspace panels by
            # running the standalone backfill script on the cluster
            # login/head node (no GPU needed, wandb API only):
            #
            #     source $VENV_PATH/bin/activate
            #     source $ENV_VARIABLES_PATH   # exports WANDB_API_KEY
            #     python scripts/wandb_pareto_plot.py \
            #         --entity <wandb-entity> --project <wandb-project>
            #
            # See the "Refresh Pareto frontier panels" section in README.md
            # for details and options (``--dry-run``, ``--skip-panel``, ...).
    else:
        cli_evaluate(args=lm_eval_args)
