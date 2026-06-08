import os
import json
import logging

from lm_eval.config.evaluate_config import EvaluatorConfig


def run_lm_eval_harness(config_path: str):
    """
    This is taken as is from lm_eval/_cli/run.py (version: 0.4.12). We pass the `EvaluatorConfig` object directly to this function,
    so we skip the CLI argument parsing and validation steps.
    """
    cfg = EvaluatorConfig.from_config(config_path)

    # Create and validate config (most validation now occurs in EvaluationConfig)
    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker, TrackioLogger, WandbLogger
    from lm_eval.utils import handle_non_serializable, make_table

    # Set up logging
    if cfg.wandb_args:
        wandb_logger = WandbLogger(cfg.wandb_args, cfg.wandb_config_args)
        wandb_logger.run.log_artifact(config_path, name="config")
    if cfg.trackio_args:
        trackio_logger = TrackioLogger(cfg.trackio_args)

    # Set up evaluation tracker
    if cfg.output_path:
        cfg.hf_hub_log_args["output_path"] = cfg.output_path

    if os.environ.get("HF_TOKEN", None):
        cfg.hf_hub_log_args["token"] = os.environ.get("HF_TOKEN")

    evaluation_tracker = EvaluationTracker(**cfg.hf_hub_log_args)

    # Create task manager (metadata already set up in config validation)
    task_manager = cfg.process_tasks(cfg.metadata)

    eval_logger = logging.getLogger(__name__)

    # Validation warnings (keep these in CLI as they're logging-specific)
    if "push_samples_to_hub" in cfg.hf_hub_log_args and not cfg.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set.")

    # Log task selection (tasks already processed in config)
    if cfg.include_path is not None:
        eval_logger.info("Including path: %s", cfg.include_path)
    eval_logger.info("Selected Tasks: %s", cfg.tasks)

    # Run evaluation
    results = simple_evaluate(
        model=cfg.model,
        model_args=cfg.model_args,
        tasks=cfg.tasks,
        num_fewshot=cfg.num_fewshot,
        batch_size=cfg.batch_size,
        max_batch_size=cfg.max_batch_size,
        device=cfg.device,
        use_cache=cfg.use_cache,
        cache_requests=cfg.cache_requests.get("cache_requests", False),
        rewrite_requests_cache=cfg.cache_requests.get("rewrite_requests_cache", False),
        delete_requests_cache=cfg.cache_requests.get("delete_requests_cache", False),
        limit=cfg.limit,
        samples=cfg.samples,
        check_integrity=cfg.check_integrity,
        write_out=cfg.write_out,
        log_samples=cfg.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=cfg.system_instruction,
        apply_chat_template=cfg.apply_chat_template,
        fewshot_as_multiturn=cfg.fewshot_as_multiturn,
        gen_kwargs=cfg.gen_kwargs,
        task_manager=task_manager,
        verbosity=cfg.verbosity,
        predict_only=cfg.predict_only,
        random_seed=cfg.seed[0] if cfg.seed else None,
        numpy_random_seed=cfg.seed[1] if cfg.seed else None,
        torch_random_seed=cfg.seed[2] if cfg.seed else None,
        fewshot_random_seed=cfg.seed[3] if cfg.seed else None,
        confirm_run_unsafe_code=cfg.confirm_run_unsafe_code,
        metadata=cfg.metadata,
    )

    # Process results
    if results is not None:
        if cfg.log_samples:
            samples = results.pop("samples")

        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        if cfg.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # W&B logging
        if cfg.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if cfg.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:  # noqa: BLE001
                eval_logger.info("Logging to W&B failed: %s", e)

        # Trackio logging
        if cfg.trackio_args:
            try:
                trackio_logger.post_init(results)
                trackio_logger.log_eval_result()
                if cfg.log_samples:
                    trackio_logger.log_eval_samples(samples)
            except Exception as e:  # noqa: BLE001
                eval_logger.info("Logging to Trackio failed: %s", e)

        # Save results
        evaluation_tracker.save_results_aggregated(results=results, samples=samples if cfg.log_samples else None)

        if cfg.log_samples:
            for task_name in results["configs"]:
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        # Print results
        cfg.model_args.pop("trust_remote_code", None)
        print(
            f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), "
            f"limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
            f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if cfg.wandb_args:
            wandb_logger.run.finish()

        if cfg.trackio_args:
            trackio_logger.finish()
