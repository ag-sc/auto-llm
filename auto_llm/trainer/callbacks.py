from transformers import TrainerCallback


class KeepBestAdapterCallback(TrainerCallback):
    """
    Saves the model (LoRA adapter or full model) to ``output_dir`` *only* when the
    monitored validation metric improves, overwriting the previous save. This keeps
    **at most one** checkpoint on disk at any time during training, and that single
    copy is always the best one seen so far.

    It also implements early stopping itself: training is stopped once the metric has
    not improved for ``patience`` consecutive evaluations.

    This deliberately avoids HuggingFace's ``load_best_model_at_end`` /
    ``EarlyStoppingCallback`` combination, which relies on the native checkpointing and
    can transiently keep two checkpoints on disk (the best plus the most recent). That
    matters under memory/disk-constrained full fine-tuning, where full-model checkpoints
    are large. Use it with ``save_strategy="no"`` so HuggingFace writes no checkpoints of
    its own.

    The owning wrapper must set ``callback.trainer`` to the ``Trainer`` instance after
    construction (``Trainer.save_model`` handles PEFT adapters, model unwrapping and the
    main-process guard).
    """

    def __init__(self, output_dir, metric_name="eval_loss", greater_is_better=False, patience=3, threshold=0.0):
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.num_bad_evals = 0
        self.trainer = None  # set by the wrapper after the Trainer is built

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.greater_is_better:
            return value > self.best + self.threshold
        return value < self.best - self.threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric_name not in metrics:
            return control
        value = metrics[self.metric_name]

        if self._is_improvement(value):
            self.best = value
            self.num_bad_evals = 0
            if self.trainer is not None:
                # Overwrites the single copy at output_dir -> at most one checkpoint on disk.
                self.trainer.save_model(self.output_dir)
        else:
            self.num_bad_evals += 1
            if self.num_bad_evals >= self.patience:
                control.should_training_stop = True

        return control
