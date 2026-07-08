import logging.config
import os
import sys

import yaml


def setup_logging():
    with open("logging.yml", "rt") as file:
        cfg = yaml.safe_load(file.read())
        logging.config.dictConfig(cfg)


def disable_wandb_console_capture() -> None:
    """Force ``WANDB_CONSOLE=off`` before W&B is imported/initialised.

    In its default ("wrap") console mode, W&B monkeypatches ``stdout``/``stderr``
    ``write()`` and reads them through a non-blocking pipe. Under SLURM that pipe
    is O_NONBLOCK, so heavy tqdm/eval output makes the wrapped write raise
    BlockingIOError and kill the run. Disabling console capture avoids the patch
    entirely; force_blocking_std_streams then covers the plain-tqdm path. Set via
    setdefault so an explicit env override is still honoured.
    """
    os.environ.setdefault("WANDB_CONSOLE", "off")


def force_blocking_std_streams() -> None:
    """Clear O_NONBLOCK on stdout/stderr.

    Some launchers (notably srun) hand the process stdout/stderr with O_NONBLOCK
    set on the underlying open-file description. That flag is shared by every fd
    pointing to the same OFD, so any rapid/large writer -- tqdm progress bars,
    big prints, flush(), or W&B's console capture -- gets EAGAIN, which Python
    raises as BlockingIOError and kills the job. Clearing it at startup makes
    writes block briefly and complete instead of crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            os.set_blocking(stream.fileno(), True)
        except (OSError, ValueError, AttributeError):
            pass
