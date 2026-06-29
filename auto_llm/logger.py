import logging.config
import os
import sys

import yaml


def setup_logging():
    with open("logging.yml", "rt") as file:
        cfg = yaml.safe_load(file.read())
        logging.config.dictConfig(cfg)


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
