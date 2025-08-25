import logging.config
import yaml


def setup_logging():
    with open("logging.yml", "rt") as file:
        cfg = yaml.safe_load(file.read())
        logging.config.dictConfig(cfg)
