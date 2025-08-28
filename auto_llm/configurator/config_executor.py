from typing import List

from auto_llm.configurator.config_generator import ConfiguratorOutput


class ConfigExecutor:
    def __init__(self, configurator_outputs: List[ConfiguratorOutput]):
        self._configurator_outputs = configurator_outputs

    def execute(self):
        # sort config outputs based on priority
        # separate evaluator and trainer cfgs
        # run pre-trained eval configs -> prio 1
        # run trainer cfgs -> prio 2
        # how to decide ddp or fsdp?
        # run evaluator configs for ft models -> prio 3
        for output in self._configurator_outputs:
            ...
