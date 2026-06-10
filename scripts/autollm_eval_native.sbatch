#!/bin/bash

# set the following variables
CONFIG_PATH=$1
VENV_PATH=$2
ENV_VARIABLES_PATH=$3

export LOGLEVEL=DEBUG
source $VENV_PATH/bin/activate
source $ENV_VARIABLES_PATH

python -m auto_llm.evaluator.run --config_path $CONFIG_PATH
