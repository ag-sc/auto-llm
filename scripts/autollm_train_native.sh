#!/bin/bash

# set the following variables
CONFIG_PATH=$1
VENV_PATH=$2
ENV_VARIABLES_PATH=$3
PARALLELISM=$4

source $VENV_PATH/bin/activate
source $ENV_VARIABLES_PATH  # /homes/vsudhi/env.sh

if [ "$PARALLELISM" == "ddp" ]; then
    accelerate launch --config_file config_files/accelerator_configs/accelerate_config_ddp.yaml -m auto_llm.trainer.run --config_path $CONFIG_PATH
elif [ "$PARALLELISM" == "fsdp" ]; then
    accelerate launch --config_file config_files/accelerator_configs/accelerate_config_ddp.yaml -m auto_llm.trainer.run --config_path $CONFIG_PATH
else
    echo "No Parallelism mode passed. Running python -m instead."
    python -m auto_llm.trainer.run --config_path $CONFIG_PATH
fi