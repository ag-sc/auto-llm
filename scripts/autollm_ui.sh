ENV_VARIABLES_PATH=$1

source $ENV_VARIABLES_PATH  # /homes/vsudhi/env.sh
cd ui && reflex run
