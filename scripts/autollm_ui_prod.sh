cd ui
fuser -k 3000/tcp
fuser -k 8000/tcp

ENV_VARIABLES_PATH=/home/ubuntu/env.sh
source $ENV_VARIABLES_PATH

reflex export
reflex run --env prod