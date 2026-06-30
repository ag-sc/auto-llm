cd ui
fuser -k 3000/tcp
fuser -k 8000/tcp

ENV_VARIABLES_PATH=/home/ubuntu/env.sh
source $ENV_VARIABLES_PATH

reflex run --frontend-port 3000 --backend-port 8000