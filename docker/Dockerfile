# Image for the AutoLLM Reflex UI only.
# The repo itself is bind-mounted at runtime (same path as on the host), so this image
# only holds the Python environment. Training/eval jobs still run on the host through
# the host's task-spooler queue.

# Same Ubuntu as the host, so the host's `ts` binary (mounted in) finds a compatible glibc.
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl unzip git psmisc \
    && rm -rf /var/lib/apt/lists/*

# uv provides Python 3.10 (what the host venv uses) without deadsnakes.
COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --python 3.10 /opt/venv

# CPU-only torch: the UI never trains, so skip the multi-GB CUDA wheels.
# Version matches the host venv (torch 2.12.0).
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0

# constraints.txt pins every package to the version in the host venv.
COPY requirements.txt /tmp/requirements.txt
COPY docker/constraints.txt /tmp/constraints.txt
RUN uv pip install -r /tmp/requirements.txt -c /tmp/constraints.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt

# Bun for building/serving the Reflex frontend. Reflex only downloads it during
# `reflex init`, which it skips because ui/.web already exists in the mounted repo.
# Version matches reflex 0.8.27 (reflex/constants/installer.py: Bun.VERSION).
ENV BUN_INSTALL=/opt/bun \
    PATH=/opt/bun/bin:$PATH
RUN curl -fsSL https://bun.sh/install | bash -s "bun-v1.3.6" \
    && bun --version

# Run as the same uid/gid as the host user so files in the mounted repo/cache stay owned
# by them and the host ts server accepts the socket connection.
ARG UID=1000
ARG GID=1000
RUN (getent group ${GID} || groupadd -g ${GID} ubuntu) \
    && (getent passwd ${UID} || useradd -m -u ${UID} -g ${GID} -d /home/ubuntu ubuntu) \
    && mkdir -p /home/ubuntu/.local/share/reflex \
    && chown -R ${UID}:${GID} /home/ubuntu
ENV HOME=/home/ubuntu

COPY docker/entrypoint.sh /usr/local/bin/autollm-ui-entrypoint
RUN chmod +x /usr/local/bin/autollm-ui-entrypoint

USER ${UID}:${GID}
ENTRYPOINT ["autollm-ui-entrypoint"]
