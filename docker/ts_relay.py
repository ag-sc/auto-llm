#!/usr/bin/env python3
"""Host-side relay that lets the UI container submit task-spooler jobs to the host.

task-spooler runs a job inside the `ts` process that submitted it, not inside the ts
server. If the UI container called `ts` directly, its jobs would run in the container
(no GPUs, no flash-attn, killed on every UI restart). Instead the container's `ts`
(docker/ts_client.py) forwards its arguments here, and this relay runs the real `ts`
on the host, so jobs run on the host just like before the UI moved to Docker.

Run on the host (scripts/start_ts_queue.sh starts it):
    python3 docker/ts_relay.py

Protocol: one JSON request per connection, {"cwd": str, "args": [str, ...]}, answered with
{"rc": int, "stdout": str, "stderr": str}.
"""
import json
import os
import socketserver
import subprocess
import sys

RELAY_DIR = os.environ.get("TS_RELAY_DIR", "/home/ubuntu/.autollm_ts_relay")
RELAY_SOCKET = os.path.join(RELAY_DIR, "relay.sock")
TS_SOCKET = os.environ.get("TS_SOCKET", "/tmp/shared_gpu_queue")
TS_BIN = os.environ.get("TS_BIN", "ts")
# Jobs may only be submitted from these directories (the generated run.sh cds into the repo).
ALLOWED_ROOTS = [
    p for p in os.environ.get("TS_RELAY_ALLOWED_ROOTS", "/home/ubuntu/auto-llm:/home/ubuntu/.auto_llm_cache").split(":") if p
]
SUBMIT_TIMEOUT_SECONDS = 60


def _ts_server_listening() -> bool:
    """True if a server listens on TS_SOCKET. Reads /proc instead of connecting, because a
    `ts` call that cannot connect would start a second server on the same path."""
    try:
        with open("/proc/net/unix") as f:
            return any(line.split()[-1] == TS_SOCKET for line in f if len(line.split()) >= 8)
    except OSError:
        return False


def _allowed_cwd(cwd: str) -> bool:
    real = os.path.realpath(cwd)
    return any(real == root or real.startswith(root.rstrip("/") + "/") for root in ALLOWED_ROOTS)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request = json.loads(self.rfile.readline())
            cwd, args = request["cwd"], request["args"]
            if not isinstance(cwd, str) or not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ValueError("bad request")
        except Exception as e:
            return self._reply(2, "", f"ts-relay: invalid request: {e}\n")

        if not _allowed_cwd(cwd):
            return self._reply(2, "", f"ts-relay: submitting from {cwd} is not allowed\n")
        if not _ts_server_listening():
            return self._reply(2, "", f"ts-relay: no task-spooler server on {TS_SOCKET}; run scripts/start_ts_queue.sh on the host\n")

        env = dict(os.environ, TS_SOCKET=TS_SOCKET)
        try:
            result = subprocess.run(
                [TS_BIN, *args], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=SUBMIT_TIMEOUT_SECONDS,
            )
        except Exception as e:
            return self._reply(2, "", f"ts-relay: running ts failed: {e}\n")
        print(f"ts-relay: ts {' '.join(args)!r} (cwd={cwd}) -> rc={result.returncode}", flush=True)
        self._reply(result.returncode, result.stdout, result.stderr)

    def _reply(self, rc: int, stdout: str, stderr: str):
        self.wfile.write((json.dumps({"rc": rc, "stdout": stdout, "stderr": stderr}) + "\n").encode())


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    os.makedirs(RELAY_DIR, mode=0o700, exist_ok=True)
    os.chmod(RELAY_DIR, 0o700)  # only this user can reach the socket
    if os.path.exists(RELAY_SOCKET):
        os.unlink(RELAY_SOCKET)
    with Server(RELAY_SOCKET, Handler) as server:
        os.chmod(RELAY_SOCKET, 0o600)
        print(f"ts-relay: listening on {RELAY_SOCKET}, forwarding to TS_SOCKET={TS_SOCKET}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
