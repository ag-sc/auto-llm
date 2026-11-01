#!/opt/venv/bin/python
"""`ts` inside the UI container: forwards the call to docker/ts_relay.py on the host,
so the job runs on the host (see ts_relay.py for why)."""
import json
import os
import socket
import sys

RELAY_SOCKET = os.path.join(os.environ.get("TS_RELAY_DIR", "/home/ubuntu/.autollm_ts_relay"), "relay.sock")


def main() -> int:
    request = json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}) + "\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(90)
            s.connect(RELAY_SOCKET)
            s.sendall(request.encode())
            reply = json.loads(s.makefile().readline())
    except Exception as e:
        print(f"ts: cannot reach the host ts relay at {RELAY_SOCKET} ({e}). Run scripts/start_ts_queue.sh on the host.", file=sys.stderr)
        return 2
    sys.stdout.write(reply["stdout"])
    sys.stderr.write(reply["stderr"])
    return reply["rc"]


if __name__ == "__main__":
    sys.exit(main())
