import datetime
import json
import os
from typing import Any, Dict, List


from ..backend import CONFIGS_DIR


class Workbench:
    def __init__(self, path: str = CONFIGS_DIR):
        self.path = path

    def get_jobs(self, username: str) -> List[Dict[str, Any]]:
        job_details = []
        for x in os.walk(self.path):
            for file in x[2]:
                if "settings" not in file:
                    continue

                settings_path = os.path.join(x[0], file)
                state_path = os.path.join(x[0], file.replace("settings", "configure_state"))
                with open(settings_path, "r") as f:
                    settings = json.load(f)

                with open(state_path, "r") as f:
                    state = json.load(f)

                dt_object = datetime.datetime.strptime(settings["timestamp"], "%Y-%m-%d_%H-%M-%S")

                path = state_path.replace("configure_state.json", "")

                if settings["username"] == username:
                    job_details.append({"job_path": path, "timestamp": dt_object})

        job_details.sort(key=lambda x: x["timestamp"], reverse=True)

        return job_details
