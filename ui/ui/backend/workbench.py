import datetime
import json
import os
from typing import Any, Dict, List, Optional


from ..backend import CONFIGS_DIR, ROOT_DIR


class Workbench:
    def __init__(self, path: str = CONFIGS_DIR):
        self.path = path

    def get_jobs(self, username: str, team: Optional[str] = None, user_teams: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Returns the jobs of `username`, or, if `team` is given, the jobs of every user created for that team.
        `user_teams` maps usernames to their organization and is used for older jobs that did not record a team.
        """
        user_teams = user_teams or {}
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

                if team is not None:
                    job_team = settings.get("team") or user_teams.get(settings["username"], "")
                    is_visible = job_team.strip() == team
                else:
                    is_visible = settings["username"] == username

                if is_visible:
                    run_group = state.get("run_group") or self._read_run_group(path)
                    # run_group is "<dataset_name>_<hash>" (see TrainEvalRunConfigurator); dataset names may contain underscores
                    dataset_name, _, run_group_hash = run_group.rpartition("_")
                    job_details.append(
                        {
                            "job_path": path,
                            "job_path_short": path.replace(ROOT_DIR, ""),
                            "run_group": run_group,
                            "dataset_name": dataset_name,
                            "run_group_hash": run_group_hash,
                            "timestamp": dt_object,
                        }
                    )

        job_details.sort(key=lambda x: x["timestamp"], reverse=True)

        return job_details

    @staticmethod
    def _read_run_group(job_path: str) -> str:
        """Older jobs did not save run_group in configure_state.json; fall back to the job's run_group.json."""
        try:
            with open(os.path.join(job_path, "run_group.json"), "r") as f:
                return json.load(f)["run_group"]
        except (OSError, ValueError, KeyError):
            return ""
