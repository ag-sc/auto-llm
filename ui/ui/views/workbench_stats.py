import json
from typing import Any, Dict, List

import reflex as rx

from ..pages.configure import AppState
from ..state.user import User, UserModel
from ..backend.workbench import Workbench
from ..backend import ROOT_DIR
from .. import styles


class WorkbenchState(rx.State):
    jobs: List[Dict[str, Any]] = []
    is_auto_refresh: bool = False

    @rx.event
    async def get_jobs(self):
        user_state = await self.get_state(User)

        if not user_state.is_admin:
            self.jobs = Workbench().get_jobs(username=user_state.username)
            return

        # Admins see the jobs of every user of the currently selected team
        with rx.session() as session:
            user_teams = {u.username: u.organization for u in session.exec(UserModel.select()).all()}

        self.jobs = Workbench().get_jobs(username=user_state.username, team=user_state.active_team, user_teams=user_teams)

    @rx.event
    async def view_job_details(self, job_path: str):
        print("Viewing job details for:", job_path)
        configure_state_path = f"{job_path}/configure_state.json"

        try:
            with open(configure_state_path, "r") as f:
                configure_state = json.load(f)

            # Older jobs were saved without run_group/configs_path; recover them from the job folder
            if not configure_state.get("run_group"):
                with open(f"{job_path}/run_group.json", "r") as f:
                    configure_state["run_group"] = json.load(f)["run_group"]
            if not configure_state.get("configs_path"):
                configure_state["configs_path"] = job_path

            app_state = await self.get_state(AppState)
            app_state.load_from_json(configure_state)

            yield
            yield rx.redirect("/configure")

        except Exception as e:
            print(f"Error loading job: {e}")
            yield rx.window_alert("Could not load job details.")


def render_individual_job(job):
    """
    This function defines how a SINGLE job appears.
    Reflex will call this for every item in WorkbenchState.jobs.
    """
    return rx.card(
        rx.hstack(
            rx.text(job["job_path_short"]),
            rx.spacer(),
            rx.moment(job["timestamp"], from_now=True),
            rx.button(rx.icon("eye"), variant="soft", size="1", on_click=lambda: WorkbenchState.view_job_details(job["job_path"])),
            width="100%",
            align="center",
        ),
        width="100%",
    )


def workbench_stats_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.hstack(rx.icon("folder-kanban", size=25), rx.heading("Workbench", size="5", weight="bold"), align="center"),
                rx.button(
                    rx.icon(tag="refresh-cw", size=15),
                    on_click=WorkbenchState.get_jobs,
                ),
                width="100%",
                justify="between",
                align="center",
            ),
            rx.scroll_area(
                # Use foreach instead of a manual list append
                rx.foreach(WorkbenchState.jobs, render_individual_job),
                type="always",
                scrollbars="vertical",
                height="30vh",
            ),
            spacing="3",
            width="100%",
        ),
        width="100%",
        box_shadow=styles.box_shadow_style,
        on_mount=WorkbenchState.get_jobs,
    )
