import asyncio
from datetime import datetime
from typing import Any, Dict, List

import reflex as rx
from reflex.components.radix.themes.base import LiteralAccentColor

from ..backend.workbench import Workbench

from .. import styles


class WorkbenchState(rx.State):
    jobs: List[Dict[str, Any]] = []
    is_auto_refresh: bool = False

    @rx.event
    def get_jobs(self):
        self.jobs = Workbench().get_jobs()
        print("jobs", self.jobs)


def render_individual_job(job):
    """
    This function defines how a SINGLE job appears.
    Reflex will call this for every item in WorkbenchState.jobs.
    """
    return rx.card(
        rx.hstack(
            rx.text(job["job_path"]),
            rx.spacer(),
            rx.moment(job["timestamp"], from_now=True),
            rx.button(rx.icon("eye"), variant="soft", size="1"),
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
                rx.button(rx.icon(tag="refresh-cw", size=20), on_click=WorkbenchState.get_jobs),
                width="100%",
                justify="between",
                align="center",
            ),
            rx.scroll_area(
                # Use foreach instead of a manual list append
                rx.foreach(WorkbenchState.jobs, render_individual_job),
                type="always",
                scrollbars="vertical",
                height="40vh",
            ),
            spacing="3",
            width="100%",
        ),
        width="80%",
        box_shadow=styles.box_shadow_style,
    )
