"""The profile page."""

import reflex as rx

from auto_llm.registry.tracker_registry import WANDB_EVAL_REPORT_URL
from ..state.user import User
from ..templates import template


@template(route="/monitor", title="Monitor", on_load=User.check_logged_in)
def monitor() -> rx.Component:
    return rx.center(
            rx.html(
                f'<iframe id="my-frame" src={WANDB_EVAL_REPORT_URL} style="border:none;height:1024px;width:100%"></iframe>',
                width="100%",
            ),
            width="100%",
        )