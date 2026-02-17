"""The overview page of the app."""

import reflex as rx

from .. import styles
from ..state.user import User
from ..templates import template

def overview() -> rx.Component:
    return rx.vstack(
        rx.flex(
            rx.vstack(
                rx.markdown(f"# 👋 Hi, {User.username_display}!"),
                rx.input(
                    rx.input.slot(rx.icon("search"), padding_left="0"),
                    placeholder="Search here...",
                    size="3",
                    width="100%",
                    max_width="450px",
                    radius="large",
                    style=styles.ghost_input_style,
                ),
            ),


        ),
        width="100%",
    )


@template(route="/overview", title="Overview", on_load=User.check_logged_in)
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.

    """
    return overview()
