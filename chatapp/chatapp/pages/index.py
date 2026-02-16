"""The overview page of the app."""

import reflex as rx

from .. import styles
from ..templates import template


@template(route="/", title="Overview",)
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.

    """
    return rx.vstack(
        rx.flex(
            rx.input(
                rx.input.slot(rx.icon("search"), padding_left="0"),
                placeholder="Search here...",
                size="3",
                width="100%",
                max_width="450px",
                radius="large",
                style=styles.ghost_input_style,
            ),

    )
    )
