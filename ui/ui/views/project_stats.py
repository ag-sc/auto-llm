import reflex as rx
from reflex.components.radix.themes.base import LiteralAccentColor

from .. import styles


def stats_card(project_name: str, project_url: str, num_runs: int, runtime: float, emission: float) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.badge(
                    rx.icon(tag="users", size=28),
                    color_scheme="blue",
                    radius="full",
                    padding="0.7rem",
                ),
                rx.vstack(
                    rx.link(
                        rx.text(project_name, font_weight="bold"),
                        href=project_url,
                        is_external=True,
                    ),
                    spacing="1",
                    height="100%",
                    align_items="start",
                    width="100%",
                ),
                height="100%",
                spacing="4",
                align="center",
                width="100%",
            ),
            rx.hstack(
                rx.hstack(
                    # num runs
                    rx.icon(
                        tag="list-ordered",
                        size=20,
                        color=rx.color("black", 9),
                    ),
                    rx.text(f"{num_runs} runs", weight="medium"),
                    # runtime
                    rx.icon(
                        tag="hourglass",
                        size=20,
                        color=rx.color("blue", 9),
                    ),
                    rx.text(f"{runtime} hours", weight="medium"),
                    # emission
                    rx.icon(
                        tag="leaf",
                        size=20,
                        color=rx.color("green", 9),
                    ),
                    rx.text(f"{emission} CO2", weight="medium"),
                    spacing="2",
                    align="center",
                ),
                align="center",
                width="100%",
            ),
        ),
        size="2",
        width="100%",
        # box_shadow=styles.box_shadow_style,
    )


def stats_cards(list_stats_cards) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(rx.icon("panels-top-left", size=25), rx.heading("Projects", size="5", weight="bold"), align="center"),
            rx.grid(
                list_stats_cards,
                gap="1rem",
                grid_template_columns=[
                    "1fr",
                    "repeat(1, 1fr)",
                    "repeat(2, 1fr)",
                    "repeat(3, 1fr)",
                    "repeat(3, 1fr)",
                ],
                width="100%",
            ),
        ),
        width="80%",
        box_shadow=styles.box_shadow_style,
    )
