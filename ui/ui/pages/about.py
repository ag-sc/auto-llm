"""The about page."""

import reflex as rx

from .. import styles
from ..state.user import User
from ..templates import template

LINKS = [
    ("Website", "globe", "https://llm4kmu.de/"),
    ("Reports", "chart-line", "https://wandb.ai/llm4kmu/projects"),
    ("LinkedIn", "linkedin", "https://www.linkedin.com/company/llm4kmu/"),
    ("Code", "github", "https://github.com/ag-sc/auto-llm"),
]

FEATURES = [
    (
        "boxes",
        "Choose your model",
        "Pick from a wide range of open-source LLMs and find the one that fits your application.",
    ),
    (
        "settings-2",
        "Train automatically",
        "Inspired by Auto-ML, AutoLLM determines the optimal model, architecture and training method for your problem.",
    ),
    (
        "scale",
        "Evaluate and compare",
        "Evaluate the trained models on the metrics that matter to you and compare them side by side.",
    ),
]

HASHTAGS = ["#efre", "#efrenrw", "#EUinmyRegion"]


def link_button(label: str, icon: str, href: str) -> rx.Component:
    return rx.link(
        rx.button(
            rx.icon(icon, size=16),
            label,
            variant="soft",
            size="3",
            cursor="pointer",
        ),
        href=href,
        is_external=True,
    )


def bordered_card(*children, padding: str) -> rx.Component:
    return rx.box(
        *children,
        border=f"1px solid {rx.color('gray', 6)}",
        border_radius="var(--radius-4)",
        background_color=rx.color("gray", 1),
        box_shadow=styles.box_shadow_style,
        padding=padding,
        width="100%",
    )


def feature_card(icon: str, title: str, description: str) -> rx.Component:
    return bordered_card(
        rx.vstack(
            rx.box(
                rx.icon(icon, size=22, color=rx.color("accent", 11)),
                background_color=rx.color("accent", 3),
                border_radius="var(--radius-3)",
                padding="0.6em",
                display="flex",
            ),
            rx.heading(title, size="4"),
            rx.text(description, size="2", color_scheme="gray"),
            spacing="3",
        ),
        padding="1.5em",
    )


def hero() -> rx.Component:
    return rx.vstack(
        rx.color_mode_cond(
            rx.image(src="/AutoLLM_logo.png", height="7em"),
            rx.image(src="/AutoLLM_dark.png", height="7em"),
        ),
        # rx.heading("AutoLLM", size="9", weight="bold"),
        rx.text(
            "⚙️ Train and ⚖️ Evaluate your LMs effortlessly!",
            size="5",
            weight="medium",
            color_scheme="gray",
        ),
        rx.hstack(
            *[link_button(*link) for link in LINKS],
            spacing="3",
            wrap="wrap",
            justify="center",
            padding_top="0.5em",
        ),
        align="center",
        text_align="center",
        spacing="4",
        width="100%",
        padding_y="2em",
    )


def intro() -> rx.Component:
    return rx.text(
        rx.text.strong("AutoLLM"),
        " supports you in finding the ",
        rx.text.strong("right"),
        " open-source model, architecture and training method for your application. "
        "Choose from different open-source models, training techniques and evaluation metrics, "
        "and let AutoLLM train and evaluate them for you.",
        size="4",
        text_align="center",
        max_width="48em",
    )


def project_card() -> rx.Component:
    return bordered_card(
        rx.flex(
            rx.image(src="/LLM4KMU_Logo_RGB.svg", height="4.5em", flex_shrink="0"),
            rx.vstack(
                rx.heading("Developed as part of the project: LLM4KMU", size="5"),
                rx.text(
                    "Optimierter Einsatz von Open Source Large Language Models (LLMs) in kleinen "
                    "und mittelständischen Unternehmen (KMUs). Mit Mitteln der Europäischen Union gefördert.",
                    size="3",
                    color_scheme="gray",
                    font_style="italic",
                ),
                rx.hstack(
                    *[rx.badge(tag, variant="surface", size="2") for tag in HASHTAGS],
                    spacing="2",
                    wrap="wrap",
                ),
                spacing="3",
            ),
            direction=rx.breakpoints(initial="column", md="row"),
            align=rx.breakpoints(initial="start", md="center"),
            spacing="6",
        ),
        padding="2em",
    )


@template(route="/about", title="About", on_load=User.check_logged_in)
def about() -> rx.Component:
    return rx.vstack(
        hero(),
        intro(),
        rx.grid(
            *[feature_card(*feature) for feature in FEATURES],
            columns=rx.breakpoints(initial="1", md="3"),
            spacing="4",
            width="100%",
            padding_y="1.5em",
        ),
        project_card(),
        align="center",
        spacing="5",
        width="100%",
        max_width="64em",
        margin_x="auto",
    )
