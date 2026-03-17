import reflex as rx

from . import styles
from .pages import *

app = rx.App(
    style=styles.base_style,
    stylesheets=styles.base_stylesheets,
    head_components=[
        rx.el.link(rel="icon", href="/favicon.ico"),
        rx.el.title("AutoLLM"),
    ],
)
