import reflex as rx

from ..state.user import User


@rx.page(route="/", title="Login", on_load=User.check_login)
def login() -> rx.Component:
    return rx.center(
        rx.card(
        rx.vstack(
            # Centering the image specifically
            rx.box(
                rx.image(src="/AutoLLM.png", height="6em"),
                width="100%",
                display="flex",
                justify_content="center",
                padding_bottom="1.5em",
            ),
            rx.form(
                rx.vstack(
                    rx.center(
                        rx.heading(
                            "Sign in to your account",
                            size="6",
                            as_="h2",
                            text_align="center",
                            width="100%",
                        ),
                        direction="column",
                        spacing="5",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.text(
                            "Username",
                            size="3",
                            weight="medium",
                            text_align="left",
                            width="100%",

                        ),
                        rx.input(
                            placeholder="e.g. username", size="3", width="100%", name="username",
                            on_change=User.setvar("username"), value=User.username,
                        ),
                        justify="start",
                        spacing="2",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.text("Password", size="3", weight="medium"),
                            justify="between",
                            width="100%",
                        ),
                        rx.input(
                            placeholder="Enter your password",
                            type="password",
                            size="3",
                            width="100%",
                            name="password",
                            on_change=User.setvar("password"), value=User.password,
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.button("Sign in", size="3", width="100%", type="submit"),
                    spacing="6",
                    width="100%",
                ),
                on_submit=User.handle_sign_in,
            ),
        ),
        size="5",
        max_width="28em",
        width="100%",
    ),
        height="100vh",
        width="100%",
        bg=rx.color("mauve", 10),
    )