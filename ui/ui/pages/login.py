import reflex as rx

from ..state.user import User


@rx.page(route="/", title="Login", on_load=User.check_login)
def login() -> rx.Component:
    return rx.box(
        # Background Video Component
        rx.el.video(
            src="/LLM4KMU_Verlauf-bewegt_1.mp4",
            auto_play=True,
            loop=True,
            muted=True,
            plays_inline=True,
            position="fixed",
            top="0",
            left="0",
            width="100vw",
            height="100vh",
            object_fit="cover",
            z_index="-1",  # Keeps the video behind UI content
        ),
        # Login Card Layout
        rx.center(
            rx.card(
                rx.vstack(
                    # Centering the image specifically
                    rx.box(
                        rx.image(src="/AutoLLM_logo.png", height="6em"),
                        width="100%",
                        display="flex",
                        justify_content="center",
                        padding_bottom="1.5em",
                    ),
                    rx.card(
                        rx.vstack(
                            rx.vstack(
                                rx.text(
                                    "Username",
                                    size="3",
                                    weight="medium",
                                    text_align="left",
                                    width="100%",
                                ),
                                rx.input(
                                    placeholder="Enter your username",
                                    size="3",
                                    width="100%",
                                    name="username",
                                    on_change=User.setvar("username"),
                                    value=User.username,
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
                                    on_change=User.setvar("password"),
                                    value=User.password,
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            rx.button(
                                "Sign in",
                                size="3",
                                width="100%",
                                type="button",
                                bg="#4a4d9b",
                                on_click=User.handle_sign_in,
                                margin_top="1em",
                            ),
                            rx.vstack(
                                # rx.text("Powered by", weight="medium"),
                                rx.box(
                                    rx.image(src="/LLM4KMU_Logo_RGB.svg", height="5em"),
                                    width="95%",
                                    display="flex",
                                    justify_content="center",
                                ),
                                rx.text(
                                    "© 2026 LLM4KMU. All rights reserved.",
                                    size="1",
                                    color_scheme="gray",
                                    margin_top="1em",
                                ),
                                spacing="2",
                                width="100%",
                                align="center",
                                padding_top="1.5em",
                            ),
                            # rx.button(
                            #     "Sign Up",
                            #     size="3",
                            #     width="100%",
                            #     type="button",
                            #     on_click=User.handle_sign_up,
                            # ),
                            # spacing="6",
                            # width="100%",
                        ),
                        spacing="5",
                        width="100%",
                        variant="ghost",
                        align="center",
                    ),
                ),
                size="5",
                max_width="28em",
                width="100%",
            ),
            height="100vh",
            width="100%",
        ),
        width="100%",
        position="relative",
    )
