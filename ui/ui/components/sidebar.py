import reflex as rx

from .. import styles
from ..state.user import User


def view_user_profile():
    return rx.dialog.root(
        rx.dialog.trigger(
            rx.icon_button(
                "user",
                size="2",
                radius="full",
            ),
        ),
        rx.dialog.content(
            rx.vstack(
                rx.vstack(
                    rx.avatar(
                        size="7",
                        variant="soft",
                        color_scheme="violet",
                        fallback=f"{User.firstname_proc[0]}{User.lastname_proc[0]}",
                        radius="full",
                    ),
                    rx.hstack(
                        rx.text(f"{User.firstname_proc} {User.lastname_proc}", weight="medium"),
                        align="center",
                    ),
                    rx.hstack(
                        rx.text(User.organization_proc, weight="medium"),
                        align="center",
                    ),
                    align="center",
                    width="100%",
                    padding="2em",
                ),
                rx.card(
                    rx.vstack(
                        rx.text("You can change your password below.", weight="medium"),
                        rx.input(
                            placeholder="Enter your username",
                            size="3",
                            width="100%",
                            name="username",
                            on_change=User.setvar("username"),
                            value=User.username,
                            disabled=True,
                        ),
                        rx.input(
                            placeholder="Enter your current password",
                            type="password",
                            size="3",
                            width="100%",
                            name="password",
                            on_change=User.setvar("current_password"),
                        ),
                        rx.input(
                            placeholder="Enter your new password",
                            type="password",
                            size="3",
                            width="100%",
                            name="password",
                            on_change=User.setvar("new_password"),
                        ),
                        rx.button(
                            "Change Password",
                            size="3",
                            width="100%",
                            type="button",
                            bg="#4a4d9b",
                            on_click=User.change_password,
                        ),
                    ),
                    align="center",
                    width="100%",
                ),
                rx.card(
                    rx.vstack(
                        rx.text("You can log out from the current session below.", weight="medium"),
                        rx.button(
                            "Log Out",
                            size="3",
                            width="100%",
                            type="button",
                            bg="#4a4d9b",
                            on_click=User.handle_sign_out,
                        ),
                    ),
                    align="center",
                    width="100%",
                ),
                # rx.hstack(
                #     rx.dialog.close(rx.button("Close", variant="soft")),
                #     justify="center",
                #     width="100%",
                # ),
            ),
            width="30vw",
            height="75vh",
            max_width="30vw",
            max_height="75vh",
            padding="1em",
        ),
    )


def sidebar_header() -> rx.Component:
    """Sidebar header."""
    return rx.hstack(
        rx.color_mode_cond(
            rx.image(src="/AutoLLM_logo.png", height="5em"),
            rx.image(src="/AutoLLM_dark.png", height="5em"),
        ),
        rx.spacer(),
        align="center",
        width="100%",
        padding="0.35em",
        margin_bottom="1em",
    )


def sidebar_footer() -> rx.Component:
    """Sidebar footer.

    Returns:
        The sidebar footer component.

    """
    return rx.vstack(
        rx.hstack(
            rx.link(
                rx.text("Docs", size="3"),
                href="https://ag-sc.github.io/auto-llm/",
                color_scheme="gray",
                underline="none",
            ),
            rx.spacer(),
            rx.color_mode.button(style={"opacity": "0.8", "scale": "0.95"}, size="3"),
            justify="end",
            align="center",
            width="100%",
            padding="0.35em",
        ),
        rx.hstack(
            view_user_profile(),
            rx.text(User.firstname_proc, weight="bold", size="2"),
            align="center",
            spacing="2",
            justify="end",
            width="100%",
        ),
        width="100%",
        align_items="column",
    )


def sidebar_item_icon(icon: str) -> rx.Component:
    return rx.icon(icon, size=18)


def sidebar_item(text: str, url: str) -> rx.Component:
    """Sidebar item.

    Args:
        text: The text of the item.
        url: The URL of the item.

    Returns:
        rx.Component: The sidebar item component.

    """
    # Whether the item is active.
    active = (rx.State.router.page.path == url.lower()) | ((rx.State.router.page.path == "/") & text == "Overview")

    return rx.link(
        rx.hstack(
            rx.match(
                text,
                ("Overview", sidebar_item_icon("home")),
                ("Configure", sidebar_item_icon("table-2")),
                # ("Monitor", sidebar_item_icon("book-open")),
                # ("Chat", sidebar_item_icon("user")),
                ("About", sidebar_item_icon("settings")),
                sidebar_item_icon("layout-dashboard"),
            ),
            rx.text(text, size="3", weight="regular"),
            color=rx.cond(
                active,
                styles.accent_text_color,
                styles.text_color,
            ),
            style={
                "_hover": {
                    "background_color": rx.cond(
                        active,
                        styles.accent_bg_color,
                        styles.gray_bg_color,
                    ),
                    "color": rx.cond(
                        active,
                        styles.accent_text_color,
                        styles.text_color,
                    ),
                    "opacity": "1",
                },
                "opacity": rx.cond(
                    active,
                    "1",
                    "0.95",
                ),
            },
            align="center",
            border_radius=styles.border_radius,
            width="100%",
            spacing="2",
            padding="0.35em",
        ),
        underline="none",
        href=url,
        width="100%",
    )


def sidebar() -> rx.Component:
    """The sidebar.

    Returns:
        The sidebar component.
    """
    from reflex.page import DECORATED_PAGES

    ordered_page_routes = [
        "/overview",
        "/configure",
        # "/monitor",
        # "/chat",
        "/about",
    ]

    pages = [page_dict for page_list in DECORATED_PAGES.values() for _, page_dict in page_list]

    ordered_pages = sorted(
        pages,
        key=lambda page: (ordered_page_routes.index(page["route"]) if page["route"] in ordered_page_routes else len(ordered_page_routes)),
    )

    return rx.flex(
        rx.vstack(
            sidebar_header(),
            rx.vstack(
                *[
                    sidebar_item(
                        text=page.get("title", page["route"].strip("/").capitalize()),
                        url=page["route"],
                    )
                    for page in ordered_pages
                    if page["route"] != "/"
                ],
                spacing="1",
                width="100%",
            ),
            rx.spacer(),
            sidebar_footer(),
            justify="end",
            align="end",
            width=styles.sidebar_content_width,
            height="100dvh",
            padding="1em",
        ),
        # display="flex",
        # display=["flex", "flex", "flex", "flex", "flex", "flex"],
        display={"sm": "flex", "md": "flex", "lg": "flex"},
        max_width=styles.sidebar_width,
        width="auto",
        height="100%",
        position="sticky",
        justify="end",
        top="0px",
        left="0px",
        flex="1",
        bg=rx.color("gray", 2),
    )
