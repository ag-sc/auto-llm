import reflex as rx

from ..state.user import User, get_wandb_client
from .. import styles
from ..templates import template
from ..views.project_stats import stats_card, stats_cards, stats_cards_empty
from ..views.workbench_stats import workbench_stats_card


class OverviewState(rx.State):
    entity: str = ""
    project_details: list[dict] = []
    is_loading: bool = False

    @rx.event
    async def fetch_overview_data(self):
        self.is_loading = True
        yield

        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)
        self.entity = client.get_entity()
        self.project_details = client.get_project_details()
        self.is_loading = False


def overview() -> rx.Component:
    # Use rx.foreach to dynamically render the stats cards based on the state
    return rx.flex(
        rx.vstack(
            rx.flex(
                rx.vstack(
                    rx.markdown(f"# 👋 Hi {User.firstname_proc}!"),
                ),
            ),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.hstack(
                            rx.icon("panels-top-left", size=25),
                            rx.heading("Projects", size="5", weight="bold"),
                            rx.code(OverviewState.entity, size="2"),
                            align="center",
                        ),
                        rx.button(
                            rx.icon(tag="refresh-cw", size=15),
                            on_click=lambda: OverviewState.fetch_overview_data,
                        ),
                        width="100%",
                        align="center",
                        justify="between",
                    ),
                    rx.cond(
                        OverviewState.is_loading,
                        stats_cards_empty(),
                        stats_cards(
                            rx.foreach(
                                OverviewState.project_details,
                                lambda item: stats_card(
                                    project_name=item["name"],
                                    project_url=item["url"],
                                    num_runs=item["num_runs"],
                                    runtime=item["runtime"],
                                    emission=0,
                                ),
                            ),
                        ),
                    ),
                    width="100%",
                    box_shadow=styles.box_shadow_style,
                ),
                spacing="5",
                width="100%",
                # on_mount=OverviewState.fetch_overview_data,
            ),
            workbench_stats_card(username=User.username),
            spacing="5",
            width="60vw",
            height="80vh",
            # flex="1",
        ),
        width="100%",
        height="100%",
        direction={"sm": "row", "md": "row"},
    )


@template(
    route="/overview",
    title="Overview",
    on_load=[
        User.check_logged_in,
    ],
)
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.

    """
    return overview()
