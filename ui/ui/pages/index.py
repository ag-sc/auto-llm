import asyncio
import time
import reflex as rx

from ..state.user import User, get_wandb_client
from .. import styles
from ..templates import template
from ..views.project_stats import stats_card, stats_cards, stats_cards_empty
from ..views.workbench_stats import workbench_stats_card
from concurrent.futures import ThreadPoolExecutor

# A shared executor to handle blocking W&B API network calls safely
_executor = ThreadPoolExecutor(max_workers=5)

# Project details are shared across all sessions of the same team (W&B entity) for a short while
_PROJECT_DETAILS_TTL_SECONDS = 300
_project_details_cache: dict[str, tuple[float, list[dict]]] = {}


def _get_project_details(client, force: bool = False) -> list[dict]:
    entity = client.get_entity()
    cached = _project_details_cache.get(entity)
    if not force and cached and time.monotonic() - cached[0] < _PROJECT_DETAILS_TTL_SECONDS:
        return cached[1]

    project_details = client.get_project_details()
    _project_details_cache[entity] = (time.monotonic(), project_details)
    return project_details


class OverviewState(rx.State):
    entity: str = ""
    project_details: list[dict] = []
    fetched_data: bool = False
    is_loading: bool = False  # Prevent concurrent execution of the event handler

    @rx.event
    async def refresh_overview_data(self):
        async for event in self._load_overview_data(force=True):
            yield event

    @rx.event
    async def fetch_overview_data(self):
        async for event in self._load_overview_data(force=False):
            yield event

    async def _load_overview_data(self, force: bool):
        user_state = await self.get_state(User)
        client = get_wandb_client(user=user_state)

        # Prevent concurrent execution or redundant fetching (refetch when an admin switched teams)
        if self.is_loading or (not force and self.fetched_data and self.entity == client.get_entity()):
            return

        self.is_loading = True
        yield

        loop = asyncio.get_event_loop()

        try:
            project_details = await loop.run_in_executor(_executor, _get_project_details, client, force)

            self.entity = client.get_entity()
            self.project_details = project_details
            self.fetched_data = True
        except Exception as e:
            print(f"Error loading W&B overview data: {e}")
            yield rx.toast.warning("Failed to fetch project details. Please retry later!")
        finally:
            self.is_loading = False
            yield  # CRITICAL: Forces UI to register that loading has finished


def overview() -> rx.Component:
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
                            rx.cond(
                                User.is_admin,
                                rx.select(
                                    User.teams,
                                    value=User.selected_team,
                                    on_change=User.set_team,
                                    size="1",
                                ),
                                rx.code(OverviewState.entity, size="2"),
                            ),
                            align="center",
                        ),
                        # 3. ROBUSTNESS: Disable the refresh button during active fetching
                        rx.button(
                            rx.icon(tag="refresh-cw", size=15),
                            on_click=OverviewState.refresh_overview_data,
                            loading=OverviewState.is_loading,
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
                on_mount=OverviewState.fetch_overview_data,
            ),
            workbench_stats_card(),
            spacing="5",
            width="60vw",
            height="80vh",
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
        # OverviewState.fetch_overview_data,
    ],
)
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.

    """
    return overview()
