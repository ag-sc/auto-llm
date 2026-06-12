import reflex as rx


from ..state.user import User
from ..templates import template
from ..views.project_stats import stats_card, stats_cards
from ..views.workbench_stats import workbench_stats_card

from ..backend.wandb_client import Client


def overview() -> rx.Component:
    project_details = Client.get_project_details()
    list_stats_card = []
    for item in project_details:
        list_stats_card.append(
            stats_card(
                project_name=item["name"],
                project_url=item["url"],
                num_runs=item["num_runs"],
                runtime=item["runtime"],
                emission=0,
            )
        )

    return rx.vstack(
        rx.flex(
            rx.vstack(
                rx.markdown(f"# 👋 Hi, {User.username_display}!"),
            ),
        ),
        stats_cards(list_stats_card),
        workbench_stats_card(username=User.username),
        spacing="5",
        width="100%",
    )


@template(route="/overview", title="Overview", on_load=User.check_logged_in)
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.

    """
    return overview()
