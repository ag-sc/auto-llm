import reflex as rx


def _badge(icon: str, text: str, color_scheme: str, width: str = None):
    return rx.badge(
        rx.icon(icon, size=16),
        text,
        color_scheme=color_scheme,
        radius="full",
        variant="soft",
        size="3",
        width=width,  # Forces the entire badge to this width if given
        justify="center",
    )


STATUS_WIDTH = "100px"


def status_badge(status: str):
    badge_mapping = {
        # Priorities
        "1": ("info", "Prio 1", "green"),
        "2": ("info", "Prio 2", "amber"),
        "3": ("info", "Prio 3", "crimson"),
        # Configurations
        "TRAINER_RUN_CFG": ("beaker", "TRAIN", "green"),
        "EVALUATOR_RUN_CFG": ("beaker", "EVAL", "blue"),
        # Status
        "running": ("loader", "Running", "blue", STATUS_WIDTH),
        "finished": ("check", "Finished", "green", STATUS_WIDTH),
        "failed": ("circle_alert", "Failed", "red", STATUS_WIDTH),
        "crashed": ("circle_alert", "Crashed", "orange", STATUS_WIDTH),
        "killed": ("circle_alert", "Killed", "gray", STATUS_WIDTH),
        "pending": ("loader", "Pending", "yellow", STATUS_WIDTH),
    }
    return _badge(
        *badge_mapping.get(
            status,
            ("loader", "Pending", "yellow", "200px"),
        )
    )
