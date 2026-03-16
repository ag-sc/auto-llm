import reflex as rx


def _badge(icon: str, text: str, color_scheme: str):
    return rx.badge(
        rx.icon(icon, size=16),
        text,
        color_scheme=color_scheme,
        radius="full",
        variant="soft",
        size="3",
    )


def status_badge(status: str):
    badge_mapping = {
        "1": ("info", "Prio 1", "green"),
        "2": ("info", "Prio 2", "amber"),
        "3": ("info", "Prio 3", "crimson"),
        "TRAINER_RUN_CFG": ("beaker", "TRAIN", "green"),
        "EVALUATOR_RUN_CFG": ("beaker", "EVAL", "blue"),
    }
    return _badge(*badge_mapping.get(status, ("loader", "Pending", "yellow")))
