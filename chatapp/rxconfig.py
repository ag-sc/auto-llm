import reflex as rx

config = rx.Config(
    app_name="chatapp",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    # state_auto_setters=True,
)