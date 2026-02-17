import reflex as rx

config = rx.Config(
    app_name="ui",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    # state_auto_setters=True,
)