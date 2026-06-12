import reflex as rx

config = rx.Config(
    app_name="ui",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    deploy_url="https://autollm.llm4kmu.de",
    api_url="https://autollm.llm4kmu.de",
)
