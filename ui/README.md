# Deploy AutoLLM UI

- In `auto-llm/ui/.web/vite.config.js`, add `allowedHosts` as below:


```
server: {
  port: process.env.PORT,
  allowedHosts: ["autollm.llm4kmu.de"],
```

- In ``/home/ubuntu/auto-llm/ui/rxconfig.py``, add `deploy_url` and `api_url`

```
config = rx.Config(
    app_name="ui",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    deploy_url="https://autollm.llm4kmu.de",
    api_url="https://autollm.llm4kmu.de",
)
```


## Database

```
reflex db init
reflex db makemigrations
reflex db migrate
```