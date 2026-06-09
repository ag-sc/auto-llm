import datetime
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb


class WandbClient:
    def __init__(self, entity: str = "llm4kmu"):
        """
        Initialize the W&B client.
        :param entity: Your username or team name.
        """
        self.entity = entity
        self.api = wandb.Api()

    def login(self, key: str = None):
        """Authenticates with W&B."""
        try:
            wandb.login(key=key)
            print("Successfully logged into Weights & Biases.")
        except Exception as e:
            print(f"Login failed: {e}")

    def get_project_details(self) -> List[Dict[str, Any]]:
        """
        Fetches all runs from the project and returns them as a list of dictionaries.
        """
        project_details = []
        for project in self.api.projects(entity=self.entity):
            runs = self.api.runs(f"{self.entity}/{project.name}")
            num_runs = runs.length

            # Total runtime (W&B stores runtime in seconds)
            total_seconds = sum(run.summary.get("_wandb", {}).get("runtime", 0) for run in runs)

            project_details.append({"name": project.name, "num_runs": num_runs, "runtime": round(total_seconds / 3600, 2), "url": f"https://wandb.ai/{self.entity}/{project.name}"})

        return project_details

    def get_run_details(self, run_name: str, project_name: str, dt_object: datetime.datetime, user_name: str):
        buffer_time = dt_object - datetime.timedelta(seconds=30)
        iso_timestamp = buffer_time.isoformat()

        filters = {
            "$and": [
                {"display_name": run_name},  # Find run by name
                # {"user": {"$eq": user_name}},  # Filter by user name
                {"created_at": {"$gte": iso_timestamp}},  # Created after threshold
            ]
        }

        runs = self.api.runs(path=f"{self.entity}/{project_name}")  # , filters=filters)
        print("runs", runs)

        run = runs[0]
        return run

    @staticmethod
    def get_run_plot_html(run):
        history = run.history()
        fig = px.line(history, title=f"Loss for {run.name}")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def get_run(self, run_id: str, project_name: str):
        run = self.api.run(path=f"{self.entity}/{project_name}/{run_id}")
        return run

    def get_run_url(self, run_id: str, project_name: str):
        return f"https://wandb.ai/{self.entity}/{project_name}/runs/{run_id}"

    def get_run_state(self, run_id: str, project_name: str):
        run = self.get_run(run_id=run_id, project_name=project_name)
        return run.state

    def get_eval_runs_of_group(self, group: str, project_name: str):
        runs = self.api.runs(path=f"{self.entity}/{project_name}")

        results = []
        metric_names = []
        for run in runs:
            if run.job_type != "evaluation":
                continue

            if run.group != group:
                continue

            result = {}
            for key in run.summary.keys():
                if "stderr" in key:
                    metric_name = key.replace("_stderr", "")
                    if metric_name not in metric_names:
                        metric_names.append(metric_name)

                    metric_value = run.summary[metric_name]

                    result.update({metric_name: metric_value})

            result.update({"run": run.name})
            results.append(result)

        if len(results) == 0:
            return ""

        df = pd.DataFrame(results)
        df = df.drop_duplicates(subset="run", keep="last")

        df_melted = df.melt(id_vars=["run"], value_vars=metric_names, var_name="Metric", value_name="Value")

        # Update the figure to use the melted data
        fig = px.bar(
            df_melted,
            x="Metric",
            y="Value",
            color="run",  # This creates the different bars
            barmode="group",  # This groups them side-by-side
            labels={
                "Metric": "Evaluation Metric",
                "Value": "Score",
                "run": "Run Configuration",
            },
            template="plotly_white",
        )

        # Update text labels to show values on top of each bar
        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")

        num_metrics = len(df_melted["Metric"].unique())
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.5)",
            ),
            yaxis=dict(range=[0, df_melted["Value"].max() * 1.2]),
            margin=dict(t=60, l=60, r=150, b=40),
            # height=500,
            autosize=True,
            bargap=0.15,  # Spacing between groups of bars
            bargroupgap=0.1,  # Spacing between individual bars within a group
            # If only one metric, make the bars occupy less horizontal space
            xaxis=dict(domain=[0.2, 0.8] if num_metrics == 1 else [0, 1], tickmode="array", tickvals=metric_names, ticktext=[m.split("/")[-1] for m in metric_names]),
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def get_loss_plot(self, run_id: str, project_name: str):
        run = self.get_run(run_id=run_id, project_name=project_name)
        df = run.history()

        fig = go.Figure()
        train_df = df.dropna(subset=["train/loss"])
        if not train_df.empty:
            fig.add_trace(go.Scatter(x=train_df["_step"], y=train_df["train/loss"], name="Train Loss", mode="lines+markers"))

        eval_df = df.dropna(subset=["eval/loss"])
        if not eval_df.empty:
            fig.add_trace(go.Scatter(x=eval_df["_step"], y=eval_df["eval/loss"], name="Eval Loss", mode="lines+markers", line=dict(color="red")))

        fig.update_layout(title="Loss Curves", xaxis_title="Step", yaxis_title="Loss")
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
