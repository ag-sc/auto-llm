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
        for run in runs:
            if run.job_type != "evaluation":
                continue

            for key in run.summary.keys():
                if "stderr" in key:
                    metric_name = key.replace("_stderr", "")
                    metric_value = run.summary[metric_name]

                    results.append({"run": run.name, metric_name: metric_value})
                    break

        df = pd.DataFrame(results)
        df = df.drop_duplicates(subset="run", keep="last")

        fig = px.bar(
            df,
            x="run",
            y="pubmed_mcqa/acc",
            title="PubMed MCQA Accuracy",
            labels={"run": "Run Configuration", "pubmed_mcqa/acc": "Accuracy"},
            template="plotly_white",
            color="run",
        )

        # 4. Improve layout for long labels
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            margin=dict(t=50, l=50, r=50, b=150),  # Add bottom margin for labels
            height=500,
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
