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
        # print("W&B Client logging with:", entity)
        self.api = wandb.Api(timeout=60)

    def get_entity(self) -> str:
        return self.entity

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
        print("Getting proejct details:", self.entity)
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

        explanation = ""
        examples_df = None

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

        explanation = ""
        grouped = df_melted.groupby("Metric")
        for metric_name, group_df in grouped:
            sorted_group = group_df.sort_values(by="Value", ascending=False)

            metric_name = metric_name.split("/")[-1]

            top_value_raw = sorted_group.head(1)["Value"].values[0]
            top_run_names = group_df[group_df["Value"] == top_value_raw]["run"].tolist()

            print("top_run_names", top_run_names)

            value = round(sorted_group.head(1)["Value"].values[0] * 100, 2)
            value_str = f"{value:.2f}%"

            num_runs = len(group_df)

            # TODO: avoid hard-coding and map run names to model types more robustly
            best_models = []
            for run in top_run_names:
                if "pre" in run:
                    best_models.append("Pre-trained model")
                elif "fft" in run:
                    best_models.append("Full Weights fine-tuned model")
                elif "lora" in run:
                    best_models.append("LoRa fine-tuned model")
                else:
                    best_models.append(run)

            if len(best_models) == num_runs:
                explanation += f"* All models perform similarly for the metric `{metric_name}` with a value of ``{value_str}%``. This implies there is no particular gain from fine-tuning for this task for this metric.\n"

            if len(best_models) == 1:
                explanation += f"* Out of the `{num_runs}` evaluation run(s), the ``{best_models[0]}`` ({top_run_names[0]}) yields the best result for the metric `{metric_name}` with a value of ``{value}%``.\n"

            if len(best_models) > 1:
                explanation += f"* Out of the `{num_runs}` evaluation run(s), ``{', '.join(best_models)}`` ({', '.join(top_run_names)}) yield the best result for the metric `{metric_name}` with a value of ``{value}%``.\n"

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
                "run": "Legend",
            },
            template="plotly_white",
        )

        # Update text labels to show values on top of each bar
        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside", textfont=dict(size=9))

        num_metrics = len(df_melted["Metric"].unique())
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.5)",
            ),
            margin=dict(t=10, l=40, r=120, b=10),
            # height=500,
            height=300,
            # autosize=True,
            bargap=0.15,  # Spacing between groups of bars
            bargroupgap=0.1,  # Spacing between individual bars within a group
            yaxis=dict(
                range=[0, df_melted["Value"].max() * 1.2],
                tickfont=dict(size=9),
            ),
            xaxis=dict(
                domain=[0.2, 0.8] if num_metrics == 1 else [0, 1],
                tickmode="array",
                tickvals=metric_names,
                ticktext=[m.split("/")[-1] for m in metric_names],
                tickfont=dict(size=9),
            ),
        )

        examples_df = self.get_examples_of_group(group=group, project_name=project_name)

        return fig.to_html(full_html=False, include_plotlyjs="cdn"), explanation, examples_df

    def get_examples_of_group(self, group: str, project_name: str):
        runs = self.api.runs(path=f"{self.entity}/{project_name}")

        examples_df = []
        for run in runs:
            if run.job_type != "evaluation":
                continue

            if run.group != group:
                continue

            for key in run.summary.keys():
                if "_eval_results" in key:
                    examples_key = key
                    break

            table_artifact = None
            for artifact in run.logged_artifacts():
                if examples_key in artifact.name:
                    table_artifact = artifact
                    break

            table = table_artifact.get(examples_key)
            eval_results_df = table.get_dataframe().head(5)
            eval_results_df["run_name"] = run.name
            examples_df.append(eval_results_df)

        combined_df = pd.concat(examples_df, axis=0)
        combined_df.columns = [str(col) for col in combined_df.columns]
        combined_df = combined_df.reset_index(drop=True)
        combined_df = combined_df.fillna("")
        return combined_df

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
