import datetime
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

from auto_llm.dto.builder_config import SftDatasetType

RUN_KIND_LABELS = {"pre": "Pre-trained", "fft": "Full fine-tuned", "lora": "LoRA fine-tuned"}

# Metrics where a smaller value is better; everything else is treated as higher-is-better
LOWER_IS_BETTER_METRICS = {"perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte", "ter"}

# Run overview: config values worth showing, as (label, candidate flattened config keys); the first key present wins
RUN_VIEW_CONFIG_KEYS = [
    ("Model", ["auto_llm_trainer_args.model_name", "_name_or_path", "model_name", "model"]),
    ("Epochs", ["num_train_epochs"]),
    ("Learning rate", ["learning_rate"]),
    ("Batch size / device", ["per_device_train_batch_size"]),
    ("Grad. accumulation", ["gradient_accumulation_steps"]),
    ("Max length", ["max_length"]),
    ("LoRA rank", ["peft_config.r", "lora_r"]),
]
# Logged keys not worth a chart or card (throughput, step counters used as the x-axis)
RUN_VIEW_HIDDEN_KEYS = {"runtime", "samples_per_second", "steps_per_second", "epoch", "global_step", "total_flos"}
# Keys shown first, in this order; everything else follows alphabetically
RUN_VIEW_KEY_ORDER = ["loss", "mean_token_accuracy", "accuracy", "learning_rate", "grad_norm"]
# A key needs at least this many logged points to be drawn as a curve; otherwise it is shown as a value
RUN_VIEW_MIN_CURVE_POINTS = 3


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
        print("Getting project details:", self.entity)
        projects = list(self.api.projects(entity=self.entity))
        if not projects:
            return []

        with ThreadPoolExecutor(max_workers=min(len(projects), 8)) as executor:
            return list(executor.map(self._get_single_project_details, projects))

    def _get_single_project_details(self, project) -> Dict[str, Any]:
        # lazy=False fetches the summaries with each page of runs; lazy runs would issue one extra request per run
        runs = self.api.runs(f"{self.entity}/{project.name}", per_page=500, lazy=False)

        # Total runtime (W&B stores runtime in seconds)
        total_seconds = sum(run.summary_metrics.get("_runtime", 0) or 0 for run in runs)

        return {"name": project.name, "num_runs": len(runs), "runtime": round(total_seconds / 3600, 2), "url": f"https://wandb.ai/{self.entity}/{project.name}"}

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

    def get_run_states(self, run_ids: List[str], project_name: str) -> Dict[str, str]:
        """
        Fetches the states of several runs with a single query.
        Runs that have not been created in W&B yet are missing from the result.
        """
        if not run_ids:
            return {}
        runs = self.api.runs(path=f"{self.entity}/{project_name}", filters={"name": {"$in": list(run_ids)}}, per_page=len(run_ids))
        return {run.id: run.state for run in runs}

    def get_eval_runs_of_group(self, group: str, project_name: str):

        fig = None
        explanation = ""
        examples_df = pd.DataFrame()

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
            return "", [], examples_df

        df = pd.DataFrame(results)
        df = df.drop_duplicates(subset="run", keep="last")

        df_melted = df.melt(id_vars=["run"], value_vars=metric_names, var_name="Metric", value_name="Value")

        insights = self.build_metric_insights(df_melted, group=group)

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

        try:
            examples_df = self.get_examples_of_group(group=group, project_name=project_name)
        except Exception as e:
            print(f"Error loading examples: {e}")
        fig = fig.to_html(full_html=False, include_plotlyjs="cdn")

        return fig, insights, examples_df

    @staticmethod
    def describe_run(run_name: str, group: str) -> Dict[str, str]:
        """
        Splits a run name "<pre|fft|lora>_<dataset>_<model>_<dataset_type>" into its model kind and model name.
        """
        kind, _, rest = run_name.partition("_")
        if kind not in RUN_KIND_LABELS:
            return {"kind": "other", "kind_label": run_name, "model": ""}

        dataset_name = group.rpartition("_")[0]
        if dataset_name and rest.startswith(dataset_name + "_"):
            rest = rest[len(dataset_name) + 1 :]
        for dataset_type in SftDatasetType:
            if rest.endswith("_" + dataset_type.value):
                rest = rest[: -len(dataset_type.value) - 1]
                break
        return {"kind": kind, "kind_label": RUN_KIND_LABELS[kind], "model": rest}

    @staticmethod
    def _relative_bar(value: float, best_value: float, lower_is_better: bool) -> int:
        if lower_is_better:
            return int(round(100 * best_value / value)) if value > 0 else 0
        return int(round(100 * value / best_value)) if best_value > 0 else 0

    def build_metric_insights(self, df_melted: pd.DataFrame, group: str) -> List[Dict[str, Any]]:
        """
        Summarizes each metric: every run's score, the best run(s), and how fine-tuning compares to the pre-trained model.
        """
        insights = []
        for metric_key, group_df in df_melted.groupby("Metric"):
            metric_name = metric_key.split("/")[-1]
            # lm-eval appends the filter name ("acc,none"); "none" carries no information
            if metric_name.endswith(",none"):
                metric_name = metric_name[: -len(",none")]
            group_df = group_df.dropna(subset=["Value"])
            if group_df.empty:
                continue

            lower_is_better = metric_name.split(",")[0] in LOWER_IS_BETTER_METRICS
            group_df = group_df.sort_values(by="Value", ascending=lower_is_better)
            best_value = group_df["Value"].iloc[0]

            runs = []
            for _, row in group_df.iterrows():
                is_best = abs(row["Value"] - best_value) <= 1e-9
                runs.append(
                    {
                        **self.describe_run(row["run"], group),
                        "run_name": row["run"],
                        "value": f"{row['Value']:.3f}",
                        # bar length relative to the best score, so the best model always has the full bar
                        "bar": self._relative_bar(row["Value"], best_value, lower_is_better),
                        "is_best": is_best,
                    }
                )

            best_runs = [r for r in runs if r["is_best"]]
            pre_values = group_df[[self.describe_run(r, group)["kind"] == "pre" for r in group_df["run"]]]["Value"]

            if len(best_runs) == len(runs) and len(runs) > 1:
                tone = "gray"
                verdict = f"All {len(runs)} models score the same. Fine-tuning brings no gain on this metric."
            elif all(r["kind"] == "pre" for r in best_runs):
                tone = "amber"
                verdict = "The pre-trained model is best. Fine-tuning does not improve this metric."
            elif any(r["kind"] == "pre" for r in best_runs):
                tone = "gray"
                verdict = "Fine-tuning matches but does not beat the pre-trained model on this metric."
            else:
                tone = "green"
                winners = " and ".join(f"{r['kind_label']}" + (f" ({r['model']})" if r["model"] else "") for r in best_runs)
                verdict = f"{winners} {'are' if len(best_runs) > 1 else 'is'} best"
                if not pre_values.empty:
                    best_pre = pre_values.min() if lower_is_better else pre_values.max()
                    gain = (best_pre - best_value) if lower_is_better else (best_value - best_pre)
                    verdict += f", {gain:.3f} better than the best pre-trained model"
                verdict += "."

            insights.append(
                {
                    "metric": metric_name,
                    "best_value": f"{best_value:.3f}",
                    "direction": "lower is better" if lower_is_better else "higher is better",
                    "runs": runs,
                    "verdict": verdict,
                    "tone": tone,
                }
            )
        return insights

    @staticmethod
    def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        flat = {}
        for key, value in d.items():
            if isinstance(value, dict):
                flat.update(WandbClient._flatten(value, f"{prefix}{key}."))
            else:
                flat[f"{prefix}{key}"] = value
        return flat

    @staticmethod
    def _split_key(key: str):
        """'train/loss' -> ('train', 'loss'). Long task-specific prefixes (eval runs) collapse into 'results'."""
        section, _, name = key.rpartition("/")
        if len(section) > 20:
            section = "results"
        return section, name

    @staticmethod
    def _format_number(value: float) -> str:
        if value == 0 or 1e-3 <= abs(value) < 1e5:
            return f"{value:.4g}"
        return f"{value:.3e}"

    def get_run_view(self, run_id: str, project_name: str) -> Dict[str, Any]:
        """
        Collects everything for the in-app run overview, generically for any run:
        run info and config highlights, keys with few logged points as values, and the rest as curves.
        """
        run = self.api.run(path=f"{self.entity}/{project_name}/{run_id}")

        runtime = int(run.summary.get("_runtime") or 0)
        info = [
            {"label": "Type", "value": run.job_type or "-"},
            {"label": "Created", "value": run.created_at.replace("T", " ").replace("Z", " UTC") if run.created_at else "-"},
            {"label": "Runtime", "value": str(datetime.timedelta(seconds=runtime))},
        ]
        config = self._flatten(run.config)
        for label, candidates in RUN_VIEW_CONFIG_KEYS:
            value = next((config[k] for k in candidates if config.get(k) is not None), None)
            if value is not None:
                info.append({"label": label, "value": str(value)})

        history = run.history(samples=500, pandas=True)
        summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

        def is_shown(key: str) -> bool:
            name = key.rpartition("/")[2]
            is_hidden = any(name == h or name.endswith("_" + h) for h in RUN_VIEW_HIDDEN_KEYS)  # also "train_runtime" etc.
            return not key.startswith("_") and not key.endswith("_stderr") and not is_hidden

        numeric_history = [c for c in history.columns if pd.api.types.is_numeric_dtype(history[c])]
        keys = [k for k in numeric_history if is_shown(k)]
        keys += [k for k in summary if is_shown(k) and k not in keys]  # summary-only values, e.g. final scores
        keys.sort(key=lambda k: (RUN_VIEW_KEY_ORDER.index(k.rpartition("/")[2]) if k.rpartition("/")[2] in RUN_VIEW_KEY_ORDER else len(RUN_VIEW_KEY_ORDER), k))

        x_key = "train/global_step" if "train/global_step" in history.columns else "_step"
        curves, stats = [], []
        for key in keys:
            points = history[[x_key, key]].dropna() if key in history.columns and x_key in history.columns else pd.DataFrame()
            section, name = self._split_key(key)
            if len(points) >= RUN_VIEW_MIN_CURVE_POINTS:
                curves.append((section, name, key, points))
                continue
            value = summary.get(key, points[key].iloc[-1] if len(points) else None)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            stderr = summary.get(f"{key}_stderr")
            stats.append(
                {
                    "section": section,
                    "name": name,
                    "value": self._format_number(value),
                    "stderr": f"± {self._format_number(stderr)}" if isinstance(stderr, (int, float)) and not math.isnan(stderr) else "",
                }
            )

        fig = None
        if curves:
            cols = 2 if len(curves) > 1 else 1
            rows = math.ceil(len(curves) / cols)
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"{section}/{name}" if section else name for section, name, _, _ in curves],
                horizontal_spacing=0.08,
                vertical_spacing=min(0.12, 0.6 / rows),
            )
            for idx, (_, name, key, points) in enumerate(curves):
                fig.add_trace(
                    go.Scatter(x=points[x_key], y=points[key], mode="lines+markers", name=name, marker=dict(size=4)),
                    row=idx // cols + 1,
                    col=idx % cols + 1,
                )
            fig.update_layout(
                template="plotly_white",
                showlegend=False,
                height=230 * rows,
                margin=dict(t=30, l=40, r=10, b=30),
                font=dict(size=10),
            )
            fig.update_annotations(font_size=11)
            fig.update_xaxes(title_text="step" if x_key == "_step" else "global step", title_font_size=9)

        return {"url": run.url, "state": run.state, "info": info, "stats": stats, "figure": fig}

    def get_examples_of_group(self, group: str, project_name: str):
        runs = self.api.runs(path=f"{self.entity}/{project_name}")

        examples_df = []
        for run in runs:
            if run.job_type != "evaluation":
                continue

            if run.group != group:
                continue

            examples_key = None
            for key in run.summary.keys():
                if "_eval_results" in key:
                    examples_key = key
                    break

            if examples_key is None:
                continue

            table_artifact = None
            for artifact in run.logged_artifacts():
                if examples_key in artifact.name:
                    table_artifact = artifact
                    break

            if table_artifact is None:
                continue

            table = table_artifact.get(examples_key)
            eval_results_df = table.get_dataframe().head(5)
            eval_results_df["run_name"] = run.name
            examples_df.append(eval_results_df)

        if not examples_df:
            return pd.DataFrame()

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
