import pandas as pd
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

api = wandb.Api()
runs = api.runs("llm4kmu/german-ner-eval")

results = []
for run in tqdm(runs):
    if run.state == "finished":
        config = run.config

        if not config:
            continue

        model_args = config.get("cli_configs", None).get("model_args", None)

        pretrained_model = model_args.get("pretrained")
        pretrained_model = pretrained_model.split("/")[-1]

        lora_model = 0
        peft_path = model_args.get("peft")
        if peft_path:
            if "lora" in peft_path:
                lora_model = 1

        if lora_model:
            model_prefix = "lora_" + pretrained_model
        else:
            model_prefix = pretrained_model

        task_name = ""
        for key, value in run.summary.items():
            if "alias" in key:
                task_name = value
                # print(task_name)

                f1 = run.summary.get(f"{task_name}/f1_score_all_labels")
                exact_match = run.summary.get(f"{task_name}/exact_match_all_labels")
                fuzzy_match = run.summary.get(f"{task_name}/fuzzy_match_all_labels")
                # print(f1)
                results.append(dict(task_name=task_name, model=model_prefix, f1=f1, exact_match=exact_match, fuzzy_match=fuzzy_match))

results_df = pd.DataFrame(results)
print(results_df)

# 1. Filter and melt for F1 score only
df_long = results_df.melt(
    id_vars=["task_name", "model"],
    value_vars=["f1"],
    var_name="metric",
    value_name="score",
)

name_mapping = {
    "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    "Ministral-3-8B-Instruct-2512-BF16": "Ministral 8B",
    "lora_Ministral-3-8B-Instruct-2512-BF16": "Ministral 8B\n(LoRA)",
    "Qwen3.5-9B": "Qwen 3.5 9B",
    "lora_Qwen3.5-9B": "Qwen 3.5 9B\n(LoRA)",
    "gemma-4-12B-it": "Gemma 4 12B",
    "lora_gemma-4-12B-it": "Gemma 4 12B\n(LoRA)",
}

# Apply short names to the dataframe
df_long["short_model"] = df_long["model"].map(name_mapping)

short_model_order = [
    "GPT-4o Mini",
    "Ministral 8B",
    "Ministral 8B\n(LoRA)",
    "Qwen 3.5 9B",
    "Qwen 3.5 9B\n(LoRA)",
    "Gemma 4 12B",
    "Gemma 4 12B\n(LoRA)",
]

color_mapping = {
    "GPT-4o Mini": "#4A5568",
    "Ministral 8B": "#3182CE",
    "Ministral 8B\n(LoRA)": "#3182CE",
    "Qwen 3.5 9B": "#DD6B20",
    "Qwen 3.5 9B\n(LoRA)": "#DD6B20",
    "Gemma 4 12B": "#319795",
    "Gemma 4 12B\n(LoRA)": "#319795",
}

# 5. Create output directory
# output_dir = "evaluation_plots"
# os.makedirs(output_dir, exist_ok=True)


# wandb.init(
#     entity="llm4kmu",
#     project="german-ner-eval",
#     name="f1-evaluation-plots",
#     job_type="visualization",
# )

VISUALIZATION_RUN_ID = "5z5m4rtn"
wandb.init(
    entity="llm4kmu",
    project="german-ner-eval",
    id=VISUALIZATION_RUN_ID,  # Targets the existing run
    resume="must",  # Forces W&B to log to this run or error out if not found
)

# 6. Loop and save plots to your machine
for task, group in df_long.groupby("task_name"):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

    # Generate the barplot (legend set to False to eliminate it entirely)
    sns.barplot(
        data=group,
        x="short_model",
        y="score",
        hue="short_model",
        order=short_model_order,
        palette=color_mapping,
        ax=ax,
        legend=False,
    )

    # Hatch Logic: Match bars securely to their actual X-axis tick labels
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]

    for bar in ax.patches:
        bar_x = bar.get_x() + bar.get_width() / 2.0

        # Guard against empty/unrendered bars throwing errors
        if len(ax.get_xticks()) > 0:
            closest_tick_idx = min(range(len(ax.get_xticks())), key=lambda i: abs(ax.get_xticks()[i] - bar_x))
            corresponding_model = tick_labels[closest_tick_idx]

            if "(LoRA)" in corresponding_model:
                bar.set_hatch("//")
                bar.set_edgecolor("white")
                bar.set_linewidth(1.5)

    # Add text labels on top of the bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=10,
                weight="bold",
            )

    # Labels and Titles
    ax.set_title(f"F1: {task}", fontsize=14, pad=15)
    ax.set_xlabel("Model", fontsize=12, labelpad=10)
    ax.set_ylabel("F1 Score", fontsize=12, labelpad=10)
    ax.set_ylim(0, 1.05)

    # Cleaner x-axis layout
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()

    # Save to machine
    # safe_task_name = task.replace("/", "_").replace(" ", "_")
    # file_path = os.path.join(output_dir, f"{safe_task_name}_f1_clean.png")

    # plt.savefig(file_path, dpi=300, bbox_inches="tight")
    # print(f"Saved: {file_path}")

    wandb.log({f"f1_results/{task}": wandb.Image(fig)}, step=0)
    # wandb.run.summary[f"f1_results/{task}"] = wandb.Image(fig)

    plt.close(fig)
