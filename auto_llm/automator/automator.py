from typing import List

import datasets
import pandas as pd

from huggingface_hub import HfApi
from auto_llm.tasks.sequence_to_sequence_task import SequenceToSequenceTask


class Automator:
    """
    This class is used to automate AutoLLM training and evaluation runs based on the (minimal) user input.

    User Input:
    - task category: Seq2Seq, SeqLabelling, Seq2Label
    - dataset: path in HF or local
    - hardware: type and number of GPUs

    Function sof the class:
    - Model finding: Given task category and hardware details, find model
    - Template building: Given task category, populate templates
    -
    """

    def __init__(
        self, task_type: str, dataset: str, hardware_type: str, hardware_count: int
    ) -> None:
        # TODO: task_type should be of type Task
        self.task_type = task_type

        self.dataset = dataset
        self.hardware_type = hardware_type
        self.hardware_count = hardware_count

        self.model_names = self.get_model_names()

    def get_models_df(self, top_k: int = 5) -> pd.DataFrame:
        ds = datasets.load_dataset("open-llm-leaderboard/contents").get("train")
        df = ds.to_pandas()

        # select only Official Models
        df = df[df["Official Providers"] == True]

        # select only pre-trained and instruction tuned models
        types = ["🟢 pretrained", "💬 chat models (RLHF, DPO, IFT, ...)"]
        df = df[df["Type"].isin(types)]

        # TODO: fix model_param_range based on the architecture passed
        model_param_range = (1, 4)
        df = df[df["#Params (B)"].between(model_param_range[0], model_param_range[1])]

        is_moe = False
        df = df[df["MoE"] == is_moe]

        model_providers = [
            "google",
            "mistralai",
            "tiiuae",
            "ibm",
            "deepseek-ai",
            "microsoft",
            "openai-community",
            "meta-llama",
            "HuggingFaceTB",
            "Qwen",
            "EleutherAI",
            "nvidia",
            "ibm-granite",
        ]
        pattern = "|".join(model_providers)
        df = df[df["Base Model"].str.contains(f"^({pattern})", case=False, na=False)]

        # TODO: for different tasks, sort based on different benchmarks, now considering Average scores
        # sort by the column "Average ⬆️".
        df = df.sort_values(by="IFEval", ascending=False)
        df = df.head(top_k)

        df = df[
            [
                "fullname",
                "#Params (B)",
                "Average ⬆️",
                "CO₂ cost (kg)",
                "IFEval",
                "BBH",
                "GPQA",
                "MUSR",
                "MMLU-PRO",
                "MATH Lvl 5",
            ]
        ]

        return df

    def get_model_names(
        self,
    ) -> List[str]:
        df = self.get_models_df()

        model_names = df["fullname"].tolist()
        return model_names

    @staticmethod
    def get_datasets() -> List[str]:
        api = HfApi()
        datasets = api.list_datasets(author="llm-4-kmu")
        dataset_names = [d.id for d in datasets]
        return dataset_names


if __name__ == "__main__":
    task_type = SequenceToSequenceTask
    dataset = "llm-4-kmu/qa_pubmed_mcqa"
    hardware_type = "NVIDIA H200"
    hardware_count = 1

    Automator.get_datasets()

    automator = Automator(
        task_type=task_type,
        dataset=dataset,
        hardware_type=hardware_type,
        hardware_count=hardware_count,
    )
