import os

import datasets


def push_dataset_to_hub(dataset_dir: str, dataset_name: str):
    ds_dict = datasets.load_from_disk(dataset_dir)
    ds_dict.push_to_hub(
        repo_id=f"llm-4-kmu/{dataset_name}",
        token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    dataset_dir = "/vol/auto_llm/processed_datasets/pico/Covid19"
    dataset_name = "pico_covid19"

    push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)
