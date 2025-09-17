import os

import datasets


def push_dataset_to_hub(
    dataset_dir: str, dataset_name: str, hf_repo_id: str = "llm-4-kmu"
):
    """
    Helper function to push datasets to HuggingFace Repo.

    :param dataset_dir: local path of the dataset
    :param dataset_name: name of the dataset
    :param hf_repo_id: HuggingFace repo ID

    Example:
    ```python
    from auto_llm.builder.utils import push_dataset_to_hub

    dataset_dir = "/vol/auto_llm/processed_datasets/pico/Covid19"
    dataset_name = "pico_covid19"
    push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)
    ```
    """
    ds_dict = datasets.load_from_disk(dataset_dir)
    ds_dict.push_to_hub(
        repo_id=f"{hf_repo_id}/{dataset_name}",
        token=os.getenv("HF_TOKEN"),
    )
