from datasets import DatasetDict, Dataset, load_dataset

SEED = 0


class DataBuilder:
    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.3,
        val_size: float = 0.1,
    ):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.val_size = val_size

    def build(self) -> DatasetDict:
        # TODO: add support for other file formats
        # TODO: load dataset from HF Hub

        dataset = load_dataset("json", data_files=self.dataset_path, split="all")
        dataset = self.split_dataset(
            dataset=dataset, test_size=self.test_size, val_size=self.val_size
        )
        return dataset

    @staticmethod
    def split_dataset(
        dataset: Dataset,
        test_size: float,
        val_size: float,
    ) -> DatasetDict:
        # get train and the test+val splits
        ds_dict_sp_one = dataset.train_test_split(test_size=test_size, seed=SEED)

        # split test+val splits to get test and val splits
        ds_dict_sp_two = ds_dict_sp_one["test"].train_test_split(
            test_size=val_size, seed=SEED
        )

        ds_dict = DatasetDict(
            {
                "train": ds_dict_sp_one["train"],
                "test": ds_dict_sp_two["train"],
                "val": ds_dict_sp_two["test"],
            }
        )

        return ds_dict
