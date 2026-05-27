from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from auto_llm.builder.task_data_builder.task_data_builder import SEED, TaskDataBuilder
from auto_llm.dto.builder_config import DatasetSplit, TaskDatasetFeatures


MEDMCQA_SUBSAMPLE_TOTAL = 20_000
MEDMCQA_TEST_RATIO = 0.1
PUBMEDQA_OVERSAMPLE_FACTOR = 2

MCQ_INSTRUCTION = (
    "Answer the following medical question by selecting the correct option "
    "(A, B, C, or D)."
)
PUBMEDQA_INSTRUCTION = (
    "Answer the following medical research question with yes, no, or maybe "
    "based on the provided abstract."
)

OPTION_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


class MedMixedDataBuilder(TaskDataBuilder):
    """Mixed Open Medical LLM benchmark training corpus.

    Combines MedQA (4-options), MedMCQA (stratified subsample), and PubMedQA
    (`pqa_labeled`, 2x oversampled) into one DatasetDict with unified
    {input_text, output_text} schema. Per-sample task instructions baked into
    `input_text` so the trainer's instruction_template stays generic.
    """

    def build(self) -> DatasetDict:
        medqa_dd = self._build_medqa()
        medmcqa_dd = self._build_medmcqa()
        pubmedqa_dd = self._build_pubmedqa()

        train = self._concat_and_shuffle(
            [medqa_dd[DatasetSplit.TRAIN], medmcqa_dd[DatasetSplit.TRAIN], pubmedqa_dd[DatasetSplit.TRAIN]]
        )
        validation = self._concat_and_shuffle(
            [medqa_dd[DatasetSplit.VALIDATION], medmcqa_dd[DatasetSplit.VALIDATION], pubmedqa_dd[DatasetSplit.VALIDATION]]
        )
        test = self._concat_and_shuffle(
            [medqa_dd[DatasetSplit.TEST], medmcqa_dd[DatasetSplit.TEST], pubmedqa_dd[DatasetSplit.TEST]]
        )

        return DatasetDict(
            {
                DatasetSplit.TRAIN: train,
                DatasetSplit.VALIDATION: validation,
                DatasetSplit.TEST: test,
            }
        )

    @staticmethod
    def _concat_and_shuffle(datasets_list: list[Dataset]) -> Dataset:
        return concatenate_datasets(datasets_list).shuffle(seed=SEED)

    def _build_medqa(self) -> DatasetDict:
        ds_dict = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_4options_source",
            trust_remote_code=True,
        )
        return DatasetDict(
            {
                DatasetSplit.TRAIN: self._format_medqa_split(ds_dict["train"]),
                DatasetSplit.VALIDATION: self._format_medqa_split(ds_dict["validation"]),
                DatasetSplit.TEST: self._format_medqa_split(ds_dict["test"]),
            }
        )

    @staticmethod
    def _format_medqa_split(ds: Dataset) -> Dataset:
        samples = []
        for item in ds:
            options_text = "\n".join(
                [f"{opt['key']}. {opt['value']}" for opt in item["options"]]
            )
            input_text = (
                f"{MCQ_INSTRUCTION}\n"
                f"Question: {item['question']}\n"
                f"Options:\n{options_text}"
            )
            samples.append(
                {
                    TaskDatasetFeatures.INPUT_TEXT: input_text,
                    TaskDatasetFeatures.OUTPUT_TEXT: item["answer_idx"],
                }
            )
        return Dataset.from_list(samples)

    def _build_medmcqa(self) -> DatasetDict:
        ds_dict = load_dataset("openlifescienceai/medmcqa", trust_remote_code=True)
        raw_train = ds_dict["train"]
        raw_val = ds_dict["validation"]

        raw_train_encoded = raw_train.class_encode_column("subject_name")

        subsample_ratio = MEDMCQA_SUBSAMPLE_TOTAL / len(raw_train_encoded)
        subsampled = raw_train_encoded.train_test_split(
            test_size=subsample_ratio,
            stratify_by_column="subject_name",
            seed=SEED,
        )["test"]

        split = subsampled.train_test_split(
            test_size=MEDMCQA_TEST_RATIO,
            stratify_by_column="subject_name",
            seed=SEED,
        )

        train_formatted = self._format_medmcqa_split(split["train"])
        test_formatted = self._format_medmcqa_split(split["test"])
        val_formatted = self._format_medmcqa_split(raw_val)

        return DatasetDict(
            {
                DatasetSplit.TRAIN: train_formatted,
                DatasetSplit.VALIDATION: val_formatted,
                DatasetSplit.TEST: test_formatted,
            }
        )

    @staticmethod
    def _format_medmcqa_split(ds: Dataset) -> Dataset:
        samples = []
        for item in ds:
            input_text = (
                f"{MCQ_INSTRUCTION}\n"
                f"Question: {item['question']}\n"
                f"Options:\n"
                f"A. {item['opa']}\n"
                f"B. {item['opb']}\n"
                f"C. {item['opc']}\n"
                f"D. {item['opd']}"
            )
            samples.append(
                {
                    TaskDatasetFeatures.INPUT_TEXT: input_text,
                    TaskDatasetFeatures.OUTPUT_TEXT: OPTION_MAP[item["cop"]],
                }
            )
        return Dataset.from_list(samples)

    def _build_pubmedqa(self) -> DatasetDict:
        ds_dict = load_dataset(
            "qiaojin/PubMedQA", name="pqa_labeled", trust_remote_code=True
        )
        formatted = self._format_pubmedqa_split(ds_dict["train"])
        split_dd = self.split_ds(ds=formatted)

        oversampled_train = concatenate_datasets(
            [split_dd[DatasetSplit.TRAIN]] * PUBMEDQA_OVERSAMPLE_FACTOR
        )

        return DatasetDict(
            {
                DatasetSplit.TRAIN: oversampled_train,
                DatasetSplit.VALIDATION: split_dd[DatasetSplit.VALIDATION],
                DatasetSplit.TEST: split_dd[DatasetSplit.TEST],
            }
        )

    @staticmethod
    def _format_pubmedqa_split(ds: Dataset) -> Dataset:
        samples = []
        for item in ds:
            context = "\n".join(item["context"]["contexts"])
            input_text = (
                f"{PUBMEDQA_INSTRUCTION}\n"
                f"Abstract: {context}\n"
                f"Question: {item['question']}"
            )
            samples.append(
                {
                    TaskDatasetFeatures.INPUT_TEXT: input_text,
                    TaskDatasetFeatures.OUTPUT_TEXT: item["final_decision"],
                }
            )
        return Dataset.from_list(samples)
