import os

from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit

keys = ["ART", "CON", "LOC", "MAT", "PER", "SPE"]

NER_FEATURES = Features({
    TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
    TaskDatasetFeatures.OUTPUT_TEXT: {
        key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None)
        for key in keys
    },
})

RAW_DATA_DIR = "/vol/auto_llm/raw_datasets/Archaeo-NER-data-English-Dutch-German"
FOLD_DIR = f"{RAW_DATA_DIR}/German/5-folds-test-train-val-split/fold1"


class ArchaeoNerDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/alexbrandsen/Archaeo-NER-data-English-Dutch-German
    Clone the repository into /vol/auto_llm/raw_datasets before running the builder:
        git clone https://github.com/alexbrandsen/Archaeo-NER-data-English-Dutch-German \
            /vol/auto_llm/raw_datasets/Archaeo-NER-data-English-Dutch-German
    Uses the German subset, fold 1 of the 5-fold test/train/val split.
    PER = Time Periods (not persons).
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        train_docs = self._parse_bio_file(f"{FOLD_DIR}/train.bio")
        val_docs   = self._parse_bio_file(f"{FOLD_DIR}/val.bio")
        test_docs  = self._parse_bio_file(f"{FOLD_DIR}/test.bio")

        ds_dict = DatasetDict({
            DatasetSplit.TRAIN.value:      Dataset.from_list(self._docs_to_rows(train_docs), features=NER_FEATURES),
            DatasetSplit.VALIDATION.value: Dataset.from_list(self._docs_to_rows(val_docs),   features=NER_FEATURES),
            DatasetSplit.TEST.value:       Dataset.from_list(self._docs_to_rows(test_docs),  features=NER_FEATURES),
        })
        print(ds_dict)
        return ds_dict

    def _parse_bio_file(self, path):
        documents = []
        current_tokens, current_labels = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip() == "":
                    if current_tokens:
                        documents.append((current_tokens, current_labels))
                        current_tokens, current_labels = [], []
                else:
                    parts = line.split()
                    # Handle rare merged lines (e.g. "der DET ODas DET O" = two tokens on one line)
                    if len(parts) == 6:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[2])
                        current_tokens.append(parts[3])
                        current_labels.append(parts[5])
                    elif len(parts) == 3:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[2])
        if current_tokens:
            documents.append((current_tokens, current_labels))
        return documents

    def _docs_to_rows(self, documents):
        rows = []
        for tokens, bio_labels in documents:
            spans = {key: [] for key in keys}
            i = 0
            while i < len(bio_labels):
                lbl = bio_labels[i]
                if lbl.startswith("B-"):
                    base = lbl[2:]
                    j = i + 1
                    while j < len(bio_labels) and bio_labels[j] == f"I-{base}":
                        j += 1
                    span_text = " ".join(tokens[i:j])
                    if span_text not in spans[base]:
                        spans[base].append(span_text)
                    i = j
                else:
                    i += 1
            rows.append({
                TaskDatasetFeatures.INPUT_TEXT: " ".join(tokens),
                TaskDatasetFeatures.OUTPUT_TEXT: spans,
            })
        return rows


if __name__ == "__main__":
    builder = ArchaeoNerDataBuilder()
    ds_dict = builder.build()

    repo_id = "llm-4-kmu/ArchaeoNER"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
