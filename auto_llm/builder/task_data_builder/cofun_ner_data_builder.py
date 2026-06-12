import ast
import os

from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit

keys = ["Auslagerung", "Ort", "Software", "Unternehmen"]

NER_FEATURES = Features({
    TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
    TaskDatasetFeatures.OUTPUT_TEXT: {
        key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None)
        for key in keys
    },
})

RAW_DATA_DIR = "/vol/auto_llm/raw_datasets/CO-Fun"
CRF_DIR = f"{RAW_DATA_DIR}/prepared-data-and-code/NER/CRF"


class CofunNerDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/NedaForoutan/CO-Fun
    Clone the repository into /vol/auto_llm/raw_datasets before running the builder:
        git clone https://github.com/NedaForoutan/CO-Fun /vol/auto_llm/raw_datasets/CO-Fun
    German fund prospectus NER dataset. Train/val/test splits are provided by the dataset.
    Applies a latin-1/UTF-8 mojibake fix to tokens.
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        train_docs = self._parse_crf_file(f"{CRF_DIR}/train_set.txt")
        val_docs   = self._parse_crf_file(f"{CRF_DIR}/dev_set.txt")
        test_docs  = self._parse_crf_file(f"{CRF_DIR}/test_set.txt")

        ds_dict = DatasetDict({
            DatasetSplit.TRAIN.value:      Dataset.from_list(self._docs_to_rows(train_docs), features=NER_FEATURES),
            DatasetSplit.VALIDATION.value: Dataset.from_list(self._docs_to_rows(val_docs),   features=NER_FEATURES),
            DatasetSplit.TEST.value:       Dataset.from_list(self._docs_to_rows(test_docs),  features=NER_FEATURES),
        })
        print(ds_dict)
        return ds_dict

    def _parse_crf_file(self, path):
        # Format: Python list-of-tuples per document.
        # train_set.txt ends documents with ')],', dev/test with '],' on its own line.
        documents = []
        current_lines = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("["):
                    current_lines = [stripped]
                else:
                    current_lines.append(stripped)
                if stripped.endswith(")],") or stripped == "],":
                    text = "\n".join(current_lines).rstrip(",")
                    documents.append(ast.literal_eval(text))
                    current_lines = []
        return documents

    @staticmethod
    def _fix_encoding(text):
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    def _docs_to_rows(self, documents):
        rows = []
        for doc in documents:
            tokens     = [self._fix_encoding(t) for t, _, _ in doc]
            bio_labels = [l for _, _, l in doc]
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
    builder = CofunNerDataBuilder()
    ds_dict = builder.build()

    repo_id = "llm-4-kmu/COFun"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
