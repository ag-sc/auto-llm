from typing import Dict, List, Optional

from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit


class EbmPicoDataBuilder(TaskDataBuilder):
    """
    Constructs EBM PICO data from HF dataset bigbio/ebm_pico.
    """

    def __init__(self, val_ratio: float = 0.1, splits: Optional[List[str]] = None):
        self.val_ratio = val_ratio
        self.splits = splits

    def build(self) -> DatasetDict:
        raw = load_dataset("bigbio/ebm_pico", name="ebm_pico_bigbio_kb",trust_remote_code=True)
        splits_to_process = self.splits or list(raw.keys())
        processed = {}

        # features define the schema of the dataset. If not passed, the order of PICO keys would change.
        features = Features(
            {
                TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
                TaskDatasetFeatures.OUTPUT_TEXT: {
                    "P": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                    "I": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                    "O": Sequence(
                        feature=Value(dtype="string", id=None), length=-1, id=None
                    ),
                },
            }
        )
        for split in splits_to_process:
            ds = raw[split]
            self.logger.info(f"Processing split '{split}' with {len(ds)} samples...")

            processed_dict = {
                TaskDatasetFeatures.INPUT_TEXT: [],
                TaskDatasetFeatures.OUTPUT_TEXT: [],
            }
            for sample in ds:
                out = self._process_sample(sample)
                processed_dict[TaskDatasetFeatures.INPUT_TEXT].append(
                    out[TaskDatasetFeatures.INPUT_TEXT]
                )
                processed_dict[TaskDatasetFeatures.OUTPUT_TEXT].append(
                    out[TaskDatasetFeatures.OUTPUT_TEXT]
                )

            new_ds = Dataset.from_dict(processed_dict, features=features)

            if split == DatasetSplit.TRAIN:
                new_ds = new_ds.shuffle(seed=42)
                val_size = int(len(new_ds) * self.val_ratio)
                processed[DatasetSplit.VALIDATION] = new_ds.select(range(val_size))
                processed[DatasetSplit.TRAIN] = new_ds.select(
                    range(val_size, len(new_ds))
                )
            else:
                processed[split] = new_ds

        return DatasetDict(processed)

    @staticmethod
    def _extract_entities(entities: List[dict]) -> Dict[str, List[str]]:
        buckets = {"P": [], "I": [], "O": []}
        for ent in entities or []:
            ent_type = ent.get("type", "").lower()
            ent_text = " ".join(ent.get("text") or []).strip()
            if not ent_text:
                continue

            if ent_type.startswith("participant"):
                buckets["P"].append(ent_text)
            elif ent_type.startswith("intervention"):
                buckets["I"].append(ent_text)
            elif ent_type.startswith("outcome"):
                buckets["O"].append(ent_text)

        return buckets

    def _process_sample(self, sample: dict) -> dict:
        passages = sample.get("passages", [])
        text_parts = []
        for p in passages:
            text_parts.extend(p.get("text", []))
        input_text = " ".join(text_parts).strip()

        entities = self._extract_entities(sample.get("entities", []))

        return {
            TaskDatasetFeatures.INPUT_TEXT: input_text,
            TaskDatasetFeatures.OUTPUT_TEXT: entities,
        }
