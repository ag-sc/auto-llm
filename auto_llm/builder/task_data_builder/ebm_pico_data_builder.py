from typing import Dict, List, Optional

from datasets import load_dataset, Dataset, DatasetDict

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit


class EbmPicoDataBuilder(TaskDataBuilder):
    """
    Constructs EBM PICO data from HF dataset bigbio/ebm_pico.
    """

    def __init__(self, splits: Optional[List[str]] = None):
        self.splits = splits

    def build(self) -> DatasetDict:
        raw = load_dataset("bigbio/ebm_pico", trust_remote_code=True)
        splits_to_process = self.splits or list(raw.keys())
        processed = {}
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

            new_ds = Dataset.from_dict(processed_dict)
            processed[split] = new_ds

        return DatasetDict(processed)

    @staticmethod
    def _map_label(label: str) -> Optional[str]:
        if not label:
            return None
        l = label.lower()
        if "participant" in l or "population" in l or "patient" in l:
            return "Population"
        if "intervention" in l:
            return "Intervention"
        if "outcome" in l:
            return "Outcome"
        return None

    def _extract_from_entities_list(self, ents: List[dict]) -> Dict[str, List[str]]:
        buckets = {"Population": [], "Intervention": [], "Outcome": []}
        for ent in ents or []:
            text = ent.get(TaskDatasetFeatures.INPUT_TEXT) or ""
            label = ent.get("annotation_type") or ""
            mapped = self._map_label(label)
            if mapped and text.strip():
                buckets[mapped].append(text.strip())
        return buckets

    def _extract_from_fields(self, sample: dict) -> Dict[str, List[str]]:
        pop = sample.get("population") or []
        intr = sample.get("intervention") or []
        out = sample.get("outcome") or []
        if isinstance(pop, str):
            pop = [pop]
        if isinstance(intr, str):
            intr = [intr]
        if isinstance(out, str):
            out = [out]
        return {
            "Population": pop,
            "Intervention": intr,
            "Outcome": out,
        }

    def _process_sample(self, sample: dict) -> dict:
        text = (
            sample.get(TaskDatasetFeatures.INPUT_TEXT) or sample.get("document") or ""
        )
        if isinstance(text, list):
            text = " ".join(text)
        if TaskDatasetFeatures.OUTPUT_TEXT in sample and isinstance(
            sample[TaskDatasetFeatures.OUTPUT_TEXT], list
        ):
            entities = self._extract_from_entities_list(
                sample[TaskDatasetFeatures.OUTPUT_TEXT]
            )
        else:
            entities = self._extract_from_fields(sample)
        return {
            TaskDatasetFeatures.INPUT_TEXT: text.strip(),
            TaskDatasetFeatures.OUTPUT_TEXT: entities,
        }
