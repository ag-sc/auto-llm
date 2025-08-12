from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset, DatasetDict


class EbmPicoDataBuilder:
    def __init__(self, save_dir: str, splits: Optional[List[str]] = None):
        self.save_dir = Path(save_dir)
        self.splits = splits

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
            text = ent.get("text") or ""
            label = ent.get("annotation_type") or ""
            mapped = self._map_label(label)
            if mapped and text.strip():
                buckets[mapped].append(text.strip())
        return buckets

    def _extract_from_fields(self, sample: dict) -> Dict[str, List[str]]:
        pop = sample.get("population") or []
        intr = sample.get("intervention") or []
        out = sample.get("outcome") or []
        if isinstance(pop, str): pop = [pop]
        if isinstance(intr, str): intr = [intr]
        if isinstance(out, str): out = [out]
        return {
            "Population": pop,
            "Intervention": intr,
            "Outcome": out,
        }

    def _process_sample(self, sample: dict) -> dict:
        text = sample.get("text") or sample.get("document") or ""
        if isinstance(text, list):
            text = " ".join(text)
        if "entities" in sample and isinstance(sample["entities"], list):
            entities = self._extract_from_entities_list(sample["entities"])
        else:
            entities = self._extract_from_fields(sample)
        return {"text": text.strip(), "entities": entities}

    def build(self) -> DatasetDict:
        raw = load_dataset("bigbio/ebm_pico")
        splits_to_process = self.splits or list(raw.keys())
        processed = {}
        for split in splits_to_process:
            ds = raw[split]
            print(f"Processing split '{split}' with {len(ds)} samples...")

            processed_dict = {"text": [], "entities": []}
            for sample in ds:
                out = self._process_sample(sample)
                processed_dict["text"].append(out["text"])
                processed_dict["entities"].append(out["entities"])

            new_ds = Dataset.from_dict(processed_dict)
            processed[split] = new_ds

        return DatasetDict(processed)

    def save(self, ds_dict: DatasetDict):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_ds in ds_dict.items():
            out_path = self.save_dir / f"{split_name}.arrow"
            split_ds.save_to_disk(str(out_path))
            print(f"Saved split '{split_name}' to {out_path}")


if __name__ == "__main__":
    output_dir = "/homes/kczok/auto-llm/auto-llm/data/PICO/ebm"
    builder = EbmPicoDataBuilder(save_dir=output_dir)
    dataset_dict = builder.build()
    builder.save(dataset_dict)
