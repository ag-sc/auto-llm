import os

from datasets import DatasetDict, Dataset


class ADPicoDataBuilder:
    def __init__(self, path: str):
        self.path = path

    def build(self) -> DatasetDict:
        train_samples = {"text": [], "entities": []}
        dev_samples = {"text": [], "entities": []}
        test_samples = {"text": [], "entities": []}

        for subdir, dirs, files in os.walk(self.path):
            if not len(dirs):
                for file in files:
                    data_path = os.path.join(subdir, file)
                    print("Checking", data_path)
                    samples = self.construct_pico_data(data_path=data_path)
                    if "train" in file:
                        train_samples["text"].extend(samples["text"])
                        train_samples["entities"].extend(samples["entities"])
                    elif "dev" in file:
                        dev_samples["text"].extend(samples["text"])
                        dev_samples["entities"].extend(samples["entities"])
                    elif "test" in file:
                        test_samples["text"].extend(samples["text"])
                        test_samples["entities"].extend(samples["entities"])

        train_ds = Dataset.from_dict(train_samples)
        dev_ds = Dataset.from_dict(dev_samples)
        test_ds = Dataset.from_dict(test_samples)

        ds_dict = DatasetDict({"train": train_ds, "dev": dev_ds, "test": test_ds})
        return ds_dict

    def save(self, ds: DatasetDict, path: str):
        ds.save_to_disk(dataset_dict_path=path)
        print(f"Saved to {path}")

    @staticmethod
    def read_data_file(data_path: str):
        with open(data_path, "r") as f:
            data = f.readlines()

        return data

    def construct_pico_data(self, data_path: str):
        data = self.read_data_file(data_path)
        samples = {"text": [], "entities": []}
        texts = []
        entities = []
        for line in data:
            if "-DOCSTART-" in line:
                if not len(texts) > 1:
                    continue
                samples["text"].append(" ".join(texts))
                # samples["texts"].append(texts)
                # samples["entities"].append(entities)

                entities_form = []
                for idx, entity in enumerate(entities):
                    if "B-" in entity:
                        entity_key = entity.split("-")[-1]
                        start_idx = idx
                        stop_idx = idx
                        for next_idx in range(idx + 1, len(entities)):
                            if entities[next_idx] == "O":
                                stop_idx = next_idx
                                break
                        idx = stop_idx
                        entities_form.append(
                            {
                                "entity_key": entity_key,
                                "start_idx": start_idx,
                                "stop_idx": stop_idx,
                            }
                        )

                extracted_entities = {"P": [], "I": [], "C": [], "O": []}
                for form in entities_form:
                    entity_text = " ".join(texts[form["start_idx"] : form["stop_idx"]])

                    if entity_text not in extracted_entities[form["entity_key"]]:
                        extracted_entities[form["entity_key"]].append(entity_text)

                samples["entities"].append(extracted_entities)

                texts = []
                entities = []

            else:
                line = line.strip()
                if line:
                    sp = line.split("\t")
                    texts.append(sp[0])
                    try:
                        entities.append(sp[1])
                    except IndexError:
                        entities.append("-NA-")
        return samples


if __name__ == "__main__":
    path = "/homes/vsudhi/llm4kmu_datasets/section_specific_annotation_of_PICO/data/AD"
    builder = ADPicoDataBuilder(path=path)
    ds_dict = builder.build()

    out_path = (
        "/homes/vsudhi/llm4kmu_datasets/section_specific_annotation_of_PICO/data/AD/ds"
    )
    builder.save(ds=ds_dict, path=out_path)
    ...
