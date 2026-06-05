import string
from typing import List

from datasets import DatasetDict, Dataset, Features, Value, Sequence
from collections import OrderedDict

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures


class GerPSNerDataBuilder(TaskDataBuilder):
    """
    Data from https://zenodo.org/records/10822682.
    Save the files from https://zenodo.org/records/10822682
    in the path ``/vol/auto_llm/raw_datasets`` before running the builder.
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:

        from pathlib import Path

        # Specify your folder path
        folder_path = Path("/vol/auto_llm/raw_datasets/GerPS-NER_code_dataset/code/GerPS-NER/GerPS-NER/")

        # List only files
        files = [f.resolve() for f in folder_path.iterdir() if f.is_file() and f.name.endswith(".conll")]
        print(len(files))

        data = []
        for file in files:
            with open(file, "r") as f:
                x = f.readlines()
            data.extend(x)

        keys = []
        for x in data:
            k = x.split(" ")[-1].strip()
            k = k.replace("B-", "")
            k = k.replace("I-", "")
            if k not in keys:
                keys.append(k)

        print(keys)
        keys.remove("O")  # remove "O" from keys
        keys.remove("")  # remove empty key
        print(keys)

        # keys = ["Mitwirkender", "Handlungsgrundlage", "Ergebnisempfänger", "Hauptakteur", "Aktion", "Dokument", "Signalwort", "Frist", "Bedingung", "Datenfeld"]
        feature_keys = {key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None) for key in keys}

        # features define the schema of the dataset. If not passed, the order of keys would change.
        NER_FEATURES = Features({TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None), TaskDatasetFeatures.OUTPUT_TEXT: feature_keys})

        doc_start = "-DOCSTART-\n"
        new_data = []
        for x in data:
            new_data.append(x)
            if x == "\n":
                new_data.append(doc_start)

        samples = self.parse_data(data=new_data, keys=keys)
        print("len samples", len(samples.get(TaskDatasetFeatures.INPUT_TEXT, [])))
        ds = Dataset.from_dict(samples, features=NER_FEATURES)
        ds_dict = self.split_ds(ds=ds)

        return ds_dict

    def parse_data(self, data, keys):
        samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        texts = []
        entities = []
        for line in data:
            if "-DOCSTART-" in line:
                inp_text = self._join_tokens(texts)

                if not len(texts) > 1:
                    continue

                if inp_text in samples[TaskDatasetFeatures.INPUT_TEXT]:
                    continue

                samples[TaskDatasetFeatures.INPUT_TEXT].append(inp_text)

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

                extracted_entities = OrderedDict({key: [] for key in keys})
                for form in entities_form:
                    entity_text = self._join_tokens(texts[form["start_idx"] : form["stop_idx"]])
                    # print(form["entity_key"], entity_text)

                    if entity_text not in extracted_entities[form["entity_key"]]:
                        extracted_entities[form["entity_key"]].append(entity_text)

                samples[TaskDatasetFeatures.OUTPUT_TEXT].append(extracted_entities)

                texts = []
                entities = []

            else:
                line = line.strip()
                if line:
                    sp = line.split(" ")
                    # print("line", line)
                    # print("sp[0]", sp[0])
                    texts.append(sp[0])
                    try:
                        entities.append(sp[1])
                    except IndexError:
                        entities.append("-NA-")
        return samples

    def _join_tokens(self, tokens: List[str]) -> str:
        """
        Joins tokens, ensuring punctuation does not have leading spaces.
        """
        punctuation = set(string.punctuation)
        result = []

        for i, token in enumerate(tokens):
            # Add a space if it's not the first token AND it's not punctuation
            if i > 0 and token not in punctuation:
                result.append(" ")
            result.append(token)

        return "".join(result)


if __name__ == "__main__":
    builder = GerPSNerDataBuilder()
    ds_dict = builder.build()

    print(ds_dict)
    # repo_id = "llm-4-kmu/ger-ps-ner"  # f"{hf_repo_id}/{dataset_name}"
    # ds_dict.push_to_hub(
    #     repo_id=repo_id,
    #     token=os.getenv("HF_TOKEN"),
    # )
