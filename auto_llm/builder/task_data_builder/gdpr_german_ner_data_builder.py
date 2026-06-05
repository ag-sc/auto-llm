from datasets import DatasetDict, Dataset, Features, Value, Sequence
from collections import OrderedDict

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures

keys = [
    "PD",
    "P",
    "RP",
    "DC",
    "NPD",
    "TP",
    "CONS",
    "TM",
    "R",
    "DS",
    "LB",
    "DSO",
    "OM",
    "LI",
    "RET",
    "SNEU",
    "RI",
    "DP",
    "CONT",
    "A",
    "ADM",
    "SEU",
    "DSR17",
    "DSR15",
    "DPO",
    "DSR16",
    "DSR21",
    "NRP",
    "DSR18",
    "LC",
    "DSR20",
    "DSR19",
    "DSR22",
]

feature_keys = {key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None) for key in keys}


# features define the schema of the dataset. If not passed, the order of keys would change.
NER_FEATURES = Features({TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None), TaskDatasetFeatures.OUTPUT_TEXT: feature_keys})


class GdprGermanNerDataBuilder(TaskDataBuilder):
    """
    Data from https://huggingface.co/datasets/PaDaS-Lab/gdpr-compliant-ner
    Save conll file from https://huggingface.co/datasets/PaDaS-Lab/gdpr-compliant-ner as "gdpr-compliant-ner.conll"
    in the path /vol/auto_llm/raw_datasets before running the builder.
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        with open("/vol/auto_llm/raw_datasets/gdpr-compliant-ner.conll", "r") as f:
            data = f.readlines()

        new_data = []
        for x in data:
            doc_start = "-DOCSTART-\n"
            new_data.append(x)

            if ".\tO\n" in x:
                new_data.append(doc_start)

        samples = self.parse_data(data=new_data)
        ds = Dataset.from_dict(samples, features=NER_FEATURES)
        ds_dict = self.split_ds(ds=ds)
        print(ds_dict)

        return ds_dict

    def parse_data(self, data):
        samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }
        texts = []
        entities = []
        for line in data:
            if "-DOCSTART-" in line:
                inp_text = " ".join(texts)

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
                    entity_text = " ".join(texts[form["start_idx"] : form["stop_idx"]])

                    if entity_text not in extracted_entities[form["entity_key"]]:
                        extracted_entities[form["entity_key"]].append(entity_text)

                samples[TaskDatasetFeatures.OUTPUT_TEXT].append(extracted_entities)

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
    builder = GdprGermanNerDataBuilder()
    ds_dict = builder.build()

    print(ds_dict["TRAIN"].select(range(5))[0])

    # repo_id = "llm-4-kmu/german-ler-ner"  # f"{hf_repo_id}/{dataset_name}"
    # ds_dict.push_to_hub(
    #     repo_id=repo_id,
    #     token=os.getenv("HF_TOKEN"),
    # )
