import os

import xml.etree.ElementTree as ET
from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures

from pathlib import Path

keys = ["PERS", "ORG", "PLACE"]

feature_keys = {key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None) for key in keys}

# features define the schema of the dataset. If not passed, the order of keys would change.
NER_FEATURES = Features({TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None), TaskDatasetFeatures.OUTPUT_TEXT: feature_keys})


class ZurichStateArchiveNerDataBuilder(TaskDataBuilder):
    """
    Data from https://github.com/EHRI/EHRI-NER/blob/main/dataset/iob/de/ehri_de.txt
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:

        # Specify your folder path
        folder_path = Path("/vol/auto_llm/raw_datasets/named-entity-recognition_staatsarchiv/data/training_data")

        # List only files
        files = [f.resolve() for f in folder_path.rglob("*.xml") if f.is_file()]
        print(len(files))

        samples = {
            TaskDatasetFeatures.INPUT_TEXT: [],
            TaskDatasetFeatures.OUTPUT_TEXT: [],
        }

        for file_path in files:
            ner_data = extract_entities_from_xml(file_path=file_path)
            for item in ner_data:
                samples[TaskDatasetFeatures.INPUT_TEXT].append(item["text"])
                samples[TaskDatasetFeatures.OUTPUT_TEXT].append(item["entities"])

        ds = Dataset.from_dict(samples, features=NER_FEATURES)
        ds_dict = self.split_ds(ds=ds)

        return ds_dict


def extract_entities_from_xml(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        xml_content = file.read()

    # Parse the XML content
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    # Define the TEI namespace
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    ner_data = []

    # 1. Find all paragraph (<p>) tags anywhere in the document
    p_elements = root.findall(".//tei:p", ns)

    for i, p_element in enumerate(p_elements, start=1):
        # Extract all text inside this specific paragraph (including nested tags)
        p_text = "".join(p_element.itertext()).strip()

        # Initialize a dictionary for this paragraph's entities
        p_entities = {key: [] for key in keys}

        # 2. Search for entities ONLY inside this specific paragraph element
        for key in keys:
            # We use './' to search relative to the current paragraph
            search_path = f"./tei:{key.lower()}Name"

            for entity in p_element.findall(search_path, ns):
                et = entity.text
                if et:
                    et = et.strip()
                    if et not in p_entities[key]:
                        p_entities[key].append(et)

        # 3. Combine the text and entities into a single structured object
        ner_data.append(
            {
                "text": p_text,
                "entities": p_entities,
            }
        )

    return ner_data


if __name__ == "__main__":
    builder = ZurichStateArchiveNerDataBuilder()
    ds_dict = builder.build()
    print(ds_dict)

    repo_id = "llm-4-kmu/zurich-state-archive-ner"  # f"{hf_repo_id}/{dataset_name}"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
