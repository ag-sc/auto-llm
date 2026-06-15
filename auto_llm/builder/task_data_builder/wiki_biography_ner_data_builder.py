import glob
import os
import random
import xml.etree.ElementTree as ET

from datasets import DatasetDict, Dataset, Features, Value, Sequence

from auto_llm.builder.task_data_builder.task_data_builder import TaskDataBuilder
from auto_llm.dto.builder_config import TaskDatasetFeatures, DatasetSplit

keys = ["loc", "org", "person", "temp", "undef_ne"]

NER_FEATURES = Features({
    TaskDatasetFeatures.INPUT_TEXT: Value(dtype="string", id=None),
    TaskDatasetFeatures.OUTPUT_TEXT: {
        key: Sequence(feature=Value(dtype="string", id=None), length=-1, id=None)
        for key in keys
    },
})

RAW_DATA_DIR = "/vol/auto_llm/raw_datasets/wiki-biography/WikiBiography/mmax"
VALID_TYPES = {"person", "loc", "org", "temp", "undef_ne"}
SEED = 42


class WikiBiographyNerDataBuilder(TaskDataBuilder):
    """
    Data from https://www.h-its.org/wp-content/uploads/2014/12/wiki-biography.tar.gz
    Download and extract into /vol/auto_llm/raw_datasets before running the builder:
        wget https://www.h-its.org/wp-content/uploads/2014/12/wiki-biography.tar.gz \
            -O /vol/auto_llm/raw_datasets/wiki-biography.tar.gz
        tar -xzf /vol/auto_llm/raw_datasets/wiki-biography.tar.gz \
            -C /vol/auto_llm/raw_datasets/
    German Wikipedia biography corpus in MMAX2 XML format.
    NE annotations are automatically generated (not human-annotated).
    Split is done at document level (80/10/10) to avoid sentence-level leakage.
    Some NE type field values are corrupt gender markers (m, f, null, uni) — these are filtered out.
    """

    def __init__(self): ...

    def build(self) -> DatasetDict:
        all_ids = sorted({
            os.path.basename(p).replace("_words.xml", "")
            for p in glob.glob(f"{RAW_DATA_DIR}/bdp/*_words.xml")
        })

        random.seed(SEED)
        doc_ids = list(all_ids)
        random.shuffle(doc_ids)
        n = len(doc_ids)
        train_ids = set(doc_ids[:int(0.8 * n)])
        val_ids   = set(doc_ids[int(0.8 * n):int(0.9 * n)])
        test_ids  = set(doc_ids[int(0.9 * n):])

        train_rows, val_rows, test_rows = [], [], []
        for text_id in all_ids:
            try:
                words, sentences, ne_spans = self._parse_document(text_id)
                rows = self._build_rows(words, sentences, ne_spans)
                if text_id in train_ids:
                    train_rows.extend(rows)
                elif text_id in val_ids:
                    val_rows.extend(rows)
                else:
                    test_rows.extend(rows)
            except Exception:
                pass

        ds_dict = DatasetDict({
            DatasetSplit.TRAIN.value:      Dataset.from_list(train_rows, features=NER_FEATURES),
            DatasetSplit.VALIDATION.value: Dataset.from_list(val_rows,   features=NER_FEATURES),
            DatasetSplit.TEST.value:       Dataset.from_list(test_rows,  features=NER_FEATURES),
        })
        print(ds_dict)
        return ds_dict

    def _parse_document(self, text_id):
        words_path = f"{RAW_DATA_DIR}/bdp/{text_id}_words.xml"
        sent_path  = f"{RAW_DATA_DIR}/markables/{text_id}_sentence_level.xml"
        unit_path  = f"{RAW_DATA_DIR}/markables/{text_id}_unit_level.xml"

        words = {}
        for w in ET.parse(words_path).getroot():
            num = int(w.attrib["id"].split("_")[1])
            words[num] = w.text or ""

        sentences = []
        for m in ET.parse(sent_path).getroot():
            sentences.append(self._parse_span(m.attrib["span"]))
        sentences.sort()

        ne_spans = []
        for m in ET.parse(unit_path).getroot():
            ne_type = m.attrib.get("type", "")
            if ne_type not in VALID_TYPES:
                continue
            start, end = self._parse_span(m.attrib["span"])
            ne_spans.append((start, end, ne_type))

        return words, sentences, ne_spans

    @staticmethod
    def _parse_span(span_str):
        if ".." in span_str:
            a, b = span_str.split("..")
            return int(a.split("_")[1]), int(b.split("_")[1])
        else:
            n = int(span_str.split("_")[1])
            return n, n

    def _build_rows(self, words, sentences, ne_spans):
        rows = []
        for sent_start, sent_end in sentences:
            token_nums = [n for n in range(sent_start, sent_end + 1) if n in words]
            tokens = [words[n] for n in token_nums]
            spans = {key: [] for key in keys}
            for ne_start, ne_end, ne_type in ne_spans:
                if ne_start >= sent_start and ne_end <= sent_end:
                    span_tokens = [words[n] for n in range(ne_start, ne_end + 1) if n in words]
                    if span_tokens:
                        span_text = " ".join(span_tokens)
                        if span_text not in spans[ne_type]:
                            spans[ne_type].append(span_text)
            rows.append({
                TaskDatasetFeatures.INPUT_TEXT: " ".join(tokens),
                TaskDatasetFeatures.OUTPUT_TEXT: spans,
            })
        return rows


if __name__ == "__main__":
    builder = WikiBiographyNerDataBuilder()
    ds_dict = builder.build()

    repo_id = "llm-4-kmu/WikiBiography"
    ds_dict.push_to_hub(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
    )
