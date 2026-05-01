from datasets import DatasetDict

from auto_llm.builder.task_data_builder.ad_covid_pico_data_builder import (
    AdCovidPicoDataBuilder,
)
from auto_llm.builder.task_data_builder.ebm_pico_data_builder import EbmPicoDataBuilder
from auto_llm.builder.task_data_builder.pubmed_gen_qa_data_builder import (
    PubMedGenQaDataBuilder,
)
from auto_llm.builder.task_data_builder.pubmed_mcqa_data_builder import (
    PubMedMcqaDataBuilder,
)
from auto_llm.builder.task_data_builder.medmcqa_data_builder import (
    MedmcqaDataBuilder,
)
from auto_llm.dto.builder_config import DatasetSplit, TaskDatasetFeatures
from auto_llm.builder.utils import push_dataset_to_hub

from auto_llm.builder.task_data_builder.med_mcqa_data_builder import MedmcqaDataBuilder
from auto_llm.builder.task_data_builder.med_qa_data_builder import MedQaDataBuilder


def _generic_task_data_builder_tests(ds_dict: DatasetDict):
    assert DatasetSplit.TRAIN in ds_dict.keys()
    assert DatasetSplit.TEST in ds_dict.keys()
    assert DatasetSplit.VALIDATION in ds_dict.keys()

    assert TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.TRAIN].column_names
    assert TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.TRAIN].column_names

    assert TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.TEST].column_names
    assert TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.TEST].column_names

    assert (
        TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.VALIDATION].column_names
    )
    assert (
        TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.VALIDATION].column_names
    )

    # check for data contamination. All splits should be unique - they should not have duplicate items.
    all_samples = []
    all_samples.extend(ds_dict[DatasetSplit.TRAIN][TaskDatasetFeatures.INPUT_TEXT])
    all_samples.extend(ds_dict[DatasetSplit.VALIDATION][TaskDatasetFeatures.INPUT_TEXT])

    assert len(all_samples) == len(
        set(all_samples)
    ), "Train and Validation splits should be unique - they should not have duplicate items."

    all_samples = []
    all_samples.extend(ds_dict[DatasetSplit.TRAIN][TaskDatasetFeatures.INPUT_TEXT])
    all_samples.extend(ds_dict[DatasetSplit.TEST][TaskDatasetFeatures.INPUT_TEXT])

    assert len(all_samples) == len(
        set(all_samples)
    ), "Train and Test splits should be unique - they should not have duplicate items."

    all_samples = []
    all_samples.extend(ds_dict[DatasetSplit.VALIDATION][TaskDatasetFeatures.INPUT_TEXT])
    all_samples.extend(ds_dict[DatasetSplit.TEST][TaskDatasetFeatures.INPUT_TEXT])

    assert len(all_samples) == len(set(all_samples))


def test_ad_pico_data_builder():
    raw_data_path = "/vol/auto_llm/raw_datasets"
    builder = AdCovidPicoDataBuilder(raw_data_path=raw_data_path)
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    out_path = "/vol/auto_llm/processed_datasets/pico/AD"
    builder.save(ds_dict=ds_dict, path=out_path)


def test_covid_19_pico_data_builder():
    raw_data_path = "/vol/auto_llm/raw_datasets/COVID-19"
    builder = AdCovidPicoDataBuilder(raw_data_path=raw_data_path)
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    out_path = "/vol/auto_llm/processed_datasets/pico/Covid19"
    builder.save(ds_dict=ds_dict, path=out_path)


def test_ebm_pico_data_builder():
    # pip install datasets==3.6.0
    builder = EbmPicoDataBuilder()
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    output_dir = "/vol/auto_llm/processed_datasets/pico/EBM"
    # builder.save(ds_dict=ds_dict, path=output_dir)


def test_ebm_pico_data_builder_wo_duplicates():
    # pip install datasets==3.6.0
    builder = EbmPicoDataBuilder(keep_duplicate_entities=False)
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    output_dir = "/vol/auto_llm/processed_datasets/pico/EBM-NoDuplicates"
    # builder.save(ds_dict=ds_dict, path=output_dir)


def test_pubmed_gen_qa_data_builder():
    builder = PubMedGenQaDataBuilder()
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    output_dir = "/vol/auto_llm/processed_datasets/qa/pubmed_gen_qa"
    builder.save(ds_dict=ds_dict, path=output_dir)


def test_pubmed_mcqa_data_builder():
    builder = PubMedMcqaDataBuilder()
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    output_dir = "/vol/auto_llm/processed_datasets/qa/pubmed_mcqa"
    builder.save(ds_dict=ds_dict, path=output_dir)

def test_med_qa_data_builder():
    builder = MedQaDataBuilder()
    ds_dict = builder.build()

    _generic_task_data_builder_tests(ds_dict=ds_dict)

    output_dir = "/vol/auto_llm/processed_datasets/qa/med_qa"
    builder.save(ds_dict=ds_dict, path=output_dir)


def test_medmcqa_data_builder():
    builder = MedmcqaDataBuilder()
    ds_dict = builder.build()

    assert DatasetSplit.TRAIN in ds_dict.keys()
    assert DatasetSplit.VALIDATION in ds_dict.keys()

    assert TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.TRAIN].column_names
    assert TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.TRAIN].column_names

    assert (
        TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.VALIDATION].column_names
    )
    assert (
        TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.VALIDATION].column_names
    )

    # check for data contamination. All splits should be unique - they should not have duplicate items.
    all_samples = []
    all_samples.extend(ds_dict[DatasetSplit.TRAIN][TaskDatasetFeatures.INPUT_TEXT])
    all_samples.extend(ds_dict[DatasetSplit.VALIDATION][TaskDatasetFeatures.INPUT_TEXT])

    assert len(all_samples) == len(
        set(all_samples)
    ), "Train and Validation splits should be unique - they should not have duplicate items."

    output_dir = "/vol/auto_llm/processed_datasets/open-medical-llm-benchmark/medmcqa"
    builder.save(ds_dict=ds_dict, path=output_dir)

def test_medmcqa_data_builder():
    builder = MedmcqaDataBuilder()
    ds_dict = builder.build()

    assert DatasetSplit.TRAIN in ds_dict.keys()
    assert DatasetSplit.VALIDATION in ds_dict.keys()

    assert TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.TRAIN].column_names
    assert TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.TRAIN].column_names

    assert (
        TaskDatasetFeatures.INPUT_TEXT in ds_dict[DatasetSplit.VALIDATION].column_names
    )
    assert (
        TaskDatasetFeatures.OUTPUT_TEXT in ds_dict[DatasetSplit.VALIDATION].column_names
    )

    # check for data contamination. All splits should be unique - they should not have duplicate items.
    all_samples = []
    all_samples.extend(ds_dict[DatasetSplit.TRAIN][TaskDatasetFeatures.INPUT_TEXT])
    all_samples.extend(ds_dict[DatasetSplit.VALIDATION][TaskDatasetFeatures.INPUT_TEXT])

    assert len(all_samples) == len(
        set(all_samples)
    ), "Train and Validation splits should be unique - they should not have duplicate items."

    output_dir = "/vol/auto_llm/processed_datasets/open-medical-llm-benchmark/medmcqa"
    builder.save(ds_dict=ds_dict, path=output_dir)


def test_push_dataset_to_hub():
    # dataset_dir = "/vol/auto_llm/processed_datasets/pico/AD"
    # dataset_name = "pico_ad"
    # push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)
    #
    # dataset_dir = "/vol/auto_llm/processed_datasets/pico/Covid19"
    # dataset_name = "pico_covid19"
    # push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)

    # dataset_dir = "/vol/auto_llm/processed_datasets/qa/pubmed_gen_qa"
    # dataset_name = "qa_pubmed_gen_qa"
    # push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)

    # dataset_dir = "/vol/auto_llm/processed_datasets/qa/pubmed_mcqa"
    # dataset_name = "qa_pubmed_mcqa"
    # push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)

    dataset_dir = "/vol/auto_llm/processed_datasets/qa/med_qa"
    dataset_name = "qa_med_qa"
    push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)

    dataset_dir = "/vol/auto_llm/processed_datasets/qa/med_mcqa"
    dataset_name = "qa_med_mcqa"
    push_dataset_to_hub(dataset_dir=dataset_dir, dataset_name=dataset_name)

