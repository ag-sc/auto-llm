from datasets import DatasetDict

from auto_llm.builder.task_data_builder.ad_covid_pico_data_builder import (
    AdCovidPicoDataBuilder,
)
from auto_llm.builder.task_data_builder.ebm_pico_data_builder import EbmPicoDataBuilder
from auto_llm.dto.builder_config import DatasetSplit, TaskDatasetFeatures


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
    builder.save(ds_dict=ds_dict, path=output_dir)
