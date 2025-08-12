from auto_llm.builder.trainer_data_builder.sft_data_builder import (
    DataBuilder,
    SftDatasetType,
)


def test_data_builder():
    dataset_dir_or_path = "/homes/vsudhi/llm4kmu_datasets/section_specific_annotation_of_PICO/data/AD/processed"
    instruction_template = 'Given the text "Text", extract the PICO tags in the JSON format "Format". Do not modify the sentences.\nFormat:\n```json\n{\n  "P": ["value for P"],\n  "I": ["value for I"],\n  "C": ["value for C"],\n  "O": ["value for O"],\n}\n```\n'

    input_template = (
        f"""{{examples}}Text: {{text}}\nPICO tags according to the format:\n"""
    )

    output_template = f"""
    ```json\n{{entities}}\n```
    """

    # builder = DataBuilder(
    #     dataset_dir_or_path=dataset_dir_or_path,
    #     instruction_template=instruction_template,
    #     input_template=input_template,
    #     output_template=output_template,
    #     dataset_type=SftDatasetType.PROMPT_COMPLETIONS,
    #     instruction_input_separator="\n",
    #     parse_output_as_json=True,
    #     num_few_shot_examples=5,
    #     few_shot_examples_split="dev",
    # )

    builder = DataBuilder(
        dataset_dir_or_path=dataset_dir_or_path,
        instruction_template=instruction_template,
        input_template=input_template,
        output_template=output_template,
        dataset_type=SftDatasetType.CONVERSATIONAL,
        instruction_input_separator="\n",
        use_system_message=True,
        parse_output_as_json=True,
        num_few_shot_examples=5,
        few_shot_examples_split="dev",
    )

    ds_dict = builder.build()
    print(ds_dict)
    ...
