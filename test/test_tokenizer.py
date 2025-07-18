import os
from typing import Dict, List

import pytest
from transformers import AutoTokenizer

from auto_llm.builder.sft_data_builder import preprocess_fn


@pytest.fixture
def examples() -> Dict[str, List[str]]:
    return {
        "instruction": [
            "this is instruction 1",
        ],
        "input": [
            "input text 1",
        ],
        "expected_output": [
            "output text 1",
        ],
    }


def test_tokenize_base_model(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2-2b",
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=False,
        apply_chat_template=False,
        truncation=False,
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=True,
        apply_chat_template=False,
        truncation=False,
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=True,
        apply_chat_template=False,
        truncation=True,
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=10,
        only_completion_loss=False,
        apply_chat_template=False,
        truncation=True,
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248]
    assert encoded_inputs["labels"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=10,
        only_completion_loss=True,
        apply_chat_template=False,
        truncation=True,
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on


def test_tokenize_instruct_model(examples):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2-2b-it",
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=False,
        apply_chat_template=True,
        truncation=False,
        keep_instruction_with_system_content=False,  # for gemma no system turn
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 107, 108, 106, 2516, 108, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["labels"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 107, 108, 106, 2516, 108, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=True,
        apply_chat_template=True,
        truncation=False,
        keep_instruction_with_system_content=False,  # for gemma no system turn
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 107, 108, 106, 2516, 108,  4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=False,
        apply_chat_template=True,
        truncation=True,
        keep_instruction_with_system_content=False,  # for gemma no system turn
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 107, 108, 106, 2516, 108, 4328]
    assert encoded_inputs["labels"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 107, 108, 106, 2516, 108, 4328]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = preprocess_fn(
        examples=examples,
        tokenizer=tokenizer,
        max_length=20,
        only_completion_loss=True,
        apply_chat_template=True,
        truncation=True,
        keep_instruction_with_system_content=False,  # for gemma no system turn
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 108, 2675, 2793, 235248, 235274, 107, 108, 106, 2516, 108, 4328]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on
