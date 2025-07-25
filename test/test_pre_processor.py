import os
from typing import Dict, List

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from auto_llm.pre_processor.sft_pre_procesor import SftPreProcessor


@pytest.fixture
def non_conversational_examples() -> Dict[str, List]:
    return {
        "prompt": [
            "this is instruction 1",
        ],
        "completion": [
            "output text 1",
        ],
    }


@pytest.fixture
def conversational_examples() -> Dict[str, List]:
    return {
        "messages": [
            [
                {"role": "user", "content": "this is instruction 1"},
                {"role": "assistant", "content": "output text 1"},
            ],
            [
                {"role": "user", "content": "this is another instruction"},
                {"role": "assistant", "content": "some other output text"},
            ],
        ]
    }


@pytest.fixture
def gemma_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2-2b", token=os.getenv("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


@pytest.fixture
def gemma_instruct_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="google/gemma-2-2b-it",
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def test_pre_processor_not_completion_only_loss(
    gemma_tokenizer, gemma_instruct_tokenizer, non_conversational_examples
):
    # Gemma base tokenizer, Not completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_tokenizer, completion_only_loss=False
    )

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=20, truncation=False
    )

    # fmt: off
    assert len(encoded_inputs["input_ids"][0].tolist()) == 20
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=10, truncation=True
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248]
    assert encoded_inputs["labels"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    # Gemma instruct tokenizer, Not completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_instruct_tokenizer, completion_only_loss=False
    )

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=20, truncation=False
    )

    # fmt: off
    assert len(encoded_inputs["input_ids"][0].tolist()) == 20
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=10, truncation=True
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248]
    assert encoded_inputs["labels"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on


def test_pre_processor_with_completion_only_loss(
    gemma_tokenizer, gemma_instruct_tokenizer, non_conversational_examples
):
    # Gemma base tokenizer, completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_tokenizer, completion_only_loss=True
    )

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=20, truncation=False
    )

    # fmt: off
    assert len(encoded_inputs["input_ids"][0].tolist()) == 20
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=10, truncation=True
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    # Gemma instruct tokenizer, completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_instruct_tokenizer, completion_only_loss=True
    )

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=20, truncation=False
    )

    # fmt: off
    assert len(encoded_inputs["input_ids"][0].tolist()) == 20
    assert encoded_inputs["input_ids"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248, 235274, 1]
    assert encoded_inputs["attention_mask"][0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = pre_processor.pre_process(
        examples=non_conversational_examples, max_length=10, truncation=True
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 883, 603, 14239, 235248, 235274, 108, 4328, 2793, 235248]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on


def test_pre_processor_with_conversational_ds_not_completion_only_loss(
    gemma_tokenizer, gemma_instruct_tokenizer, conversational_examples
):
    # Gemma base tokenizer, Not completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_tokenizer, completion_only_loss=False
    )

    with pytest.raises(Exception):
        pre_processor.pre_process(
            examples=conversational_examples, max_length=20, truncation=False
        )

    # Gemma instruct tokenizer, Not completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_instruct_tokenizer, completion_only_loss=False
    )

    encoded_inputs = pre_processor.pre_process(
        examples=conversational_examples, max_length=20, truncation=False
    )

    # fmt: off
    assert len(encoded_inputs["input_ids"][0].tolist()) == 20
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 107, 108, 106, 2516, 108, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["labels"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 107, 108, 106, 2516, 108, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = pre_processor.pre_process(
        examples=conversational_examples, max_length=10, truncation=True
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 107]
    assert encoded_inputs["labels"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 107]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on


def test_pre_processor_with_conversational_ds_completion_only_loss(
    gemma_tokenizer, gemma_instruct_tokenizer, conversational_examples
):
    # Gemma base tokenizer, Not completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_tokenizer, completion_only_loss=True
    )

    with pytest.raises(Exception):
        pre_processor.pre_process(
            examples=conversational_examples, max_length=20, truncation=False
        )

    # Gemma instruct tokenizer, Not completion only loss
    pre_processor = SftPreProcessor(
        tokenizer=gemma_instruct_tokenizer, completion_only_loss=True
    )

    encoded_inputs = pre_processor.pre_process(
        examples=conversational_examples, max_length=20, truncation=False
    )

    # fmt: off
    assert len(encoded_inputs["input_ids"][0].tolist()) == 20
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 107, 108, 106, 2516, 108, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 4328, 2793, 235248, 235274, 107, 108]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on

    encoded_inputs = pre_processor.pre_process(
        examples=conversational_examples, max_length=10, truncation=True
    )

    # fmt: off
    assert encoded_inputs["input_ids"][0].tolist() == [2, 106, 1645, 108, 883, 603, 14239, 235248, 235274, 107]
    assert encoded_inputs["labels"][0].tolist() == [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
    assert encoded_inputs["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on
