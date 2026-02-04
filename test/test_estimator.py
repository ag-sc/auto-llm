import pytest

from auto_llm.estimator.emission_estimator import EmissionEstimator
from auto_llm.estimator.inference_flops_estimator import InferenceFlopsEstimator
from auto_llm.estimator.runtime_estimator import RuntimeEstimator
from auto_llm.estimator.trainer_flops_estimator import TrainerFlopsEstimator
from auto_llm.estimator.utils import get_model_params, get_gpu_params


@pytest.fixture
def models_meta():
    return get_model_params()


@pytest.fixture
def gpu_params():
    return get_gpu_params()


def test_inference_flops_estimator(models_meta):
    # test for a task already part of lm-eval-harness
    config_path = "config_files/evaluator_configs/truthful_qa_gemma-2-2b.yaml"
    estimator = InferenceFlopsEstimator(
        config_path=config_path, models_meta=models_meta
    )
    flops = estimator.estimate()
    assert flops == 13123076029415424

    # test for a custom task
    config_path = "config_files/evaluator_configs/pico_ad_gemma-2-2b.yaml"
    estimator = InferenceFlopsEstimator(
        config_path=config_path, models_meta=models_meta
    )
    flops = estimator.estimate()
    assert flops == 2248752318382080


def test_trainer_flops_estimator(models_meta):
    config_path = (
        "config_files/trainer_configs/pico/ad/pico_ad_gemma-2-2b-it_conv_few-shot.yaml"
    )
    estimator = TrainerFlopsEstimator(config_path=config_path, models_meta=models_meta)
    flops = estimator.estimate()
    assert flops == 477859851571200


def test_runtime_estimator(models_meta, gpu_params):
    config_path = "config_files/evaluator_configs/pico_ad_gemma-2-2b-sft.yaml"
    gpu_name = list(gpu_params.keys())[0]

    flops_estimator = InferenceFlopsEstimator(
        config_path=config_path, models_meta=models_meta
    )

    gpu_params = get_gpu_params()
    runtime_estimator = RuntimeEstimator(
        flops_estimator=flops_estimator,
        gpu_params=gpu_params,
        gpu_name=gpu_name,
    )

    runtime = runtime_estimator.estimate()

    assert round(runtime, 4) == 1.1363


def test_emission_estimator(models_meta, gpu_params):
    config_path = "config_files/evaluator_configs/pico_ad_gemma-2-2b-sft.yaml"
    gpu_name = list(gpu_params.keys())[0]

    flops_estimator = InferenceFlopsEstimator(
        config_path=config_path, models_meta=models_meta
    )

    runtime_estimator = RuntimeEstimator(
        flops_estimator=flops_estimator,
        gpu_params=gpu_params,
        gpu_name=gpu_name,
    )

    emission_estimator = EmissionEstimator(
        runtime_estimator=runtime_estimator,
        gpu_params=gpu_params,
        gpu_name=gpu_name,
    )

    emissions = emission_estimator.estimate()

    assert round(emissions, 4) == 70.9245
