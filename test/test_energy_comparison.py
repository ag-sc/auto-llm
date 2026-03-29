"""Tests for GPU name resolution and the emission comparator."""

import csv
import json
import os
from unittest.mock import patch, MagicMock

import pytest

from auto_llm.estimator.emission_comparator import EmissionComparator
from auto_llm.estimator.utils import resolve_gpu_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GPU_PARAMS = {
    "NVIDIA A40": {"tflops": 150, "tdp": 300},
    "NVIDIA A100": {"tflops": 312, "tdp": 400},
}


# ---------------------------------------------------------------------------
# resolve_gpu_name
# ---------------------------------------------------------------------------


class TestResolveGpuName:
    """Tests for the hybrid GPU name resolver."""

    def test_explicit_exact_match(self):
        """Explicit name that exactly matches a gpu_params key."""
        result = resolve_gpu_name(gpu_name="NVIDIA A40", gpu_params=SAMPLE_GPU_PARAMS)
        assert result == "NVIDIA A40"

    def test_explicit_case_insensitive(self):
        """Explicit name matches case-insensitively."""
        result = resolve_gpu_name(gpu_name="nvidia a40", gpu_params=SAMPLE_GPU_PARAMS)
        assert result == "NVIDIA A40"

    def test_explicit_no_match_falls_back_to_auto(self):
        """Invalid explicit name falls through to auto-detection."""
        mock_device = MagicMock(return_value="NVIDIA A100-SXM4-80GB")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", mock_device):
            result = resolve_gpu_name(
                gpu_name="NonExistentGPU", gpu_params=SAMPLE_GPU_PARAMS
            )
        assert result == "NVIDIA A100"

    def test_auto_detect_substring_match(self):
        """Auto-detection matches via substring."""
        mock_device = MagicMock(return_value="NVIDIA A40")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", mock_device):
            result = resolve_gpu_name(gpu_name=None, gpu_params=SAMPLE_GPU_PARAMS)
        assert result == "NVIDIA A40"

    def test_auto_detect_no_cuda(self):
        """Returns None when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            result = resolve_gpu_name(gpu_name=None, gpu_params=SAMPLE_GPU_PARAMS)
        assert result is None

    def test_auto_detect_no_match(self):
        """Returns None when device name doesn't match any key."""
        mock_device = MagicMock(return_value="AMD Instinct MI250X")
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_name", mock_device):
            result = resolve_gpu_name(gpu_name=None, gpu_params=SAMPLE_GPU_PARAMS)
        assert result is None

    def test_none_gpu_name_none_gpu_params_no_cache(self):
        """Returns None when gpu_params cannot be loaded."""
        with patch(
            "auto_llm.estimator.utils.get_gpu_params",
            side_effect=FileNotFoundError("no cache"),
        ):
            result = resolve_gpu_name(gpu_name=None, gpu_params=None)
        assert result is None


# ---------------------------------------------------------------------------
# EmissionComparator
# ---------------------------------------------------------------------------


class TestEmissionComparator:
    """Tests for the estimated-vs-actual emission comparator."""

    @staticmethod
    def _make_estimate(**overrides):
        """Build a sample estimate dict with sensible defaults."""
        base = {
            "estimated_flops": 1_000_000,
            "estimated_runtime_s": 100.0,
            "estimated_co2_g": 50.0,
            "gpu_name": "NVIDIA A40",
            "carbon_intensity_g_per_kWh": 328,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        base.update(overrides)
        return base

    @staticmethod
    def _make_actual(**overrides):
        """Build a sample actual-emissions dict (typed, not CSV strings)."""
        base = {
            "duration": 120.0,
            "emissions": 0.060,  # 60 g (CodeCarbon uses kg)
            "energy_consumed": 0.5,
            "gpu_power": 280.0,
            "cpu_power": 100.0,
            "ram_power": 20.0,
            "gpu_model": "NVIDIA A40",
        }
        base.update(overrides)
        return base

    def test_compare_success(self):
        """Happy path: both dicts provided, comparison computed correctly."""
        estimate = self._make_estimate()
        actual = self._make_actual()

        comparator = EmissionComparator(
            estimated_emissions=estimate,
            actual_emissions=actual,
        )
        result = comparator.compare()

        assert result is not None
        # runtime: |100 - 120| / 120 * 100 = 16.67%
        assert round(result["runtime_error_pct"], 2) == 16.67
        # CO₂: |50 - 60| / 60 * 100 = 16.67%
        assert round(result["co2_error_pct"], 2) == 16.67
        # ratio: 100 / 120 = 0.833...
        assert round(result["estimated_vs_actual_runtime_ratio"], 3) == 0.833
        # CO₂ ratio: 50 / 60 = 0.833...
        assert round(result["estimated_vs_actual_co2_ratio"], 3) == 0.833

    def test_compare_missing_estimate(self):
        """Returns None when estimated_emissions is empty."""
        comparator = EmissionComparator(
            estimated_emissions={},
            actual_emissions=self._make_actual(),
        )
        assert comparator.compare() is None

    def test_compare_missing_actual(self):
        """Returns None when actual_emissions is empty."""
        comparator = EmissionComparator(
            estimated_emissions=self._make_estimate(),
            actual_emissions={},
        )
        assert comparator.compare() is None

    def test_compare_zero_actuals(self):
        """Error percentages are None when actuals are zero (no division by zero)."""
        actual = self._make_actual(duration=0.0, emissions=0.0)

        comparator = EmissionComparator(
            estimated_emissions=self._make_estimate(),
            actual_emissions=actual,
        )
        result = comparator.compare()
        assert result is not None
        assert result["runtime_error_pct"] is None
        assert result["co2_error_pct"] is None

    def test_save_comparison(self, tmp_path):
        """save_comparison writes emission_comparison.json correctly."""
        estimate = self._make_estimate()
        actual = self._make_actual()

        comparator = EmissionComparator(
            estimated_emissions=estimate,
            actual_emissions=actual,
        )
        result = comparator.compare()
        assert result is not None

        EmissionComparator.save_comparison(result, str(tmp_path))

        comparison_path = tmp_path / "emission_comparison.json"
        assert comparison_path.exists()
        with open(comparison_path) as f:
            written = json.load(f)
        assert written["runtime_error_pct"] == result["runtime_error_pct"]

    def test_normalize_csv_row(self):
        """normalize_csv_row converts string CSV values to typed floats."""
        csv_row = {
            "duration": "120.0",
            "emissions": "0.060",
            "energy_consumed": "0.5",
            "gpu_power": "280.0",
            "cpu_power": "100.0",
            "ram_power": "20.0",
            "gpu_model": "NVIDIA A40",
        }
        normalised = EmissionComparator.normalize_csv_row(csv_row)
        assert normalised["duration"] == 120.0
        assert normalised["emissions"] == 0.060
        assert normalised["gpu_model"] == "NVIDIA A40"  # non-numeric stays string

    def test_get_wandb_metrics(self):
        """get_wandb_metrics returns None before compare, dict after."""
        comparator = EmissionComparator(
            estimated_emissions=self._make_estimate(),
            actual_emissions=self._make_actual(),
        )
        assert comparator.get_wandb_metrics() is None

        comparator.compare()
        metrics = comparator.get_wandb_metrics()
        assert metrics is not None
        assert "emissions/comparison/runtime_error_pct" in metrics
