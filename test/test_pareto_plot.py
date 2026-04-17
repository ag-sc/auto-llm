import numpy as np

from auto_llm.evaluator.plots.pareto_plot import compute_pareto_indices


def test_basic_frontier():
    energies = np.array([10.0, 20.0, 15.0, 30.0])
    accuracies = np.array([60.0, 80.0, 70.0, 75.0])
    assert compute_pareto_indices(energies, accuracies) == [0, 2, 1]


def test_tie_energy_prefers_higher_accuracy():
    # Two runs at identical energy — only the higher-accuracy one should
    # land on the frontier, regardless of the input ordering.
    energies = np.array([10.0, 10.0])
    accuracies_low_first = np.array([70.0, 85.0])
    accuracies_high_first = np.array([85.0, 70.0])

    assert compute_pareto_indices(energies, accuracies_low_first) == [1]
    assert compute_pareto_indices(energies, accuracies_high_first) == [0]


def test_single_point():
    assert compute_pareto_indices(np.array([5.0]), np.array([50.0])) == [0]


def test_dominated_points_dropped():
    energies = np.array([5.0, 10.0, 15.0])
    accuracies = np.array([90.0, 80.0, 70.0])
    assert compute_pareto_indices(energies, accuracies) == [0]
