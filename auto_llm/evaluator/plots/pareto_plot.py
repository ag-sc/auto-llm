"""Pareto frontier plot: Fine-Tuning Energy (Wh) vs Benchmark Accuracy (%)."""

from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_pareto_indices(
    energies: np.ndarray,
    accuracies: np.ndarray,
) -> List[int]:
    """Return indices of Pareto-optimal points (lower energy, higher accuracy).

    A point is Pareto-optimal if no other point has both strictly lower energy
    and strictly higher accuracy. Ties in energy resolve toward the higher-
    accuracy point so a dominated tie never lands on the frontier.
    """
    # Secondary key (-accuracies) sorts ties so the best accuracy is seen first.
    sorted_indices = np.lexsort((-accuracies, energies))
    pareto_indices = []
    max_accuracy = -np.inf
    for idx in sorted_indices:
        if accuracies[idx] > max_accuracy:
            pareto_indices.append(idx)
            max_accuracy = accuracies[idx]
    return pareto_indices


_compute_pareto_frontier = compute_pareto_indices


def energy_accuracy_plot(
    results: List[Dict[str, Any]],
    output_path: str = "pareto_frontier.png",
    title: str = "Energy vs Accuracy Pareto Frontier",
    highlight_pareto: bool = True,
    figsize: tuple = (12, 8),
) -> str:
    """Generate a Pareto frontier scatter plot of energy vs accuracy.

    Args:
        results: List of dicts, each with keys:
            - ``model_name`` (str): label for the data point.
            - ``energy_wh`` (float): total fine-tuning energy in Wh.
            - ``accuracy_pct`` (float): benchmark accuracy in %.
        output_path: File path to save the plot image.
        title: Plot title.
        highlight_pareto: Whether to draw the Pareto frontier line.
        figsize: Figure size as (width, height) in inches.

    Returns:
        The path to the saved plot file.
    """
    if not results:
        raise ValueError("results must be a non-empty list")

    names = [r["model_name"] for r in results]
    energies = np.array([r["energy_wh"] for r in results], dtype=float)
    accuracies = np.array([r["accuracy_pct"] for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter all points
    ax.scatter(energies, accuracies, s=100, zorder=5, edgecolors="black", linewidths=0.5)

    # Annotate each point with model name
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (energies[i], accuracies[i]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    # Draw Pareto frontier
    if highlight_pareto and len(results) > 1:
        pareto_idx = _compute_pareto_frontier(energies, accuracies)
        pareto_energies = energies[pareto_idx]
        pareto_accuracies = accuracies[pareto_idx]

        ax.scatter(
            pareto_energies,
            pareto_accuracies,
            s=120,
            zorder=6,
            edgecolors="red",
            facecolors="none",
            linewidths=2,
            label="Pareto optimal",
        )
        ax.step(
            pareto_energies,
            pareto_accuracies,
            where="post",
            color="red",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
        )
        ax.legend(fontsize=10)

    ax.set_xlabel("Total Fine-Tuning Energy (Wh)", fontsize=12)
    ax.set_ylabel("Benchmark Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
