"""Plot original model simulations against their SINDy fits."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
EXPERIMENTAL_FILE = PROJECT_ROOT / "pcbi.1007688.s007.xlsx"
EXPERIMENTAL_SHEET = "COLO858-Vem live cell count"


def load_experimental_total(dosage: float) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalize the experimental total-cell measurements."""
    measurements = pd.read_excel(
        EXPERIMENTAL_FILE,
        sheet_name=EXPERIMENTAL_SHEET,
        header=None,
    )
    headers = measurements.iloc[1]
    dosage_label = f"Vem ({dosage:g} µM)"
    columns = [
        index
        for index, value in headers.items()
        if value == dosage_label
    ]
    if not columns:
        raise ValueError(f"No experimental columns found for dosage {dosage:g}.")

    time = pd.to_numeric(measurements.iloc[2:, 0], errors="coerce").to_numpy()
    counts = measurements.iloc[2:, columns].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_rows = ~np.isnan(time) & counts.notna().any(axis=1).to_numpy()
    time = time[valid_rows]
    mean_count = counts.loc[valid_rows].mean(axis=1).to_numpy()
    initial_count = mean_count[0]
    if initial_count <= 0:
        raise ValueError(f"Invalid initial experimental count for dosage {dosage:g}.")
    return time, mean_count / initial_count


def simulate_dosage(dosage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, original-model data, and SINDy-predicted data."""
    parameters = main.PARAMETERS[dosage]
    for name, value in parameters.items():
        setattr(main, name, value)

    time, original_data = main.simulate_system()
    sindy_model = main.fit_sindy_model(time, original_data)
    sindy_data = sindy_model.simulate(
        original_data[0],
        time,
        u=lambda current_time: np.array([current_time]),
    )
    return time, original_data, sindy_data


def plot_dosage(dosage: float) -> None:
    """Save an S/R comparison plot for one drug dosage."""
    time, original_data, sindy_data = simulate_dosage(dosage)
    experimental_time, experimental_total = load_experimental_total(dosage)

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(
        time,
        original_data.sum(axis=1),
        label="Original total (S + R)",
    )
    axes[0].plot(
        time,
        sindy_data.sum(axis=1),
        "--",
        label="SINDy total (S + R)",
    )
    axes[0].scatter(
        experimental_time,
        experimental_total,
        s=10,
        alpha=0.6,
        label="Experimental total",
    )
    axes[0].set_ylabel("Total population")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    labels = ("Sensitive cells (S)", "Resistant cells (R)")
    for index, axis in enumerate(axes[1:]):
        axis.plot(time, original_data[:, index], label="Original model")
        axis.plot(time, sindy_data[:, index], "--", label="SINDy fit")
        axis.set_ylabel(labels[index])
        axis.grid(alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Time")
    figure.suptitle(f"Original model vs. SINDy fit ({dosage:g} dosage)")
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / f"comparison_{dosage:g}.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved {output_path}")


def main_plot() -> None:
    """Generate comparison plots for every configured dosage."""
    for dosage in main.PARAMETERS:
        plot_dosage(dosage)


if __name__ == "__main__":
    main_plot()
