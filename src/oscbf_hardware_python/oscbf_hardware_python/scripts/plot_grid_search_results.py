#!/usr/bin/env python3
"""Plot grid-search collision results as heatmaps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def rgb(r: int, g: int, b: int) -> tuple[float, float, float]:
    return (r / 255, g / 255, b / 255)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir / "grid_search_results.csv"
    default_output = script_dir / "grid_search_results.png"

    parser = argparse.ArgumentParser(
        description="Plot OSCBF grid-search collision results."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Path to the input CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Path to save the figure (default: {default_output})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in a window after saving it.",
    )
    return parser.parse_args()


def load_results(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "N": int(row["N"]),
                    "Radius": float(row["Radius"]),
                    "Speed": float(row["Speed"]),
                    "Total_Collisions": float(row["Total_Collisions"]),
                }
            )
    if not rows:
        raise ValueError(f"No data found in {csv_path}")
    return rows


def aggregate_results(
    rows: list[dict[str, float]],
) -> dict[int, dict[tuple[float, float], float]]:
    grouped: dict[int, dict[tuple[float, float], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["N"]][(row["Radius"], row["Speed"])].append(
            row["Total_Collisions"]
        )

    aggregated: dict[int, dict[tuple[float, float], float]] = {}
    for obs_count, entries in grouped.items():
        aggregated[obs_count] = {
            key: float(np.mean(values)) for key, values in entries.items()
        }
    return aggregated


def plot_results(
    aggregated: dict[int, dict[tuple[float, float], float]],
    output_path: Path,
    show_plot: bool,
) -> None:
    obstacle_counts = sorted(aggregated)
    radii = sorted({radius for values in aggregated.values() for radius, _ in values})
    speeds = sorted({speed for values in aggregated.values() for _, speed in values})

    fig, axes = plt.subplots(
        1,
        len(obstacle_counts),
        figsize=(5 * len(obstacle_counts), 4.5),
        constrained_layout=True,
        squeeze=False,
    )

    vmax = max(
        collision_count
        for values in aggregated.values()
        for collision_count in values.values()
    )
    cmap = LinearSegmentedColormap.from_list(
        "collisions",
        [
            rgb(46, 125, 50),    # green
            rgb(249, 224, 127),  # yellow
            rgb(127, 0, 0),      # dark red
        ],
    )

    image = None
    for axis, obs_count in zip(axes[0], obstacle_counts):
        grid = np.full((len(radii), len(speeds)), np.nan)
        for row_idx, radius in enumerate(radii):
            for col_idx, speed in enumerate(speeds):
                key = (radius, speed)
                if key in aggregated[obs_count]:
                    grid[row_idx, col_idx] = aggregated[obs_count][key]

        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=vmax if vmax > 0 else 1,
        )

        axis.set_title(f"N = {obs_count}")
        axis.set_xlabel("Obstacle Speed")
        axis.set_ylabel("Obstacle Radius")
        axis.set_xticks(np.arange(len(speeds)))
        axis.set_xticklabels([f"{speed:.2f}" for speed in speeds])
        axis.set_yticks(np.arange(len(radii)))
        axis.set_yticklabels([f"{radius:.2f}" for radius in radii])

        for row_idx in range(len(radii)):
            for col_idx in range(len(speeds)):
                value = grid[row_idx, col_idx]
                if np.isnan(value):
                    continue
                text_color = "white" if value > 0.5 * max(vmax, 1) else "black"
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[0], shrink=0.95)
        colorbar.set_label("Average Total Collisions")

    fig.suptitle("Grid Search Results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_results(args.csv)
    aggregated = aggregate_results(rows)
    plot_results(aggregated, args.output, args.show)


if __name__ == "__main__":
    main()
