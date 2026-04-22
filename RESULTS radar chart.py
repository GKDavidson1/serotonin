from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Edit these values before running the script.
CSV_FILE = Path(r"D:\New folder\serotonin\Model RUNS\rescaled_difference_vectors\vector_normalized\SEROTONIN BATCH on run_001_seed_12345_20260408_003506\difference_vectors_vector_normalized.csv")
LABEL_COLUMN = "area_name"
VALUE_COLUMNS = ["model_change_vector_normalized", "target_change_vector_normalized"]
VALID_COLUMN = "valid"
FILTER_VALID_ONLY = True
OUTPUT_FILE: Path | None = None
CHART_TITLE: str | None = None
CSV_ENCODING = "utf-8-sig"
CSV_DELIMITER = ","
OUTPUT_DPI = 300
SHOW_CHART = False
SHOW_GRID = True
HIGHLIGHT_ZERO = True


def validate_columns(
    frame: pd.DataFrame,
    label_column: str,
    value_columns: list[str],
    valid_column: str,
    filter_valid_only: bool,
) -> None:
    required_columns = [label_column, *value_columns]
    if filter_valid_only:
        required_columns.append(valid_column)

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        available = ", ".join(map(str, frame.columns))
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {available}")


def is_valid_flag(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    return normalized in {"valid", "true", "1", "yes", "y", "t"}


def prepare_data(
    frame: pd.DataFrame,
    label_column: str,
    value_columns: list[str],
    valid_column: str,
    filter_valid_only: bool,
) -> tuple[list[str], dict[str, list[float]]]:
    selected_columns = [label_column, *value_columns]
    if filter_valid_only:
        selected_columns.append(valid_column)

    data = frame[selected_columns].copy()
    data[label_column] = data[label_column].astype(str).str.strip()
    for value_column in value_columns:
        data[value_column] = pd.to_numeric(data[value_column], errors="coerce")

    data = data.dropna(subset=value_columns)
    data = data[data[label_column] != ""]
    if filter_valid_only:
        data = data[data[valid_column].apply(is_valid_flag)]

    if data.empty:
        raise ValueError("No usable rows remain after filtering the valid flag, empty labels, and non-numeric values.")

    labels = data[label_column].tolist()
    series_by_column = {value_column: data[value_column].tolist() for value_column in value_columns}
    return labels, series_by_column


def default_output_path(csv_path: Path, value_columns: list[str]) -> Path:
    safe_columns = [
        "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value_column)
        for value_column in value_columns
    ]
    return csv_path.with_name(f"{csv_path.stem}_{'_vs_'.join(safe_columns)}_radar.png")


def create_radar_chart(
    labels: list[str],
    series_by_column: dict[str, list[float]],
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
    show_grid: bool,
    highlight_zero: bool,
) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + [angles[0]]
    all_values = [value for values in series_by_column.values() for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    radial_offset = -min_value if min_value < 0 else 0.0
    transformed_series = {
        column_name: [value + radial_offset for value in values]
        for column_name, values in series_by_column.items()
    }

    figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for column_name, values in transformed_series.items():
        closed_values = values + [values[0]]
        axis.plot(closed_angles, closed_values, linewidth=2, label=column_name)
        axis.fill(closed_angles, closed_values, alpha=0.15)

    axis.set_xticks(angles)
    axis.set_xticklabels(labels)
    axis.set_title(title, pad=24)
    axis.grid(show_grid)
    axis.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    transformed_values = [value for values in transformed_series.values() for value in values]
    min_radius = min(transformed_values)
    max_radius = max(transformed_values)
    if min_radius == max_radius:
        padding = abs(max_radius) * 0.1 if max_radius != 0 else 1.0
    else:
        padding = (max_radius - min_radius) * 0.05

    display_min = max(0.0, min_radius - padding)
    display_max = max_radius + padding
    zero_radius = radial_offset
    if highlight_zero:
        display_min = min(display_min, zero_radius)
        display_max = max(display_max, zero_radius)

    axis.set_ylim(display_min, display_max)

    tick_positions = np.linspace(display_min, display_max, num=5)
    tick_labels = [f"{tick - radial_offset:.3g}" for tick in tick_positions]
    axis.set_yticks(tick_positions)
    axis.set_yticklabels(tick_labels)

    if highlight_zero and display_min <= zero_radius <= display_max:
        zero_angles = np.linspace(0, 2 * np.pi, 512)
        axis.plot(zero_angles, np.full_like(zero_angles, zero_radius), color="black", linestyle="--", linewidth=1.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(figure)


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    if not LABEL_COLUMN.strip():
        raise ValueError("LABEL_COLUMN must be set.")
    if len(VALUE_COLUMNS) < 2:
        raise ValueError("VALUE_COLUMNS must contain at least two column names.")
    if any(not column.strip() for column in VALUE_COLUMNS):
        raise ValueError("VALUE_COLUMNS cannot contain blank names.")
    if FILTER_VALID_ONLY and not VALID_COLUMN.strip():
        raise ValueError("VALID_COLUMN must be set when FILTER_VALID_ONLY is enabled.")

    frame = pd.read_csv(CSV_FILE, encoding=CSV_ENCODING, sep=CSV_DELIMITER)
    validate_columns(frame, LABEL_COLUMN, VALUE_COLUMNS, VALID_COLUMN, FILTER_VALID_ONLY)
    labels, series_by_column = prepare_data(frame, LABEL_COLUMN, VALUE_COLUMNS, VALID_COLUMN, FILTER_VALID_ONLY)

    output_path = OUTPUT_FILE or default_output_path(CSV_FILE, VALUE_COLUMNS)
    title = CHART_TITLE or f"{' vs '.join(VALUE_COLUMNS)} by {LABEL_COLUMN}"
    create_radar_chart(
        labels,
        series_by_column,
        title,
        output_path,
        OUTPUT_DPI,
        SHOW_CHART,
        SHOW_GRID,
        HIGHLIGHT_ZERO,
    )

    print(f"Saved radar chart to: {output_path}")


if __name__ == "__main__":
    main()
