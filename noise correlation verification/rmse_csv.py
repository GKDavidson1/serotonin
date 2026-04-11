import csv
import math
from pathlib import Path


def load_numeric_csv(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                rows.append([float(value.strip()) for value in row])
            except ValueError as exc:
                raise ValueError(
                    f"{path} contains a non-numeric value on row {row_index}."
                ) from exc

    if not rows:
        raise ValueError(f"{path} does not contain any numeric data.")

    width = len(rows[0])
    for row_index, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(
                f"{path} has inconsistent column counts on row {row_index}."
            )

    return rows


def compute_rmse(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right):
        raise ValueError("CSV files must have the same number of rows.")

    if len(left[0]) != len(right[0]):
        raise ValueError("CSV files must have the same number of columns.")

    squared_error_sum = 0.0
    value_count = 0

    for row_index, (left_row, right_row) in enumerate(zip(left, right), start=1):
        if len(left_row) != len(right_row):
            raise ValueError(f"Row {row_index} has a different number of columns.")

        for left_value, right_value in zip(left_row, right_row):
            difference = left_value - right_value
            squared_error_sum += difference * difference
            value_count += 1

    return math.sqrt(squared_error_sum / value_count)


def main() -> None:
    csv_a = Path("noise.csv")
    csv_b = Path("noise 2.csv")

    left = load_numeric_csv(csv_a)
    right = load_numeric_csv(csv_b)
    rmse = compute_rmse(left, right)
    print(f"RMSE: {rmse}")


if __name__ == "__main__":
    main()
