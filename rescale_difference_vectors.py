from __future__ import annotations

import csv
import math
from pathlib import Path


# Edit these in the IDE before running if you want a different search root or filename.
SEARCH_ROOT = Path(r"D:\New folder\serotonin\Model RUNS\Difference Vector Runs")
TARGET_FILENAME = "difference_vectors.csv"
OUTPUT_ROOT = SEARCH_ROOT / "rescaled_difference_vectors"
ZERO_TOLERANCE = 1e-12

MODEL_CHANGE_COLUMN = "model_change"
TARGET_CHANGE_COLUMN = "target_change"
MODEL_VALUE_COLUMN = "model_5ht_seed_fc_z"
TARGET_VALUE_COLUMN = "target_5ht_seed_fc_z"
MODEL_REFERENCE_COLUMN = "reference_model_seed_fc_z"
TARGET_REFERENCE_COLUMN = "reference_target_seed_fc_z"
VALID_COLUMN = "valid"
RANGE_METHOD_DIRECTORY = "valid_range_scaled"
VECTOR_METHOD_DIRECTORY = "vector_normalized"
RANGE_FILE_SUFFIX = "_valid_range_scaled"
VECTOR_FILE_SUFFIX = "_vector_normalized"
VALUE_RANGE_SCALED_SUFFIX = "_valid_range_scaled"
VALUE_VECTOR_SCALED_SUFFIX = "_vector_normalized"
MODEL_RANGE_SCALED_COLUMN = "model_change_valid_range_scaled"
TARGET_RANGE_SCALED_COLUMN = "target_change_valid_range_scaled"
MODEL_VALID_RANGE_COLUMN = "model_valid_value_range"
TARGET_VALID_RANGE_COLUMN = "target_valid_value_range"
MODEL_VECTOR_SCALED_COLUMN = "model_change_vector_normalized"
TARGET_VECTOR_SCALED_COLUMN = "target_change_vector_normalized"


def find_matching_csv_files(root: Path, filename: str) -> list[Path]:
    return sorted(path for path in root.rglob(filename) if path.is_file())


def parse_float(raw_value: str, column_name: str, csv_path: Path, row_number: int) -> float:
    value = raw_value.strip()
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{csv_path} row {row_number}: could not parse {column_name}={raw_value!r} as a float."
        ) from exc


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.12g}"


def parse_valid_flag(raw_value: str) -> bool | None:
    value = raw_value.strip().lower()
    if value == "":
        return None
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def scale_by_denominator(value: float, denominator: float) -> float:
    if math.isnan(value) or math.isnan(denominator):
        return math.nan
    if abs(denominator) <= ZERO_TOLERANCE:
        return math.nan
    return value / abs(denominator)


def compute_value_range(parsed_rows: list[dict[str, float | bool | None]], value_column: str) -> float:
    values: list[float] = []
    for row in parsed_rows:
        is_valid = row[VALID_COLUMN]
        if is_valid is False:
            continue
        value = row[value_column]
        if isinstance(value, float) and not math.isnan(value):
            values.append(value)

    if not values:
        return math.nan

    value_range = max(values) - min(values)
    if value_range <= ZERO_TOLERANCE:
        return math.nan
    return value_range


def compute_vector_norm(parsed_rows: list[dict[str, float | bool | None]], value_column: str) -> float:
    squared_sum = 0.0
    count = 0
    for row in parsed_rows:
        is_valid = row[VALID_COLUMN]
        if is_valid is False:
            continue
        value = row[value_column]
        if isinstance(value, float) and not math.isnan(value):
            squared_sum += value * value
            count += 1

    if count == 0:
        return math.nan

    norm = math.sqrt(squared_sum)
    if norm <= ZERO_TOLERANCE:
        return math.nan
    return norm


def make_norm_column_name(source_column: str) -> str:
    return f"{source_column}_valid_norm"


def build_output_path(csv_path: Path, method_directory: str, file_suffix: str) -> Path:
    relative_parent = csv_path.parent.relative_to(SEARCH_ROOT)
    output_directory = OUTPUT_ROOT / method_directory / relative_parent
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / f"{csv_path.stem}{file_suffix}{csv_path.suffix}"


def load_csv_data(
    csv_path: Path,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, float | bool | None]]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row.")

        fieldnames = list(reader.fieldnames)
        required_columns = [
            MODEL_CHANGE_COLUMN,
            TARGET_CHANGE_COLUMN,
            MODEL_VALUE_COLUMN,
            TARGET_VALUE_COLUMN,
            MODEL_REFERENCE_COLUMN,
            TARGET_REFERENCE_COLUMN,
        ]
        missing_columns = [name for name in required_columns if name not in fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        raw_rows: list[dict[str, str]] = []
        parsed_rows: list[dict[str, float | bool | None]] = []
        for row_number, row in enumerate(reader, start=2):
            model_change = parse_float(row[MODEL_CHANGE_COLUMN], MODEL_CHANGE_COLUMN, csv_path, row_number)
            target_change = parse_float(row[TARGET_CHANGE_COLUMN], TARGET_CHANGE_COLUMN, csv_path, row_number)
            model_value = parse_float(row[MODEL_VALUE_COLUMN], MODEL_VALUE_COLUMN, csv_path, row_number)
            target_value = parse_float(row[TARGET_VALUE_COLUMN], TARGET_VALUE_COLUMN, csv_path, row_number)
            model_reference = parse_float(
                row[MODEL_REFERENCE_COLUMN], MODEL_REFERENCE_COLUMN, csv_path, row_number
            )
            target_reference = parse_float(
                row[TARGET_REFERENCE_COLUMN], TARGET_REFERENCE_COLUMN, csv_path, row_number
            )
            valid_flag = parse_valid_flag(row.get(VALID_COLUMN, ""))

            raw_rows.append(dict(row))
            parsed_rows.append(
                {
                    MODEL_CHANGE_COLUMN: model_change,
                    TARGET_CHANGE_COLUMN: target_change,
                    MODEL_VALUE_COLUMN: model_value,
                    TARGET_VALUE_COLUMN: target_value,
                    MODEL_REFERENCE_COLUMN: model_reference,
                    TARGET_REFERENCE_COLUMN: target_reference,
                    VALID_COLUMN: valid_flag,
                }
            )

    return fieldnames, raw_rows, parsed_rows


def build_output_rows(
    raw_rows: list[dict[str, str]],
    parsed_rows: list[dict[str, float | bool | None]],
    model_change_scaled_column: str,
    target_change_scaled_column: str,
    value_scaled_suffix: str,
    model_denominator_column: str,
    target_denominator_column: str,
    model_denominator: float,
    target_denominator: float,
) -> tuple[list[str], list[dict[str, str]]]:
    model_value_scaled_column = f"{MODEL_VALUE_COLUMN}{value_scaled_suffix}"
    target_value_scaled_column = f"{TARGET_VALUE_COLUMN}{value_scaled_suffix}"
    model_reference_scaled_column = f"{MODEL_REFERENCE_COLUMN}{value_scaled_suffix}"
    target_reference_scaled_column = f"{TARGET_REFERENCE_COLUMN}{value_scaled_suffix}"
    extra_columns = [
        model_change_scaled_column,
        target_change_scaled_column,
        model_denominator_column,
        target_denominator_column,
        model_value_scaled_column,
        target_value_scaled_column,
        model_reference_scaled_column,
        target_reference_scaled_column,
    ]
    rows: list[dict[str, str]] = []
    for row, parsed_row in zip(raw_rows, parsed_rows):
        output_row = dict(row)
        output_row[model_change_scaled_column] = format_float(
            scale_by_denominator(parsed_row[MODEL_CHANGE_COLUMN], model_denominator)
        )
        output_row[target_change_scaled_column] = format_float(
            scale_by_denominator(parsed_row[TARGET_CHANGE_COLUMN], target_denominator)
        )
        output_row[model_value_scaled_column] = format_float(
            scale_by_denominator(parsed_row[MODEL_VALUE_COLUMN], model_denominator)
        )
        output_row[target_value_scaled_column] = format_float(
            scale_by_denominator(parsed_row[TARGET_VALUE_COLUMN], target_denominator)
        )
        output_row[model_reference_scaled_column] = format_float(
            scale_by_denominator(parsed_row[MODEL_REFERENCE_COLUMN], model_denominator)
        )
        output_row[target_reference_scaled_column] = format_float(
            scale_by_denominator(parsed_row[TARGET_REFERENCE_COLUMN], target_denominator)
        )
        output_row[model_denominator_column] = format_float(model_denominator)
        output_row[target_denominator_column] = format_float(target_denominator)
        rows.append(output_row)

    return extra_columns, rows


def write_output_csv(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    extra_columns: list[str],
    method_directory: str,
    file_suffix: str,
) -> Path:
    output_fieldnames = list(fieldnames)
    for extra_column in extra_columns:
        if extra_column not in output_fieldnames:
            output_fieldnames.append(extra_column)

    output_path = build_output_path(csv_path, method_directory, file_suffix)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def build_vector_output_rows(
    raw_rows: list[dict[str, str]],
    parsed_rows: list[dict[str, float | bool | None]],
) -> tuple[list[str], list[dict[str, str]]]:
    source_columns = [
        MODEL_CHANGE_COLUMN,
        TARGET_CHANGE_COLUMN,
        MODEL_VALUE_COLUMN,
        TARGET_VALUE_COLUMN,
        MODEL_REFERENCE_COLUMN,
        TARGET_REFERENCE_COLUMN,
    ]
    norm_by_column = {
        column_name: compute_vector_norm(parsed_rows, column_name) for column_name in source_columns
    }
    scaled_by_column = {
        column_name: f"{column_name}{VALUE_VECTOR_SCALED_SUFFIX}" for column_name in source_columns
    }
    norm_column_names = {
        column_name: make_norm_column_name(column_name) for column_name in source_columns
    }
    extra_columns = [scaled_by_column[column_name] for column_name in source_columns]
    extra_columns.extend(norm_column_names[column_name] for column_name in source_columns)

    rows: list[dict[str, str]] = []
    for row, parsed_row in zip(raw_rows, parsed_rows):
        output_row = dict(row)
        for column_name in source_columns:
            output_row[scaled_by_column[column_name]] = format_float(
                scale_by_denominator(parsed_row[column_name], norm_by_column[column_name])
            )
            output_row[norm_column_names[column_name]] = format_float(norm_by_column[column_name])
        rows.append(output_row)

    return extra_columns, rows


def process_csv(csv_path: Path) -> tuple[Path, Path]:
    fieldnames, raw_rows, parsed_rows = load_csv_data(csv_path)

    model_valid_range = compute_value_range(parsed_rows, MODEL_VALUE_COLUMN)
    target_valid_range = compute_value_range(parsed_rows, TARGET_VALUE_COLUMN)
    range_extra_columns, range_rows = build_output_rows(
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        model_change_scaled_column=MODEL_RANGE_SCALED_COLUMN,
        target_change_scaled_column=TARGET_RANGE_SCALED_COLUMN,
        value_scaled_suffix=VALUE_RANGE_SCALED_SUFFIX,
        model_denominator_column=MODEL_VALID_RANGE_COLUMN,
        target_denominator_column=TARGET_VALID_RANGE_COLUMN,
        model_denominator=model_valid_range,
        target_denominator=target_valid_range,
    )
    range_output_path = write_output_csv(
        csv_path=csv_path,
        fieldnames=fieldnames,
        rows=range_rows,
        extra_columns=range_extra_columns,
        method_directory=RANGE_METHOD_DIRECTORY,
        file_suffix=RANGE_FILE_SUFFIX,
    )

    vector_extra_columns, vector_rows = build_vector_output_rows(
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
    )
    vector_output_path = write_output_csv(
        csv_path=csv_path,
        fieldnames=fieldnames,
        rows=vector_rows,
        extra_columns=vector_extra_columns,
        method_directory=VECTOR_METHOD_DIRECTORY,
        file_suffix=VECTOR_FILE_SUFFIX,
    )

    return range_output_path, vector_output_path


def main() -> None:
    matching_files = find_matching_csv_files(SEARCH_ROOT, TARGET_FILENAME)
    if not matching_files:
        raise FileNotFoundError(
            f"No files named {TARGET_FILENAME!r} were found under {SEARCH_ROOT}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(matching_files)} file(s) named {TARGET_FILENAME!r} under {SEARCH_ROOT}")
    print(f"Writing rescaled files to: {OUTPUT_ROOT}")
    for csv_path in matching_files:
        range_output_path, vector_output_path = process_csv(csv_path)
        print(f"Processed: {csv_path}")
        print(f"Saved valid-range output to: {range_output_path}")
        print(f"Saved vector-normalized output to: {vector_output_path}")


if __name__ == "__main__":
    main()
