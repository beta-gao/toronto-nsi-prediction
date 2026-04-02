from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


RAW_DATASET_DIR = Path("dataset")
CLEANED_DATASET_DIR = Path("cleaned_dataset")
SUMMARY_FILE = Path("cleaned_dataset_summary.csv")

CRIME_FILES = (
    "Assault_Open_Data_8153726331132086508.csv",
    "Robbery_Open_Data_1445483808884001275.csv",
    "Break_and_Enter_Open_Data_1917561619063639397.csv",
    "Auto_Theft_Open_Data_3636037395679683176.csv",
    "Theft_Over_Open_Data_7952014260427331473.csv",
)

DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass
class CleaningStats:
    file_name: str
    original_rows: int = 0
    duplicate_rows_removed: int = 0
    invalid_report_date_rows_removed: int = 0
    invalid_occ_date_rows_removed: int = 0
    cleaned_rows: int = 0
    trimmed_string_cells: int = 0
    repaired_report_parts: int = 0
    repaired_occ_parts: int = 0
    rows_with_report_parts_repaired: int = 0
    rows_with_occ_parts_repaired: int = 0
    rows_with_invalid_report_components_fixed: int = 0
    rows_with_invalid_occ_components_fixed: int = 0
    zero_coordinate_rows_remaining: int = 0
    nsa_neighbourhood_rows_remaining: int = 0


def parse_datetime(raw_value: str) -> datetime | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None

    for fmt in (DATE_FORMAT, "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw_value, fmt)
        except ValueError:
            continue
    return None


def trim_row_strings(row: dict[str, str]) -> int:
    trimmed_cells = 0
    for key, value in row.items():
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed != value:
                row[key] = trimmed
                trimmed_cells += 1
    return trimmed_cells


def sync_date_parts(row: dict[str, str], prefix: str, parsed_dt: datetime) -> int:
    hour_12 = parsed_dt.strftime("%I").lstrip("0") or "0"
    expected_values = {
        f"{prefix}_YEAR": str(parsed_dt.year),
        f"{prefix}_MONTH": parsed_dt.strftime("%B"),
        f"{prefix}_DAY": str(parsed_dt.day),
        f"{prefix}_DOY": str(parsed_dt.timetuple().tm_yday),
        f"{prefix}_DOW": parsed_dt.strftime("%A"),
        f"{prefix}_DATE": (
            f"{parsed_dt.month}/{parsed_dt.day}/{parsed_dt.year} "
            f"{hour_12}:{parsed_dt.strftime('%M:%S %p')}"
        ),
    }

    repaired = 0
    for column, expected in expected_values.items():
        current = row.get(column, "")
        if current != expected:
            row[column] = expected
            repaired += 1
    return repaired


def is_zero_coordinate_row(row: dict[str, str]) -> bool:
    try:
        return float(row.get("LONG_WGS84", "")) == 0.0 and float(row.get("LAT_WGS84", "")) == 0.0
    except ValueError:
        return False


def has_nsa_neighbourhood(row: dict[str, str]) -> bool:
    return row.get("HOOD_158", "") == "NSA" or row.get("NEIGHBOURHOOD_158", "") == "NSA"


def normalize_object_ids(rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows, start=1):
        if "OBJECTID" in row:
            row["OBJECTID"] = str(index)


def dedupe_key(row: dict[str, str], fieldnames: list[str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in fieldnames if field != "OBJECTID")


def has_invalid_date_components(row: dict[str, str], prefix: str) -> bool:
    try:
        datetime.strptime(
            f"{row.get(f'{prefix}_YEAR', '')} {row.get(f'{prefix}_MONTH', '')} {row.get(f'{prefix}_DAY', '')}",
            "%Y %B %d",
        )
        return False
    except ValueError:
        return True


def clean_file(input_path: Path, output_path: Path) -> CleaningStats:
    stats = CleaningStats(file_name=input_path.name)

    with input_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = list(reader.fieldnames or [])
        cleaned_rows: list[dict[str, str]] = []
        seen_rows: set[tuple[str, ...]] = set()

        for row in reader:
            stats.original_rows += 1
            stats.trimmed_string_cells += trim_row_strings(row)

            report_dt = parse_datetime(row.get("REPORT_DATE", ""))
            if report_dt is None:
                stats.invalid_report_date_rows_removed += 1
                continue

            occ_dt = parse_datetime(row.get("OCC_DATE", ""))
            if occ_dt is None:
                stats.invalid_occ_date_rows_removed += 1
                continue

            had_invalid_report_components = has_invalid_date_components(row, "REPORT")
            had_invalid_occ_components = has_invalid_date_components(row, "OCC")

            repaired_report_cells = sync_date_parts(row, "REPORT", report_dt)
            repaired_occ_cells = sync_date_parts(row, "OCC", occ_dt)

            stats.repaired_report_parts += repaired_report_cells
            stats.repaired_occ_parts += repaired_occ_cells

            if repaired_report_cells:
                stats.rows_with_report_parts_repaired += 1
            if repaired_occ_cells:
                stats.rows_with_occ_parts_repaired += 1
            if had_invalid_report_components:
                stats.rows_with_invalid_report_components_fixed += 1
            if had_invalid_occ_components:
                stats.rows_with_invalid_occ_components_fixed += 1

            row_key = dedupe_key(row, fieldnames)
            if row_key in seen_rows:
                stats.duplicate_rows_removed += 1
                continue

            seen_rows.add(row_key)
            cleaned_rows.append(row)

    normalize_object_ids(cleaned_rows)

    for row in cleaned_rows:
        if is_zero_coordinate_row(row):
            stats.zero_coordinate_rows_remaining += 1
        if has_nsa_neighbourhood(row):
            stats.nsa_neighbourhood_rows_remaining += 1

    stats.cleaned_rows = len(cleaned_rows)

    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    return stats


def write_summary(stats_rows: list[CleaningStats]) -> None:
    with SUMMARY_FILE.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "file_name",
                "original_rows",
                "duplicate_rows_removed",
                "invalid_report_date_rows_removed",
                "invalid_occ_date_rows_removed",
                "cleaned_rows",
                "trimmed_string_cells",
                "repaired_report_parts",
                "repaired_occ_parts",
                "rows_with_report_parts_repaired",
                "rows_with_occ_parts_repaired",
                "rows_with_invalid_report_components_fixed",
                "rows_with_invalid_occ_components_fixed",
                "zero_coordinate_rows_remaining",
                "nsa_neighbourhood_rows_remaining",
            ]
        )
        for stats in stats_rows:
            writer.writerow(
                [
                    stats.file_name,
                    stats.original_rows,
                    stats.duplicate_rows_removed,
                    stats.invalid_report_date_rows_removed,
                    stats.invalid_occ_date_rows_removed,
                    stats.cleaned_rows,
                    stats.trimmed_string_cells,
                    stats.repaired_report_parts,
                    stats.repaired_occ_parts,
                    stats.rows_with_report_parts_repaired,
                    stats.rows_with_occ_parts_repaired,
                    stats.rows_with_invalid_report_components_fixed,
                    stats.rows_with_invalid_occ_components_fixed,
                    stats.zero_coordinate_rows_remaining,
                    stats.nsa_neighbourhood_rows_remaining,
                ]
            )


def main() -> None:
    CLEANED_DATASET_DIR.mkdir(exist_ok=True)

    stats_rows: list[CleaningStats] = []
    for file_name in CRIME_FILES:
        input_path = RAW_DATASET_DIR / file_name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {input_path}")

        output_path = CLEANED_DATASET_DIR / file_name
        stats = clean_file(input_path, output_path)
        stats_rows.append(stats)
        print(
            f"{file_name}: kept {stats.cleaned_rows}/{stats.original_rows}, "
            f"removed_duplicates={stats.duplicate_rows_removed}, "
            f"removed_invalid_report_dates={stats.invalid_report_date_rows_removed}, "
            f"removed_invalid_occ_dates={stats.invalid_occ_date_rows_removed}"
        )

    write_summary(stats_rows)
    print(f"Wrote cleaned files to {CLEANED_DATASET_DIR}")
    print(f"Wrote summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
