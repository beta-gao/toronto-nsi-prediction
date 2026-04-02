from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import median


CLEANED_DATASET_DIR = Path("cleaned_dataset")
OUTPUT_FILE = Path("monthly_neighbourhood_safety_from_cleaned.csv")
COORDINATE_OUTPUT_FILE = Path("neighbourhood_158_coordinates.csv")
MIN_OCC_DATE = date(2014, 1, 1)
EXCLUDED_NEIGHBOURHOODS = {"", "NSA"}

# Weights are defined by the project specification.
CRIME_FILES = {
    "Assault_Open_Data_8153726331132086508.csv": ("Assault", 5),
    "Robbery_Open_Data_1445483808884001275.csv": ("Robbery", 4),
    "Break_and_Enter_Open_Data_1917561619063639397.csv": ("Break and Enter", 3),
    "Auto_Theft_Open_Data_3636037395679683176.csv": ("Auto Theft", 2),
    "Theft_Over_Open_Data_7952014260427331473.csv": ("Theft Over", 1),
}


def parse_occ_date(raw_value: str) -> date | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None

    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw_value, fmt).date()
        except ValueError:
            continue
    return None


def parse_float(raw_value: str) -> float | None:
    try:
        return float(raw_value)
    except ValueError:
        return None


def compute_safety_index(score: int, min_score: int, max_score: int) -> float:
    if max_score == min_score:
        return 0.5
    return 1 - ((score - min_score) / (max_score - min_score))


def load_neighbourhood_coordinates() -> dict[str, tuple[float, float]]:
    longitude_samples: dict[str, list[float]] = defaultdict(list)
    latitude_samples: dict[str, list[float]] = defaultdict(list)

    for file_name in CRIME_FILES:
        file_path = CLEANED_DATASET_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing cleaned dataset file: {file_path}")

        with file_path.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                neighbourhood = (row.get("NEIGHBOURHOOD_158") or "").strip()
                if neighbourhood in EXCLUDED_NEIGHBOURHOODS:
                    continue

                longitude = parse_float((row.get("LONG_WGS84") or "").strip())
                latitude = parse_float((row.get("LAT_WGS84") or "").strip())
                if longitude is None or latitude is None:
                    continue
                if longitude == 0.0 and latitude == 0.0:
                    continue

                longitude_samples[neighbourhood].append(longitude)
                latitude_samples[neighbourhood].append(latitude)

    coordinates: dict[str, tuple[float, float]] = {}
    for neighbourhood in sorted(longitude_samples):
        coordinates[neighbourhood] = (
            median(longitude_samples[neighbourhood]),
            median(latitude_samples[neighbourhood]),
        )
    return coordinates


def write_neighbourhood_coordinates(coordinates: dict[str, tuple[float, float]]) -> None:
    with COORDINATE_OUTPUT_FILE.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["neighbourhood_158", "longitude_wgs84", "latitude_wgs84"])
        for neighbourhood in sorted(coordinates):
            longitude, latitude = coordinates[neighbourhood]
            writer.writerow([neighbourhood, f"{longitude:.8f}", f"{latitude:.8f}"])


def main() -> None:
    monthly_total_scores: dict[tuple[str, str], int] = defaultdict(int)
    monthly_total_incidents: dict[tuple[str, str], int] = defaultdict(int)
    monthly_crime_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for file_name, (crime_type, weight) in CRIME_FILES.items():
        file_path = CLEANED_DATASET_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing cleaned dataset file: {file_path}")

        with file_path.open("r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                occ_date = parse_occ_date(row.get("OCC_DATE", ""))
                neighbourhood = (row.get("NEIGHBOURHOOD_158") or "").strip()

                if occ_date is None or occ_date < MIN_OCC_DATE:
                    continue
                if neighbourhood in EXCLUDED_NEIGHBOURHOODS:
                    continue

                month_key = (occ_date.strftime("%Y-%m"), neighbourhood)
                monthly_total_scores[month_key] += weight
                monthly_total_incidents[month_key] += 1
                monthly_crime_counts[month_key][crime_type] += 1

    if not monthly_total_scores:
        raise ValueError("No monthly neighbourhood scores were generated from cleaned_dataset.")

    coordinates = load_neighbourhood_coordinates()
    write_neighbourhood_coordinates(coordinates)

    all_scores = list(monthly_total_scores.values())
    global_min_score = min(all_scores)
    global_max_score = max(all_scores)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "occ_month",
                "neighbourhood_158",
                "longitude_wgs84",
                "latitude_wgs84",
                "assault_count",
                "robbery_count",
                "break_and_enter_count",
                "auto_theft_count",
                "theft_over_count",
                "incident_count",
                "total_crime_score",
                "global_min_score",
                "global_max_score",
                "safety_index",
            ]
        )

        for month_key in sorted(monthly_total_scores):
            occ_month, neighbourhood = month_key
            crime_counts = monthly_crime_counts[month_key]
            total_crime_score = monthly_total_scores[month_key]
            longitude, latitude = coordinates[neighbourhood]

            writer.writerow(
                [
                    occ_month,
                    neighbourhood,
                    f"{longitude:.8f}",
                    f"{latitude:.8f}",
                    crime_counts["Assault"],
                    crime_counts["Robbery"],
                    crime_counts["Break and Enter"],
                    crime_counts["Auto Theft"],
                    crime_counts["Theft Over"],
                    monthly_total_incidents[month_key],
                    total_crime_score,
                    global_min_score,
                    global_max_score,
                    f"{compute_safety_index(total_crime_score, global_min_score, global_max_score):.6f}",
                ]
            )

    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {COORDINATE_OUTPUT_FILE}")
    print(
        "Coordinate algorithm: median longitude/latitude of all valid incident points "
        "for each NEIGHBOURHOOD_158, excluding NSA and 0,0 coordinates."
    )
    print(f"Global min total crime score: {global_min_score}")
    print(f"Global max total crime score: {global_max_score}")


if __name__ == "__main__":
    main()
