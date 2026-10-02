from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MARKER_MAP_DIR = DATA_DIR / "marker_maps"

EXPERIMENTS = {
    "F-30": {
        "csv": "F-30_001.csv",
        "marker_map": "F-30map.json",
    },
    "F-60": {
        "csv": "F-60_001.csv",
        "marker_map": "F-60map.json",
    },
    "F-90": {
        "csv": "F-90_001.csv",
        "marker_map": "F-90map.json",
    },
    "F-110": {
        "csv": "F-110_001.csv",
        "marker_map": "F-110map.json",
    },
    "F0": {
        "csv": "F0_001.csv",
        "marker_map": "F0map.json",
    },
    "F30": {
        "csv": "F30_001.csv",
        "marker_map": "F30map.json",
    },
    "F60": {
        "csv": "F60_001.csv",
        "marker_map": "F60map.json",
    },
    "F90": {
        "csv": "F90_002.csv",
        "marker_map": "F90map.json",
    },
    "F110": {
        "csv": "F110_001.csv",
        "marker_map": "F110map.json",
    },
    "R-2": {
        "csv": "R-2_001.csv",
        "marker_map": "R-2map.json",
    },
    "R0": {
        "csv": "R0_001.csv",
        "marker_map": "R0map.json",
    },
    "R2": {
        "csv": "R2_001.csv",
        "marker_map": "R2map.json",
    },
    "SW": {
        "csv": "SW_002.csv",
        "marker_map": "SWmap.json",
    },
}


def get_experiment(name: str) -> tuple[Path, Path]:
    """Return the CSV and marker-map paths for an experiment."""
    # Check if the experiment name is valid
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS)
        raise ValueError(
            f"Unknown experiment '{name}'. "
            f"Available experiments: {available}"
        )

    experiment = EXPERIMENTS[name]

    # Construct the full paths to the CSV and marker-map files
    csv_path = DATA_DIR / experiment["csv"]
    marker_map_path = MARKER_MAP_DIR / experiment["marker_map"]

    return csv_path, marker_map_path