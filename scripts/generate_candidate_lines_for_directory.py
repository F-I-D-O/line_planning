import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from lineplanning.candidate_lines import (
    generate_candidate_lines,
    DEFAULT_NUMBER_OF_STOPS,
    DEFAULT_NB_LINES,
    DEFAULT_MIN_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_START_END_DISTANCE,
    DEFAULT_DETOUR_SKELETON,
    DEFAULT_AREA_NAME,
)


def _load_yaml_if_possible(path: Path) -> Dict[str, Any]:
    """
    Load YAML config if PyYAML is available; otherwise raise ImportError.
    """
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at top-level in {path}, got {type(data).__name__}."
        )
    return data


def _resolve_area_path_from_config(instance_dir: Path) -> Path | None:
    """
    Resolve area_path from config.yaml if present.

    Supports either:
    - config['area_dir']: directory relative to instance_dir
    - config['area_path']: directory relative to instance_dir (or absolute)
    """
    config_path = instance_dir / "config.yaml"
    if not config_path.exists():
        return None

    try:
        config = _load_yaml_if_possible(config_path)
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read config.yaml. "
            "Install with `pip install pyyaml` and rerun."
        ) from exc

    raw = config.get("area_dir")

    if not isinstance(raw, str) or not raw.strip():
        return None

    area_path = Path(raw)
    if not area_path.is_absolute():
        area_path = (instance_dir / area_path).resolve()
    return area_path


def process_instance_directory(
    instance_dir: Path,
    number_of_stops: int,
    nb_lines: int,
    min_length: int,
    max_length: int,
    min_start_end_distance: int,
    detour_skeleton: int,
    area_name: str | None,
    export_geopackage: bool = True,
    geopackage_path: Path | None = None,
) -> None:
    """
    Generate candidate lines for a single instance directory.

    The underlying graph + distance matrix live in the area directory referenced
    by the instance config (config['area_path'] or config['area_dir']).
    Writes lines.txt and (when applicable) candidate_lines.gpkg under instance_dir.
    """
    logging.info("Processing instance directory: %s", instance_dir)

    area_path = _resolve_area_path_from_config(instance_dir)
    if area_path is None:
        # Fallback: assume the instance directory itself is the area directory.
        logging.info(
            "No area_dir/area_path in config.yaml for %s, using instance directory as area.",
            instance_dir,
        )
        area_path = instance_dir

    generate_candidate_lines(
        area_path=area_path,
        number_of_stops=number_of_stops,
        nb_lines=nb_lines,
        min_length=min_length,
        max_length=max_length,
        min_start_end_distance=min_start_end_distance,
        detour_skeleton=detour_skeleton,
        area_name=area_name,
        output_path=instance_dir,
        export_geopackage=export_geopackage,
        geopackage_path=geopackage_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate candidate lines for all instance subdirectories in a given directory.\n\n"
            "Each instance directory must contain a distance matrix file "
            "(dm.h5, dm.hdf5, dm.csv, or dm.dm) and a road network at map/edges.csv.\n"
            "For each valid instance directory, lines.txt is written; if map/nodes.csv exists "
            "and geopandas is installed, candidate_lines.gpkg (QGIS) is written as well."
        )
    )
    parser.add_argument(
        "instances_root",
        type=str,
        help=(
            "Path to the directory containing instance subdirectories, e.g.\n"
            r"C:\Google Drive AIC\My Drive\AIC Experiment Data\Line Planning\Instances\manhattan-2_h-10_percent"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.",
    )
    parser.add_argument(
        "--number-of-stops",
        type=int,
        default=DEFAULT_NUMBER_OF_STOPS,
        help=f"Number of stops to sample per instance (default: {DEFAULT_NUMBER_OF_STOPS}).",
    )
    parser.add_argument(
        "--nb-lines",
        type=int,
        default=DEFAULT_NB_LINES,
        help=f"Number of candidate lines to generate per instance (default: {DEFAULT_NB_LINES}).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help=f"Minimum number of stops per line (default: {DEFAULT_MIN_LENGTH}).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum number of stops per line (default: {DEFAULT_MAX_LENGTH}).",
    )
    parser.add_argument(
        "--min-start-end-distance",
        type=int,
        default=DEFAULT_MIN_START_END_DISTANCE,
        help=f"Minimum travel-time distance between start and end stops (default: {DEFAULT_MIN_START_END_DISTANCE}).",
    )
    parser.add_argument(
        "--detour-skeleton",
        type=int,
        default=DEFAULT_DETOUR_SKELETON,
        help=f"Maximum detour factor for intermediate stops (default: {DEFAULT_DETOUR_SKELETON}).",
    )
    parser.add_argument(
        "--area-name",
        type=str,
        default=DEFAULT_AREA_NAME,
        help=(
            "Optional area name for OpenStreetMap download (overrides map/edges.csv). "
            "Default: use map/edges.csv if available, otherwise None."
        ),
    )
    parser.add_argument(
        "--no-geopackage",
        action="store_true",
        help="Do not write candidate_lines.gpkg (QGIS export).",
    )
    parser.add_argument(
        "--geopackage-path",
        type=str,
        default=None,
        help=(
            "Path to the output .gpkg file. Default: "
            "<instance_dir>/candidate_lines.gpkg when processing subdirectories, "
            "or <root>/candidate_lines.gpkg when the root is a single instance."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    root = Path(args.instances_root)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Instances root is not a directory: {root}")

    # Process all immediate subdirectories that look like instances
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    gpkg_arg = Path(args.geopackage_path) if args.geopackage_path else None

    if not subdirs:
        # If there are no subdirectories, treat the root itself as a single instance directory
        logging.info(
            "No subdirectories found under %s. Treating it as a single instance directory.",
            root,
        )
        process_instance_directory(
            root,
            number_of_stops=args.number_of_stops,
            nb_lines=args.nb_lines,
            min_length=args.min_length,
            max_length=args.max_length,
            min_start_end_distance=args.min_start_end_distance,
            detour_skeleton=args.detour_skeleton,
            area_name=args.area_name,
            export_geopackage=not args.no_geopackage,
            geopackage_path=gpkg_arg,
        )
        return

    for instance_dir in subdirs:
        try:
            if gpkg_arg is not None and len(subdirs) > 1:
                out_gpkg = gpkg_arg.parent / f"{instance_dir.name}_{gpkg_arg.name}"
            else:
                out_gpkg = gpkg_arg
            process_instance_directory(
                instance_dir,
                number_of_stops=args.number_of_stops,
                nb_lines=args.nb_lines,
                min_length=args.min_length,
                max_length=args.max_length,
                min_start_end_distance=args.min_start_end_distance,
                detour_skeleton=args.detour_skeleton,
                area_name=args.area_name,
                export_geopackage=not args.no_geopackage,
                geopackage_path=out_gpkg,
            )
        except Exception as e:
            logging.error(
                "Failed to generate candidate lines for %s: %s", instance_dir, e
            )


if __name__ == "__main__":
    main()

