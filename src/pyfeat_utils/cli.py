from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pyfeat_utils.config import ConfigError, load_config, write_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyfeat-utils",
        description="Batch py-feat processing and descriptive analysis utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starter config.")
    init_parser.add_argument("--config", type=Path, default=None)
    init_parser.add_argument("--data-dir", type=Path, default=None)

    process_parser = subparsers.add_parser("process", help="Process images/videos.")
    process_parser.add_argument("--config", type=Path, required=True)
    process_parser.add_argument("--visualize", action="store_true")

    stats_parser = subparsers.add_parser("stats", help="Compute descriptive stats.")
    stats_parser.add_argument("--config", type=Path, required=True)
    stats_parser.add_argument("--plots", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    try:
        if args.command == "init":
            config_path = write_default_config(args.config, args.data_dir)
            print(f"Created config: {config_path}")
            return 0

        config = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.command == "stats":
        from pyfeat_utils.statistics import (
            compute_directory_statistics,
            write_statistics_summary,
        )

        statistics = compute_directory_statistics(config.data_dir)
        output_path = config.data_dir / "statistics_summary.csv"
        write_statistics_summary(statistics, output_path)
        print(f"Wrote statistics summary: {output_path}")
        return 0

    if args.command == "process":
        from pyfeat_utils.processing import process_directory

        results = process_directory(config)
        for result in results:
            print(
                f"Processed {result.media_type}: "
                f"{result.input_path} -> {result.output_path}"
            )
        print(f"Processed {len(results)} file(s).")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
