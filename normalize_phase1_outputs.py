from __future__ import annotations

import argparse
from pathlib import Path

from phase2_results import DEFAULT_INPUT, DEFAULT_OUTPUT, discover_input_files, normalize_phase1_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize RUBRIC-MME Phase 1 sample outputs into stable Phase 2 dialogue-level and round-level artifacts."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[str(DEFAULT_INPUT)],
        help="One or more phase-1 samples.jsonl files or directories to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Directory to write standardized Phase 2 outputs.",
    )
    parser.add_argument(
        "--glob",
        default="*_samples.jsonl",
        help="Filename pattern used when an input path is a directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on malformed JSON lines instead of skipping them.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear existing Phase 2 output files before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.input, args.glob)
    if not input_files:
        raise FileNotFoundError("No phase-1 samples.jsonl files were found.")
    normalize_phase1_outputs(
        input_files,
        Path(args.output_dir),
        strict=args.strict,
        clear_output=not args.keep_existing,
    )


if __name__ == "__main__":
    main()
