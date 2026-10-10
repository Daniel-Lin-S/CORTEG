"""Render aggregate CORTEG FingerFlex training-monitor figures.

Input: a model-fit artifact root containing history.jsonl files.
Output: one PNG per monitor under plots/training_monitors/ by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plotting.fingerflex_training_monitors import (
    generate_training_monitor_plots,
)


def main() -> None:
    """Parse paths and save aggregate monitor figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in generate_training_monitor_plots(
        args.root, args.output_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
