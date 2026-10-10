"""Render FingerFlex regression curves from completed report artifacts.

Input: ``--root`` points to a campaign artifact with
``reports/summary.json`` and ``provenance/resolved_config.json``.
Output: PNG plots under ``<root>/reports/plots/`` by default, including
``subjects/<subject>/<metric>.png`` and ``aggregate.png``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plotting.fingerflex import generate_plots


def main() -> None:
    """Parse artifact paths and render complete saved results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in generate_plots(args.root, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
