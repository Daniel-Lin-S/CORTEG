"""Resolve reusable machine compute limits and available GPU devices.

Input: an ignored JSON file containing gpu_workers, cpu_workers,
cpu_threads_per_worker, and min_gpu_free_mib. Output: validated
resource limits and GPU indices selected from current host status.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

RESOURCE_KEYS = (
    "gpu_workers", "cpu_workers", "cpu_threads_per_worker",
    "min_gpu_free_mib",
)


def load_resources(path: Path) -> dict[str, int]:
    """Validate positive machine-specific worker limits."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Local compute configuration is missing: {path}."
        )
    values = json.loads(path.read_text())
    if set(values) != set(RESOURCE_KEYS):
        raise ValueError(
            f"Expected compute keys {RESOURCE_KEYS}, got "
            f"{sorted(values)}."
        )
    for key, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"Compute setting {key} must be a positive integer, "
                f"got {value}."
            )
    return values


def available_gpus(resources: dict[str, int]) -> list[int]:
    """Select configured GPU count by current utilization and free RAM."""
    query = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    available = []
    for line in query.splitlines():
        index, free, busy = (
            int(value.strip()) for value in line.split(",")
        )
        if free >= resources["min_gpu_free_mib"]:
            available.append((busy, -free, index))
    if len(available) < resources["gpu_workers"]:
        raise RuntimeError(
            f"Need {resources['gpu_workers']} GPUs with at least "
            f"{resources['min_gpu_free_mib']} MiB free; "
            f"found {len(available)}."
        )
    return [
        row[2] for row in sorted(available)[:resources["gpu_workers"]]
    ]
