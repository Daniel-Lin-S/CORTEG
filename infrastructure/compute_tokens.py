"""Share machine worker slots across detached training campaigns.

Input: ignored local worker limits and a common ignored lock directory.
Output: a held CPU/GPU slot number that releases automatically on exit.
The lock files contain no scientific data or campaign-specific settings.
"""
from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

POLL_SECONDS = 1.0


@contextmanager
def worker_slot(folder: Path, kind: str,
                indices: list[int]) -> Iterator[int]:
    """Acquire one advisory worker slot shared across processes.

    Parameters
    ----------
    folder : Path
        Machine-local directory shared by both campaigns.
    kind : str
        ``cpu`` or ``gpu``.
    indices : list[int]
        Configured CPU slot IDs or available GPU device IDs.

    Yields
    ------
    int
        Acquired slot or physical GPU index.
    """
    if kind not in ("cpu", "gpu") or not indices:
        raise ValueError("Worker slots need cpu/gpu and nonempty indices.")
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        for index in indices:
            path = folder / f"{kind}_{index}.lock"
            stream = path.open("a+")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield index
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
                stream.close()
            return
        time.sleep(POLL_SECONDS)
