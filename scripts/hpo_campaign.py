"""Launch and resume an adapter-backed validation search.

Input is portable JSON with an ``adapter`` Python module and task settings.
Output is an adapter-defined search root with provenance/, status/, and
selection.json. Fit artifacts and comparisons are owned by the adapter.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.hpo import select_gpus


def load_adapter(module_name: str) -> object:
    """Import a task adapter from a configured dotted module name.

    Parameters
    ----------
    module_name : str
        Importable Python module containing coordinator callbacks.

    Returns
    -------
    object
        Loaded adapter module.
    """
    if not module_name or any(
        not part.isidentifier() for part in module_name.split(".")
    ):
        raise ValueError("HPO adapter must be a Python module name.")
    return importlib.import_module(module_name)


def coordinate(adapter: object, search: Path, spec: dict,
               base: dict) -> None:
    """Complete screening, freeze validation selection, and run finals."""
    gpus = select_gpus(spec["runtime"])
    (search / "status" / "gpu_assignment.json").write_text(
        json.dumps({"physical_gpu_indices": gpus}, indent=2) + "\n"
    )
    adapter.dispatch(search, spec, "screen", gpus)
    adapter.select(search, spec)
    adapter.dispatch(search, spec, "final", gpus)
    comparison = adapter.finish_comparison(search, spec, base)
    print(comparison.resolve(), flush=True)


def main() -> None:
    """Initialize, launch, coordinate, or execute an adapter fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("init", "launch", "coordinate", "fit"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--subject")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("screen", "score", "final"))
    args = parser.parse_args()
    if args.command == "fit":
        if args.search_root is None:
            raise ValueError("Fit requires --search-root.")
        saved = args.search_root / "provenance" / "resolved_search.json"
        spec = json.loads(saved.read_text())
        load_adapter(spec["adapter"]).run_fit(args)
        return
    if args.config is None:
        raise ValueError("Search initialization requires --config.")
    config = json.loads(args.config.read_text())
    adapter = load_adapter(config["adapter"])
    spec, fit_study, base, roots = adapter.load_spec(args.config)
    search = adapter.ensure_search(spec, fit_study, base, roots)
    print(search.resolve(), flush=True)
    if args.command == "init":
        return
    if args.command == "launch":
        select_gpus(spec["runtime"])
        if not adapter.pending_jobs(search, spec, "screen") and (
            search / "selection.json"
        ).is_file() and not adapter.pending_jobs(
            search, spec, "final"
        ):
            print("All selected fits are complete.", flush=True)
            return
        session = f"hpo_{search.name[-12:]}"
        command = shlex.join([
            sys.executable, str(Path(__file__).resolve()),
            "coordinate", "--config", str(args.config.resolve()),
        ])
        log = search / "status" / "coordinator.log"
        shell = (
            f"cd {shlex.quote(str(REPO))} && {command} "
            f"> {shlex.quote(str(log))} 2>&1"
        )
        subprocess.run([
            base["tmux_executable"], "new-session", "-d",
            "-s", session, shell,
        ], check=True)
        print(f"tmux session: {session}", flush=True)
        return
    try:
        coordinate(adapter, search, spec, base)
    except Exception:
        (search / "status" / "coordinator_error.log").write_text(
            traceback.format_exc()
        )
        raise


if __name__ == "__main__":
    main()
