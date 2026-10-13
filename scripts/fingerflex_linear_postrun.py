"""Resume interrupted FingerFlex fits and finalize the detached report.

Input: configs/fingerflex_linear_fusion.json, ignored local settings,
existing fit roots and two detached coordinator sessions. Output:
quarantined incomplete attempts, retried fit artifacts, the final
comparison report, and a postrun log. Temporary unhashed aliases are
removed after every fit and the final comparison complete.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from infrastructure.compute import load_resources
from scripts.fingerflex_comparison import (
    find_fit_roots,
    load_comparison_study,
    report,
)
from scripts.fingerflex_linear_fusion import (
    CONFIG,
    LINEAR,
    NEURAL,
    RESOURCES,
    coordinate,
)
from scripts.fingerflex_linear_identity import ridge_root

WAIT_SECONDS = 30
NEURAL_SESSION = "fingerflex_linear_neural"
LINEAR_SESSION = "fingerflex_linear_linear"


def _wait_session(executable: str, session: str) -> None:
    """Wait until one detached coordinator has joined all its workers."""
    while subprocess.run(
        [executable, "has-session", "-t", session],
        capture_output=True, check=False,
    ).returncode == 0:
        time.sleep(WAIT_SECONDS)


def _remove_aliases(base: dict) -> None:
    """Remove only the expected temporary links to hashed ridge roots."""
    parent = Path(base["artifact_base"]) / "fits"
    for model in LINEAR:
        alias = parent / model
        target = ridge_root(base, model)
        if not alias.is_symlink() or alias.resolve() != target:
            raise ValueError(
                f"Expected temporary {model} link to {target}."
            )
        alias.unlink()
        print(f"Removed temporary alias {alias}", flush=True)


def _missing(
    roots: dict[str, Path], study: dict, models: tuple[str, ...],
) -> list[Path]:
    """List incomplete fit markers for one resource class."""
    pending = []
    for model in models:
        fractions = (0.5,) if model in NEURAL else study["fractions"]
        seeds = study["seeds"] if model in NEURAL else (None,)
        for fraction in fractions:
            for subject in study["subjects"]:
                for seed in seeds:
                    repeat = (
                        "deterministic" if seed is None
                        else f"seed{seed}"
                    )
                    result = (
                        roots[model] / "runs"
                        / f"{int(fraction * 100)}pct" / model
                        / subject / repeat / "result.json"
                    )
                    if not result.is_file():
                        pending.append(result)
    return pending


def main() -> None:
    """Retry pending fits after original coordinators release resources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--local-resources", type=Path, default=RESOURCES,
    )
    args = parser.parse_args()
    study, base = load_comparison_study(args.config)
    resources = load_resources(args.local_resources)
    executable = base["tmux_executable"]
    _wait_session(executable, LINEAR_SESSION)
    roots = find_fit_roots(study, base)
    if _missing(roots, study, LINEAR):
        coordinate("linear", study, roots, resources)
    _wait_session(executable, NEURAL_SESSION)
    if _missing(roots, study, NEURAL):
        coordinate("neural", study, roots, resources)
    missing = _missing(roots, study, (*NEURAL, *LINEAR))
    if missing:
        raise RuntimeError(
            f"Postrun retry left {len(missing)} incomplete fits; "
            f"first is {missing[0]}."
        )
    comparisons = list(
        (Path(base["artifact_base"]) / "comparisons").glob(
            study["name"] + "__*"
        )
    )
    if len(comparisons) != 1:
        raise ValueError(
            f"Expected one comparison root, got {comparisons}."
        )
    summary = report(comparisons[0])
    if not summary["complete"]:
        raise RuntimeError("Comparison report is incomplete after retry.")
    _remove_aliases(base)
    print(f"Final comparison: {comparisons[0]}", flush=True)


if __name__ == "__main__":
    main()
