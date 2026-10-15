"""Complete saved HGA-only CNN-GRU fits for a report-only comparison.

Input: a portable comparison JSON, existing fit roots with prepared data,
and an ignored local compute JSON. Output: missing 50% fits in the existing
CNN-GRU root, fit status events, and a completed comparison with plots.
Each completed new fit also stores source hashes beside its result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.fingerflex_cache import preparation_id, sha256_file
from infrastructure.compute import available_gpus, load_resources
from infrastructure.compute_tokens import worker_slot
from infrastructure.training_lifecycle import (
    coordinator_scope, fit_task, register_campaign,
)
from scripts.fingerflex_comparison import (
    _job_path, ensure_comparison, find_fit_roots,
    load_comparison_study, report,
)
from scripts.fingerflex_fit_campaigns import (
    _event, _run_fit_attempt,
)
from scripts.fingerflex_prediction_examples import generate_examples

CONFIG = REPO / 'configs/fingerflex_corteg_cnn_gru_comparison.json'
RESOURCES = REPO / 'configs/compute_local.json'
SESSION = 'fingerflex_cnn_gru_hga_50'
MODEL = 'cnn_gru'
FRACTION = 0.5
SOURCES = (
    'models/temporal_spatial_cnn_gru.py',
    'experiments/fingerflex_campaign.py',
    'experiments/regression_trainer.py',
    'train/lr_schedule.py',
)


def _record_selectors(root: Path, study: dict) -> None:
    """Append requested selectors to the existing fit provenance."""
    path = root / 'provenance/selections.json'
    selected = json.loads(path.read_text())
    for key in ('subjects', 'fractions', 'seeds'):
        selected[key] = list(dict.fromkeys(
            selected[key] + study[key]
        ))
    path.write_text(json.dumps(selected, indent=2) + '\n')


def _context(config_path: Path) -> tuple[dict, dict, dict, Path]:
    """Resolve existing roots and register every expected comparison fit."""
    study, base = load_comparison_study(config_path)
    roots = find_fit_roots(study, base)
    root = roots[MODEL]
    index = json.loads((root / 'index.json').read_text())
    if index.get('model') != MODEL or (
        index.get('preprocessing_id') != preparation_id(base)
    ) or index.get('semantics', {}).get('delay_ms') != 0:
        raise ValueError(f'Unexpected HGA-only fit semantics: {root}.')
    training = json.loads(
        (root / 'provenance/resolved_config.json').read_text()
    )['training']
    if any(
        index['semantics']['training'][key] != training[key]
        for key in index['semantics']['training']
    ):
        raise ValueError(f'HGA-only training settings differ: {root}.')
    _record_selectors(root, study)
    comparison = ensure_comparison(study, base, roots)
    tasks = [
        fit_task(
            fit_root,
            _job_path(fit_root, model, fraction, subject, seed)
            / 'result.json',
            model=model, subject=subject,
            fraction=fraction, seed=seed,
        )
        for model, fit_root in roots.items()
        for fraction, subject, seed in itertools.product(
            study['fractions'], study['subjects'], study['seeds'],
        )
    ]
    register_campaign(comparison, tasks)
    return study, base, roots, comparison


def _source_provenance(output: Path) -> None:
    """Record source identity for one newly completed fit."""
    if not (output / 'result.json').is_file():
        return
    path = output / 'source_provenance.json'
    if path.is_file():
        return
    revision = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    value = {
        'git_revision': revision,
        'source_sha256': {
            source: sha256_file(REPO / source)
            for source in SOURCES
        },
    }
    path.write_text(json.dumps(value, indent=2) + '\n')


def _attempt(
    root: Path, subject: str, seed: int,
    gpu: int, lock: threading.RLock,
    resources: dict, slots: Path,
) -> None:
    """Run one missing fit under a shared GPU slot and record failure."""
    output = _job_path(root, MODEL, FRACTION, subject, seed)
    if (output / 'result.json').is_file():
        return
    events = root / 'status/jobs.jsonl'
    attempt_id = uuid.uuid4().hex
    try:
        with worker_slot(slots, 'gpu', [gpu]):
            _run_fit_attempt(
                MODEL, root, (subject, FRACTION, seed),
                events, lock, gpu, resources, attempt_id,
            )
        _source_provenance(output)
    except Exception as error:
        _event(events, {
            'state': 'failed', 'model': MODEL,
            'subject': subject, 'fraction': FRACTION,
            'seed': seed, 'attempt_id': attempt_id,
            'exit_code': None,
            'error': f'{type(error).__name__}: {error}',
        }, lock)


def _pending(root: Path, subjects: list[str],
             seeds: list[int]) -> list[tuple[str, int]]:
    """List only fits without completion markers."""
    return [
        (subject, seed)
        for subject, seed in itertools.product(subjects, seeds)
        if not (_job_path(
            root, MODEL, FRACTION, subject, seed,
        ) / 'result.json').is_file()
    ]


def run(
    study: dict, roots: dict, comparison: Path,
    config_path: Path, resources_path: Path,
    selected: list[tuple[str, int]] | None = None,
) -> None:
    """Coordinate missing 50% fits and render completed comparisons."""
    resources = load_resources(resources_path)
    root = roots[MODEL]
    jobs = _pending(root, study['subjects'], study['seeds'])
    if selected is not None:
        jobs = [item for item in jobs if item in selected]
    if not jobs:
        if selected is None:
            finish(comparison, config_path)
        return
    gpus = available_gpus(resources)
    lock = threading.RLock()
    events = root / 'status/jobs.jsonl'
    slots = REPO / 'logs/training/shared_slots'
    scope = (
        coordinator_scope(comparison, 'hga_only_extension', (MODEL,))
        if selected is None else nullcontext()
    )
    with scope:
        for subject, seed in jobs:
            _event(events, {
                'state': 'queued', 'model': MODEL,
                'subject': subject, 'fraction': FRACTION,
                'seed': seed,
            }, lock)
        with ThreadPoolExecutor(
            max_workers=resources['gpu_workers']
        ) as pool:
            futures = [
                pool.submit(
                    _attempt, root, subject, seed,
                    gpus[index % len(gpus)], lock, resources, slots,
                )
                for index, (subject, seed) in enumerate(jobs)
            ]
            for future in futures:
                future.result()
    if selected is None:
        finish(comparison, config_path)


def finish(comparison: Path, config_path: Path) -> None:
    """Build reports and examples when every saved fit is complete."""
    summary = report(comparison)
    if summary['complete']:
        config = json.loads(config_path.read_text())
        generate_examples((comparison,), config)
    print(f'{comparison.resolve()} complete={summary["complete"]}',
          flush=True)


def launch(
    base: dict, comparison: Path, config_path: Path,
    resources_path: Path,
) -> None:
    """Start one resumable coordinator in a detached local session."""
    resources = load_resources(resources_path)
    available_gpus(resources)
    executable = base['tmux_executable']
    live = subprocess.run(
        [executable, 'has-session', '-t', SESSION],
        capture_output=True, check=False,
    )
    if live.returncode == 0:
        raise RuntimeError(f'Coordinator already live: {SESSION}.')
    log = comparison / 'coordinator.log'
    command = shlex.join([
        sys.executable, str(Path(__file__).resolve()),
        'coordinate', '--config', str(config_path.resolve()),
        '--local-resources', str(resources_path.resolve()),
    ])
    shell = (
        f'cd {shlex.quote(str(REPO))} && {command} '
        f'> {shlex.quote(str(log))} 2>&1'
    )
    subprocess.run(
        [executable, 'new-session', '-d', '-s', SESSION, shell],
        check=True,
    )
    print(f'Detached {SESSION}; log {log.resolve()}', flush=True)


def main() -> None:
    """Dispatch saved-root initialization, fitting, and reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'command', choices=('init', 'fit-one', 'launch',
                            'coordinate', 'report'),
    )
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--local-resources', type=Path,
                        default=RESOURCES)
    parser.add_argument('--subject')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    study, base, roots, comparison = _context(args.config)
    if args.command == 'fit-one':
        if args.subject not in study['subjects'] or (
            args.seed not in study['seeds']
        ):
            raise ValueError('Fit-one needs a selected subject and seed.')
        run(
            study, roots, comparison, args.config,
            args.local_resources,
            selected=[(args.subject, args.seed)],
        )
    elif args.command == 'launch':
        launch(base, comparison, args.config, args.local_resources)
    elif args.command == 'coordinate':
        run(study, roots, comparison, args.config,
            args.local_resources)
    elif args.command == 'report':
        finish(comparison, args.config)
    else:
        print(comparison.resolve(), flush=True)


if __name__ == '__main__':
    main()
