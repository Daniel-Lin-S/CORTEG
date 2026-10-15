"""Check saved-source comparison resolution and fit resumption.

Input: synthetic comparison indexes and fit markers under pytest's temporary
folder. Output: assertions only; no real fit artifacts are modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import fingerflex_comparison as comparison
from scripts.fingerflex_cnn_gru_existing import (
    _pending, _record_selectors,
)
from scripts.fingerflex_prediction_examples import _comparison_models


def _source(
    base: Path, name: str, model: str,
    fit_id: str, preprocessing_id: str,
) -> Path:
    """Write one completed synthetic source comparison."""
    fit = base / 'fits' / f'{model}__{fit_id}'
    fit.mkdir(parents=True)
    (fit / 'index.json').write_text(json.dumps({
        'model': model, 'semantic_id': fit_id,
    }))
    root = base / 'comparisons' / f'{name}__sourcehash'
    (root / 'reports').mkdir(parents=True)
    (root / 'reports/summary.json').write_text(
        json.dumps({'complete': True})
    )
    (root / 'index.json').write_text(json.dumps({
        'name': name, 'semantic_id': 'sourcehash',
        'fit_roots': {model: str(fit)},
        'semantics': {
            'preprocessing_id': preprocessing_id,
            'fit_ids': {model: fit_id},
        },
    }))
    return fit


def test_resolve_sources_without_fit_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Find selected saved roots without an ID in portable settings."""
    monkeypatch.setattr(comparison, 'preparation_id',
                        lambda _: 'prepared')
    first = _source(tmp_path, 'first', 'corteg', 'a123', 'prepared')
    second = _source(tmp_path, 'second', 'cnn_gru', 'b456',
                     'prepared')
    study = {'source_comparisons': {
        'corteg': 'first', 'cnn_gru': 'second',
    }}
    roots = comparison.find_fit_roots(
        study, {'artifact_base': str(tmp_path)},
    )
    assert roots == {'corteg': first, 'cnn_gru': second}
    assert _comparison_models({
        'name': 'study', 'models': ['corteg', 'cnn_gru'],
    }) == {'study': ['corteg', 'cnn_gru']}
    with pytest.raises(ValueError, match='found 0'):
        comparison.find_fit_roots(
            {'source_comparisons': {'cnn_gru': 'absent'}},
            {'artifact_base': str(tmp_path)},
        )


def test_pending_skips_completed_fit(tmp_path: Path) -> None:
    """Only missing 50% seeds are enqueued for extension."""
    done = (tmp_path / 'runs/50pct/cnn_gru/bp/seed0')
    done.mkdir(parents=True)
    (done / 'result.json').write_text('{}')
    assert _pending(tmp_path, ['bp'], [0, 1]) == [('bp', 1)]


def test_selector_provenance_preserves_existing_fractions(
    tmp_path: Path,
) -> None:
    """Append 50% without dropping completed 10%, 25%, or 100%."""
    path = tmp_path / 'provenance/selections.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        'subjects': ['bp'], 'fractions': [1.0, 0.25, 0.1],
        'seeds': [0, 1, 2],
    }))
    study = {
        'subjects': ['bp', 'cc'],
        'fractions': [0.25, 0.5, 1.0],
        'seeds': [0, 1, 2],
    }
    _record_selectors(tmp_path, study)
    _record_selectors(tmp_path, study)
    saved = json.loads(path.read_text())
    assert saved['fractions'] == [1.0, 0.25, 0.1, 0.5]
    assert saved['subjects'] == ['bp', 'cc']
    assert saved['seeds'] == [0, 1, 2]
