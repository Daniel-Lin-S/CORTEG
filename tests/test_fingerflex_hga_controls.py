"""Check filterbank boundaries, CV recovery, and comparison safety.

Input: synthetic one-second windows and temporary fit directories.
Output: assertions only; no scientific campaign artifacts are written.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from evaluation.regression import regression_metrics
from experiments.hga_control_cv import (
    candidates, select_branch,
)
from experiments.ridge_cv import convex_weights
from experiments.fingerflex_hga_controls import _legacy_cv_resolver
from models.hga_regressors import make_svr
from preprocessing.channel_time_features import ChannelTimeScaler
from preprocessing.naplib_filterbank import (
    analytic_kernels, center_frequencies, window_amplitudes,
)
from plotting.fingerflex_comparison import _draw_curves
from scripts.fingerflex_comparison import _check_alignment, _read_fit
from scripts.fingerflex_hga_identity import (
    fit_root, fit_semantics, semantic_hash,
)
from scripts.fingerflex_hga_controls import (
    _fit_root_with_legacy_grid, _guard_attempt, _new_root,
)


def test_filterbank_component_and_time_boundaries() -> None:
    """Require eleven naplib centers and equal 5-sample time bins."""
    centers = center_frequencies(70, 200)
    assert len(centers) == 11
    assert np.isclose(centers[0], 71.98537593)
    assert np.isclose(centers[-1], 193.77026266)
    time = np.arange(1000) / 1000
    window = np.stack((
        np.sin(2 * np.pi * 80 * time),
        np.sin(2 * np.pi * 160 * time),
    ), axis=1).astype(np.float32)
    kernels = analytic_kernels(1000, 1000, centers)
    bands = window_amplitudes(window, kernels, 200)
    assert bands.shape == (2, 11, 200)
    assert int(bands[0].mean(axis=1).argmax()) == 1
    assert int(bands[1].mean(axis=1).argmax()) == 8
    assert np.isfinite(bands).all()
    with pytest.raises(ValueError, match="not divisible"):
        window_amplitudes(window, kernels, 128)


def test_scaler_uses_only_fold_training_rows() -> None:
    """Keep validation and held-out magnitudes out of moments."""
    training = np.array([
        [[1., 3.], [10., 14.]],
        [[3., 5.], [12., 16.]],
    ])
    held_out = np.full((1, 2, 2), 1000.0)
    scaler = ChannelTimeScaler.fit(training)
    assert np.allclose(scaler.mean.ravel(), [3, 13])
    assert np.all(scaler.transform(held_out) > 100)


def test_invalid_cnn_geometry_is_recorded() -> None:
    """Exclude all dropout settings for depth-four stride-three."""
    settings = {
        "cnn_gru": {
            "depths": [2, 3, 4], "strides": [2, 3],
            "dropouts": [0, 0.1, 0.3, 0.5],
        }
    }
    valid, excluded = candidates(
        "cnn_gru_late", settings, 2, 200,
    )
    assert len(valid) == 20
    assert len(excluded) == 4
    assert all(
        item["candidate"]["n_convolution_layers"] == 4
        and item["candidate"]["conv_stride"] == 3
        for item in excluded
    )
    configured = json.loads((
        Path(__file__).parents[1]
        / "configs/fingerflex_hga_controls.json"
    ).read_text())
    reduced, invalid = candidates(
        "cnn_gru_late", configured, 2, 200,
    )
    assert len(reduced) == 9
    assert not invalid
    assert set(
        item["conv_stride"] for item in reduced
    ) == {2}
    assert set(
        item["drop_out"] for item in reduced
    ) == {0.1, 0.3, 0.5}


def test_cnn_resume_grid_reuses_original_cv_unit_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse a saved fold and keep full-grid candidate IDs."""
    grid = [
        {
            "n_convolution_layers": depth,
            "conv_stride": 2,
            "drop_out": 0.1,
        }
        for depth in (2, 3)
    ]
    def fake_candidates(
        _model: str, settings: dict, *_args: object,
    ) -> tuple[list[dict], list]:
        return (
            grid if len(settings["cnn_gru"]["depths"]) == 2
            else [grid[1]], []
        )

    monkeypatch.setattr(
        "experiments.hga_control_cv.candidates",
        fake_candidates,
    )
    monkeypatch.setattr(
        "experiments.fingerflex_hga_controls.candidates",
        fake_candidates,
    )
    rows = np.arange(50, dtype=np.float32)
    x = rows[:, None, None]
    y = np.repeat(rows[:, None], 5, axis=1)
    saved = tmp_path / "candidate1/seed0/fold0"
    saved.mkdir(parents=True)
    np.save(saved / "prediction.npy", y[:10])
    (saved / "score.json").write_text(json.dumps({
        "candidate": grid[1], "seed": 0, "fold": 0,
        "training_rows": 40, "held_out_rows": 10,
        "score": 1.0,
    }))
    partial = tmp_path / "reduced_grid/candidate0/seed0/fold1"
    partial.mkdir(parents=True)
    (partial / "checkpoint.tmp").write_text("incomplete")
    called = []

    def fake_fit(
        model: str, candidate: dict, seed: int | None,
        train_x: np.ndarray, train_y: np.ndarray,
        validation_x: np.ndarray, validation_y: np.ndarray,
        prediction_x: np.ndarray, output: Path,
        settings: dict, device: torch.device,
        persist_model: bool = True,
    ) -> tuple[np.ndarray, dict]:
        called.append(output)
        output.mkdir(parents=True)
        (output / "best.pt").write_bytes(b"checkpoint")
        predicted = np.repeat(
            prediction_x[:, 0, 0:1], 5, axis=1,
        )
        return predicted, {}

    monkeypatch.setattr(
        "experiments.hga_control_cv.fit_branch", fake_fit,
    )
    settings = {
        "folds": 5, "purge_rows": 0,
        "cnn_gru": {
            "depths": [3], "strides": [2],
            "dropouts": [0.1],
        },
        "cnn_gru_legacy_grid": {
            "depths": [2, 3], "strides": [2],
            "dropouts": [0.1],
        },
    }
    resolver, indices = _legacy_cv_resolver(
        "cnn_gru_late", settings, tmp_path, 1, 1,
    )
    candidate, oof, history = select_branch(
        "cnn_gru_late", x, y, x[:10], y[:10], tmp_path,
        settings, [0], torch.device("cpu"),
        unit_resolver=resolver,
    )
    assert candidate == grid[1]
    assert history["candidate_indices"] == [0]
    assert indices == [1]
    assert history["selected_index"] == 0
    assert np.array_equal(oof[0], y)
    assert len(called) == 4
    assert all(path.parts[-3] == "candidate0" for path in called)
    assert not (tmp_path / "candidate0").exists()
    assert len(list((tmp_path / "quarantine").glob("*fold1*"))) == 1


def test_cv_ties_and_incomplete_unit_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer candidate order and quarantine a partial fold."""
    rows = np.arange(50, dtype=np.float32)
    x = np.repeat(rows[:, None, None], 2, axis=2)
    y = np.repeat(rows[:, None], 5, axis=1)
    validation_x = x[:10]
    validation_y = y[:10]
    partial = (
        tmp_path / "candidate0/deterministic/fold0"
    )
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("incomplete")

    def fake_fit(
        model: str, candidate: dict, seed: int | None,
        train_x: np.ndarray, train_y: np.ndarray,
        val_x: np.ndarray, val_y: np.ndarray,
        prediction_x: np.ndarray, output: Path,
        settings: dict, device: torch.device,
        persist_model: bool = True,
    ) -> tuple[np.ndarray, dict]:
        output.mkdir(parents=True, exist_ok=True)
        if persist_model:
            joblib.dump({}, output / "best.joblib")
        predicted = np.repeat(
            prediction_x[:, 0, :1], 5, axis=1,
        )
        return predicted, {}

    monkeypatch.setattr(
        "experiments.hga_control_cv.fit_branch", fake_fit,
    )
    settings = {
        "folds": 5, "purge_rows": 0,
        "svr": {
            "C": [0.1, 1.0], "gamma_factors": [1.0],
            "epsilon": 0.1,
        },
    }
    chosen, oof, history = select_branch(
        "svr_late", x, y, validation_x, validation_y,
        tmp_path, settings, [], torch.device("cpu"),
    )
    assert chosen["C"] == 0.1
    assert history["selected_index"] == 0
    assert np.array_equal(oof[None], y)
    assert list((tmp_path / "quarantine").iterdir())
    assert not list(tmp_path.glob("candidate*/**/best.joblib"))
    chosen_again, _, _ = select_branch(
        "svr_late", x, y, validation_x, validation_y,
        tmp_path, settings, [], torch.device("cpu"),
    )
    assert chosen_again == chosen


def test_shared_kernel_svr_matches_independent_rbf(tmp_path: Path) -> None:
    """Keep five-target fits and restored predictions numerically exact."""
    rng = np.random.default_rng(314)
    design = rng.normal(size=(50, 36)).astype(np.float32)
    targets = rng.normal(size=(50, 5))
    held_out = rng.normal(size=(8, 36)).astype(np.float32)
    gamma = 0.1 / (design.shape[1] * float(design.var()))
    independent = MultiOutputRegressor(
        SVR(C=1, gamma=gamma, epsilon=0.1, cache_size=256),
        n_jobs=1,
    ).fit(design, targets)
    shared = make_svr(design, 1, 0.1, 0.1, 256)
    shared.fit(design, targets)
    expected = independent.predict(held_out)
    np.testing.assert_allclose(shared.predict(held_out), expected,
                               rtol=1e-10, atol=1e-10)
    saved = tmp_path / "svr.joblib"
    joblib.dump(shared, saved)
    restored = joblib.load(saved)
    np.testing.assert_allclose(restored.predict(held_out), expected,
                               rtol=1e-10, atol=1e-10)


def test_seed_specific_convex_weights() -> None:
    """Keep neural seed predictions separate when fitting fusion."""
    target = np.repeat(
        np.array([[0.], [1.], [2.], [3.]]), 5, axis=1,
    )
    low = target + np.array([[-1.], [1.], [-1.], [1.]])
    high_first = target + 2
    high_second = target + 0.5
    first = convex_weights(low, high_first, target)
    second = convex_weights(low, high_second, target)
    assert not np.array_equal(first, second)


def test_saved_metrics_and_row_alignment(tmp_path: Path) -> None:
    """Reject metric changes and cross-model target identity changes."""
    truth = np.arange(50, dtype=np.float32).reshape(10, 5)
    predicted = truth + 0.1
    result = {
        "model": "svr_late", "subject": "bp",
        "fraction": 0.25, "seed": None, "delay_ms": 0,
        "test_scored_after_restore": True,
        "metrics": regression_metrics(truth, predicted),
        "boundaries": {
            key: "aligned" for key in (
                "fit_indices_sha256",
                "validation_indices_sha256",
                "test_grid_sha256", "target_values_sha256",
            )
        },
    }
    (tmp_path / "best.joblib").write_bytes(b"saved")
    np.savez(tmp_path / "test_predictions.npz",
             y=truth, y_hat=predicted)
    (tmp_path / "result.json").write_text(json.dumps(result))
    loaded = _read_fit(
        tmp_path, "svr_late", "bp", 0.25,
        None, validate_scores=True,
    )
    _check_alignment({"svr_late": [loaded], "ridge_late": [loaded]})
    changed = dict(loaded)
    changed["truth_sha256"] = "different"
    with pytest.raises(ValueError, match="disagree"):
        _check_alignment({
            "svr_late": [loaded], "ridge_late": [changed],
        })
    result["metrics"]["overall"]["mse"] += 1
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Saved mse differs"):
        _read_fit(
            tmp_path, "svr_late", "bp", 0.25,
            None, validate_scores=True,
        )


def test_semantic_identity_excludes_invocation_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep selectors and machine limits out of model fit identity."""
    monkeypatch.setattr(
        "scripts.fingerflex_hga_identity.preparation_id",
        lambda base: "prepared",
    )
    source = json.loads((
        Path(__file__).parents[1]
        / "configs/fingerflex_hga_controls.json"
    ).read_text())
    base = {
        "artifact_base": str(tmp_path / "first"),
        "gpu_workers": 4,
    }
    baseline = semantic_hash(fit_semantics(base, "svr_late", source))
    source["subjects"] = ["mv"]
    base.update({
        "artifact_base": str(tmp_path / "second"),
        "gpu_workers": 1,
    })
    assert semantic_hash(fit_semantics(base, "svr_late", source)) == baseline
    source["svr"]["C"] = [0.1, 1.0]
    assert semantic_hash(fit_semantics(base, "svr_late", source)) != baseline


def test_cnn_reduced_grid_reuses_existing_fit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep completed fits while recording the changed search grid."""
    monkeypatch.setattr(
        "scripts.fingerflex_hga_identity.preparation_id",
        lambda base: "prepared",
    )
    monkeypatch.setattr(
        "scripts.fingerflex_hga_controls.preparation_id",
        lambda base: "prepared",
    )
    source = json.loads((
        Path(__file__).parents[1]
        / "configs/fingerflex_hga_controls.json"
    ).read_text())
    base = {"artifact_base": str(tmp_path)}
    reduced = semantic_hash(fit_semantics(
        base, "cnn_gru_late", source,
    ))
    previous = json.loads(json.dumps(source))
    previous["cnn_gru"] = {
        "depths": [2, 3, 4],
        "strides": [2, 3],
        "dropouts": [0, 0.1, 0.3, 0.5],
    }
    original = semantic_hash(fit_semantics(
        base, "cnn_gru_late", previous,
    ))
    assert reduced != original
    root = tmp_path / "fits" / f"cnn_gru_late__{original}"
    root.mkdir(parents=True)
    (root / "index.json").write_text(json.dumps({
        "semantics": fit_semantics(base, "cnn_gru_late", previous),
    }))
    resolved, grid = _fit_root_with_legacy_grid(
        base, source, "cnn_gru_late",
    )
    assert resolved == root
    assert grid == previous["cnn_gru"]
    (tmp_path / "fits/semantic_aliases.json").write_text(
        json.dumps({f"cnn_gru_late__{reduced}": root.name})
    )
    assert fit_root(base, "cnn_gru_late", source) == root
    aliased, grid = _fit_root_with_legacy_grid(
        base, source, "cnn_gru_late",
    )
    assert aliased == root
    assert grid == previous["cnn_gru"]
    prepared = tmp_path / "preprocessing/prepared/subjects"
    prepared.mkdir(parents=True)
    assert _new_root(base, source, "cnn_gru_late") == root
    version = (
        root / "provenance/semantic_versions"
        / f"{reduced}.json"
    )
    assert json.loads(version.read_text())["semantics"] == (
        json.loads(json.dumps(fit_semantics(
            base, "cnn_gru_late", source,
        )))
    )


def test_explicit_feature_labels_in_legend() -> None:
    """Show each model's LFS and HGA representation in the legend."""
    labels = {
        "corteg": "CORTEG (LFS waveform + single-band HGA amplitude)",
        "svr_late": (
            "RBF-SVR late fusion "
            "(LFS waveform + single-band HGA amplitude)"
        ),
    }
    figure, axis = plt.subplots()
    _draw_curves(
        axis, np.array([0.25, 0.5, 1.0]),
        np.ones((2, 3)), np.full((2, 3), np.nan),
        "pearson_r", tuple(labels), labels,
    )
    assert axis.get_legend_handles_labels()[1] == list(labels.values())
    plt.close(figure)


def test_worker_failure_has_attempt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attribute coordinator-side exceptions to a saved worker log."""
    events = []

    def fail(*args: object) -> None:
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(
        "scripts.fingerflex_hga_controls._attempt", fail,
    )
    monkeypatch.setattr(
        "scripts.fingerflex_hga_controls.append_fit_event",
        lambda root, event, lock: events.append(event),
    )
    _guard_attempt(
        "svr_late", "bp", 0.25, tmp_path,
        {"seeds": [0, 1, 2]}, {}, {}, tmp_path,
        threading.RLock(), tmp_path, [], tmp_path,
    )
    assert len(events) == 1
    assert events[0]["attempt_id"]
    assert events[0]["state"] == "failed"
    log = Path(events[0]["attempt_log"])
    assert log.is_file()
    assert "synthetic worker failure" in log.read_text()
