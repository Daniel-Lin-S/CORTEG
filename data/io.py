"""Load prepared ECoG windows and select raw electrode coordinates.

Input: <root>/<subject>_features.pkl with six train/test arrays, and either
<root>/<subject>_electrode_loc.mat or <root>/<subject>.mat containing
channelName plus <raw_root>/<subject>/<subject>_fingerflex.mat with locs.
Output: SubjectData with aligned (C, 3) electrode coordinates when requested.
The raw recording directory is read only.
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.io


@dataclass
class SubjectData:
    """Canonical subject record loaded from *_features.pkl (+ optional electrode_loc.mat)."""
    subject: str
    X_feat_tr: np.ndarray  # (N,C,200,2) [high, low]
    y_tr: np.ndarray       # (N,5)
    X_feat_te: np.ndarray  # (N,C,200,2)
    y_te: np.ndarray       # (N,5)
    X_raw_tr: np.ndarray   # (N,C,128)
    X_raw_te: np.ndarray   # (N,C,128)
    ecog_xyz_mm: Optional[np.ndarray] = None  # (C,3) in mm


def load_features_pkl(pkl_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a 6-array features pickle: (X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te)."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, (tuple, list)) and len(obj) == 6):
        raise ValueError(f"Expected 6 arrays in {pkl_path}, got type={type(obj)} len={len(obj) if isinstance(obj,(tuple,list)) else 'NA'}")
    X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te = obj
    return X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te


def load_ecog_xyz_mm(mat_path: str, key: str = "electrodes") -> np.ndarray:
    """Load electrode XYZ coordinates (mm) from a .mat file. Returns float32 array of shape (C, 3)."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)
    mat = scipy.io.loadmat(mat_path)
    if key not in mat:
        raise KeyError(f"Missing key '{key}' in {mat_path}. Keys={list(mat.keys())}")
    xyz = mat[key]
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected (C,3) xyz in mm, got {xyz.shape} from {mat_path}")
    return xyz.astype(np.float32, copy=False)


def load_raw_xyz_for_channels(
    file_root: str,
    raw_root: str,
    subject: str,
    n_channels: int,
) -> np.ndarray:
    """Select raw locs using channelName saved by MATLAB cleaning.

    Parameters
    ----------
    file_root : str
        Directory containing <subject>.mat from MATLAB preprocessing.
    raw_root : str
        Directory containing <subject>/<subject>_fingerflex.mat.
    subject : str
        Two-letter Stanford subject identifier.
    n_channels : int
        Channel count C of the prepared (N, C, T) windows.

    Returns
    -------
    np.ndarray
        Selected electrode coordinates, shape (C, 3), in raw channel order.

    Raises
    ------
    ValueError
        If channel names are missing, duplicated, malformed, or incompatible
        with the raw recording or prepared features.
    """
    cleaned_path = os.path.join(file_root, f"{subject}.mat")
    raw_path = os.path.join(
        raw_root, subject, f"{subject}_fingerflex.mat"
    )
    if not os.path.isfile(cleaned_path):
        raise FileNotFoundError(
            f"Missing {cleaned_path}; it must contain channelName from "
            "the MATLAB cleaning step to align raw locs."
        )
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(
            f"Missing raw recording {raw_path}; provide --raw_root or "
            "CORTEG_STANFORD_RAW_ROOT."
        )
    cleaned = scipy.io.loadmat(
        cleaned_path, variable_names=["channelName"], squeeze_me=True
    )
    if "channelName" not in cleaned:
        raise KeyError(
            f"Missing channelName in {cleaned_path}; cannot select "
            "raw electrode locations."
        )
    if n_channels <= 0:
        raise ValueError(
            f"Expected a positive prepared channel count, got "
            f"{n_channels}."
        )
    labels = np.asarray(cleaned["channelName"]).reshape(-1)
    if len(labels) != n_channels:
        raise ValueError(
            f"{cleaned_path}: expected {n_channels} channel names to "
            f"match prepared windows, got {len(labels)}."
        )
    raw = scipy.io.loadmat(raw_path, variable_names=["locs"])
    if "locs" not in raw:
        raise KeyError(f"Missing locs in raw recording {raw_path}.")
    locs = np.asarray(raw["locs"], dtype=np.float64)
    if locs.ndim != 2 or locs.shape[1] != 3:
        raise ValueError(
            f"{raw_path}: expected locs shape (C, 3), got {locs.shape}."
        )
    indices = []
    for label in labels:
        if isinstance(label, np.ndarray) and label.size == 1:
            label = label.item()
        name = str(label).strip()
        match = re.fullmatch(r"ch([1-9][0-9]*)", name)
        if match is None:
            raise ValueError(
                f"{cleaned_path}: expected channel name chN, got "
                f"{name!r}."
            )
        index = int(match.group(1)) - 1
        if index >= len(locs):
            raise ValueError(
                f"{cleaned_path}: {name} exceeds {len(locs)} raw "
                f"electrodes in {raw_path}."
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(
            f"{cleaned_path}: channelName contains duplicate channels."
        )
    selected = locs[indices]
    if not np.isfinite(selected).all():
        raise ValueError(
            f"{raw_path}: selected locs contain NaN or infinity."
        )
    return selected.astype(np.float32)


def load_subject(
    file_root: str,
    subject: str,
    *,
    pkl_suffix: str = "_features.pkl",
    mat_suffix: str = "_electrode_loc.mat",
    require_xyz: bool = False,
    raw_root: str = "",
) -> SubjectData:
    """Load prepared windows and aligned electrode coordinates.

    Parameters
    ----------
    file_root : str
        Directory with prepared feature pickle and cleaned MATLAB file.
    subject : str
        Stanford subject identifier.
    pkl_suffix : str, optional
        Feature-pickle suffix, default ``_features.pkl``.
    mat_suffix : str, optional
        Legacy coordinate-file suffix, default
        ``_electrode_loc.mat``.
    require_xyz : bool, optional
        Require coordinates, default False.
    raw_root : str, optional
        Read-only raw recording root, default empty. When provided,
        select locs using channelName even if a legacy file exists.

    Returns
    -------
    SubjectData
        Prepared arrays and optional (C, 3) coordinates.
    """
    pkl_path = os.path.join(file_root, f"{subject}{pkl_suffix}")
    X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te = load_features_pkl(pkl_path)

    mat_path = os.path.join(file_root, f"{subject}{mat_suffix}")
    ecog_xyz_mm = None
    source = raw_root or os.environ.get(
        "CORTEG_STANFORD_RAW_ROOT", ""
    )
    if require_xyz and (source or not os.path.isfile(mat_path)):
        if X_raw_tr.ndim != 3:
            raise ValueError(
                f"{subject}: expected X_raw_tr shape (N, C, T), "
                f"got {X_raw_tr.shape}."
            )
        source = source or os.path.join(
            file_root, "raw", "Stanford"
        )
        ecog_xyz_mm = load_raw_xyz_for_channels(
            file_root, source, subject, int(X_raw_tr.shape[1])
        )
    elif os.path.isfile(mat_path):
        ecog_xyz_mm = load_ecog_xyz_mm(mat_path)

    # shape checks (canonical contract)
    if X_feat_tr.ndim != 4 or X_feat_tr.shape[-1] != 2:
        raise ValueError(f"{subject}: X_feat_tr must be (N,C,200,2), got {X_feat_tr.shape}")
    if X_raw_tr.ndim != 3:
        raise ValueError(f"{subject}: X_raw_tr must be (N,C,128), got {X_raw_tr.shape}")
    if y_tr.ndim != 2 or y_tr.shape[1] != 5:
        raise ValueError(f"{subject}: y_tr must be (N,5), got {y_tr.shape}")

    C = int(X_raw_tr.shape[1])
    if ecog_xyz_mm is not None and int(ecog_xyz_mm.shape[0]) != C:
        raise ValueError(f"{subject}: xyz has C={ecog_xyz_mm.shape[0]} but X_raw has C={C}")

    return SubjectData(
        subject=subject,
        X_feat_tr=np.asarray(X_feat_tr),
        y_tr=np.asarray(y_tr),
        X_feat_te=np.asarray(X_feat_te),
        y_te=np.asarray(y_te),
        X_raw_tr=np.asarray(X_raw_tr),
        X_raw_te=np.asarray(X_raw_te),
        ecog_xyz_mm=ecog_xyz_mm,
    )
