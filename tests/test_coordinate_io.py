"""Check Stanford channel-name selection of read-only raw coordinates.

Input: temporary cleaned MATLAB files, raw locs, and prepared feature pickles.
Output: assertions on coordinate order and absence of raw-root writes.
"""
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from data.io import load_raw_xyz_for_channels, load_subject


class ChannelLocationTests(unittest.TestCase):
    """Check raw coordinate selection after channel removal."""

    def test_channel_names_select_raw_locs_without_new_file(self) -> None:
        """Select channels one and three in prepared array order."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            raw = root / "raw"
            prepared.mkdir()
            (raw / "bp").mkdir(parents=True)
            savemat(
                prepared / "bp.mat",
                {"channelName": np.array(["ch3", "ch1"], dtype=object)},
            )
            locs = np.arange(12, dtype=float).reshape(4, 3)
            savemat(raw / "bp" / "bp_fingerflex.mat", {"locs": locs})
            arrays = (
                np.zeros((2, 2, 200, 2)),
                np.zeros((2, 5)),
                np.zeros((1, 2, 200, 2)),
                np.zeros((1, 5)),
                np.zeros((2, 2, 128)),
                np.zeros((1, 2, 128)),
            )
            with (prepared / "bp_features.pkl").open("wb") as stream:
                pickle.dump(arrays, stream)
            before = set(raw.rglob("*"))
            subject = load_subject(
                str(prepared), "bp", require_xyz=True,
                raw_root=str(raw),
            )
            np.testing.assert_array_equal(
                subject.ecog_xyz_mm, locs[[2, 0]]
            )
            self.assertEqual(before, set(raw.rglob("*")))
            self.assertFalse(
                (prepared / "bp_electrode_loc.mat").exists()
            )

    def test_duplicate_channel_name_is_rejected(self) -> None:
        """Reject ambiguous channel mapping before returning coordinates."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw" / "bp").mkdir(parents=True)
            savemat(
                root / "bp.mat",
                {"channelName": np.array(["ch1", "ch1"])},
            )
            savemat(
                root / "raw" / "bp" / "bp_fingerflex.mat",
                {"locs": np.ones((2, 3))},
            )
            with self.assertRaisesRegex(ValueError, "duplicate channels"):
                load_raw_xyz_for_channels(
                    str(root), str(root / "raw"), "bp", 2
                )

    def test_native_output_cannot_write_inside_raw_root(self) -> None:
        """Reject an output folder within the raw download."""
        from data.stanford_native import main as native_main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            with self.assertRaisesRegex(ValueError, "inside raw root"):
                native_main([
                    "--subjects", "bp",
                    "--raw_root", str(raw),
                    "--out_root", str(raw / "derived"),
                    "--data_root", str(root / "prepared"),
                ])
            self.assertFalse((raw / "derived").exists())
