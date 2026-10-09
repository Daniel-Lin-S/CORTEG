"""
DESCRIPTION:
    Prepare ECoG windows paired with finger-flexion targets.
    Input: cleaned subject MATLAB files and resolved split config.
    Output: one feature pickle per subject with six train/test arrays.

"""

from scipy.io import loadmat
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from data.stanford_split import load_split_config, split_boundary
import pickle
import numpy as np
import mne
from mne.filter import resample
from scipy.signal import hilbert

# metadata for different datasets
datasets = {
    "Stanford": {
        "subjects": 9,
        "subject_name": ['bp', 'cc', 'ht', 'jc', 'jp', 'mv', 'wc', 'wm', 'zt'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        # file path to the preprocessed data from MATLAB
        "path": str(REPO_ROOT / 'preprocessed_data' / 'Stanford')
    },
}

def HGALFS_feature_extractor(
    data, traj, fs, window_size=1000, step_size=50, T=200, delay=None
):
    """
    INPUT:
        data: [time, channel] ECoG data, 1 kHz
        traj: [time, 5] trajectory of 5 fingers, 1 kHz
        window_size: window size of ECoG in ms
        step_size: step size of sliding windows in ms
        T: number of time points after downsampling
        delay: optional time delay between ECoG and trajectory

    OUTPUT:
        X: [nEpoch, nChannel, T, nBand]
        Y: [nEpoch, 5]
        X0: [nEpoch, nChannel, window_size/fs * 128 Hz]
    """
    assert data.shape[0] == traj.shape[0], (
        "ECoG and trajectory must have the same sample count."
    )
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    if delay is not None:
        data, traj = data[delay:, :], traj[:-delay, :]

    # Downsample trajectory to match step size
    traj_ds = resample(traj.T, down=step_size, npad='auto').T

    X, Y  = [], []
    X0 = []
    n_samples = len(traj_ds)
    for k in range(n_samples):
        end = k * step_size
        start = end - window_size
        if start < 0:
            continue
        data_win = data[start:end].T
        traj_point = traj_ds[k]

        n_channels, n_times = data_win.shape
        subwin_len = n_times // T  # for downsampling

        # extract HGA (n_channels, T)
        iir_params = dict(order=4, ftype='butter')
        data_filtered = mne.filter.filter_data(data_win, sfreq=fs,
                                          l_freq=70 , h_freq=200,
                                          method='iir', iir_params=iir_params,
                                          verbose=False)
        analytic = hilbert(data_filtered, axis=1)
        envelope = np.abs(analytic)  # (n_channels, n_times)
        envelope = envelope[..., :subwin_len * T]

        HGA = envelope.reshape(n_channels, T, subwin_len).mean(axis=2)

        # extract LFS (n_channels, T)
        LFS = resample(data_win, down=5, npad='auto')

        # -> (n_channels, T, n_bands)
        features = np.stack([HGA, LFS], axis=-1)

        X.append(features)
        Y.append(traj_point)

        data_win_ds = resample(data_win, up=16, down=125, npad='auto')
        X0.append(data_win_ds)

    return np.array(X), np.array(Y), np.array(X0)

def main() -> None:
    """Extract paired windows from each prepared Stanford recording.

    Input
    -----
    ``preprocessed_data/Stanford/<subject>.mat`` and ``resolved_config.json``.

    Output
    ------
    ``<subject>_features.pkl`` in the same folder, containing six arrays:
    train ECoG and flexion, test ECoG and flexion, and train/test 128 Hz ECoG.
    """
    selected = 'Stanford'
    file_root = Path(datasets[selected]['path'])
    fs_ecog = datasets[selected]['fs_ecog']
    fs_dg = datasets[selected]['fs_dg']
    split = load_split_config(file_root)

    for subject in datasets[selected]['subject_name']:
        input_path = file_root / f'{subject}.mat'
        recording = loadmat(input_path)
        ecog = recording['data'].T
        trajectory = recording['flex']
        if ecog.shape[0] != trajectory.shape[0]:
            raise ValueError(
                f'{input_path}: ECoG and flexion sample counts differ.'
            )
        boundary, _ = split_boundary(ecog.shape[0], fs_ecog, split)
        ecog_train = ecog[:boundary, :]
        ecog_test = ecog[boundary:, :]
        trajectory_train = trajectory[:boundary, :]
        trajectory_test = trajectory[boundary:, :]
        train = HGALFS_feature_extractor(
            ecog_train, trajectory_train, fs_ecog,
            window_size=fs_ecog, step_size=fs_ecog // fs_dg,
            T=200, delay=40,
        )
        test = HGALFS_feature_extractor(
            ecog_test, trajectory_test, fs_ecog,
            window_size=fs_ecog, step_size=fs_ecog // fs_dg,
            T=200, delay=40,
        )
        output_path = file_root / f'{subject}_features.pkl'
        with output_path.open('wb') as stream:
            pickle.dump(
                [train[0], train[1], test[0], test[1],
                 train[2], test[2]],
                stream,
            )
        print(f'Finished subject: {subject}; saved {output_path}', flush=True)

if __name__ == "__main__":
    main()


