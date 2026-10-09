# Regression dataset preparation

This note covers data cleaning, train/test splitting, and construction of
neural-signal/target pairs for the two regression datasets. It describes only
steps supported by the released files. Model inputs and training are outside
its scope.

## Stanford finger trajectories (public)

The `fingerflex` recordings contain nine patients (`bp`, `cc`, `ht`, `jc`, `jp`,
`mv`, `wc`, `wm`, `zt`). ECoG is sampled at 1,000 Hz. The five-finger data-glove
signal was acquired at 25 Hz but is stored on a 1,000 Hz grid alongside ECoG;
the stored samples do not represent new 1,000 Hz glove measurements. Source:
[Stanford Digital Repository](https://searchworks.stanford.edu/view/zk881ps0522).

### Cleaning

`data_preprocessing_Stanford.m` reads each raw `<subject>_fingerflex.mat`,
transposes and scales ECoG by 0.0298 and applies 59.5–60.5, 119.5–120.5,
and 179.5–180.5 Hz band-stop filters. It then computes each channel’s variance
over **training samples only**, before common-average reference. The median is
taken over all channels. A channel is excluded if its variance is strictly
above `varianceHighMultiplier × median` or strictly below
`varianceLowMultiplier × median`; defaults are `5` and `0.2`, respectively, in
`configs/stanford_preprocessing.json`. The JSON includes defaults and
overrides; the merged settings are saved to
`preprocessed_data/Stanford/resolved_config.json`.
The remaining channels receive common-average reference. The same retained
channel set applies to training and test data.

The cleaned `<subject>.mat` saves `data` (channels × time), `flex`
(time × five fingers), `fs`, retained `channelName`, removed
`badChannelName`, per-channel variance, the median variance, training-sample
count, and both thresholds. The recording root is read only. This automatic
rule replaces the earlier interactive selection, for which no complete
per-subject exclusion list was saved; data regenerated now may have different
channels from earlier reported runs. No separate electrode-location file is
needed: retained `channelName` identifies rows of raw `locs` at runtime.

### Split and sample extraction

`prepare_taskFormatedData_Stanford.py` splits each subject chronologically,
before window extraction. If the recording lasts at least 600 seconds, the
first 400 seconds are training and the remainder is test. Otherwise, the first
two thirds are training and the final third is test. Windows are generated
separately within those segments, so none crosses the split boundary.
The split parameters mean:

| Parameter | Meaning |
| --- | --- |
| `split.duration_cutoff_seconds` | Duration at which the fixed-time rule applies |
| `split.at_or_above_cutoff.train_seconds` | Training duration for recordings at or above the cutoff |
| `split.below_cutoff.train_parts` | Number of equal whole-sample parts used for training below the cutoff |
| `split.below_cutoff.total_parts` | Total equal whole-sample parts below the cutoff |

Within each segment, the script aligns arrays with `data = data[40:]` and
`flex = flex[:-40]`, then resamples `flex` from its stored 1,000 Hz grid to
25 Hz. For index `k >= 25`, the target is resampled five-finger flexion at
`k × 40` ms relative to that segment's start. Its paired ECoG window has
1,000 samples, advances by 40 samples per pair, and covers original segment
indices `[k × 40 - 960, k × 40 + 40)`. Thus the neural window reaches about
40 ms **after** the target time; the code's delay direction matters when
assessing causal prediction.

Each ECoG window yields two `(channels, 200)` arrays: a 70–200 Hz filtered
Hilbert-magnitude envelope averaged into 200 bins, and cleaned ECoG resampled
to 200 Hz. A further cleaned-ECoG copy is resampled to 128 Hz. The extractor
does not explicitly apply a 1 Hz high-pass to the 200 Hz or 128 Hz copies.
Per subject, `<subject>_features.pkl` stores six arrays in this order:

| Array | Contents and shape |
| --- | --- |
| `ECoG_train` | Train windows, `(N_train, C, 200, 2)`; last axis is envelope, resampled ECoG |
| `trajectory_train` | Aligned train targets, `(N_train, 5)` |
| `ECoG_test` | Test windows, `(N_test, C, 200, 2)` |
| `trajectory_test` | Aligned test targets, `(N_test, 5)` |
| `ECoG_train_128` | Train windows of resampled ECoG, `(N_train, C, 128)` |
| `ECoG_test_128` | Test windows of resampled ECoG, `(N_test, C, 128)` |

Within either split, row `i` of every neural array pairs with row `i` of its
five-value trajectory array. The feature files themselves are not included in
this repository; the two preprocessing scripts and `tutorial_Stanford.md`
document their creation.

## Ghent audio envelope (private)

The release identifies 16 subjects and reports a single audio-envelope
regression target per sample. The recordings, target arrays, data-cleaning
code, and sample-extraction code are not released.

| Data step | Available information |
| --- | --- |
| Cleaning and audio-envelope derivation | Unavailable. |
| Train/test boundaries and validation split | Boundaries unavailable. An archived Ghent foundation-model result records `val_ratio=0.1`, but the split implementation is unavailable. |
| Neural-window duration and target alignment | Unavailable. |
| Sample spacing | The demo builder reports a 46.875 ms step (six samples at 128 Hz) for its Ghent traces; the underlying loader is unavailable for verification. |
| Final paired arrays | Unavailable; released result and demo files cannot reconstruct the training and test pairs. |

The Ghent statements above come from `REPRODUCING.md`,
`build_demo_data.py`, and the released Ghent result and demo files. They do
not establish a reproducible preprocessing recipe.
