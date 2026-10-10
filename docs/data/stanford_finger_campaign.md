# Stanford FingerFlex campaign data

Raw files are read from the registered Stanford dataset and never modified.
Each patient file supplies 1 kHz `data`, five `flex` channels on the 25 Hz
finger grid (stored at 1 kHz), and electrode coordinates in `locs`.

The Python cleaner multiplies ECoG by 0.0298, then applies three third-order
Butterworth band-stop filters at 59.5–60.5, 119.5–120.5, and 179.5–180.5 Hz.
It ports FieldTrip's `onepass` behavior: for each pass it demeans each channel,
subtracts the first sample before forward filtering, applies the filter's DC
gain to that sample, and restores the channel mean. The implementation follows
[FieldTrip's band-stop filter](https://github.com/fieldtrip/fieldtrip/blob/master/preproc/ft_preproc_bandstopfilter.m)
and its [one-pass edge correction](https://github.com/fieldtrip/fieldtrip/blob/master/preproc/private/filter_with_correction.m).
SciPy and MATLAB filter coefficients can differ slightly numerically.

A chronological training boundary is 400 s for recordings at least 600 s long;
otherwise it is two-thirds of the recording. Channel variance is computed on
**the entire training segment** after notch filtering, using sample variance.
Channels outside 0.2–5 times the across-channel median variance are removed.
The remaining channels receive common-average reference. This channel choice
uses training data beyond the selected 25% or 10% fit prefix; it never uses
test data. Coordinates are selected directly from `locs` with the same channel
indices. No separate coordinate file is required in the raw dataset.

In each training and test segment, targets are the original finger values at
1 kHz indices `1000, 1040, …` up to (but excluding) `segment_length − 40`.
Thus the target rows are 25 Hz, with no target resampling. Each neural window
has 1,000 samples. At 0 ms it ends at the target index; at 40 ms it ends 40
samples later. The target indices and values are identical at both delays.
Windows never cross the chronological train/test boundary.

The final `floor(10%)` of training windows are validation rows. The fit set
uses the earliest 100%, 25%, or 10% of the remaining windows. Target and
feature normalization statistics use only those selected fit rows. Saved
per-patient metadata includes the split boundary, retained channels, channel
variance, and target-grid/value hashes. Prepared feature arrays are shared
across seeds and training fractions; the 40 ms feature arrays are separate.
