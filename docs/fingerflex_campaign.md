# FingerFlex campaign configuration and cache

[`configs/stanford_preprocessing.json`](../configs/stanford_preprocessing.json)
contains the cleaning and chronological split settings. Its `defaults` and
`overrides` are resolved before preparing samples. The sample grid and feature
transform are versioned in `data/fingerflex_cache.py`.

[`configs/fingerflex_campaign.json`](../configs/fingerflex_campaign.json)
contains the 0 ms study name, data and artifact locations, subject/model/
fraction/seed selectors, checkpoint, and training hyperparameters.
``configs/fingerflex_delay40ms.json`` selects the three-subject, CORTEG-only
100% probe with the same training and preprocessing settings. Both configs
reference the same preprocessing config without duplicating its settings.

Prepared samples live at
`/data/lyh/CORTEG/fingerflex_stanford/preprocessing/<preprocessing-id>/subjects`.
The preprocessing ID depends on resolved preprocessing settings and the sample
extraction version, not on learning rates, epochs, seeds, or subjects. Existing
validated arrays are linked from the completed original campaign without
copying them. A training artifact's `prepared` entry points to this cache.

A training artifact has a separate semantic ID based on the preprocessing ID,
training settings, checkpoint hash, and model source contents. Seeds and
subjects can be added to the same artifact root. Nonzero delay changes the
training identity but not the shared preprocessing cache. The 162 main fits
and nine 40 ms probe fits are linked from the original completed artifact;
launching the same settings finds them complete and starts no GPU work. Exact
launch
configurations are retained under `provenance/launches/`; resolved
preprocessing and training settings are saved under `provenance/`.

Preparation can be run independently. A cache hit verifies input hashes and
returns without extracting samples. Then launch a study from its JSON file:

```bash
/data/lyh/conda_envs/EEG-FM-Bench/bin/python \
  /home/lyh/codes/CORTEG/scripts/fingerflex_campaign.py prepare \
  --config /absolute/path/to/new_training_config.json
/data/lyh/conda_envs/EEG-FM-Bench/bin/python \
  /home/lyh/codes/CORTEG/scripts/fingerflex_campaign.py launch \
  --config /absolute/path/to/new_training_config.json
```

Generate 27 subject plots and a three-metric aggregate plot from the completed
0 ms report:

```bash
MPLCONFIGDIR=/tmp/corteg_matplotlib \
/data/lyh/conda_envs/EEG-FM-Bench/bin/python \
  /home/lyh/codes/CORTEG/scripts/plot_fingerflex_campaign.py \
  --root /data/lyh/CORTEG/fingerflex_stanford/stanford_finger_small_hilo_0ms__c4bff97e9f9d
```

Subject bands show sample standard deviation across seeds. Aggregate bands
show standard deviation across nine subject seed means. Aggregate brackets
show paired subject sign-flip tests at Holm-adjusted p < 0.05, with Holm
correction across the three metrics within each training proportion.
