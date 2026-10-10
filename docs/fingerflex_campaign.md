# FingerFlex campaign configuration and cache

[`configs/stanford_preprocessing.json`](../configs/stanford_preprocessing.json)
contains the cleaning and chronological split settings. Its `defaults` and
`overrides` are resolved before preparing samples. The sample grid and feature
transform are versioned in `data/fingerflex_cache.py`.

[`configs/fingerflex_campaign.json`](../configs/fingerflex_campaign.json)
contains the study name, data and artifact locations, subject/fraction/seed
selectors, checkpoint, and training hyperparameters. It only **references**
the preprocessing config; it does not duplicate its settings.

Prepared samples live at
`/data/lyh/CORTEG/fingerflex_stanford/preprocessing/<preprocessing-id>/subjects`.
The preprocessing ID depends on resolved preprocessing settings and the sample
extraction version, not on learning rates, epochs, seeds, or subjects. Existing
validated arrays are linked from the completed original campaign without
copying them. A training artifact's `prepared` entry points to this cache.

A training artifact has a separate semantic ID based on the preprocessing ID,
training settings, checkpoint hash, and model source contents. Seeds and
subjects can be added to the same artifact root. The 171 completed fits from
the original campaign are linked into the new identity, so launching the same
settings finds them complete and starts no GPU work. Exact launch
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
