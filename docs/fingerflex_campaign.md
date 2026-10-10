# FingerFlex campaign settings

The tracked `configs/fingerflex_campaign.json` and
`configs/fingerflex_delay40ms.json` contain subject, model, fraction,
seed, delay, and training settings. Both reference
`configs/stanford_preprocessing.json` for cleaning and splitting.

Copy `configs/fingerflex_local.example.json` to the ignored
`configs/fingerflex_local.json` and fill the raw dataset root, artifact
base, checkpoint, HiLoFuseNet model source, and tmux executable. These
machine-specific paths do not belong in tracked configs. The runner hashes
the checkpoint bytes automatically; its SHA-256 is saved in the resolved
artifact provenance and participates in the training semantic ID.

Prepared arrays live under
`<artifact_base>/preprocessing/<preprocessing-id>/subjects`.
The preprocessing ID depends on resolved cleaning and sample extraction,
so campaigns with the same preparation reuse these arrays. Training
artifacts have separate semantic IDs, and completed seed fits are reused.

From the repository root, with the local Python environment active:

``bash
python scripts/fingerflex_campaign.py prepare \
  --config configs/fingerflex_campaign.json
python scripts/fingerflex_campaign.py launch \
  --config configs/fingerflex_campaign.json
python scripts/fingerflex_campaign.py launch \
  --config configs/fingerflex_delay40ms.json
``

The plot command reads completed results and writes subject curves and an
aggregate figure under the artifact's `reports/plots/` directory:

``bash
python scripts/plot_fingerflex_campaign.py --root "$ARTIFACT_ROOT"
``

Subject bands show seed standard deviation. Aggregate bands show
between-subject standard deviation. Aggregate brackets mark paired
subject sign-flip tests with Holm correction across the three metrics
within each training proportion.
