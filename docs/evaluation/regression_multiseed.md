# Regression experiments: identity, seeds, and paired tests

`scripts/multiseed_regression.py` coordinates
per-subject regression jobs and evaluates saved results. It supports any task
with per-subject result JSON, including Stanford finger trajectories and
Ghent audio envelopes when their result files exist. The private Ghent data
and preparation code are unavailable in this release.

## Named semantic identity

Run the example from the repository root and save its specification
there so relative paths resolve correctly.

The JSON specification needs a readable top-level `name`, `semantic_config`,
and nonempty `tasks`. Each task needs `name`, `protocol_id`, `dataset`,
`semantic_config`, and `models`. Each model needs `name` and
`semantic_config`. The semantic config at each level can be a concrete JSON
object or `{ "defaults": {...}, "overrides": {...} }`; overrides are merged
recursively, and unknown override keys are errors. A nested
`{ "$json": "/absolute/path/to/config.json" }` loads another JSON config,
including its defaults and overrides. This lets an experiment use the
Stanford preprocessing settings in
`configs/stanford_preprocessing.json` without copying
them into the experiment file.

The artifact ID is a SHA-256 hash of resolved semantic configs, task dataset
and protocol, and launch templates after invocation controls are removed.
Seed, device, subject selection, data paths, output paths, worker count,
result paths, config-file paths, and the readable name do not enter the hash.
For config files passed to commands, include their resolved contents in
`semantic_config` using `$json`; the path alone is not semantic. The directory is
`OUTPUT_BASE/<readable-name-slug>__<first-16-hash-digits>/`. Its
`experiment_identity.json` stores the full hash and exact resolved semantic
configuration; `spec_versions/<full-spec-hash>.json` preserves each invocation
specification, including subjects, seeds, paths, and device settings. When
launched result JSON files include an `args` object, `aggregate.json` also
records the effective runner arguments after removing invocation fields and
rejects differences across that model’s subject and seed runs. Declare
all settings that affect interpretation, such as alignment, features,
checkpoint identity, model architecture, and training choices. A command's
remaining semantic arguments also enter the hash. Changing a semantic setting
creates a new artifact root.

For a launched model, `subjects` is a task list and `seeds` is a model list.
The `command` argument vector must include `{subject}`, `{seed}`, and
`{save_root}`. Example:

```json
{
  "name": "Finger regression comparison",
  "semantic_config": {
    "metric": "mean Pearson r over five finger targets"
  },
  "tasks": [
    {
      "name": "stanford_finger",
      "dataset": "Stanford",
      "protocol_id": "chronological_40ms_v1",
      "semantic_config": {
        "preprocessing": {
          "$json": "configs/stanford_preprocessing.json"
        },
        "target_alignment_ms": 40
      },
      "subjects": ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"],
      "models": [
        {
          "name": "STEEGFormer_small",
          "semantic_config": {
            "architecture": "ST-EEGFormer-small",
            "checkpoint_release": "ST-EEGFormer-small checkpoint-300",
            "epochs": 200,
            "model_kwargs": {
              "$json": "/path/to/validated/small_config.json"
            }
          },
          "seeds": [1, 2, 42],
          "command": [
            "python",
            "experiments/run_regression_hilo_clean.py",
            "--dataset", "Stanford",
            "--train_mode", "per_subject",
            "--data_root", "/path/to/prepared/Stanford",
            "--raw_root", "<RAW_ROOT>",
            "--model_kwargs_json", "/path/to/validated/small_config.json",
            "--subjects", "{subject}",
            "--seed", "{seed}",
            "--save_root", "{save_root}"
          ]
        }
      ]
    }
  ]
}
```

The paths under `/path/to` must be replaced. The existing small model config
must point to an actual checkpoint before a real run; the coordinator does not
change model settings or repair checkpoint paths. Other models can use their
own command so long as each job writes one-subject `results_persub.json` under
its save root. `result_file` can name another relative JSON path. Existing
full-cohort result files can instead be supplied with a model's `results`
mapping from seed string to JSON path. An experiment may contain both launched
models and existing results; each task uses a shared protocol ID.

```bash
python scripts/multiseed_regression.py run \
  --spec /path/to/experiment.json \
  --output-root /path/to/output_base \
  --max-parallel 2

python scripts/multiseed_regression.py evaluate \
  --spec /path/to/experiment.json \
  --output-root /path/to/output_base
```

A run goes under `task/model/subjects/<subject>/seedN/` in the named artifact
root, with `run.log` and its usual outputs. Reusing the same spec skips
completed subject and seed jobs. Extending `subjects` or `seeds` adds only new
folders under the same root and records a new spec revision. An incomplete
existing job folder is preserved and reported for inspection. The manifest
appends new job records. `evaluate` never trains. The raw FingerFlex root is
an input and is never used as an output.

## Evaluation contract

Each result JSON needs a nonempty `per_subject` map. Every subject entry needs
`corr_mean` (finite mean Pearson correlation across targets) and `n` (positive
held-out window count). A launched job's result must contain only its assigned
subject. A task's `subjects` list, when provided, is the required complete
set; models and seeds must have exactly those subjects and identical test
window counts per subject. No subject intersection or missing-run fallback is
used. For archived results, list the subject IDs explicitly when possible.
A private task may be evaluated from saved results without access to its raw
recordings.

`aggregate.json` gives each subject's mean and sample SD across seeds, the
cohort mean and sample SD across subjects, and the sample SD of cohort means
across seeds. An SD is `null` when only one observation exists.
`comparisons.json` gives every model pair's subject differences, mean and
median differences, a two-sided paired Wilcoxon signed-rank test, and Holm
adjustment over model pairs within the task. Subjects are independent test
units; overlapping windows and random seeds do not inflate test sample size.
Seeds are averaged within subject before the paired test. Fewer than two
subjects cannot support the paired test.

Matching subject IDs and window counts cannot prove identical target samples
or preprocessing from aggregate scores alone. Use one verified test grid and
record its alignment, stride, and other semantics in the JSON configuration;
do not combine results from different test grids under one protocol ID.
