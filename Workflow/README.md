# Unified Workflow (single-entry orchestration)

This folder supports two orchestration entries:

- `Workflow/RunUnifiedPipeline.C`: run `train`, `wp`, `analysis` or `all` stages.

## Entry point

- ROOT macro: `Workflow/RunUnifiedPipeline.C`
- Shell script: `Workflow/run_pipeline.sh`

Examples:

```bash
cd ROOTWorkFlow/CodeSpace
bash Workflow/run_pipeline.sh configs/general_config.json all false
bash Workflow/run_pipeline.sh configs/general_config.json train true
```

Arguments for `run_pipeline.sh`:

1. config path (default: `configs/general_config.json`; passing another config prints an info hint)
2. stage: `train | wp | analysis | all` (default: `all`)
3. dry-run: `true | false` (default: `false`)

## Config tree

Top-level config is `configs/general_config.json`.

- `common`: global paths/tree names/physics parameters/shared options
- `execution`: runtime switches and mode routing
- `preprocess.bdt`: BDT training config buffer
- `preprocess.wp`: WP extraction config buffer
- `analysis`: unified binning/selection/fit/correction/systematics
- `checks`: check switches and variable list
- `output`: output controls and compatibility switches

Mode routing in `execution`:

- `training_mode`: drives BDT training
- `wp_mode`: drives WP extraction
- `analysis_mode`: drives ProcessAnalysis

For every executed stage, a manifest is written to `Workflow/manifests/run_manifest_<stage>.json`.
Add `Workflow/manifests/run_manifest_*.json` to `.gitignore` to keep runtime artifacts out of commits.
