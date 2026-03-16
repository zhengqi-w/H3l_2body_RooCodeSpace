# Unified Workflow (single-entry orchestration)

This folder now runs analysis through one entry macro:

- `Tasks/ProcessAnalysis.C`

The engine executes new unified task code directly.

## Entry point

- ROOT macro: `Workflow/ProcessWorkflow.C`
- Shell script: `Workflow/run_workflow.sh`

Example:

```bash
cd ROOTWorkFlow/CodeSpace
bash Workflow/run_workflow.sh configs/general_config.json
```

## Config tree

Top-level config is `configs/general_config.json`.

- `common`: global paths/tree names/physics parameters/shared options
- `execution`: runtime mode/MT/stop-on-error
- `analysis`: unified binning/selection/fit/correction/systematics
- `checks`: check switches and variable list
- `output`: output controls and compatibility switches
- optional `workflow.order`: run a list of modes in one workflow call

No task-level generated effective config indirection is required in the new execution path.
