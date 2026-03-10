# Unified Workflow (single-entry orchestration)

This folder provides a single-entry workflow runner for the three main extraction tasks:

- `bdt_spectrum` -> `Tasks/ProcessBdtSpectrum.C`
- `topology_spectrum` -> `Tasks/ProcessTopologySpectrum.C`
- `ct_extraction` -> `Tasks/ProcessCtSpectrum.C`
- `ct_single` -> `Tasks/ProcessCtSingleSpectrum.C`

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
- `bdt_spectrum` / `topology_spectrum` / `ct_extraction` / `ct_single`: task sections
- optional `workflow.order`: execution order override
- optional `workflow.stop_on_error`: fail-fast switch
- optional `workflow.generated_config_dir`: generated task-config output dir

The orchestrator writes effective configs (for legacy task macros) to `Workflow/generated_configs/` and dispatches all tasks from the single general config.
