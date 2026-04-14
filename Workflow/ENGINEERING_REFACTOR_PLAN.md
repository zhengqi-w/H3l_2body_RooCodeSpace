# Unified Analysis Engineering Refactor Plan

## 1. Goals

- One command to run full pipeline or selected stage.
- One source of truth for configuration.
- Clear module boundaries for maintainability and testability.
- Backward compatibility during migration.

## 2. Current Pain Points and Direct Fixes

### Pain 1: Three configs are hard to keep in sync

Current files:
- `configs/PreProcess/config_BDT.yaml`
- `configs/PreProcess/config_WP.json`
- `configs/general_config.json`

Fix:
- Introduce a layered config model:
  - `configs/pipeline/base.yaml`: global defaults, tree names, common physics constants.
  - `configs/pipeline/profiles/{bdt_spectrum,topology_spectrum,pt_ct,ct_single}.yaml`: mode-specific binning and switches.
  - `configs/pipeline/periods/{period_name}.yaml`: period/path overrides.
  - `configs/pipeline/local.template.yaml`: user-local machine paths.

### Pain 2: Need to run 3 macros manually

Current execution flow:
- `PreProcess/BDTPreProces.py`
- `PreProcess/ProcessWP.C`
- `Tasks/ProcessAnalysis.C`

Fix:
- Add a unified CLI runner:
  - `Workflow/pipeline_cli.py` (or shell wrapper)
  - stages: `train`, `wp`, `analysis`, `all`
  - options: `--mode`, `--config`, `--dry-run`, `--resume`

### Pain 3: API and code organization are scattered

Current large orchestrator:
- `Tools/tasks/UnifiedTaskRunner.cxx`

Fix:
- Split into focused modules (see section 4 and section 5).

## 3. Target Directory Layout

```text
CodeSpace/
  Workflow/
    pipeline_cli.py
    run_pipeline.sh
    README.md
  configs/
    pipeline/
      base.yaml
      periods/
        LHC23_PbPb_pass5.yaml
      profiles/
        bdt_spectrum.yaml
        topology_spectrum.yaml
        pt_ct.yaml
        ct_single.yaml
      schemas/
        pipeline.schema.json
      local.template.yaml
  Tools/
    pipeline/
      ConfigLoader.h
      ConfigLoader.cxx
      ConfigValidator.h
      ConfigValidator.cxx
      StageDispatcher.h
      StageDispatcher.cxx
    fit/
      MassFitService.h
      MassFitService.cxx
      LifetimeFitService.h
      LifetimeFitService.cxx
    corrections/
      AcceptanceService.h
      AcceptanceService.cxx
      AbsorptionService.h
      AbsorptionService.cxx
      BinningCorrectionService.h
      BinningCorrectionService.cxx
    io/
      SnapshotReader.h
      SnapshotReader.cxx
      RootOutputWriter.h
      RootOutputWriter.cxx
    systematics/
      SystematicsRunner.h
      SystematicsRunner.cxx
    plotting/
      SpectrumPlotService.h
      SpectrumPlotService.cxx
```

Note:
- Keep old paths temporarily and provide adapters until migration is complete.

## 4. File Migration Map (Old -> New)

### Entry macros and scripts
- `PreProcess/BDTPreProces.py` -> `Workflow/pipeline_cli.py` stage `train` + `Tools/pipeline/StageDispatcher` hooks.
- `PreProcess/ProcessWP.C` -> `Workflow/pipeline_cli.py` stage `wp` + dedicated WP service module.
- `Tasks/ProcessAnalysis.C` -> `Workflow/pipeline_cli.py` stage `analysis`.

### Core analysis runner
- `Tools/tasks/UnifiedTaskRunner.cxx` -> split into:
  - `Tools/pipeline/StageDispatcher.cxx` (high-level orchestration)
  - `Tools/fit/LifetimeFitService.cxx` (exponential, blast-wave post-fit)
  - `Tools/fit/MassFitService.cxx` (mass fit entry)
  - `Tools/systematics/SystematicsRunner.cxx` (trails and filtering)
  - `Tools/io/RootOutputWriter.cxx` (write ROOT and PDF export)

### Helper headers
- `Tools/AcceptanceHelper.h` -> `Tools/corrections/AcceptanceService.*`
- `Tools/AbsorptionHelper.h` -> `Tools/corrections/AbsorptionService.*`
- `Tools/tasks/SpectrumPlotHelper.h` -> `Tools/plotting/SpectrumPlotService.*`
- `Tools/GeneralHelper.hpp` -> split by concern:
  - config parsing helpers -> `Tools/pipeline/ConfigLoader.*`
  - generic string/path utils -> `Tools/common/StringPathUtils.*`

### Config files
- `configs/PreProcess/config_BDT.yaml` + `configs/PreProcess/config_WP.json` + `configs/general_config.json`
  -> `configs/pipeline/base.yaml` + profile + period override.

## 5. Function-Level Refactor Boundaries

### Keep in UnifiedTaskRunner only
- mode dispatch and stage lifecycle.
- top-level error handling.

### Move out of UnifiedTaskRunner
- config extraction and defaults.
- acceptance/absorption cache building.
- lifetime fit and post-fit canvas generation.
- output directory and ROOT object writing logic.
- systematics trail loops.

This reduces function length and makes each module unit-testable.

## 6. Naming Conventions

### Files and directories
- Use lowercase snake_case for new files and directories.
- Use noun-based module names by domain, not by historical step names.

### C++ APIs
- Classes: `PascalCase`.
- Methods and variables: `camelCase`.
- Constants: `kPascalCase`.
- Avoid abbreviations like `cfg`, `abso` in public interfaces; use full words.

### Config keys
- Use snake_case consistently.
- Keep backward-compatible aliases for one transition cycle only.

## 7. Config Validation Rules to Add

- Required path keys must exist and be readable before run.
- For `cen_bins` and `pt_bins_by_centrality`: size relation must be `len(pt_bins_by_centrality) == len(cen_bins) - 1`.
- For `pt_bins` and `ct_bins_by_pt`: same relation check.
- Working point file existence check per mode.
- Disallow unknown mode names at validation stage.

## 8. Unified CLI Spec (Minimal)

Examples:
- `python Workflow/pipeline_cli.py run --stage all --mode bdt_spectrum --config configs/pipeline/base.yaml`
- `python Workflow/pipeline_cli.py run --stage analysis --mode ct_single --resume`
- `python Workflow/pipeline_cli.py validate --config configs/pipeline/base.yaml`

Behavior:
- `--dry-run`: print resolved config and commands only.
- `--resume`: skip stage if output marker exists.
- write `run_manifest.json` for each stage.

## 9. Incremental Migration Plan

### Phase 1 (Low risk, 1-2 days)
- Add unified CLI wrapper that calls existing scripts/macros unchanged.
- Add config validator script.
- Add stage manifests.

### Phase 2 (Medium risk, 3-5 days)
- Move config loading to one module.
- Keep old config reader as compatibility adapter.
- Introduce profile overlays.

### Phase 3 (Medium-high risk, 1 week)
- Split `UnifiedTaskRunner.cxx` into fit/systematics/output services.
- Keep behavior parity checks using same input snapshots.

### Phase 4 (Cleanup)
- Remove deprecated config keys and old wrappers.
- Update README and examples.

## 10. Acceptance Criteria

- Single command runs full chain successfully for at least one mode.
- New config passes validation and reproduces old outputs within expected tolerance.
- Lifetime fit outputs (tau and chi2/ndf) unchanged within floating tolerance.
- Team can add a new mode/profile by editing only profile config and one registration table.
