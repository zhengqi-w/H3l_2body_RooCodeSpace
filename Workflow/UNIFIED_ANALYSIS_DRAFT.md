# Unified Analysis Workflow Draft

## 1. Goals

This draft defines a major refactor from multi-task, multi-config execution to a single-entry analysis engine that runs different physics blocks by mode and binning policy.

Primary goals:
- One entry macro for all main analysis modes.
- One config schema (no generated effective configs).
- Shared bin planner and shared mass-fit / WP lookup kernel.
- Built-in checks framework, including:
  - pre/post BDT candidate distribution checks,
  - checks from merged snapshots or raw data,
  - MC candidate distribution checks.

## 2. Current Pain Points

- Task duplication across `ProcessBdtSpectrum`, `ProcessTopologySpectrum`, `ProcessCtSpectrum`, `ProcessCtSingleSpectrum`.
- Config indirection (`general_config -> generated configs -> task config`) increases maintenance burden.
- Similar loops reimplemented by task (bin traversal, snapshot loading, output writing).
- Checks are not unified with analysis execution modes and data sources.

## 3. Target Architecture

Single entry:
- `Tasks/ProcessAnalysis.C`

Single core engine:
- `Tools/AnalysisEngine.h/.cxx`

Mode policies:
- `Tools/policies/ModePolicy.h`
- `Tools/policies/SpectrumModePolicy.h`
- `Tools/policies/TopologyModePolicy.h`
- `Tools/policies/CrossSectionModePolicy.h`
- `Tools/policies/CtSingleModePolicy.h`

Checks framework:
- `Tools/checks/ChecksEngine.h/.cxx`
- `Tools/checks/DistributionChecks.h/.cxx`
- `Tools/checks/ChecksConfig.h`

Data access layer:
- `Tools/io/DataSource.h/.cxx`

Bin planning:
- `Tools/binning/BinPlan.h`
- `Tools/binning/BinPlanBuilder.h/.cxx`

Output layer:
- `Tools/output/OutputWriter.h/.cxx`

Compatibility wrappers (temporary):
- `Tasks/ProcessBdtSpectrum.C` calls `ProcessAnalysis` with mode override.
- Same for topology/ct/ct-single wrappers.

## 4. Unified Config Schema

Use one file, e.g. `configs/analysis_unified.json`.

Top-level sections:
- `common`
- `execution`
- `analysis`
- `checks`
- `output`

### 4.1 common
- shared paths/tree names/physics constants
- reweight and absorption paths

### 4.2 execution
- `mode`: `spectrum | topology_spectrum | crosssection | ct_single`
- `enable_implicit_mt`
- `stop_on_error`

### 4.3 analysis
- `binning`
  - unified dimensions (`cen`, `pt`, `ct`) with optional axes
  - policy controls axis usage by mode
- `selection`
  - base selection and optional topology selection arrays
- `working_point`
  - source file and key mapping policy
- `fit`
  - signal/background functions and fit ranges
- `correction`
  - branching ratio, delta rapidity, matter handling
- `systematics`
  - trails and thresholds

### 4.4 checks
- `enabled`
- `source_mode`: `snapshot | rawdata | both`
- `merge_scope`:
  - `none` (per-bin),
  - `within_mode` (merge all bins in current mode),
  - `custom_groups`.
- `variables`: list of variables to check.
- `pre_bdt`: bool
- `post_bdt`: bool
- `mc`: bool
- `hist`: binning/range settings
- `outputs`: PDF/ROOT/CSV

### 4.5 output
- directory naming conventions
- root key layout
- csv summary controls

## 5. BinPlan Unification

Define one internal object:

- `BinPlanItem`
  - optional `cenMin, cenMax`
  - optional `ptMin, ptMax`
  - optional `ctMin, ctMax`
  - `label`
  - `snapshotDataPath`
  - `snapshotMcPath`
  - `mode`

- `BinPlan`
  - vector of `BinPlanItem`
  - edge arrays for each axis used by mode

`BinPlanBuilder` responsibilities:
- parse unified config and mode policy,
- produce standardized labels,
- resolve snapshot and raw-data query boundaries,
- provide stable ordering and reproducible grouping keys.

## 6. Analysis Engine Pipeline

`AnalysisEngine::Run(config)`:
1. Build bin plan.
2. Initialize data sources and shared caches.
3. Loop bin items:
   - fetch data/MC candidates,
   - resolve WP via `GeneralHelper::GetWp*` wrappers,
   - run `GeneralHelper::FitMassSpectrum`,
   - apply corrections,
   - store per-bin products.
4. Optional systematics pass.
5. Optional checks pass (same run context).
6. Write outputs through one writer.

Shared caches:
- opened files/trees by path,
- optional pre-fetched vectors for repeated trails,
- optional merged datasets for checks.

## 7. Checks Requirements Integrated

Checks are first-class and run from one checks engine.

### 7.1 Data checks: pre/post BDT

For each requested variable:
- Pre-BDT distribution:
  - from snapshot source (if includes enough columns), or
  - from raw data with bin cuts and selection reconstruction.
- Post-BDT distribution:
  - apply bin-resolved BDT threshold (WP score) then histogram.

Supported scopes:
- per-bin checks,
- merged checks across bins for a mode,
- custom merge groups (for publication/QA views).

Outputs:
- overlaid pre/post histograms,
- ratio histogram (`post/pre`),
- summary metrics (mean, RMS, KS test or chi2 test).

### 7.2 Snapshot merge vs rawdata checks

Add configurable data source strategy:
- `snapshot`: use `snapshotDataPath` for speed and consistency with training.
- `rawdata`: read AO2D/raw tree directly; apply same bin and selection expressions.
- `both`: produce and compare both; report differences.

If snapshot does not include required variable:
- fallback to rawdata path for that variable only,
- mark provenance in check output metadata.

### 7.3 MC candidate checks

For each variable:
- MC baseline distribution by bin.
- Optional post-selection MC distribution (if selection should be mirrored).
- Optional data-vs-MC shape comparison after normalization.

Outputs:
- per-bin and merged overlays,
- pull or ratio (`data/MC`) with uncertainty option,
- shape test metrics.

## 8. Checks Output Contract

Root structure proposal:
- `checks/<mode>/<scope>/<bin_or_group>/<variable>/`
  - `h_data_pre`
  - `h_data_post`
  - `h_mc_pre`
  - `h_mc_post`
  - `h_ratio_post_pre`
  - `h_ratio_data_mc`
  - `canvas_*`

CSV/JSON summary:
- one line per variable per scope with:
  - entries,
  - mean/RMS,
  - KS p-value,
  - source mode (`snapshot/rawdata/both`),
  - fallback flags.

## 9. Migration Plan (Large but Controlled)

Phase A: foundation
- Introduce unified config parser and `BinPlanBuilder`.
- Keep legacy tasks untouched.

Phase B: shared engine pilot
- Implement `ProcessAnalysis` for `spectrum` only.
- Keep old `ProcessBdtSpectrum` as reference.

Phase C: mode expansion
- Add topology mode as policy variant.
- Add crosssection and ct_single policies.

Phase D: checks integration
- Add checks engine with snapshot and rawdata source modes.
- Validate pre/post BDT checks and MC checks.

Phase E: deprecate generated configs
- Remove effective config generation in workflow.
- Keep thin compatibility wrappers for one release cycle.

## 10. Validation Strategy

Physics consistency checks:
- per-bin raw yield differences within statistical uncertainty.
- corrected spectra compatibility against old tasks.
- ct fit outputs (`tau`, fit quality) compatibility.

Checks framework validation:
- pre/post BDT checks reproducibility with fixed seed.
- snapshot vs rawdata parity where variables are common.
- MC checks stable across repeated runs.

Performance targets:
- lower wall-time through reduced repeated I/O and shared caching.
- reduced code volume in tasks (most logic moved to engine/policies).
- reduced config duplication and lower edit surface.

## 11. Immediate Next Step (Implementation-ready)

Before coding the full refactor, define and freeze:
- final unified config JSON schema,
- `BinPlanItem` fields and label rules,
- checks variable catalog and per-variable histogram settings,
- output directory/key naming contract.

Once these are frozen, implementation can proceed mode-by-mode without breaking analysis production.
