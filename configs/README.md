# Analysis Configuration Guide

This document describes the parameters in the workflow configuration files. The main variants are:

- `general_config_merged.json`: combined-period / merged-data configuration.
- `general_config_split.json`: single-period / split-data configuration.
- `general_config.json`: current default working configuration, kept aligned with the merged-data setup.

The configuration is organized by workflow scope and stage: `common`, `execution`, `preprocess`, `analysis`, `checks`, and `output`. A typical workflow command is:

```bash
bash run_pipeline.sh ../configs/general_config.json <stage> <isTest>
```

where `<stage>` can be `all`, `training`, `wp`, `analysis`, `checks`, or another stage supported by the workflow.

## `common`

Global paths, physics constants, binning, plot tags, and baseline selections.

### `common.path`

| Parameter | Description |
| --- | --- |
| `data_path` | Input data AO2D file used in single-period mode. |
| `analysisresults_path` | Input data `AnalysisResults.root` used in single-period mode. It provides event counts, centrality histograms, and QA histograms. |
| `mc_path` | Input MC AO2D file used in single-period mode for training, acceptance, BDT efficiency, and correction calculations. |
| `snapshot_dir` | Directory where training snapshots are written and analysis snapshots are read. |
| `wp_dir` | Directory containing working-point scan outputs. |
| `model_dir` | Directory containing trained BDT models. |
| `qa_dir` | Directory for BDT training and working-point QA plots. |
| `mc_file_for_absorption` | Nominal absorption-correction input file. |
| `spectrum_file` | ROOT file containing pT spectrum or reweight functions. Used by WP when `use_spectrum=true`. |
| `event_signal_loss_file` | `AnalysisResults.root` input for event loss, event splitting, and signal loss corrections. |
| `output_dir` | Top-level workflow output directory. |

### `common.periods`

List of periods used in combined-period mode. This block is used only when `execution.combine_period=true`.

| Parameter | Description |
| --- | --- |
| `tag` | Period name. Used in merged output directory names and plot labels. |
| `data_path` | Data AO2D file for this period. |
| `analysisresults_path` | Data `AnalysisResults.root` for this period. |
| `mc_path` | MC file associated with this period. The same MC file may be reused for multiple periods if needed. |

The combined-period workflow follows the unified training, unified working point, and unified analysis strategy. Only the final merged snapshot is kept.

#### Combined-period training and MC-efficiency logic

When `execution.combine_period=true`, the workflow does not use the single-period `common.path.data_path` and `common.path.mc_path` for BDT training. Instead, it loops over all entries in `common.periods`:

- Data candidates are read from each period `data_path`.
- MC signal candidates are read from each period `mc_path`.
- The same bin selection is applied to each period.
- MC signal candidates additionally require reconstructed candidates (`fIsReco == 1`) for training.
- A `fPeriodIndex` column is attached to each temporary period snapshot.
- The temporary period snapshots are merged into one final snapshot used by the unified BDT training and WP scan.
- The temporary per-period snapshots are removed after the merged snapshot is written, so only the final combined snapshot is kept.

The training mixture is controlled by `preprocess.bdt.period_weight_mode`:

| Value | Description |
| --- | --- |
| `equal_period` | Downsamples each period to the same candidate count before training, separately for MC signal and data background. This gives each period equal candidate-level weight in the classifier training. |
| `full` | Uses the direct concatenation of all period candidates without per-period downsampling. In this mode, all available MC/data candidates from all configured periods are used, and periods with larger candidate statistics naturally have larger candidate-level weight in the classifier training. |
| empty / disabled | Same behavior as `full`; kept for backward compatibility. |

This training weighting is independent of the physics correction weighting. The BDT training uses a candidate-level merged sample, while the final MC-efficiency correction uses event-count weights from the data `AnalysisResults.root` files.

In the analysis stage, combined-period MC efficiency is evaluated period by period and then averaged:

- The analysis collects all period `mc_path` entries from `common.periods`.
- Period weights are computed from the corresponding period `analysisresults_path`, using `common.event_hist.n_events_hist`.
- For `bdt_spectrum` and `topology_spectrum`, the weights are computed separately in each centrality bin. The final acceptance/efficiency in each `(centrality, pT)` bin is the event-fraction-weighted average of the per-period MC efficiencies.
- For `pt_ct`, the current implementation uses one global set of period event fractions over the configured centrality range.
- For `ct_single`, the multi-MC fallback also uses global period event fractions.
- Efficiency uncertainties from different period MC files are combined in quadrature after multiplying each uncertainty by its period weight.

Therefore, the intended merged-data strategy is:

1. unified BDT training with all periods included;
2. unified WP scan on the merged snapshots;
3. period-aware MC-efficiency correction, using data event fractions rather than raw MC-file statistics.

### `common.wp_files`

Working-point text file names for each analysis mode.

| Parameter | Description |
| --- | --- |
| `bdt_spectrum` | WP file used when `analysis_mode=bdt_spectrum`. |
| `pt_ct` | WP file used when `analysis_mode=pt_ct`. |
| `ct_single` | WP file used when `analysis_mode=ct_single`. |

### `common.tree_names`

| Parameter | Description |
| --- | --- |
| `data` | Data candidate tree name, usually `O2hypcands`. |
| `mc` | MC candidate tree name, usually `O2mchypcands`. |
| `absorption` | Tree name inside the absorption-correction input file. |

### `common.tags`

Strings used in plot labels and output naming.

| Parameter | Description |
| --- | --- |
| `collision_system` | Collision-system display string, for example `Pb--Pb`. |
| `collision_energy` | Collision-energy display string, for example `#sqrt{#it{s_{NN}}} 5.36 TeV`. |
| `use_performance` | Whether to use a performance-style label on plots. |
| `performance_label` | Text used for the performance label. |
| `period` | Period name used in single-period mode. |
| `period_mark` | Extra period marker, often used to identify tracking, PID, or cut versions. |

### `common.parameters`

| Parameter | Description |
| --- | --- |
| `branching_ratio` | Decay branching ratio used in corrected-yield normalization. |
| `delta_rap` | Rapidity interval width used in yield normalization. |
| `original_ctao_absorption` | Reference lifetime/ctau value used by the absorption input. |
| `mass_min`, `mass_max` | Invariant-mass fit range. |
| `sigma_range_mc_to_data` | Allowed data-fit sigma range relative to the MC sigma constraint. |
| `add_event_signal_loss_cen_pt` | Per-centrality switch for event/signal-loss corrections. If disabled for a bin, the corresponding correction defaults to 1. |
| `mass_nbins_mc` | Number of invariant-mass bins for MC QA and fits. |
| `mass_nbins_data` | Number of invariant-mass bins for data QA and fits. |
| `mass_fit_use_binned_data` | Whether to use binned data in the RooFit mass fit. If `false`, `RooDataSet` is used. |
| `mass_fit_prefit_sidebands` | Whether to prefit the sidebands before the full mass fit. |
| `mass_fit_sideband_exclusion_sigma` | Signal-window exclusion size, in sigma units, for the sideband prefit. |

### `common.selection`

| Parameter | Description |
| --- | --- |
| `basic_selection_data` | Baseline candidate selection applied to data and MC, for example `fDecRad > 0.8`. |
| `mc_acceptance_require_two_body` | If `true`, MC acceptance denominators and numerators require `fIsTwoBodyDecay > 0`. If `false`, the acceptance calculation does not apply the two-body decay filter. Defaults to `true`; the legacy alias `is_two_body_selected` is also accepted. |

### `common.binning`

| Parameter | Description |
| --- | --- |
| `cen_bins` | Centrality-bin edges. The number of centrality bins must match `pt_bins_by_centrality` and `related_multiplicity_center`. |
| `related_multiplicity_center` | Mean multiplicity assigned to each centrality bin. Used for the integrated-yield vs multiplicity plot. |
| `related_multiplicity_uncertainty` | Uncertainty on `related_multiplicity_center`. Used as the horizontal error bar in the integrated-yield vs multiplicity plot. Its length should match the number of centrality bins. |
| `pt_bins_by_centrality` | pT-bin edges for each centrality bin in `bdt_spectrum` and `topology_spectrum` modes. |
| `pt_bins` | pT-bin edges used in `pt_ct` mode. |
| `ct_bins_by_pt` | ct-bin edges for each pT bin in `pt_ct` mode. |
| `ct_bins_single` | ct-bin edges used in `ct_single` mode. |
| `pt_bins_single` | pT bins used by single-pT-dimension QA or auxiliary checks. |

### `common.event_hist`

| Parameter | Description |
| --- | --- |
| `n_events_hist` | Histogram path in data `AnalysisResults.root` used for event counts or centrality event distributions. |

## `execution`

Global switches controlling which workflow mode and stage are executed.

| Parameter | Description |
| --- | --- |
| `enable` | Master execution switch. Set to `false` to disable execution. |
| `combine_period` | Enables combined-period mode. When enabled, `common.periods` is used and output directories are named with combined period tags. |
| `training_mode` | Training-stage mode, for example `bdt_spectrum`. |
| `wp_mode` | Working-point-stage mode. |
| `analysis_mode` | Analysis-stage mode. This selects the matching entry in `analysis.mode_profiles`. |
| `enable_implicit_mt` | Enables ROOT implicit multi-threading. |
| `do_systematics` | Enables systematic-uncertainty evaluation in the analysis stage. |
| `save_results_to_pdf` | Enables PDF and plot output writing. |

## `preprocess.bdt`

BDT training and snapshot-building configuration.

| Parameter | Description |
| --- | --- |
| `side_band_edges` | Invariant-mass sideband boundaries used for background training samples. |
| `mc_pt_bin_var` | Variable used to pT-bin MC signal candidates, for example `fPt` or `fAbsGenPt`. |
| `training_variables` | Input variables used by the BDT model. |
| `extra_vars_save_data` | Extra variables saved in the data snapshot. |
| `extra_vars_save_mc` | Extra variables saved in the MC snapshot. |
| `test_set_size` | Test fraction used in the train/test split. |
| `bkg_fraction_max` | Maximum background-to-signal sampling ratio. |
| `random_state` | Random seed used by Python/ML preprocessing. |
| `hyperparams` | XGBoost hyperparameters. |
| `npoints_for_effi` | Number of points in the score-efficiency curve. Higher values give finer scans but increase output size and runtime. |
| `efficiency_min`, `efficiency_max` | BDT-efficiency scan range. |
| `make_training_qa` | Whether to produce training QA plots. |
| `qa_plot_bins` | Number of bins used in training QA histograms. |
| `score_efficiency_max_retrain_attempts` | Maximum number of training attempts when the hipe4ml score-efficiency array generation fails. Each retry changes the train/test split and model seed; no quantile fallback array is written. |
| `skip_existing_training_outputs` | If `true`, the training stage skips a bin when the merged data snapshot, merged MC snapshot, JSON model, PKL model, and score-efficiency array already exist. This is useful for resuming interrupted training. |
| `period_weight_mode` | Period weighting mode in combined-period training. `equal_period` gives equal candidate-level weight to each period by downsampling; `full` uses all candidates from all periods without period downsampling. Empty/disabled behaves like `full` for backward compatibility. |
| `use_training_overrides` | Enables `training_overrides`. |
| `training_overrides` | List of centrality-dependent training overrides. |

### `preprocess.bdt.hyperparams`

| Parameter | Description |
| --- | --- |
| `max_depth` | Maximum tree depth. Lower values can help reduce overtraining in high-background central bins. |
| `gamma` / `min_split_loss` | Minimum loss reduction required for a split. Both keys are kept for compatibility with XGBoost versions. |
| `learning_rate` | Learning rate. |
| `n_estimators` | Number of trees. |
| `min_child_weight` | Minimum child weight for leaf nodes. |
| `subsample` | Event subsampling fraction. |
| `colsample_bytree` | Feature subsampling fraction per tree. |
| `tree_method` | XGBoost tree-building method, for example `hist`. |
| `seed` | XGBoost random seed. |

### `preprocess.bdt.training_overrides`

Each override can contain:

| Parameter | Description |
| --- | --- |
| `name` | Override name, used only for identification. |
| `modes` | Optional list of training modes to which the override applies, for example `["cen_pt"]`, `["pt_ct"]`, or `["ct_single"]`. If omitted, the override can match any mode. |
| `bins` | Optional list of bin-matching blocks. Each block can contain `type`, `centrality_ranges`, `pt_ranges`, and/or `ct_ranges`. If omitted, the override applies to all bins in the selected modes. |
| `mc_use_full_centrality` | Whether to use full-centrality MC signal candidates for the selected centrality ranges. Useful when central-bin MC signal statistics are limited. |
| `side_band_edges` | Overrides the default sideband boundaries. |
| `training_variables` | Overrides the default training-variable list. |
| `bkg_fraction_max` | Overrides the maximum background sampling ratio. |
| `hyperparams` | Overrides part or all of the XGBoost hyperparameters. |

This block is used only when `use_training_overrides=true`. The bin matcher is shared by all analysis modes:

- `type: "cen_pt"` matches centrality-pT bins used by `bdt_spectrum` and `topology_spectrum` training.
- `type: "pt_ct"` matches pT-ct bins.
- `type: "ct_single"` matches single-ct-bin training, optionally with a common pT filter.
- `type: "pt_single"` and `type: "pt_ct_single"` are also accepted for the corresponding auxiliary modes.

Example:

```json
{
  "name": "central_low_pt",
  "modes": ["cen_pt"],
  "bins": [
    {
      "type": "cen_pt",
      "centrality_ranges": [[0, 5], [5, 10]],
      "pt_ranges": [[1.5, 2.0], [2.0, 2.5]]
    }
  ],
  "training_variables": ["fDcaV0Daug", "fDcaHe", "fDcaPi", "fCosPA", "fNSigmaHe", "fCt"],
  "bkg_fraction_max": 10
}
```

For backward compatibility, the old top-level keys `centrality_ranges`, `pt_ranges`, and `ct_ranges` are still accepted and are interpreted as one generic bin-matching block.

## `preprocess.wp`

Working-point scan configuration.

| Parameter | Description |
| --- | --- |
| `sideband_low`, `sideband_high` | Invariant-mass sideband ranges used for the background fit. |
| `signal_window_sigma` | Signal-window half width, in sigma units, used to integrate the expected background. |
| `min_entries_for_fit` | Minimum entries required for a mass fit. Can be set to 0 when fit quality is controlled by chi2. |
| `max_chi2_ndf` | Maximum allowed mass-fit chi2/ndf in the WP scan. |
| `max_sideband_rel_diff` | Maximum allowed relative difference between low and high sidebands. A large value effectively disables this requirement. |
| `performance` | Whether WP plots use performance-style labels. |
| `enable_implicit_mt` | Enables ROOT implicit multi-threading in the WP stage. |
| `use_spectrum` | Global default. If `true`, expected signal is computed from the spectrum function. If `false`, expected signal is estimated from a constant fit to `3S/BDTefficiency`. |
| `use_spectrum_override` | Optional list of per-centrality overrides. Each matched entry overrides the global `use_spectrum` value for that centrality bin. |
| `prefit_sidebands` | Whether to strictly prefit sidebands before fitting the full mass range. Recommended. |
| `save_score_fit_frames` | Whether to save fit frames for all scanned scores. The best-point frame should still be kept even when this is disabled. |
| `background_order` | Background polynomial order or `auto`. In `auto` mode the code chooses between background orders based on fit quality. |
| `background_order2_min_delta_chi2` | Minimum chi2 improvement required to select the higher-order background in `auto` mode. |
| `progress_every` | Log-print interval during the WP scan. |
| `period_text` | Period text shown in WP plots. |
| `additional_pave_text` | Additional text shown in WP plots. |
| `yield_eff_range` | Efficiency range used when fitting `3S/BDTefficiency` to estimate expected signal. |
| `target_pt_range`, `target_ct_range`, `target_cen_range` | Optional pT/ct/centrality restriction for WP processing. Empty arrays mean no restriction. |

The WP optimization target is `Expected significance (3 sigma window) * BDT efficiency`. The expected background is the background-function integral in the signal mean `± signal_window_sigma * sigma` window. The expected signal is taken either from the spectrum function or from the `3S/BDTefficiency` method.

`use_spectrum_override` can be used for per-centrality control:

```json
"use_spectrum": true,
"use_spectrum_override": [
  {"centrality_range": [70, 90], "use_spectrum": false}
]
```

For a global setting only:

```json
"use_spectrum": true
```

## `analysis`

Configuration for spectrum analysis, invariant-mass fits, corrections, systematics, and integrated yield.

### `analysis.event_signal_loss_method`

Method used for event-loss and signal-loss corrections.

| Value | Description |
| --- | --- |
| `multiplicity` | Uses multiplicity-based weights and ratios. |
| `impactparameter` | Uses impact-parameter binning and ratios. |

The final event correction includes both event loss and event splitting.

### `analysis.corrections`

Optional analysis-stage correction switches.

| Parameter | Description |
| --- | --- |
| `mc_acceptance_centrality_overrides.enable` | Enables centrality remapping for MC acceptance/efficiency in `bdt_spectrum` and `topology_spectrum` modes. Raw-yield extraction and output bin labels are unchanged; only the MC centrality range used to compute acceptance is replaced. |
| `mc_acceptance_centrality_overrides.ranges` | List of remapping entries. Each entry has `target: [cmin, cmax]` and `source: [cmin, cmax]`. For example, `target=[0,5]`, `source=[0,10]` makes the 0-5% spectrum use 0-10% MC acceptance statistics. In combined-period mode this remapping is applied per period before the event-fraction weighted average is built. |

### `analysis.selection`

| Parameter | Description |
| --- | --- |
| `is_matter` | Matter selection. Supported values are `both`, `matter`, and `antimatter`. |
| `additional_data_selection_general` | Extra analysis-stage data selection, applied on top of the mode-profile selection. |
| `add_performance` | Whether analysis plots use performance-style labels. |

### `analysis.fit`

| Parameter | Description |
| --- | --- |
| `signal_fit_func` | Invariant-mass signal function, for example `dscb` or `gaus`. |
| `bkg_fit_func` | Invariant-mass background function, for example `pol1`, `pol2`, or `expo`. |
| `integral_fit_func` | Default nominal integrated-yield fit function. Per-centrality overrides are optional and only needed when a bin must use a different nominal function. |
| `integral_fit_parameters` | Global initial values, limits, and optional fixed-parameter flags for integrated-yield fit functions. |
| `integral_fit_parameters_by_centrality` | Sparse per-centrality override of `integral_fit_parameters`. Use it only for bins whose fit parameters differ from the global defaults. |
| `integral_fit_fix_parameters_by_centrality` | Per-centrality switch that enables fixed integrated-yield fit parameters. Keys use centrality labels such as `60_70` or `70_90`. When a centrality is not enabled, any `fixed` array in that bin is ignored. The legacy location under `common.parameters` is still accepted. |

### Integrated-yield fit-function parameters

`analysis.fit.integral_fit_parameters` and `analysis.systematics.integral_fit_parameters` use the same structure:

```json
"fName": {
  "initial": [ ... ],
  "limits": [[low0, high0], [low1, high1]],
  "fixed": [false, true]
}
```

`fixed` is optional. It follows the same canonical parameter order as `initial`. A `true` entry calls `TF1::FixParameter` with the configured initial value, but only for centralities enabled in `analysis.fit.integral_fit_fix_parameters_by_centrality`.

Centrality-specific overrides use this structure:

```json
"integral_fit_parameters_by_centrality": {
  "60_70": {
    "fBGBW": {
      "initial": [6.602e-1, 1.642e-1, 1.0, 5.0e2],
      "limits": [[2.0e-1, 9.0e-1], [1.0e-1, 6.0e-1], [1.0e-2, 5.0e0], [1.0e-8, 1.0e8]],
      "fixed": [true, true, false, false]
    }
  }
}
```

| Function | Parameter order | Description |
| --- | --- | --- |
| `fBGBW` | `[beta_t, T_kin, n, norm]` | Boltzmann-Gibbs Blast-Wave. |
| `fLevi` | `[T, n, norm]` | Levy/Tsallis-like pT shape. |
| `fBoltzmann` | `[T, norm]` | Boltzmann exponential-like shape. |
| `fPtExp` | `[T, norm]` | pT exponential shape. |
| `fTsallisBW` | `[beta_t, T, q, norm, extra]` | Tsallis Blast-Wave shape. Parameter meaning follows the AliPWGFunc implementation. |

The concrete function implementations come from the AliPWGFunc/AliPWGFunc-style helpers. The number of configured parameters must match the implementation.

### `analysis.systematics`

Configuration for per-pT-bin spectrum systematics and integrated-yield systematics.

| Parameter | Description |
| --- | --- |
| `random_seed` | Random seed used by systematic toys/trials. |
| `syst_ntrails` | Number of per-pT-bin systematic trials. The current key spelling is kept for compatibility. |
| `syst_thrashold_chi2ndf` | Per-bin mass-fit chi2/ndf threshold. The current key spelling is kept for compatibility. |
| `syst_thrashold_significance` | Per-bin significance threshold. Set to 0 to disable the significance cut. |
| `syst_bdt_score_npoints` | Number of BDT-score points used in systematic scans. |
| `syst_bkg_funcs` | Background-function list for systematic variations. |
| `syst_signal_funcs` | Signal-function list for systematic variations. |
| `syst_absorption_files` | Input files used for absorption systematic variations. |
| `syst_absorption_file_labels` | Plot labels for `syst_absorption_files`. Must have the same length as the file list. |
| `absorption_length` | Fraction of the absorption envelope difference assigned as the uncertainty. `0.5` means half the difference. |
| `branching_ratio_fractional_uncertainty` | Fractional systematic uncertainty from the branching ratio. |
| `n_bins_for_fit` | Number of histogram bins used in systematic mass fits. |
| `n_trails_for_integral_syst` | Number of integrated-yield trials. |
| `n_combinations_for_integral_syst_extrapolation` | Number of extrapolation toys. |
| `reject_integral_fit_func_by_chi2` | Whether to reject integrated-yield systematic fit functions by chi2/ndf. |
| `integral_fit_func_max_chi2ndf` | Chi2/ndf threshold for fit-function systematics. Values below 0 enable a dynamic threshold. |
| `integral_fit_func_fallback_fraction` | Fallback fractional integrated-yield systematic assigned if all non-nominal functions fail. |
| `integral_gauss_fit_max_chi2ndf` | Gaussian-fit quality threshold for integrated-yield trial/extrapolation distributions. If the fit fails, RMS is used. |
| `per_ptbin_gauss_fit_max_chi2ndf` | Gaussian-fit quality threshold for per-pT-bin systematic distributions. If the fit fails, RMS is used. |
| `integral_extrap_toy_max_chi2ndf` | Chi2/ndf compatibility cut between extrapolation toys and the measured spectrum. Values below 0 enable a dynamic threshold. |
| `integral_fit_range` | pT range used to fit spectrum functions. |
| `integrated_yield_range` | Final integrated-yield integration range. Measured bins are summed as `yield * bin width`; unmeasured intervals are integrated with the nominal function. |
| `integral_lowpt_max_factor` | Post-fit sanity limit for low-pT extrapolation shapes, used to reject obviously divergent low-pT behavior. |
| `integral_fit_use_minos_errors` | Whether the nominal integrated-yield fit requests Minos errors. This may produce ROOT/Minuit Minos warnings. |
| `integral_fit_funcs` | Default function list used for integrated-yield fit-function systematics. |
| `integral_fit_funcs_by_centrality` | Sparse per-centrality override of `integral_fit_funcs`; use it only where the systematic function list differs from the default. |
| `integral_fit_parameters` | Initial values and limits for each systematic fit function. |

Current integrated-yield central-value logic:

1. For measured pT bins, the code directly sums `corrected yield * bin width`.
2. For unmeasured intervals, extrapolation ranges are built from the actual input pT-bin edges, for example `[integrated_min, first_measured_edge]` and `[last_measured_edge, integrated_max]`.
3. The nominal fit function is integrated over the unmeasured intervals.
4. The final central value is the measured-bin sum plus the extrapolated integral.

Current integrated-yield statistical uncertainty is the quadrature sum of:

1. The measured pT-bin `IntegralError` contribution.
2. The nominal-fit parameter covariance propagated only to the unmeasured extrapolation intervals.

`integral_stat_fitcov_max_fraction` controls a stability guard for item 2. If the propagated extrapolation statistical
uncertainty from the fit covariance is larger than this fraction of the nominal integrated yield, the code falls back to
`extrapolated yield * measured-bin relative statistical uncertainty`. The default analysis value is `0.10`.

The extrapolation systematic is evaluated with nominal-function parameter toys. For each toy, the normalization is refitted to the measured spectrum. Compatibility with the measured spectrum is controlled by `integral_extrap_toy_max_chi2ndf`.

### `analysis.mode_profiles`

Mode-dependent dimensions and selections.

| Mode | Description |
| --- | --- |
| `bdt_spectrum` | Centrality + pT spectrum analysis using BDT working points. |
| `topology_spectrum` | Centrality + pT spectrum analysis using manual topology cuts. |
| `pt_ct` | pT + ct analysis mode, used for lifetime/cross-section-related studies. |
| `ct_single` | Single-ct-dimension analysis mode. |

Common fields:

| Parameter | Description |
| --- | --- |
| `dimensions` | Analysis dimensions used by this mode. |
| `additional_data_selection` | Extra data selection applied within this mode. |
| `centrality_selection` | Centrality selection used by `pt_ct` and `ct_single`. |
| `add_absorption_correction` | Whether absorption correction is applied. |
| `data_selection_topology` | Manual per-centrality/per-pT topology selections used by `topology_spectrum`. |

## `checks`

QA/check-stage configuration.

| Parameter | Description |
| --- | --- |
| `enabled` | Enables the checks stage. |
| `save_pdf` | Saves check outputs as PDF. |

### `checks.general`

| Parameter | Description |
| --- | --- |
| `variables` | Default variable list used by the checks stage. |
| `axis_pool` | Histogram axis settings for each variable, including `nbins`, `min`, `max`, and `title`. |

### `checks.mc_checks`

| Parameter | Description |
| --- | --- |
| `enable` | Enables MC checks. |
| `selection` | Selection applied to MC checks. |
| `tree` | MC tree name. |
| `variables` | One-dimensional MC QA variables. |
| `hist2d_pairs` | Two-dimensional MC QA variable pairs. |

### `checks.data_all_checks`

| Parameter | Description |
| --- | --- |
| `enable` | Enables all-data checks. |
| `selection` | Selection applied to data checks. |
| `variables` | One-dimensional data QA variables. |
| `hist2d_pairs` | Two-dimensional data QA variable pairs. |

### `checks.hypertriton_onthefly_checks`

| Parameter | Description |
| --- | --- |
| `enable` | Enables on-the-fly hypertriton checks. |
| `save_hypertriton_qa_tree` | Saves the hypertriton QA tree. |
| `nsigmas_mass_window` | Mass-window size in sigma units for on-the-fly checks. |
| `selection` | Extra selection applied to the check. |
| `variables` | One-dimensional QA variables. |
| `hist2d_pairs` | Two-dimensional QA variable pairs. |

## `output`

| Parameter | Description |
| --- | --- |
| `root_layout_version` | Output ROOT-file layout version tag. |
| `write_csv_summary` | Writes CSV summaries. |
| `write_pdf` | Writes PDF plots. |

## Practical Notes

- Single-period run: set `execution.combine_period=false` and check that `common.path.*` points to the target period.
- Combined-period run: set `execution.combine_period=true` and check `common.periods.tag`, `data_path`, `analysisresults_path`, and `mc_path`.
- Analysis-only run: make sure the snapshot and WP paths match the current `combine_period` setting, to avoid mixing single-period and merged-period products.
- Faster debugging: reduce `syst_ntrails`, `n_trails_for_integral_syst`, `n_combinations_for_integral_syst_extrapolation`, and `syst_bdt_score_npoints`.
- Final yield vs multiplicity plot: `related_multiplicity_center` must match the number of centrality bins in `cen_bins`.
