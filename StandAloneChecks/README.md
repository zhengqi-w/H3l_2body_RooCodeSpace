# StandAloneChecks Macros

This directory contains standalone ROOT macros used for focused cross-checks and debugging outside the main workflow.  They are intended to validate one correction, fit model, or QA observable at a time.

## How to Run

Run the macros from `ROOTWorkFlow/CodeSpace` so that relative includes to `Tools/` remain valid:

```bash
cd ROOTWorkFlow/CodeSpace
root -l -b -q 'StandAloneChecks/FitSpectrumFunctionsSimple.C()'
```

Most macros provide default input and output paths for the current local analysis setup.  Override the function arguments when running on another production, period, or selection:

```bash
root -l -b -q 'StandAloneChecks/EventSignalLossCheck.C("/path/to/AnalysisResults.root", "multiplicity", "impactparameter", "/path/to/output")'
```

Default outputs are written under:

```text
ROOTWorkFlow/Outputs/StandAloneChecks/
```

## Macro Index

| Macro | Entry point | Purpose | Typical output |
| --- | --- | --- | --- |
| `AcceptanceRecoMCCollisionCompare.C` | `AcceptanceRecoMCCollisionCompare(...)` | Compares acceptance with and without the `fIsRecoMCCollision` denominator requirement for each centrality and pT bin. | `AcceptanceRecoMCCollisionCompare/acceptance_reco_mc_collision_compare.csv` |
| `DrawMcCtEfficiency.C` | `DrawMcCtEfficiency(...)` | Checks the generated decay-radius / ct dependence of the MC efficiency. | `MCEfficiency/mc_eff_ct.pdf` and per-ct decay-radius PDFs |
| `EventSignalLossCheck.C` | `EventSignalLossCheck(...)` | Validates the combined event-loss, event-splitting, and signal-loss corrections from `EventSignalLossHelper`. Supports multiplicity and impact-parameter methods. | `EventSignalLoss/event_signal_loss_check.root`, event-loss and signal-loss QA PDFs |
| `FitSpectrumFunctionsSimple.C` | `FitSpectrumFunctionsSimple(...)` | Tests spectrum fit functions, parameter seeds, limits, chi2 behavior, and extrapolation shapes on corrected spectra. | `FitFunctionScan/fit_function_scan_*.pdf`, `fit_results.txt`, `fit_results.root` |
| `KolmogorovDecRad.C` | `KolmogorovDecRad(...)` | Runs a Kolmogorov test between reconstructed MC decay-radius and workflow candidate decay-radius histograms. | Console p-value |
| `MCEffCheck.C` | `MCEffCheck(...)` | Compares MC efficiency / acceptance between two MC inputs, with configurable centrality and pT binning. | `MCEfficiency_*/` comparison plots |
| `RunMCEffCheck_FT0C_0_90.C` | `RunMCEffCheck_FT0C_0_90()` | Convenience wrapper around `MCEffCheck` for the FT0C 0-90% comparison setup. | `MCEfficiency_FT0C_0_90_Compare4/` |

## Notes

- These macros are analysis QA tools, not part of the normal pipeline dispatch.
- Several defaults use absolute local paths.  Treat them as examples and override them for new productions.
- Generated ROOT/ACLiC build products such as `.d`, `.pcm`, `.so`, and `.o` files should not be committed.
- Physics output files and plots under `ROOTWorkFlow/Outputs/StandAloneChecks/` are generated artifacts; keep only the ones needed for documentation or review.
