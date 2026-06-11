# PlotingScrips Macros

This directory contains plotting and comparison ROOT macros used after the main workflow has produced spectra, correction QA, or reference ROOT files.  The directory name is kept as `PlotingScrips` to match the existing code paths.

## How to Run

Run the macros from `ROOTWorkFlow/CodeSpace`:

```bash
cd ROOTWorkFlow/CodeSpace
root -l -b -q 'PlotingScrips/ComparePeriodMergedSpectrumQA.C()'
```

For macros with configurable arguments, pass explicit paths when comparing a new period, selection, or workflow output:

```bash
root -l -b -q 'PlotingScrips/SpectrumVsRun2Simple.C("/path/to/run3.root", "/path/to/run2", "/path/to/bw.root", "outDir")'
```

Default outputs are written under:

```text
ROOTWorkFlow/Outputs/PlotingScrips/
```

## Macro Index

| Macro | Entry point | Purpose | Typical output |
| --- | --- | --- | --- |
| `BlastwaveFit.C` | `BlastwaveFit(...)` | Fits Run 2 / Run 3 spectra with the blast-wave reference function. | `BlastwaveFit/` |
| `BuildReweightFunc.C` | `BuildReweightFunc(...)` | Builds pT reweighting functions from reference blast-wave fits and saves QA plots. | `BuildReweightFunc/ReweightFunc.root`, QA PDF |
| `CheckMergedSnapshotPeriods.C` | `CheckMergedSnapshotPeriods(...)` | Checks period composition and model-output distributions inside merged snapshot files. | `PeriodMergedQA/SnapshotPeriodQA_*/` |
| `CompareAnalysisResultsEventQAData.C` | `CompareAnalysisResultsEventQAData(...)`, `CompareThreePeriodsEventQA(...)` | Compares event-level QA histograms between data periods. | `CompareAnalysisResultsEventQAData/` |
| `CompareAnalysisResultsEventQAMCData.C` | `CompareAnalysisResultsEventQAMCData(...)` | Compares data and MC event-level QA distributions. | `CompareAnalysisResultsEventQAMCData_*/` |
| `ComparePeriodMergedSpectrumQA.C` | `ComparePeriodMergedSpectrumQA(...)` | Compares single-period and merged spectra, raw counts, event counts, BDT efficiency, and integrated yield. | `PeriodMergedQA/` |
| `CompareTopologyBDTSpectrumQA.C` | `CompareTopologyBDTSpectrumQA()` | Compares topology-spectrum and BDT-spectrum results. | `TopologyBDTQA/` |
| `CompareV0sCustomV0sSpectrumQA.C` | `CompareV0sCustomV0sSpectrumQA()` | Compares V0s and CustomV0s spectrum outputs. | `V0sSelectionQA/` |
| `DrawAbsorptionEff.C` / `draw_absorption_eff.C` | `DrawAbsorptionEff()`, `draw_absorption_eff()` | Draws absorption-efficiency variations and nominal comparison plots. | `DrawAbsorptionEff/absorption_eff_ptbins_10_20.pdf` |
| `DrawMergedCentralitySpectra.C` | `DrawMergedCentralitySpectra(...)` | Draws merged-centrality spectra and antimatter-over-matter ratios. | `MergedCentralitySpectra/` |
| `SpectrumVsRun2Lite.C` | `SpectrumVsRun2Lite()` | Lightweight Run 3 versus Run 2 spectrum comparison. | `SpectrumvsRun2/Spectrum_vs_run2_lite_*.pdf` |
| `SpectrumVsRun2Simple.C` | `SpectrumVsRun2Simple(...)` | Builds a more structured Run 3 versus Run 2 comparison ROOT file and plots. | `SpectrumvsRun2/Spectrum_vs_run2_simple.root` |
| `extract_bwfits.C` | `extract_bwfits()` | Extracts and stores blast-wave fit functions for later reweighting or reference comparisons. | `extract_bwfits/` |

## Local Reference ROOT Files

The files `H3L_BWFit_Run3.root`, `H3L_BWFit_Run3_23.root`, and `ReweightFunc.root` are local reference inputs/outputs used by some plotting utilities.  They are not ROOT/ACLiC build products.

## Notes

- Most macros assume that the main workflow has already produced the relevant `Outputs/<period_tag>/...` files.
- Many defaults use absolute local paths.  Update the function arguments or defaults when moving to a new machine or production tag.
- Generated ROOT/ACLiC build products such as `.d`, `.pcm`, `.so`, and `.o` files should not be committed.
- Plotting outputs under `ROOTWorkFlow/Outputs/PlotingScrips/` are generated artifacts; keep only the plots needed for documentation, talks, or review.
