# CodeSpace Introduction
This CodeSpace is developed for CERN-ALICE Run 3 offline analysis(PWGLF), aiming at extracting the hypertriton spectrum, lifetime, cross section etc. from the derived data preprocessed with the alihyperloop system. 

## Structure of the code

### PreProcess
- `MC file reweight`
- `BDT process`
- `WorkingPoint hunting`

### Main Task
- `BDTSpectrum extraction`
- `TopologySpectrum extraction`
- `Lifetime extraction in different pt bins`
- `Lifetime extraction in full range`

# CodeSpace Dependencies

This directory contains ROOT-based preprocessing and analysis workflows.

## Python Dependencies

- `numpy`
- `PyYAML`
- `matplotlib`
- `uproot`
- `xgboost`
- `joblib`
- `hipe4ml`

Install example:

```bash
pip install numpy pyyaml matplotlib uproot xgboost joblib hipe4ml
```

## Runtime Requirements

- ROOT with PyROOT enabled (tested in this project with ROOT 6.30.x).
- C++ helper source for ITS decoding:
	- `ROOTWorkFlow/CodeSpace/include/its_helpers.cc`
- Blast-Wave spectrum file for MC reweighting:
	- `H3l_2body_spectrum/utils/H3L_BwFit.root`
	- object name: `BlastWave_H3L_10_30`

`PreProcess/BDTPreProces.py` now contains local implementations for:

- loading AO2D trees (`_load_all_trees_to_chain`)
- dataframe conversion (`_correct_and_convert_df`)
- MC reweighting (`_reweight_pt_spectrum`)
- train-data range matching (`_cut_elements_to_same_range`)

This removes direct runtime dependency on `H3l_2body_spectrum/utils/utils.py` for those operations.

# Run Example

```bash
python ROOTWorkFlow/CodeSpace/PreProcess/BDTPreProces.py --config-file <your_config.yaml>
```
