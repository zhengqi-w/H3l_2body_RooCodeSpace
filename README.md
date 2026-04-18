# CodeSpace Introduction
This CodeSpace is developed for CERN ALICE Run 3 offline analysis (PWGLF), aiming to extract the hypertriton spectrum, lifetime, cross section, and related observables from derived data preprocessed with the AliHyperloop system.

## Architecture of the Workflow

### PreProcess
- `MC file reweight`
- `BDT process`
- `WorkingPoint hunting`

### Main Analysis Task
- `BDTSpectrum extraction`
- `TopologySpectrum extraction`
- `Lifetime extraction in different pt bins`
- `Lifetime extraction in full range`

# CodeSpace Dependencies
The whole process is based on ROOT, except for the BDT training part, which is implemented in Python with `xgboost` and `hipe4ml`. The workflow is designed to be modular, allowing flexible execution of different stages and easy integration of new features or updates. To run the full workflow, both ROOT and the specified Python libraries must be installed and properly configured in your environment. However, once trained snapshot trees are available, the main analysis tasks can be executed with ROOT alone, without Python dependencies.

## Python Dependencies
The BDT preprocessing script `PreProcess/BDTPreProcess.py` imports the modules below.

- `ROOT` (PyROOT, provided by your ROOT installation)
- `numpy`
- `PyYAML`
- `matplotlib`
- `uproot`
- `xgboost`
- `joblib`
- `hipe4ml`

Python standard-library modules used by the script (no pip install needed): `os`, `argparse`, `json`, `pathlib`.

Install example:

```bash
python3 -m pip install numpy pyyaml matplotlib uproot xgboost joblib hipe4ml
```

## Runtime Requirements
- `ROOT-6.30` or later
- `Reweight Functions`: In this analysis, MC samples are generated with a flat pT spectrum and then reweighted to match the physical pT distribution. For Run 3 hypertriton analysis, we use the Run 2 hypertriton Blast-Wave fit functions for centrality ranges 0-10, 10-30, and 30-50. For centrality > 50, we use the Run 2 He3 (50-90) Blast-Wave fit function as the reweighting function. The same spectrum function is also used to calculate expected significance during working-point hunting. The reweighting functions are stored in `configs/ReweightFunc.root` and are read by the analysis engine.
- `Absorption Trees`: For absorption correction studies, we calculate an absorption correction factor in each pT or ct bin. The absorption tree is generated in the `o2:sim` environment with a configurable He3 absorption fraction in the GEANT4 parameterization, to simulate hypertriton absorption in ALICE detector material (mainly ITS). `absorption_tree_x1.5.root` is chosen as the standard input for analysis. The correction factor is computed as the ratio between the ct distribution of surviving He3 and absorbed He3, and is then applied as a weight in spectrum and lifetime extraction.

# Configuration Overview (Brief)
For details please refer to `configs/README.md`.
- `configs/general_config.json`: main config file for the unified workflow, containing all necessary parameters for the whole workflow.

# How To Run
## Do MC Reweighting and Validate MC Efficiency Inputs
Before running the workflow, make sure the reweighted MC file is prepared with `PreProcess/ReweightMCAO2D.C`, and validate the result with QA plots. Then use the reweighted MC file for subsequent analysis steps.

*Important note: the MC tree contains a column `fIsTwoBodyDecay` used to select two-body decay candidates. You must set `fIsTwoBodyDecay == true` to exclude three-body decay candidates from the denominator in MC efficiency calculation; otherwise, the efficiency will be incorrect.*

## The Integral Workflow
We provide a bash script `Workflow/run_pipeline.sh` to run the full workflow in one command. Internally, it calls the ROOT macro `Workflow/RunUnifiedPipeline.C`.

Pipeline stages are:

- `train`: run BDT training and snapshot rewrite
- `wp`: run Working Point hunting
- `analysis`: run physics extraction (`Tasks/ProcessAnalysis.C`)
- `all`: run `train -> wp -> analysis` in sequence

Command format:

```bash
bash Workflow/run_pipeline.sh <config_path> <stage> <dry_run>
```

Arguments:

1. `config_path`: default is `configs/general_config.json`
2. `stage`: `train | wp | analysis | all` (default: `all`)
3. `dry_run`: `true | false` (default: `false`)

Recommended usage examples:

```bash
cd ROOTWorkFlow/CodeSpace

# run full chain
bash Workflow/run_pipeline.sh configs/general_config.json all false

# run only BDT training
bash Workflow/run_pipeline.sh configs/general_config.json train false

# run only WP stage
bash Workflow/run_pipeline.sh configs/general_config.json wp false

# run only final analysis stage
bash Workflow/run_pipeline.sh configs/general_config.json analysis false

# dry-run (print command plan, do not execute stages)
bash Workflow/run_pipeline.sh configs/general_config.json all true
```

Direct ROOT entry (equivalent backend):

```bash
root -l -b -q 'Workflow/RunUnifiedPipeline.C("configs/general_config.json", "all", false)'
```

Mode routing is read from `execution` in `configs/general_config.json`:

- `execution.training_mode` controls the `train` stage
- `execution.wp_mode` controls the `wp` stage
- `execution.analysis_mode` controls the `analysis` stage

Failure behavior:

- The pipeline is strict fail-fast now: if any stage returns non-zero exit code, the whole run stops immediately.


# The Outputs 

After each run, outputs are generated by stage as follows.

Workflow manifests:

- `Workflow/manifests/run_manifest_train.json`
- `Workflow/manifests/run_manifest_wp.json`
- `Workflow/manifests/run_manifest_analysis.json`

Each manifest records command, config, stage mode, timestamp and exit code for reproducibility.

PreProcess outputs (paths are defined in `common.path`):

- `snapshot_dir`: snapshot trees per bin, such as `data_*.root` and `mc_*.root`
- `model_dir`: trained BDT models, such as `Model_BDT_*.json` and `Model_BDT_*.pkl`
- `qa_dir`: training QA figures (feature distributions, ROC, score distributions, etc.)
- `wp_dir`: score-efficiency arrays and WP txt files

Main analysis outputs (ROOT + figures):

- Base directory:
	`common.path.output_dir/<period>_<period_mark>/<analysis_mode>/<is_matter>/`
- Main ROOT file:
	`spectrum.root`
- Typical objects inside:
	raw and corrected yield histograms, efficiencies, systematic histograms, final spectrum canvases, and fit canvases
- If PDF writing is enabled (`output.write_pdf=true` and `execution.save_results_to_pdf=true`), exported plots are saved under the same output directory.

On-the-fly QA outputs:

- `Checks_hypertriton/checks_hypertriton.root`
- optional QA PDF files under `Checks_hypertriton/`

Practical check after one full run:

1. Confirm all stage manifests have `exit_code = 0`.
2. Confirm `spectrum.root` exists in the expected mode directory.
3. Confirm WP txt and score-efficiency files are updated in `wp_dir`.
4. Confirm QA plots are updated in `qa_dir`.

