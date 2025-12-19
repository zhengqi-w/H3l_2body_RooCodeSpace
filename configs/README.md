# Configurations overview

This folder hosts the JSON snippets used by the ROOT-based workflow. A few quick hints for `ct_extraction.json`:

- `data_snapshot_dir` / `snapshot_pattern` point to the per-bin snapshot ROOT files for data. Use the `%PTMIN%`, `%PTMAX%`, `%CTMIN%`, `%CTMAX%` tokens to keep names consistent with the generator script.
- `mc_snapshot_dir`, `mc_snapshot_tree_name`, and `mc_snapshot_pattern` are brand-new knobs that let you feed an aligned MC snapshot (so the DSCB tails can be stabilised by fitting on MC first). By default they reuse the data snapshot folder/pattern.
- `bdt_score_column` tells the C++ pipeline which branch already carries the BDT response (e.g. `model_output`). When this is present the workflow no longer needs to load TMVA models.
- `trial_suffix` appends a short tag to the `std` directory inside the output ROOT file (e.g. `std_tuneA`). This keeps parallel systematic variations from clobbering each other without duplicating the heavy snapshots.
- The BDT knobs (`working_point_file`, `bdt_overrides`, `bdt_score_shift`) mirror the Python pipeline. You can shift the score on the fly without regenerating snapshots.

Whenever you add new pt/ct bin edges, keep the array lengths consistent (`len(ct_bins) == len(pt_bins) - 1`). The C++ helper will throw a descriptive error if something is out of sync.
