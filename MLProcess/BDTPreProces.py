import ROOT
ROOT.ROOT.EnableImplicitMT()
import os
import numpy as np
import argparse
import yaml
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu
import matplotlib.pyplot as plt
import xgboost as xgb

import sys
sys.path.append('/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/H3l_2body_spectrum/utils')
import utils as utils
import joblib
from pathlib import Path
import uproot


class BDTPreProcess:
    """
    可配置的训练模式：
      - Mix_mode = "pt-ct"        : 原始行为，pt_bins 为 1D，ct_bins 为与 pt 对应的 2D 边界列表。
      - Mix_mode = "cen-pt"       : 新增模式，cen_bins 为 1D，pt_bins(或 pt_bins_by_centrality) 为随 centrality 变化的 2D 边界列表。
      - Mix_mode = "pt-ct-single" : 单一 pt、ct 区间训练（兼容旧版 Mix_Mode=False）。
      - Mix_mode = "pt-single"    : 仅按单一 pt 区间训练（ct 不切分）。
      - Mix_mode = "ct-single"    : 仅按单一 ct 区间训练（pt 可选过滤）。
    """

    def __init__(self, config):
        # 配置项
        self.data_path = config['data_path']
        self.mc_path = config['mc_path']
        self.tree_name_data = config['tree_name_data']
        self.tree_name_mc = config['tree_name_mc']
        self.pt_bins = config.get('pt_bins', None)
        self.ct_bins = config.get('ct_bins', None)
        self.pt_bin = config.get('pt_bin', None)   # for separate mode optional
        self.ct_bin = config.get('ct_bin', None)   # for separate mode optional
        self.training_preselections = config['training_preselections']
        self.training_variables = config['training_variables']
        self.extra_vars_used = config.get('extra_vars_used', [])
        self.test_set_size = config['test_set_size']
        self.bkg_fraction_max = config['bkg_fraction_max']
        self.random_state = config['random_state']
        self.hyperparams = config['hyperparams']
        self.npoints_for_effi = config['npoints_for_effi']
        self.side_band_edges = config.get('side_band_edges', [2.95, 3.02])
        self.cen_bins = config.get('cen_bins', None)
        self.pt_bins_by_centrality = config.get('pt_bins_by_centrality', None)
        self.snapshot_dir = Path(config.get('snapshot_dir', 'snapshots'))
        self.models_dir = Path(config.get('model_dir', 'models'))
        self.QA_dir = Path(config.get('QA_dir', 'QAPlots'))
        self.WP_dir = Path(config.get('WP_dir', 'WorkPoints'))
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.QA_dir.mkdir(parents=True, exist_ok=True)
        self.WP_dir.mkdir(parents=True, exist_ok=True)
        mix_mode_cfg = config.get('Mix_mode', config.get('Mix_Mode', 'pt-ct'))
        if isinstance(mix_mode_cfg, bool):
            mix_mode_cfg = 'pt-ct' if mix_mode_cfg else 'pt-ct-single'
        self.mix_mode = str(mix_mode_cfg).lower()

        # prepare ROOT chains and RDataFrames
        self.chian_data = ROOT.TChain(self.tree_name_data)
        self.chian_mc = ROOT.TChain(self.tree_name_mc)
        self.data_file = ROOT.TFile.Open(self.data_path)
        self.mc_file = ROOT.TFile.Open(self.mc_path)
        self.data_rdf = utils.load_all_trees_to_chain(self.data_file, self.chian_data, self.tree_name_data)
        self.mc_rdf = utils.load_all_trees_to_chain(self.mc_file, self.chian_mc, self.tree_name_mc)
        self.data_rdf = utils.correct_and_convert_df(self.data_rdf, calibrate_he3_pt = False, isMC = False, isH4L = False)
        self.mc_rdf = utils.correct_and_convert_df(self.mc_rdf, calibrate_he3_pt = False, isMC = True, isH4L = False)
        if self.training_preselections:
            self.data_rdf = self.data_rdf.Filter(self.training_preselections)
        spectrum_file = ROOT.TFile.Open("../../../H3l_2body_spectrum/utils/H3L_BwFit.root")
        he3_spectrum = spectrum_file.Get("BlastWave_H3L_10_30")
        self.mc_rdf_reweighted = utils.reweight_pt_spectrum(self.mc_rdf, "fAbsGenPt", he3_spectrum, is_rdf = True)

        # mass preselections
        # self.data_rdf = self.data_rdf.Filter("(fMassH3L<2.95 || fMassH3L>3.02)")
        # self.mc_rdf_reweighted = self.mc_rdf_reweighted.Filter("(fMassH3L>2.95 && fMassH3L<3.02)")
        
        if self.extra_vars_used:
            self.data_columns = self.training_variables + self.extra_vars_used + ["fPt", "fCt", "fMassH3L", "fIsMatter"]
        else:
            self.data_columns = self.training_variables + ["fPt", "fCt", "fMassH3L", "fIsMatter"]
        self.mc_columns = self.training_variables + ["fAbsGenPt", "fGenCt", "fMassH3L", "fIsMatter"]

    def _make_label(self, pt_range=None, ct_range=None, cen_range=None):
        parts = []
        if cen_range and None not in cen_range:
            parts.append(f"cen_{cen_range[0]}_{cen_range[1]}")
        if pt_range and None not in pt_range:
            parts.append(f"pt_{pt_range[0]}_{pt_range[1]}")
        if ct_range and None not in ct_range:
            parts.append(f"ct_{ct_range[0]}_{ct_range[1]}")
        return "_".join(parts) if parts else "all"

    def _make_description(self, pt_range=None, ct_range=None, cen_range=None):
        parts = []
        if cen_range and None not in cen_range:
            parts.append(f"centrality {cen_range[0]}-{cen_range[1]}")
        if pt_range and None not in pt_range:
            parts.append(f"pT {pt_range[0]}-{pt_range[1]} GeV/c")
        if ct_range and None not in ct_range:
            parts.append(f"ct {ct_range[0]}-{ct_range[1]} cm")
        return ", ".join(parts) if parts else "full phase space"

    def _ensure_range(self, rng, name):
        if rng is None:
            raise ValueError(f"{name} range is required for this mode.")
        if len(rng) != 2:
            raise ValueError(f"{name} range must have exactly two values.")
        return rng[0], rng[1]

    def _make_snapshot_for_bin(self, pt_range=None, ct_range=None, cen_range=None, label=None):
        pt_min, pt_max = pt_range if pt_range else (None, None)
        ct_min, ct_max = ct_range if ct_range else (None, None)
        cen_min, cen_max = cen_range if cen_range else (None, None)

        label = label or self._make_label(pt_range, ct_range, cen_range)

        sel_data_parts = []
        sel_mc_parts = []
        if pt_min is not None and pt_max is not None:
            sel_data_parts.append(f"fPt > {pt_min} && fPt < {pt_max}")
            sel_mc_parts.append(f"fAbsGenPt > {pt_min} && fAbsGenPt < {pt_max}")
        if ct_min is not None and ct_max is not None:
            sel_data_parts.append(f"fCt > {ct_min} && fCt < {ct_max}")
            sel_mc_parts.append(f"fGenCt > {ct_min} && fGenCt < {ct_max}")
        if cen_min is not None and cen_max is not None:
            sel_data_parts.append(f"fCentralityFT0C > {cen_min} && fCentralityFT0C < {cen_max}")
            sel_mc_parts.append(f"fCentralityFT0C > {cen_min} && fCentralityFT0C < {cen_max}")

        sel_data = " && ".join(sel_data_parts) if sel_data_parts else "1"
        sel_mc = " && ".join(sel_mc_parts) if sel_mc_parts else "1"
        sel_mc = sel_mc + " && fMassH3L>2.95 && fMassH3L<3.02"

        data_root = self.snapshot_dir / f"data_{label}.root"
        mc_root   = self.snapshot_dir / f"mc_{label}.root"

        try:
            if data_root.exists():
                data_root.unlink()
            self.data_rdf.Filter(sel_data).Snapshot(self.tree_name_data, str(data_root), self.data_columns)
            print(f"Saved data snapshot: {data_root}")
        except Exception as e:
            print(f"Warning: data snapshot failed for {label}: {e}")

        try:
            if mc_root.exists():
                mc_root.unlink()
            self.mc_rdf_reweighted.Filter(sel_mc).Snapshot(self.tree_name_mc, str(mc_root), self.mc_columns)
            print(f"Saved MC snapshot: {mc_root}")
        except Exception as e:
            print(f"Warning: MC snapshot failed for {label}: {e}")

        return data_root if data_root.exists() else None, mc_root if mc_root.exists() else None

    def _read_handlers(self, data_root, mc_root):
        bin_data_hdl, bin_mc_hdl = None, None
        try:
            if data_root is not None:
                bin_data_hdl = TreeHandler(data_root, self.tree_name_data)
        except Exception as e:
            print(f"Warning: failed to read data snapshot: {e}")
            bin_data_hdl = None
        try:
            if mc_root is not None:
                bin_mc_hdl = TreeHandler(mc_root, self.tree_name_mc)
        except Exception as e:
            print(f"Warning: failed to read MC snapshot: {e}")
            bin_mc_hdl = None
        return bin_data_hdl, bin_mc_hdl

    def _balance_and_prepare(self, bin_mc_hdl, bin_data_hdl):
        df_mcH = bin_mc_hdl.get_data_frame()
        df_mcH['fNSigmaHe'] = df_mcH['fNSigmaHe'] - df_mcH['fNSigmaHe'].mean()
        bin_mc_hdl.set_data_frame(df_mcH)
        df_dataH = bin_data_hdl.get_data_frame()
        mask = (df_dataH['fNSigmaHe'] > -2) & (df_dataH['fNSigmaHe'] < 1)
        if mask.any():
            mean_shift = df_dataH.loc[mask, 'fNSigmaHe'].mean()
        else:
            mean_shift = df_dataH['fNSigmaHe'].mean()
        df_dataH['fNSigmaHe'] = df_dataH['fNSigmaHe'] - mean_shift
        bin_data_hdl.set_data_frame(df_dataH)

        try:
            utils.cut_elements_to_same_range(bin_mc_hdl, bin_data_hdl, self.training_variables)
            if self.bkg_fraction_max is not None and len(bin_data_hdl) > self.bkg_fraction_max * len(bin_mc_hdl):
                bin_data_hdl.shuffle_data_frame(size=int(self.bkg_fraction_max * len(bin_mc_hdl)), inplace=True, random_state=self.random_state)
        except Exception as e:
            print(f"Warning during balancing: {e}")

        return bin_mc_hdl, bin_data_hdl

    def _train_and_save_model_per_bin(self, bin_mc_hdl, bin_data_hdl, label, pretty_label):
        try:
            train_test_data = au.train_test_generator([bin_mc_hdl, bin_data_hdl], [1, 0],
                                                     test_size=self.test_set_size, random_state=self.random_state)
        except Exception as e:
            print(f"Failed to generate train/test for {pretty_label}: {e}")
            return

        train_features = train_test_data[0]
        train_labels = train_test_data[1]
        test_features = train_test_data[2]
        test_labels = train_test_data[3]

        distr = pu.plot_distr([bin_mc_hdl, bin_data_hdl], self.training_variables + ["fMassH3L"], bins=100, labels=['Signal',"Background"],colors=["blue","red"], log=True, density=True, figsize=(18, 13), alpha=0.5, grid=False)
        plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
        plt.savefig(f"{self.QA_dir}/features_distributions_{label}.pdf", bbox_inches='tight')
        plt.close("all")
        corr = pu.plot_corr([bin_mc_hdl,bin_data_hdl], self.training_variables + ["fMassH3L"], ['Signal',"Background"])
        corr[0].savefig(f"{self.QA_dir}/correlations_mc_{label}.pdf", bbox_inches='tight')
        corr[1].savefig(f"{self.QA_dir}/correlations_data_{label}.pdf", bbox_inches='tight')
        plt.close("all")

        try:
            model_hdl = ModelHandler(xgb.XGBClassifier(), self.training_variables)
            model_hdl.set_model_params(self.hyperparams)
            model_hdl.train_test_model(train_test_data, False, output_margin=True)
        except Exception as e:
            print(f"Training failed for {pretty_label}: {e}")
            return

        bdt_out_plot = pu.plot_output_train_test(model_hdl, train_test_data, 100, True, ["Signal", "Background"], True, density=True)
        bdt_out_plot.savefig(f"{self.QA_dir}/bdt_output_{label}.pdf")
        plt.close("all")

        y_pred_test = model_hdl.predict(test_features, output_margin = True)
        y_pred_train = model_hdl.predict(train_features, output_margin = True)
        plt.hist(y_pred_test[test_labels==0], bins=100, label='background', alpha=0.5, density=True)
        plt.hist(y_pred_test[test_labels==1], bins=100, label='signal', alpha=0.5, density=True)
        plt.xlabel("test BDT_score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.QA_dir}/testset_score_distribution_split_{label}.pdf")
        plt.close("all")
        roc_plot = pu.plot_roc_train_test(test_labels, y_pred_test, train_labels, y_pred_train)
        roc_plot.savefig(f"{self.QA_dir}/roc_test_vs_train_{label}.pdf")
        plt.close("all")

        effi_arr = np.round(np.linspace(0.5, 0.99, self.npoints_for_effi),3)
        try:
            score_arr = au.score_from_efficiency_array(test_labels, y_pred_test, effi_arr)
        except Exception as e:
            print(f"Warning: failed to derive score-efficiency array for {pretty_label}: {e}")
            score_arr = np.array([])

        if score_arr.size > 0:
            np.savetxt(f"{self.WP_dir}/score_efficiency_array_{label}.txt", np.column_stack((score_arr, effi_arr)))
            plt.figure(figsize=(7,5))
            plt.plot(score_arr, effi_arr, marker='o', linestyle='-', color='C0')
            plt.xlabel('BDT score')
            plt.ylabel('BDT efficiency')
            plt.title(f'Efficiency vs Score  {pretty_label}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f"{self.QA_dir}/efficiency_vs_score_{label}.pdf", bbox_inches='tight')
            plt.close("all")
        else:
            print("No score array available to plot/save for this bin.")

        model_name = f"Model_BDT_{label}.json"
        model_name_pkl = f"Model_BDT_{label}.pkl"
        model_path = f"{self.models_dir}/{model_name}"
        model_path_pkl = f"{self.models_dir}/{model_name_pkl}"
        model_hdl.dump_model_handler(model_path_pkl)
        try:
            model_org = model_hdl.get_original_model()
            booster = model_org.get_booster()
            booster.save_model(model_path)
            print(f"Saved model via ModelHandler: {model_name}")
        except Exception as e:
            print(f"ERROR: failed to save model for {pretty_label}: {e}")
        
        return model_hdl

    def _predict_and_rewrite_data_file(self, data_root_path, model_hdl, column_name="model_output"):
        """
        Apply trained model to the data snapshot (re-open snapshot file), add a column with predictions
        and overwrite the original snapshot root file with the new tree containing the added column.
        Parameters:
          data_root_path: Path or str to the snapshot ROOT file
          model_hdl: trained ModelHandler instance (supports apply_model_handler)
          column_name: name for the prediction column to add
        """
        try:
            data_root_path = Path(data_root_path)
            if not data_root_path.exists():
                print(f"_predict_and_rewrite_data_file: snapshot file not found: {data_root_path}")
                return False
            # Load handler from snapshot file
            try:
                data_hdl = TreeHandler(str(data_root_path), self.tree_name_data)
            except Exception as e:
                print(f"_predict_and_rewrite_data_file: failed to open snapshot with TreeHandler: {e}")
                return False
            # Apply model handler to add prediction column to the internal dataframe
            # shift NsigmaHe
            try:
                df = data_hdl.get_data_frame()
                mask = (df['fNSigmaHe'] > -2) & (df['fNSigmaHe'] < 1)
                if mask.any():
                    mean_shift = df.loc[mask, 'fNSigmaHe'].mean()
                else:
                    mean_shift = df['fNSigmaHe'].mean()
                df['fNSigmaHe'] = df['fNSigmaHe'] - mean_shift
                data_hdl.set_data_frame(df)
            except Exception as e:
                print(f"_predict_and_rewrite_data_file: failed to shift fNSigmaHe: {e}")
                return False
            try:
                num_data_before = len(data_hdl)
                data_hdl.apply_model_handler(model_hdl, column_name=column_name)
                num_data_after = len(data_hdl)
                print("[Info]: ")
                print(f"_predict_and_rewrite_data_file: applied model handler, added column '{column_name}'. Entries before: {num_data_before}, after: {num_data_after} \n")
                print("***********************************")
            except Exception as e:
                print(f"_predict_and_rewrite_data_file: apply_model_handler failed: {e}")
                return False
            # Get augmented pandas DataFrame
            try:
                df_save = data_hdl.get_data_frame()
            except Exception as e:
                print(f"_predict_and_rewrite_data_file: failed to get dataframe from handler: {e}")
                return False
            # Convert pandas dtypes to uproot-friendly type strings
            branches = {}
            arrays = {}
            for col in df_save.columns:
                arr = df_save[col].to_numpy()
                # convert pandas nullable ints to numpy
                if arr.dtype == object:
                    # try to coerce to float
                    try:
                        arr = arr.astype('float64')
                    except Exception:
                        # fallback to string storage not supported by uproot in this simple method
                        print(f"_predict_and_rewrite_data_file: column {col} has object dtype; skipping")
                        continue
                arrays[col] = arr
                if np.issubdtype(arr.dtype, np.floating):
                    branches[col] = 'float64'
                elif np.issubdtype(arr.dtype, np.integer):
                    branches[col] = 'int64'
                elif np.issubdtype(arr.dtype, np.bool_):
                    branches[col] = 'bool'
                else:
                    # fallback to float
                    branches[col] = 'float64'
            # Overwrite the ROOT file: recreate tree with new branches
            try:
                # uproot v4: write a new tree by assigning a dict of arrays
                with uproot.recreate(str(data_root_path)) as f:
                    # ensure arrays have numpy types
                    write_dict = {}
                    for k, arr in arrays.items():
                        # convert pandas/numpy scalars to numpy arrays
                        write_dict[k] = np.asarray(arr)
                    f[self.tree_name_data] = write_dict
                print(f"_predict_and_rewrite_data_file: successfully rewrote snapshot {data_root_path} with column '{column_name}'")
                return True
            except Exception as e:
                print(f"_predict_and_rewrite_data_file: failed to write snapshot via uproot: {e}")
                return False
        except Exception as e:
            print(f"_predict_and_rewrite_data_file: unexpected error: {e}")
            return False

    def _process_training_unit(self, pt_range=None, ct_range=None, cen_range=None):
        label = self._make_label(pt_range, ct_range, cen_range)
        pretty_label = self._make_description(pt_range, ct_range, cen_range)
        print(f"[SNAPSHOT+ML] Processing {pretty_label}")

        data_root, mc_root = self._make_snapshot_for_bin(pt_range, ct_range, cen_range, label)
        bin_data_hdl, bin_mc_hdl = self._read_handlers(data_root, mc_root)

        if bin_mc_hdl is None or len(bin_mc_hdl) == 0:
            print(f"Skipping bin (no MC): {pretty_label}")
            return
        if bin_data_hdl is None or len(bin_data_hdl) == 0:
            print(f"Skipping bin (no data): {pretty_label}")
            return

        side_band_sel_data = f"(fMassH3L<{self.side_band_edges[0]} or fMassH3L>{self.side_band_edges[1]})"
        side_band_sel_mc = f"(fMassH3L>{self.side_band_edges[0]} and fMassH3L<{self.side_band_edges[1]})"
        bin_data_hdl.apply_preselections(side_band_sel_data)
        bin_mc_hdl.apply_preselections(side_band_sel_mc)
        bin_mc_hdl, bin_data_hdl = self._balance_and_prepare(bin_mc_hdl, bin_data_hdl)
        print(f"Training set sizes after balancing: MC={len(bin_mc_hdl)}, Data={len(bin_data_hdl)}")

        bin_model_hdl = self._train_and_save_model_per_bin(bin_mc_hdl, bin_data_hdl, label, pretty_label)
        if bin_model_hdl is not None:
            try:
                self._predict_and_rewrite_data_file(data_root, bin_model_hdl, column_name="model_output")
            except Exception as e:
                print(f"Warning: failed to predict-and-rewrite snapshot for {pretty_label}: {e}")
        print(f"Completed processing for {pretty_label}\n")

    def run(self):
        print("Using snapshot dir:", self.snapshot_dir)
        print("Using models dir:", self.models_dir)
        print("Mix_mode:", self.mix_mode)

        if self.mix_mode == 'pt-ct':
            if self.pt_bins is None or self.ct_bins is None:
                raise ValueError("Mix_mode 'pt-ct' requires 'pt_bins' (1D edges) and 'ct_bins' (list of ct edges per pt).")
            for i_pt, (pt_min, pt_max) in enumerate(zip(self.pt_bins[:-1], self.pt_bins[1:])):
                if i_pt >= len(self.ct_bins):
                    print(f"Warning: ct_bins 缺少第 {i_pt} 个元素，跳过该 pt bin")
                    continue
                for ct_min, ct_max in zip(self.ct_bins[i_pt][:-1], self.ct_bins[i_pt][1:]):
                    self._process_training_unit((pt_min, pt_max), (ct_min, ct_max), None)

        elif self.mix_mode == 'cen-pt':
            if self.cen_bins is None:
                raise ValueError("Mix_mode 'cen-pt' requires 'cen_bins'.")
            pt_by_cen = self.pt_bins_by_centrality if self.pt_bins_by_centrality is not None else self.pt_bins
            if pt_by_cen is None:
                raise ValueError("Mix_mode 'cen-pt' requires 'pt_bins' or 'pt_bins_by_centrality'.")
            if len(self.cen_bins) < 2:
                raise ValueError("cen_bins must contain at least two edges.")
            for i_cen, (cen_min, cen_max) in enumerate(zip(self.cen_bins[:-1], self.cen_bins[1:])):
                if i_cen >= len(pt_by_cen):
                    print(f"Warning: pt bins for centrality index {i_cen} not found, skip.")
                    continue
                pt_edges = pt_by_cen[i_cen]
                if pt_edges is None or len(pt_edges) < 2:
                    print(f"Warning: pt bins for centrality index {i_cen} invalid, skip.")
                    continue
                for pt_min, pt_max in zip(pt_edges[:-1], pt_edges[1:]):
                    self._process_training_unit((pt_min, pt_max), None, (cen_min, cen_max))

        elif self.mix_mode == 'pt-ct-single':
            pt_range = self.pt_bin if self.pt_bin is not None else self._ensure_range(self.pt_bins, 'pt_bins')
            ct_range = self.ct_bin if self.ct_bin is not None else self._ensure_range(self.ct_bins, 'ct_bins')
            self._process_training_unit(pt_range, ct_range, None)

        elif self.mix_mode == 'pt-single':
            pt_range = self.pt_bin if self.pt_bin is not None else self._ensure_range(self.pt_bins, 'pt_bins')
            self._process_training_unit(pt_range, None, None)

        elif self.mix_mode == 'ct-single':
            ct_range = self.ct_bin if self.ct_bin is not None else self._ensure_range(self.ct_bins, 'ct_bins')
            pt_range = None
            if self.pt_bin is not None:
                pt_range = self._ensure_range(self.pt_bin, 'pt_bin')
            elif self.pt_bins is not None and len(self.pt_bins) == 2:
                pt_range = (self.pt_bins[0], self.pt_bins[1])
            self._process_training_unit(pt_range, ct_range, None)

        else:
            raise ValueError(f"Unsupported Mix_mode '{self.mix_mode}'.")

        print("***All Training done.***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDT Preprocessing for H3l analysis')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

    proc = BDTPreProcess(config)
    proc.run()