import ROOT
ROOT.ROOT.EnableImplicitMT()
import os
import numpy as np
import argparse
import json
import yaml
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
import hipe4ml.analysis_utils as au
import hipe4ml.plot_utils as pu
import matplotlib.pyplot as plt
import xgboost as xgb

import joblib
from pathlib import Path
import uproot
import pandas as pd


def sanitize_period_tag(tag):
    return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(tag))


def combined_period_tag(periods):
    tags = []
    for iper, period in enumerate(periods):
        if isinstance(period, dict):
            tag = period.get('tag', f'period{iper}')
        else:
            tag = f'period{iper}'
        tags.append(sanitize_period_tag(tag))
    return '_'.join([t for t in tags if t]) or 'combined_period'


def combined_top_dir(path_value, tag):
    path = Path(path_value)
    return str(path.parent / tag)


def combined_sub_dir(path_value, tag):
    path = Path(path_value)
    return str(path.parent.parent / tag / path.name)


def normalize_mix_mode(mode_raw, default_mode='pt_ct'):
    if isinstance(mode_raw, bool):
        mode_raw = 'pt_ct' if mode_raw else 'pt_ct_single'
    mode = str(mode_raw if mode_raw is not None else default_mode).strip().lower().replace('-', '_')
    aliases = {
        'ptct': 'pt_ct',
        'cenpt': 'cen_pt',
        'ptctsingle': 'pt_ct_single',
        'ptsingle': 'pt_single',
        'ctsingle': 'ct_single',
    }
    return aliases.get(mode, mode)


def format_bin_edge(value):
    return f"{float(value):g}"


def resolve_bdt_config(raw_config):
    if not isinstance(raw_config, dict):
        raise ValueError('Configuration root must be a mapping object.')

    preprocess = raw_config.get('preprocess', None)
    if not isinstance(preprocess, dict) or not isinstance(preprocess.get('bdt', None), dict):
        return raw_config

    cfg = dict(preprocess['bdt'])
    common = raw_config.get('common', {}) if isinstance(raw_config.get('common', {}), dict) else {}
    execution = raw_config.get('execution', {}) if isinstance(raw_config.get('execution', {}), dict) else {}
    paths = common.get('path', {}) if isinstance(common.get('path', {}), dict) else {}
    binning = common.get('binning', {}) if isinstance(common.get('binning', {}), dict) else {}
    selection = common.get('selection', {}) if isinstance(common.get('selection', {}), dict) else {}
    tree_names = common.get('tree_names', {}) if isinstance(common.get('tree_names', {}), dict) else {}
    periods = common.get('periods', []) if isinstance(common.get('periods', []), list) else []
    combine_period = bool(execution.get('combine_period', False))

    path_map = {
        'data_path': ['data_path'],
        'mc_path': ['mc_path'],
        'snapshot_dir': ['snapshot_dir'],
        'model_dir': ['model_dir'],
        'QA_dir': ['qa_dir'],
        'WP_dir': ['wp_dir'],
    }
    for target, keys in path_map.items():
        if target in cfg:
            continue
        for key in keys:
            val = paths.get(key, None)
            if val not in (None, ''):
                cfg[target] = val
                break

    if 'tree_name_data' not in cfg and tree_names.get('data', None):
        cfg['tree_name_data'] = tree_names['data']
    if 'tree_name_mc' not in cfg and tree_names.get('mc', None):
        cfg['tree_name_mc'] = tree_names['mc']
    if combine_period and periods and 'periods' not in cfg:
        cfg['periods'] = periods
    if combine_period and periods:
        tag = combined_period_tag(periods)
        cfg['combined_period_tag'] = tag
        if cfg.get('snapshot_dir'):
            cfg['snapshot_dir'] = combined_top_dir(cfg['snapshot_dir'], tag)
        for key in ('model_dir', 'QA_dir', 'WP_dir'):
            if cfg.get(key):
                cfg[key] = combined_sub_dir(cfg[key], tag)

    if 'basic_selection_data' not in cfg:
        basic_sel = selection.get('basic_selection_data', None)
        if basic_sel:
            cfg['basic_selection_data'] = basic_sel

    for key in ['cen_bins', 'pt_bins_by_centrality', 'pt_bins', 'ct_bins_single', 'pt_bins_single']:
        if key not in cfg and key in binning:
            cfg[key] = binning[key]
    if 'ct_bins' not in cfg and 'ct_bins_by_pt' in binning:
        cfg['ct_bins'] = binning['ct_bins_by_pt']

    if 'mix_mode' not in cfg and isinstance(execution, dict):
        training_mode = execution.get('training_mode', None)
        if training_mode is not None:
            cfg['mix_mode'] = training_mode
    return cfg


class BDTPreProcess:
    """
    可配置的训练模式：
            - mix_mode = "pt_ct"        : 原始行为，pt_bins 为 1D，ct_bins 为与 pt 对应的 2D 边界列表。
            - mix_mode = "cen_pt"       : 新增模式，cen_bins 为 1D，pt_bins(或 pt_bins_by_centrality) 为随 centrality 变化的 2D 边界列表。
            - mix_mode = "pt_ct_single" : 单一 pt、ct 区间训练。
            - mix_mode = "pt_single"    : 仅按单一 pt 区间训练（ct 不切分）。
            - mix_mode = "ct_single"    : 仅按单一 ct 区间训练（pt 可选过滤）。
    """

    def __init__(self, config):
        # 配置项
        self.data_path = config['data_path']
        self.mc_path = config['mc_path']
        self.periods = config.get('periods', [])
        self.period_weight_mode = str(config.get('period_weight_mode', 'none')).strip().lower()
        self.use_training_overrides = bool(config.get('use_training_overrides', False))
        self.training_overrides = config.get('training_overrides', [])
        if not isinstance(self.training_overrides, list):
            self.training_overrides = []
        self.mc_pt_bin_var = str(config.get('mc_pt_bin_var', 'fPt')).strip() or 'fPt'
        self.tree_name_data = config['tree_name_data']
        self.tree_name_mc = config['tree_name_mc']
        self.pt_bins = config.get('pt_bins', None)
        self.ct_bins = config.get('ct_bins', None)
        self.pt_bins_single = config.get('pt_bins_single', None)
        self.ct_bins_single = config.get('ct_bins_single', None)
        self.pt_bin = config.get('pt_bin', None)   # for separate mode optional
        self.ct_bin = config.get('ct_bin', None)   # for separate mode optional
        self.basic_selection_data = config.get('basic_selection_data', None)
        self.training_variables = config['training_variables']
        self.extra_vars_save_data = config.get('extra_vars_save_data', [])
        self.extra_vars_save_mc = config.get('extra_vars_save_mc', [])
        self.test_set_size = config['test_set_size']
        self.bkg_fraction_max = config['bkg_fraction_max']
        self.random_state = config['random_state']
        self.hyperparams = config['hyperparams']
        self.npoints_for_effi = int(config['npoints_for_effi'])
        eff_range = config.get('efficiency_range', None)
        if eff_range is not None:
            if not isinstance(eff_range, (list, tuple)) or len(eff_range) != 2:
                raise ValueError("efficiency_range must be a two-element list: [min, max].")
            self.efficiency_min = float(eff_range[0])
            self.efficiency_max = float(eff_range[1])
        else:
            self.efficiency_min = float(config.get('efficiency_min', 0.5))
            self.efficiency_max = float(config.get('efficiency_max', 0.99))
        if self.npoints_for_effi < 2:
            raise ValueError("npoints_for_effi must be >= 2.")
        if not (0.0 < self.efficiency_min < self.efficiency_max <= 1.0):
            raise ValueError(
                f"Invalid BDT efficiency range: min={self.efficiency_min}, max={self.efficiency_max}. "
                "Require 0 < min < max <= 1."
            )
        self.make_training_qa = bool(config.get('make_training_qa', True))
        self.qa_plot_bins = int(config.get('qa_plot_bins', 100))
        self.use_score_efficiency_fallback = bool(config.get('use_score_efficiency_fallback', False))
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
        self.training_failures = []
        self.score_efficiency_files = []
        mix_mode_cfg = config.get('mix_mode', 'pt_ct')
        self.mix_mode = normalize_mix_mode(mix_mode_cfg, 'pt_ct')
        self._declare_its_helpers()

        # prepare ROOT chains and RDataFrames
        self.period_rdfs = []
        self._opened_files = []
        self._chains = []
        if self.periods:
            for iper, period in enumerate(self.periods):
                tag = str(period.get('tag', f'period{iper}'))
                data_path = period.get('data_path', self.data_path)
                mc_path = period.get('mc_path', self.mc_path)
                data_rdf = self._make_ready_rdf(data_path, self.tree_name_data, is_mc=False)
                mc_rdf = self._make_ready_rdf(mc_path, self.tree_name_mc, is_mc=True)
                if self.basic_selection_data:
                    data_rdf = data_rdf.Filter(self.basic_selection_data)
                    mc_rdf = mc_rdf.Filter(self.basic_selection_data)
                self.period_rdfs.append({'tag': tag, 'data_rdf': data_rdf, 'mc_rdf': mc_rdf})
            print(f"[Info] Multi-period training enabled with {len(self.period_rdfs)} periods, period_weight_mode={self.period_weight_mode}")
        else:
            self.data_rdf = self._make_ready_rdf(self.data_path, self.tree_name_data, is_mc=False)
            self.mc_rdf = self._make_ready_rdf(self.mc_path, self.tree_name_mc, is_mc=True)
            if self.basic_selection_data:
                self.data_rdf = self.data_rdf.Filter(self.basic_selection_data)
                self.mc_rdf = self.mc_rdf.Filter(self.basic_selection_data) # Ensure to synchronize the basic selection on MC
        if self.extra_vars_save_data:
            self.data_columns = self._unique_columns(self._all_configured_training_variables() + self.extra_vars_save_data)
        else:
            self.data_columns = self._unique_columns(self._all_configured_training_variables() + ["fPt", "fCt", "fMassH3L", "fIsMatter"])
        if self.extra_vars_save_mc:
            self.mc_columns = self._unique_columns(self._all_configured_training_variables() + self.extra_vars_save_mc)
        else:
            self.mc_columns = self._unique_columns(self._all_configured_training_variables() + ["fPt", "fAbsGenPt", "fGenCt", "fMassH3L", "fIsMatter"])

    @staticmethod
    def _unique_columns(columns):
        out = []
        seen = set()
        for col in columns:
            if col in seen:
                continue
            seen.add(col)
            out.append(col)
        return out

    def _all_configured_training_variables(self):
        variables = list(self.training_variables)
        if self.use_training_overrides:
            for override in self.training_overrides:
                if not isinstance(override, dict):
                    continue
                override_vars = override.get('training_variables', None)
                if isinstance(override_vars, list):
                    variables.extend(override_vars)
        return self._unique_columns(variables)

    @staticmethod
    def _centrality_override_matches(override, cen_range):
        if not isinstance(override, dict) or cen_range is None or None in cen_range:
            return False
        cen_min, cen_max = float(cen_range[0]), float(cen_range[1])
        ranges = override.get('centrality_ranges', override.get('cen_ranges', []))
        if not isinstance(ranges, list):
            return False
        for rng in ranges:
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                continue
            lo, hi = float(rng[0]), float(rng[1])
            if abs(cen_min - lo) < 1e-6 and abs(cen_max - hi) < 1e-6:
                return True
        return False

    def _active_training_config(self, cen_range=None):
        cfg = {
            'name': 'default',
            'training_variables': list(self.training_variables),
            'side_band_edges': list(self.side_band_edges),
            'bkg_fraction_max': self.bkg_fraction_max,
            'hyperparams': dict(self.hyperparams),
            'test_set_size': self.test_set_size,
            'efficiency_min': self.efficiency_min,
            'efficiency_max': self.efficiency_max,
            'npoints_for_effi': self.npoints_for_effi,
            'mc_use_full_centrality': False,
        }
        if self.use_training_overrides:
            for override in self.training_overrides:
                if not self._centrality_override_matches(override, cen_range):
                    continue
                cfg['name'] = override.get('name', cfg['name'])
                if isinstance(override.get('training_variables', None), list):
                    cfg['training_variables'] = list(override['training_variables'])
                if isinstance(override.get('side_band_edges', None), list) and len(override['side_band_edges']) == 2:
                    cfg['side_band_edges'] = list(override['side_band_edges'])
                if 'bkg_fraction_max' in override:
                    cfg['bkg_fraction_max'] = override['bkg_fraction_max']
                if isinstance(override.get('hyperparams', None), dict):
                    merged = dict(cfg['hyperparams'])
                    merged.update(override['hyperparams'])
                    cfg['hyperparams'] = merged
                for key in ('test_set_size', 'efficiency_min', 'efficiency_max', 'npoints_for_effi'):
                    if key in override:
                        cfg[key] = override[key]
                if 'mc_use_full_centrality' in override:
                    cfg['mc_use_full_centrality'] = bool(override['mc_use_full_centrality'])
                break
        cfg['npoints_for_effi'] = int(cfg['npoints_for_effi'])
        cfg['efficiency_min'] = float(cfg['efficiency_min'])
        cfg['efficiency_max'] = float(cfg['efficiency_max'])
        cfg['test_set_size'] = float(cfg['test_set_size'])
        return cfg

    @staticmethod
    def _declare_its_helpers():
        if hasattr(ROOT, 'CountITSHits') and hasattr(ROOT, 'AvgITSClusterSize'):
            return
        helper_path = Path(__file__).resolve().parents[1] / 'include' / 'its_helpers.cc'
        if not helper_path.exists():
            raise FileNotFoundError(f"Missing ITS helper source: {helper_path}")
        with open(helper_path, 'r') as f:
            ROOT.gInterpreter.Declare(f.read())

    @staticmethod
    def _load_all_trees_to_chain(file_handle, chain, treename='O2hypcands'):
        if not file_handle or file_handle.IsZombie():
            raise RuntimeError("Cannot open ROOT input file")

        for key in file_handle.GetListOfKeys():
            name_key = key.GetName()
            if 'DF_' not in name_key:
                continue
            obj = key.ReadObj()
            if not obj or not obj.InheritsFrom('TDirectory'):
                continue
            tree = obj.Get(treename)
            if tree and tree.InheritsFrom('TTree'):
                chain.Add(f"{file_handle.GetName()}/{name_key}/{treename}")
        return ROOT.RDataFrame(chain)

    def _make_ready_rdf(self, root_path, tree_name, is_mc=False):
        chain = ROOT.TChain(tree_name)
        root_file = ROOT.TFile.Open(str(root_path))
        self._opened_files.append(root_file)
        rdf = self._load_all_trees_to_chain(root_file, chain, tree_name)
        self._chains.append(chain)
        return self._correct_and_convert_df(rdf, calibrate_he3_pt=False, isMC=is_mc, isH4L=False)

    @staticmethod
    def _reweight_pt_spectrum(df, var, distribution):
        dist_global_name = '__rw_pt_reweight'
        distribution.SetName(dist_global_name)
        try:
            ROOT.gDirectory.Add(distribution)
        except Exception:
            pass

        max_bw = float(distribution.GetMaximum())
        if max_bw <= 0:
            raise ValueError('The provided distribution has non-positive maximum.')

        expr = (
            '(((gRandom->Uniform()) > ((TF1*)gDirectory->Get("{name}"))->Eval({var})/{max_bw}) ? -1 : 1)'
        ).format(name=dist_global_name, var=var, max_bw=max_bw)
        return df.Define('rej', expr)

    @staticmethod
    def _cut_elements_to_same_range(handler1, handler2, element_names):
        if isinstance(element_names, str):
            element_names = [element_names]

        df1 = handler1.get_data_frame()
        df2 = handler2.get_data_frame()

        for element_name in element_names:
            cut_min = max(df1[element_name].min(), df2[element_name].min())
            cut_max = min(df1[element_name].max(), df2[element_name].max())

            df1 = df1[(df1[element_name] >= cut_min) & (df1[element_name] <= cut_max)]
            df2 = df2[(df2[element_name] >= cut_min) & (df2[element_name] <= cut_max)]
            print(f"Applied cut to {element_name}: range [{cut_min}, {cut_max}]")

        if 'fCt' in df1.columns and 'fCt' in df2.columns:
            df1 = df1[df1['fCt'] < 50]
            df2 = df2[df2['fCt'] < 50]
            print("Applied additional cut: fCt < 50")

        handler1.set_data_frame(df1)
        handler2.set_data_frame(df2)

    @staticmethod
    def _correct_and_convert_df(df, calibrate_he3_pt=False, isMC=False, isH4L=False):
        if not hasattr(df, 'GetColumnNames') or not hasattr(df, 'Define'):
            raise TypeError('Only ROOT.RDataFrame-like input is supported in BDTPreProcess.')

        coloumn_name = list(df.GetColumnNames())
        print('Columns before correction:', coloumn_name)
        if 'fFlags' in coloumn_name:
            df = df.Define('fHePIDHypo', '(int)(fFlags >> 4)') \
                   .Define('fPiPIDHypo', '(int)(fFlags & 0xF)')
        if calibrate_he3_pt:
            df = df.Define(
                'fPtHe3',
                '((fHePIDHypo==6) ? (fPtHe3 + (-0.1286 - 0.1269 * fPtHe3 + 0.06 * fPtHe3*fPtHe3)) '
                ': (fPtHe3 + 2.98019e-02 + 7.66100e-01 * exp(-1.31641e+00 * fPtHe3))))'
            )

        df = df.Define('fPxHe3', 'fPtHe3 * cos(fPhiHe3)') \
               .Define('fPyHe3', 'fPtHe3 * sin(fPhiHe3)') \
               .Define('fPzHe3', 'fPtHe3 * sinh(fEtaHe3)') \
               .Define('fPHe3', 'fPtHe3 * cosh(fEtaHe3)') \
               .Define('fEnHe3', 'sqrt(fPHe3*fPHe3 + 2.8083916*2.8083916)') \
               .Define('fEnHe4', 'sqrt(fPHe3*fPHe3 + 3.7273794*3.7273794)')
        df = df.Define('fPxPi', 'fPtPi * cos(fPhiPi)') \
               .Define('fPyPi', 'fPtPi * sin(fPhiPi)') \
               .Define('fPzPi', 'fPtPi * sinh(fEtaPi)') \
               .Define('fPPi', 'fPtPi * cosh(fEtaPi)') \
               .Define('fEnPi', 'sqrt(fPPi*fPPi + 0.139570*0.139570)')
        df = df.Define('fPx', 'fPxHe3 + fPxPi') \
               .Define('fPy', 'fPyHe3 + fPyPi') \
               .Define('fPz', 'fPzHe3 + fPzPi') \
               .Define('fP', 'sqrt(fPx*fPx + fPy*fPy + fPz*fPz)') \
               .Define('fEn', 'fEnHe3 + fEnPi') \
               .Define('fEn4', 'fEnHe4 + fEnPi')
        df = df.Define('fPt', 'sqrt(fPx*fPx + fPy*fPy)') \
               .Define('fEta', 'acosh(fP / fPt)') \
               .Define('fCosLambda', 'fPt / fP') \
               .Define('fCosLambdaHe', 'fPtHe3 / fPHe3')
        if not isH4L:
            df = df.Define('fDecLen', 'sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)') \
                   .Define('fCt', 'fDecLen * 2.99131 / fP')
        else:
            df = df.Define('fDecLen', 'sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx + fZDecVtx*fZDecVtx)') \
                   .Define('fCt', 'fDecLen * 3.922 / fP')
        df = df.Define('fDecRad', 'sqrt(fXDecVtx*fXDecVtx + fYDecVtx*fYDecVtx)') \
               .Define('fCosPA', '(fPx * fXDecVtx + fPy * fYDecVtx + fPz * fZDecVtx) / (fP * fDecLen)') \
               .Define('fMassH3L', 'sqrt(fEn*fEn - fP*fP)') \
               .Define('fMassH4L', 'sqrt(fEn4*fEn4 - fP*fP)') \
               .Define('fTPCSignMomHe3', 'fTPCmomHe * (-1 + 2*fIsMatter)') \
               .Define('fGloSignMomHe3', 'fPHe3 / 2. * (-1 + 2*fIsMatter)')
        if isMC:
            df = df.Define('fGenDecLen', 'sqrt(fGenXDecVtx*fGenXDecVtx + fGenYDecVtx*fGenYDecVtx + fGenZDecVtx*fGenZDecVtx)') \
                   .Define('fGenPz', 'fGenPt * sinh(fGenEta)') \
                   .Define('fGenP', 'sqrt(fGenPt*fGenPt + fGenPz*fGenPz)') \
                   .Define('fAbsGenPt', 'abs(fGenPt)')
            if not isH4L:
                df = df.Define('fGenCt', 'fGenDecLen * 2.99131 / fGenP')
            else:
                df = df.Define('fGenCt', 'fGenDecLen * 3.922 / fGenP')

        if 'fITSclusterSizesHe' in coloumn_name and 'fITSclusterSizesPi' in coloumn_name:
            df = df.Define('fAvgClusterSizeHe', 'AvgITSClusterSize(fITSclusterSizesHe)') \
                   .Define('nITSHitsHe', 'CountITSHits(fITSclusterSizesHe)') \
                   .Define('fAvgClusterSizePi', 'AvgITSClusterSize(fITSclusterSizesPi)') \
                   .Define('nITSHitsPi', 'CountITSHits(fITSclusterSizesPi)') \
                   .Define('fAvgClSizeCosLambda', 'fAvgClusterSizeHe * fCosLambdaHe')

        colounm_name_after = list(df.GetColumnNames())
        print('Columns after correction:', colounm_name_after)
        return df

    def _make_label(self, pt_range=None, ct_range=None, cen_range=None):
        parts = []
        if cen_range and None not in cen_range:
            parts.append(f"cen_{format_bin_edge(cen_range[0])}_{format_bin_edge(cen_range[1])}")
        if pt_range and None not in pt_range:
            parts.append(f"pt_{format_bin_edge(pt_range[0])}_{format_bin_edge(pt_range[1])}")
        if ct_range and None not in ct_range:
            parts.append(f"ct_{format_bin_edge(ct_range[0])}_{format_bin_edge(ct_range[1])}")
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

    def _make_snapshot_for_bin(self, pt_range=None, ct_range=None, cen_range=None, label=None, active_cfg=None):
        active_cfg = active_cfg or self._active_training_config(cen_range)
        pt_min, pt_max = pt_range if pt_range else (None, None)
        ct_min, ct_max = ct_range if ct_range else (None, None)
        cen_min, cen_max = cen_range if cen_range else (None, None)

        label = label or self._make_label(pt_range, ct_range, cen_range)

        sel_data_parts = []
        sel_mc_parts = []
        if pt_min is not None and pt_max is not None:
            sel_data_parts.append(f"fPt > {pt_min} && fPt < {pt_max}")
            sel_mc_parts.append(f"{self.mc_pt_bin_var} > {pt_min} && {self.mc_pt_bin_var} < {pt_max}")
        if ct_min is not None and ct_max is not None:
            sel_data_parts.append(f"fCt > {ct_min} && fCt < {ct_max}")
            sel_mc_parts.append(f"fGenCt > {ct_min} && fGenCt < {ct_max}")
        if cen_min is not None and cen_max is not None:
            sel_data_parts.append(f"fCentralityFT0C > {cen_min} && fCentralityFT0C < {cen_max}")
            if not active_cfg.get('mc_use_full_centrality', False):
                sel_mc_parts.append(f"fCentralityFT0C > {cen_min} && fCentralityFT0C < {cen_max}")

        sel_data = " && ".join(sel_data_parts) if sel_data_parts else "1"
        sel_mc = " && ".join(sel_mc_parts) if sel_mc_parts else "1"
        sel_mc = sel_mc + " && fIsReco == 1"  # ensure only reconstructed MC is used for training

        data_root = self.snapshot_dir / f"data_{label}.root"
        mc_root   = self.snapshot_dir / f"mc_{label}.root"

        if self.period_rdfs:
            data_parts = []
            mc_parts = []
            for iper, period in enumerate(self.period_rdfs):
                tag = period['tag']
                safe_tag = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in tag)
                data_part = self.snapshot_dir / f"data_{label}__{safe_tag}.root"
                mc_part = self.snapshot_dir / f"mc_{label}__{safe_tag}.root"
                try:
                    if data_part.exists():
                        data_part.unlink()
                    period['data_rdf'].Filter(sel_data).Define('fPeriodIndex', str(iper)).Snapshot(
                        self.tree_name_data, str(data_part), self.data_columns + ['fPeriodIndex'])
                    if data_part.exists():
                        data_parts.append(data_part)
                        print(f"Saved period data snapshot: {data_part}")
                except Exception as e:
                    print(f"Warning: period data snapshot failed for {label}/{tag}: {e}")
                try:
                    if mc_part.exists():
                        mc_part.unlink()
                    period['mc_rdf'].Filter(sel_mc).Define('fPeriodIndex', str(iper)).Snapshot(
                        self.tree_name_mc, str(mc_part), self.mc_columns + ['fPeriodIndex'])
                    if mc_part.exists():
                        mc_parts.append(mc_part)
                        print(f"Saved period MC snapshot: {mc_part}")
                except Exception as e:
                    print(f"Warning: period MC snapshot failed for {label}/{tag}: {e}")

            try:
                self._merge_period_snapshots(data_parts, data_root, self.tree_name_data)
                self._merge_period_snapshots(mc_parts, mc_root, self.tree_name_mc)
            finally:
                self._cleanup_period_snapshots(data_parts + mc_parts)
            return data_root if data_root.exists() else None, mc_root if mc_root.exists() else None

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
            self.mc_rdf.Filter(sel_mc).Snapshot(self.tree_name_mc, str(mc_root), self.mc_columns)
            print(f"Saved MC snapshot: {mc_root}")
        except Exception as e:
            print(f"Warning: MC snapshot failed for {label}: {e}")

        return data_root if data_root.exists() else None, mc_root if mc_root.exists() else None

    @staticmethod
    def _merge_period_snapshots(input_paths, output_path, tree_name):
        if output_path.exists():
            output_path.unlink()
        frames = []
        for path in input_paths:
            try:
                with uproot.open(str(path)) as f:
                    if tree_name not in f:
                        print(f"Warning: tree {tree_name} not found in {path}")
                        continue
                    frames.append(f[tree_name].arrays(library='pd'))
            except Exception as e:
                print(f"Warning: failed to read period snapshot {path}: {e}")
        if not frames:
            print(f"Warning: no period snapshots to merge for {output_path}")
            return
        df = pd.concat(frames, ignore_index=True, sort=False)
        arrays = {col: df[col].to_numpy() for col in df.columns}
        with uproot.recreate(str(output_path)) as f:
            f[tree_name] = arrays
        print(f"Saved merged multi-period snapshot: {output_path} ({len(df)} rows)")

    @staticmethod
    def _cleanup_period_snapshots(input_paths):
        for path in input_paths:
            try:
                if path.exists():
                    path.unlink()
                    print(f"Removed temporary period snapshot: {path}")
            except Exception as e:
                print(f"Warning: failed to remove temporary period snapshot {path}: {e}")

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

    def _balance_and_prepare(self, bin_mc_hdl, bin_data_hdl, training_variables=None, bkg_fraction_max=None):
        training_variables = training_variables or self.training_variables
        bkg_fraction_max = self.bkg_fraction_max if bkg_fraction_max is None else bkg_fraction_max
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
            self._cut_elements_to_same_range(bin_mc_hdl, bin_data_hdl, training_variables)
            if self.period_weight_mode == 'equal_period':
                self._equalize_periods(bin_mc_hdl)
                self._equalize_periods(bin_data_hdl)
            if bkg_fraction_max is not None and len(bin_data_hdl) > bkg_fraction_max * len(bin_mc_hdl):
                bin_data_hdl.shuffle_data_frame(size=int(bkg_fraction_max * len(bin_mc_hdl)), inplace=True, random_state=self.random_state)
        except Exception as e:
            print(f"Warning during balancing: {e}")

        return bin_mc_hdl, bin_data_hdl

    def _equalize_periods(self, handler):
        df = handler.get_data_frame()
        if 'fPeriodIndex' not in df.columns or df.empty:
            return
        counts = df.groupby('fPeriodIndex').size()
        if counts.empty:
            return
        target = int(counts.min())
        if target <= 0:
            return
        pieces = []
        for _, group in df.groupby('fPeriodIndex'):
            pieces.append(group.sample(n=min(target, len(group)), random_state=self.random_state))
        if pieces:
            out = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
            handler.set_data_frame(out)

    @staticmethod
    def _score_from_efficiency_quantile(labels, scores, effi_arr):
        labels = np.asarray(labels)
        scores = np.asarray(scores, dtype=float)
        effi_arr = np.asarray(effi_arr, dtype=float)
        sig_scores = scores[labels == 1]
        sig_scores = sig_scores[np.isfinite(sig_scores)]
        if sig_scores.size == 0:
            return np.array([])
        out = []
        for eff in effi_arr:
            eff_clamped = float(np.clip(eff, 0.0, 1.0))
            # Threshold for selecting candidates with score >= threshold.
            out.append(float(np.quantile(sig_scores, 1.0 - eff_clamped)))
        return np.asarray(out, dtype=float)

    def _derive_score_efficiency_array(self, test_labels, y_pred_test, train_labels, y_pred_train, effi_arr, pretty_label):
        try:
            score_arr = au.score_from_efficiency_array(test_labels, y_pred_test, effi_arr)
            score_arr = np.asarray(score_arr, dtype=float)
            if score_arr.size == effi_arr.size and np.all(np.isfinite(score_arr)):
                return score_arr
            print(f"Warning: invalid hipe4ml score-efficiency array for {pretty_label}.")
        except Exception as e:
            print(f"Warning: failed to derive score-efficiency array with hipe4ml for {pretty_label}: {e}")

        if not self.use_score_efficiency_fallback:
            print(f"Warning: score-efficiency fallback is disabled for {pretty_label}.")
            return np.array([])

        score_arr = self._score_from_efficiency_quantile(test_labels, y_pred_test, effi_arr)
        if score_arr.size == effi_arr.size and np.all(np.isfinite(score_arr)):
            print(f"Warning: used test-sample quantile fallback for score-efficiency array in {pretty_label}.")
            return score_arr

        score_arr = self._score_from_efficiency_quantile(train_labels, y_pred_train, effi_arr)
        if score_arr.size == effi_arr.size and np.all(np.isfinite(score_arr)):
            print(f"Warning: used train-sample quantile fallback for score-efficiency array in {pretty_label}.")
            return score_arr

        return np.array([])

    def _train_and_save_model_per_bin(self, bin_mc_hdl, bin_data_hdl, label, pretty_label, active_cfg=None):
        active_cfg = active_cfg or self._active_training_config(None)
        training_variables = active_cfg['training_variables']
        try:
            train_test_data = au.train_test_generator([bin_mc_hdl, bin_data_hdl], [1, 0],
                                                     test_size=active_cfg['test_set_size'], random_state=self.random_state)
        except Exception as e:
            msg = f"Failed to generate train/test for {pretty_label}: {e}"
            print(msg)
            self.training_failures.append(msg)
            return

        train_features = train_test_data[0]
        train_labels = train_test_data[1]
        test_features = train_test_data[2]
        test_labels = train_test_data[3]

        if self.make_training_qa:
            distr = pu.plot_distr([bin_mc_hdl, bin_data_hdl], training_variables + ["fMassH3L"], bins=self.qa_plot_bins, labels=['Signal',"Background"],colors=["red","blue"], log=True, density=True, figsize=(18, 13), alpha=0.5, grid=False)
            plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
            plt.savefig(f"{self.QA_dir}/features_distributions_{label}.pdf", bbox_inches='tight')
            plt.close("all")
            corr = pu.plot_corr([bin_mc_hdl,bin_data_hdl], training_variables + ["fMassH3L"], ['Signal',"Background"])
            corr[0].savefig(f"{self.QA_dir}/correlations_mc_{label}.pdf", bbox_inches='tight')
            corr[1].savefig(f"{self.QA_dir}/correlations_data_{label}.pdf", bbox_inches='tight')
            plt.close("all")

        try:
            model_hdl = ModelHandler(xgb.XGBClassifier(), training_variables)
            model_hdl.set_model_params(active_cfg['hyperparams'])
            model_hdl.train_test_model(train_test_data, False, output_margin=True)
        except Exception as e:
            msg = f"Training failed for {pretty_label}: {e}"
            print(msg)
            self.training_failures.append(msg)
            return

        y_pred_test = model_hdl.predict(test_features, output_margin = True)
        y_pred_train = model_hdl.predict(train_features, output_margin = True)
        if self.make_training_qa:
            bdt_out_plot = pu.plot_output_train_test(model_hdl, train_test_data, self.qa_plot_bins, True, ["Background", "Signal"], True, density=True)
            bdt_out_plot.savefig(f"{self.QA_dir}/bdt_output_{label}.pdf")
            plt.close("all")

            plt.hist(y_pred_test[test_labels==0], bins=self.qa_plot_bins, label='background', alpha=0.5, density=True)
            plt.hist(y_pred_test[test_labels==1], bins=self.qa_plot_bins, label='signal', alpha=0.5, density=True)
            plt.xlabel("test BDT_score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.QA_dir}/testset_score_distribution_split_{label}.pdf")
            plt.close("all")
            roc_plot = pu.plot_roc_train_test(test_labels, y_pred_test, train_labels, y_pred_train)
            roc_plot.savefig(f"{self.QA_dir}/roc_test_vs_train_{label}.pdf")
            plt.close("all")

        effi_arr = np.unique(np.round(np.linspace(active_cfg['efficiency_min'], active_cfg['efficiency_max'], active_cfg['npoints_for_effi']), 3))
        if effi_arr.size < active_cfg['npoints_for_effi']:
            print(f"Warning: rounded efficiency grid has {effi_arr.size} unique points instead of {active_cfg['npoints_for_effi']}.")
        score_arr = self._derive_score_efficiency_array(
            test_labels, y_pred_test, train_labels, y_pred_train, effi_arr, pretty_label)

        if score_arr.size > 0:
            score_eff_path = self.WP_dir / f"score_efficiency_array_{label}.txt"
            np.savetxt(score_eff_path, np.column_stack((score_arr, effi_arr)))
            self.score_efficiency_files.append(str(score_eff_path))
            plt.figure(figsize=(7,5))
            plt.plot(score_arr, effi_arr, marker='o', linestyle='-', color='C0')
            plt.xlabel('BDT score')
            plt.ylabel('BDT efficiency')
            plt.title(f'Efficiency vs Score  {pretty_label}')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f"{self.QA_dir}/efficiency_vs_score_{label}.pdf", bbox_inches='tight')
            plt.close("all")
        else:
            msg = f"No score array available to plot/save for {pretty_label} ({label})."
            print(msg)
            self.training_failures.append(msg)

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
        active_cfg = self._active_training_config(cen_range)
        print(
            f"Training config for {pretty_label}: {active_cfg['name']} "
            f"(side_band_edges={active_cfg['side_band_edges']}, "
            f"bkg_fraction_max={active_cfg['bkg_fraction_max']}, "
            f"mc_use_full_centrality={active_cfg['mc_use_full_centrality']}, "
            f"variables={active_cfg['training_variables']})"
        )

        data_root, mc_root = self._make_snapshot_for_bin(pt_range, ct_range, cen_range, label, active_cfg)
        bin_data_hdl, bin_mc_hdl = self._read_handlers(data_root, mc_root)

        if bin_mc_hdl is None or len(bin_mc_hdl) == 0:
            print(f"Skipping bin (no MC): {pretty_label}")
            return
        if bin_data_hdl is None or len(bin_data_hdl) == 0:
            print(f"Skipping bin (no data): {pretty_label}")
            return

        side_band_edges = active_cfg['side_band_edges']
        side_band_sel_data = f"(fMassH3L<{side_band_edges[0]} or fMassH3L>{side_band_edges[1]})"
        side_band_sel_mc = f"(fMassH3L>{side_band_edges[0]} and fMassH3L<{side_band_edges[1]})"
        bin_data_hdl.apply_preselections(side_band_sel_data)
        bin_mc_hdl.apply_preselections(side_band_sel_mc)
        if len(bin_mc_hdl) == 0:
            msg = f"Skipping bin after signal-window MC preselection (no MC): {pretty_label}"
            print(msg)
            self.training_failures.append(msg)
            return
        if len(bin_data_hdl) == 0:
            msg = f"Skipping bin after side-band data preselection (no data): {pretty_label}"
            print(msg)
            self.training_failures.append(msg)
            return
        bin_mc_hdl, bin_data_hdl = self._balance_and_prepare(
            bin_mc_hdl,
            bin_data_hdl,
            training_variables=active_cfg['training_variables'],
            bkg_fraction_max=active_cfg['bkg_fraction_max'])
        if len(bin_mc_hdl) == 0 or len(bin_data_hdl) == 0:
            msg = f"Skipping bin after balancing (MC={len(bin_mc_hdl)}, Data={len(bin_data_hdl)}): {pretty_label}"
            print(msg)
            self.training_failures.append(msg)
            return
        print(f"Training set sizes after balancing: MC={len(bin_mc_hdl)}, Data={len(bin_data_hdl)}")

        bin_model_hdl = self._train_and_save_model_per_bin(bin_mc_hdl, bin_data_hdl, label, pretty_label, active_cfg)
        if bin_model_hdl is not None:
            try:
                self._predict_and_rewrite_data_file(data_root, bin_model_hdl, column_name="model_output")
            except Exception as e:
                print(f"Warning: failed to predict-and-rewrite snapshot for {pretty_label}: {e}")
        print(f"Completed processing for {pretty_label}\n")

    def run(self):
        print("Using snapshot dir:", self.snapshot_dir)
        print("Using models dir:", self.models_dir)
        print("mix_mode:", self.mix_mode)
        print(f"BDT efficiency grid: {self.efficiency_min:.3f} -> {self.efficiency_max:.3f} ({self.npoints_for_effi} points)")
        print(f"Training QA plots: {'enabled' if self.make_training_qa else 'disabled'}")
        print(f"Score-efficiency fallback: {'enabled' if self.use_score_efficiency_fallback else 'disabled'}")

        if self.mix_mode == 'pt_ct':
            if self.pt_bins is None or self.ct_bins is None:
                raise ValueError("mix_mode 'pt_ct' requires 'pt_bins' (1D edges) and 'ct_bins' (list of ct edges per pt).")
            for i_pt, (pt_min, pt_max) in enumerate(zip(self.pt_bins[:-1], self.pt_bins[1:])):
                if i_pt >= len(self.ct_bins):
                    print(f"Warning: ct_bins 缺少第 {i_pt} 个元素，跳过该 pt bin")
                    continue
                for ct_min, ct_max in zip(self.ct_bins[i_pt][:-1], self.ct_bins[i_pt][1:]):
                    self._process_training_unit((pt_min, pt_max), (ct_min, ct_max), None)

        elif self.mix_mode == 'cen_pt':
            if self.cen_bins is None:
                raise ValueError("mix_mode 'cen_pt' requires 'cen_bins'.")
            pt_by_cen = self.pt_bins_by_centrality if self.pt_bins_by_centrality is not None else self.pt_bins
            if pt_by_cen is None:
                raise ValueError("mix_mode 'cen_pt' requires 'pt_bins' or 'pt_bins_by_centrality'.")
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

        elif self.mix_mode == 'pt_ct_single':
            pt_range = self.pt_bin if self.pt_bin is not None else self._ensure_range(self.pt_bins, 'pt_bins')
            ct_range = self.ct_bin if self.ct_bin is not None else self._ensure_range(self.ct_bins, 'ct_bins')
            self._process_training_unit(pt_range, ct_range, None)

        elif self.mix_mode == 'pt_single':
            pt_ranges = []
            if self.pt_bin is not None:
                pt_ranges = [self._ensure_range(self.pt_bin, 'pt_bin')]
            elif self.pt_bins_single is not None:
                if len(self.pt_bins_single) < 2:
                    raise ValueError("pt_bins_single must contain at least two edges.")
                pt_ranges = list(zip(self.pt_bins_single[:-1], self.pt_bins_single[1:]))
            elif self.pt_bins is not None:
                if len(self.pt_bins) < 2:
                    raise ValueError("pt_bins must contain at least two edges for pt-single mode.")
                pt_ranges = list(zip(self.pt_bins[:-1], self.pt_bins[1:]))
            else:
                raise ValueError("mix_mode 'pt_single' requires 'pt_bin', 'pt_bins_single' or 'pt_bins'.")

            for pt_range in pt_ranges:
                self._process_training_unit(pt_range, None, None)

        elif self.mix_mode == 'ct_single':
            # optional pt filter for all ct bins
            pt_range = None
            if self.pt_bin is not None:
                pt_range = self._ensure_range(self.pt_bin, 'pt_bin')
            elif self.pt_bins is not None and len(self.pt_bins) == 2:
                pt_range = (self.pt_bins[0], self.pt_bins[1])

            ct_ranges = []
            if self.ct_bin is not None:
                ct_ranges = [self._ensure_range(self.ct_bin, 'ct_bin')]
            elif self.ct_bins_single is not None:
                if len(self.ct_bins_single) < 2:
                    raise ValueError("ct_bins_single must contain at least two edges.")
                ct_ranges = list(zip(self.ct_bins_single[:-1], self.ct_bins_single[1:]))
            elif self.ct_bins is not None:
                if len(self.ct_bins) < 2:
                    raise ValueError("ct_bins must contain at least two edges for ct-single mode.")
                # if a single pair is given, this gracefully handles it as well
                ct_ranges = list(zip(self.ct_bins[:-1], self.ct_bins[1:])) if isinstance(self.ct_bins[0], (int, float)) else []
                if not ct_ranges:
                    raise ValueError("ct_bins must be a 1D edge list for ct-single mode; use ct_bins_single for multiple bins.")
            else:
                raise ValueError("mix_mode 'ct_single' requires 'ct_bin', 'ct_bins_single' or 1D 'ct_bins'.")

            for ct_range in ct_ranges:
                self._process_training_unit(pt_range, ct_range, None)

        else:
            raise ValueError(f"Unsupported mix_mode '{self.mix_mode}'.")

        print("***All Training done.***")
        print(f"[Summary] Saved {len(self.score_efficiency_files)} score-efficiency arrays in {self.WP_dir}")
        if self.training_failures:
            print("[Summary] Training bins with missing score-efficiency arrays or skipped inputs:")
            for msg in self.training_failures:
                print(f"  - {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDT Preprocessing for H3l analysis')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file (YAML/JSON)')
    parser.add_argument('--mix-mode', type=str, default='', help='Optional training mode override, e.g. cen_pt')
    args = parser.parse_args()

    config_path = Path(args.config_file)
    with open(config_path, 'r') as config_file:
        if config_path.suffix.lower() == '.json':
            raw_config = json.load(config_file)
        else:
            raw_config = yaml.safe_load(config_file)

    config = resolve_bdt_config(raw_config)
    if args.mix_mode:
        config['mix_mode'] = args.mix_mode

    proc = BDTPreProcess(config)
    proc.run()
