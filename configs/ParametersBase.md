<span style="color:red;">This is a note that backup some basic parameters for the analysis configs</span>

# Binnings for each situation
## Pt differential spectrum
- 
``` json 
"cen_bins": [0, 10, 30, 50, 80],
"pt_bins_by_centrality": [
  [2, 3, 3.5, 4, 4.5, 5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 5, 8]
]
"cen_bins_0_80": [0, 80],
"pt_bins_by_centrality_0_80": [
[2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.5, 6, 6.5, 7, 8]],
```
## Ct single for lifetime extraction
``` json
"ct_bins_single": [1, 3, 5, 7, 9, 11, 13, 15, 17, 25, 38],
"ct_bins_single_inside_beampipe": [0.8, 1, 1.2, 1.4, 1.6, 1.8]
```

## Pt-Ct selection for absorption study
``` json
"pt_bins": [2, 3, 4, 5.5, 8],
"ct_bins": [[1, 3, 6, 9, 12, 18, 30],
            [1, 3, 6, 9, 12, 18, 25],
            [1, 3, 6, 9, 15, 25],
            [1, 3, 6, 10, 23]]
```

# Selection 
## Selction for Data Snapshot 
``` json
"basic_selection_data": "fDecRad > 0.8",
"basic_selection_data_beampipe": "fDecRad > 0.8 && fDecRad < 2.1 fCentralityFT0C > 0 && fCentralityFT0C < 80",
"basic_selection_data_lifetime": "fDecRad > 0.8 && fCentralityFT0C > 0 && fCentralityFT0C < 80"
```

# Pass config
## pass5
``` json
"data_path": "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/NCrossedRows/AO2D_CustomV0s_HadronPID.root",
"analysisresults_path": "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
"mc_path": "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass5/LHC25g11_G4list/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
"snapshot_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows",
"wp_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/WorkingPoint",
"model_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/TrainedModels",
"qa_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass5_CustomV0s_HadronPID_NCrossedRows/QAPlots",
"mc_file_for_absorption": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x1.5.root",
"spectrum_file": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/ReweightFunc.root",
"output_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs"
```
## pass4
``` json
"data_path": "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/pass4/NCrossedRows/AO2D_CustomV0s_HadronPID.root",
"analysisresults_path": "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/pass4/NCrossedRows/AnalysisResults_CustomV0s_HadronPID.root",
"mc_path": "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/mc/apass4/NCrossedRows/reweighted/AO2D_CustomV0s_combined_reweighted.root",
"snapshot_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows",
"wp_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/WorkingPoint",
"model_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/TrainedModels",
"qa_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs/MLProcess/LHC23_PbPb_pass4_CustomV0s_HadronPID_NCrossedRows/QAPlots",
"mc_file_for_absorption": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/AbsorptionTrees/absorption_tree_x1.5.root",
"spectrum_file": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/CodeSpace/Ploting_scrips/ReweightFunc.root",
"output_dir": "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/Outputs"
```
