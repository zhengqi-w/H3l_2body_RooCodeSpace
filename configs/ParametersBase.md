<span style="color:red;">This is a note that backup some basic parameters for the analysis configs</span>

# Binnings for each situation
## Pt differential spectrum
- 
``` json 
"cen_bins": [0, 10, 30, 50, 80],
"related_multiplicity_center": [1857, 1050, 455, 137],
"pt_bins_by_centrality": [
  [2, 3, 3.5, 4, 4.5, 5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 5, 8]
]

"ceb_bins_merged_data": [0, 5, 10, 20, 30, 40, 50, 60, 70, 90],
"related_multiplicity_center_merged_data": [2047, 1668, 1253, 848, 559, 351, 205, 110, 38.1],
"pt_bins_by_centrality_merged_data": [
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8],
  [2, 2.5, 3, 3.5, 4, 4.5, 5, 7],
  [2, 2.5, 3, 3.5, 4.5, 6.5],
  [2, 2.5, 3.5, 6]
],

"cen_bins_0_5": [0, 5],
"related_multiplicity_center_0_5": [2047],
"pt_bins_by_centrality_0_5": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
],

"cen_bins_5_10": [5, 10],
"related_multiplicity_center_5_10": [1668],
"pt_bins_by_centrality_5_10": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
],

"cen_bins_10_20": [10, 20],
"related_multiplicity_center_10_20": [1253],
"pt_bins_by_centrality_10_20": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
],

"cen_bins_20_30": [20, 30],
"related_multiplicity_center_20_30": [848],
"pt_bins_by_centrality_20_30": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
],

"cen_bins_30_40": [30, 40],
"related_multiplicity_center_30_40": [559],
"pt_bins_by_centrality_30_40": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8]
],

"cen_bins_40_50": [40, 50],
"related_multiplicity_center_40_50": [351],
"pt_bins_by_centrality_40_50": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8]
],

"cen_bins_50_60": [50, 60],
"related_multiplicity_center_50_60": [205],
"pt_bins_by_centrality_50_60": [
  [1.5, 2.5, 3, 3.5, 4, 4.5, 5, 7]
],

"cen_bins_60_70": [60, 70],
"related_multiplicity_center_60_70": [110],
"pt_bins_by_centrality_60_70": [
  [1.5, 2.5, 3, 3.5, 4, 7]
],

"cen_bins_70_90": [70, 90],
"related_multiplicity_center_70_90": [38.1],
"pt_bins_by_centrality_70_90": [
  [1.5, 2.5, 3.5, 6]
]

"cen_bins_0_80": [0, 80],
"related_multiplicity_center_0_80": [xxx],
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

# Multiplicity comparation for centrality binning 5.36 TeV PbPb collisions
| Centrality (%) | 0-5 | 5-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 |
|----------------|------|-------|-------|-------|------|--------|-------|-------|-------|-------|
| Multiplicity (dNch/deta) | 2047 ± 54 | 1668 ± 42 | 1253 ± 33 |  848 ± 25 |  559 ± 19 | 351 ± 14 | 205 ± 11 | 110 ± 8 | 53 ± 5 | 23.2 ± 2.8 |
reference: https://alice-notes.web.cern.ch/system/files/notes/analysis/1511/2025-03-27-dNch-dEta-PbPbRun3_v6.pdf (Table 5)
