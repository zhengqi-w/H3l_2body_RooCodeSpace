1. Steps to run download and merging data
    - ``` cd alice ```
    - ``` o2 ```
    - ``` bash downloadAO2D.sh input.txt```
    Note: merge steps for desired CBT data quality status(eg. CBT_hadronPID)
    - ``` bash build_merge_paths.sh --period LHC25PbPb_pass1 --runlist ./MergeRunList.txt --base ./ --out-ao2d merge_path.txt --out-ana analysis_path.txt ```
    - ``` o2-aod-merger --max-size 3000000000000 --skip-parent-files-list --skip-non-existing-files --input merge_path.txt --output ./AO2D_period.root ```
    - ``` xargs -a analysis_path.txt hadd AnalysisResults_periodName.root ```

    Note: original merge steps (unfiltered):
    - ``` find ./period -name AO2D.root > merge_path.txt ```
    - ``` o2-aod-merger --max-size 3000000000000 --skip-parent-files-list --skip-non-existing-files --input merge_path.txt --output ./AO2D_period.root ```
    - ``` find ./period -name AnalysisResults.root | xargs hadd AnalysisResults_periodName.root ```