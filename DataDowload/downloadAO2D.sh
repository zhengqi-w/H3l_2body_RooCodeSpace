#!/bin/bash

set -euo pipefail

infile=$1

###
# Usage:
# Copy the bullet points "LHCXXX all/all: /alice..." of the "Hyperloop train run finished" e-mail into a text file "input.txt"
# Run this script using ./downloadSkimTreesHyperloop.sh input.txt
# This will parse the paths and put the AnalysisResults.root and AO2D.root (the trees) into subdirectories named for each period.
###

periodName=""
declare -a runnums=()
declare -a paths=()

copy_run_files() {
   local sourcePath=$1
   local targetDir=$2
   local sanitizedPath=${sourcePath%/}
   local -a baseCandidates=("$sanitizedPath")

   mkdir -p "$targetDir"

   for basePath in "${baseCandidates[@]}"; do
      echo "[downloadAO2D] bulk copy from '$basePath/[0-9]*/' using ext_root filter"

      # Copy all .root files inside numeric subdirectories (e.g. AOD/001/).
      if alien_cp "${basePath}/[0-9]*/" -name "ext_root" "file:${targetDir}/" 2>/dev/null; then
         return
      fi

      # Fallback: allow one more numeric nesting level if present.
      if alien_cp "${basePath}/[0-9]*/[0-9]*/" -name "ext_root" "file:${targetDir}/" 2>/dev/null; then
         return
      fi
   done

   echo "[downloadAO2D] Warning: no files copied for path '$sourcePath'" >&2
}

process_runs() {
   local period=$1
   echo "[downloadAO2D] process_runs: period='$period' (runnums=${#runnums[@]}, paths=${#paths[@]})"

   if [[ -z $period || ${#runnums[@]} -eq 0 || ${#paths[@]} -eq 0 ]]; then
      return
   fi

   local runCount=${#runnums[@]}
   local pathCount=${#paths[@]}
   local pairCount=$runCount

   if (( runCount != pathCount )); then
      echo "Warning: $period has $runCount run numbers but $pathCount paths; truncating to the smaller count." >&2
      if (( pathCount < pairCount )); then
         pairCount=$pathCount
      fi
   fi

   mkdir -p "$period"

   for ((i=0; i<pairCount; i++)); do
      local run=${runnums[$i]}
      local path=${paths[$i]}

      # skip empty entries that may result from trailing commas
      if [[ -z $run || -z $path ]]; then
         echo "[downloadAO2D] Skipping empty run/path at index $i"
         continue
      fi

      local targetDir="$period/$run"
      mkdir -p "$targetDir"
      echo "[downloadAO2D] Processing run '$run' -> path='$path' -> targetDir='$targetDir'"

      copy_run_files "$path" "$targetDir"
   done

   runnums=()
   paths=()
}

while IFS= read -r line || [[ -n $line ]]; do
   # skip empty lines
   if [[ -z "$line" ]]; then
      continue
   fi

   if [[ "$line" == LHC* ]]; then
      # New period starts; process the previous block first
      process_runs "$periodName"
      periodName=${line%% *}
      echo "[downloadAO2D] Detected new period line: '$line' -> periodName='$periodName'"
      runnums=()
      paths=()
      continue
   fi

   if [[ "$line" == runnum:* ]]; then
      payload=${line#runnum:}
      payload=${payload//$'\r'/}
      # remove whitespace around commas
      payload=${payload//[[:space:]]/}
      IFS=',' read -r -a runnums <<< "$payload"
      echo "[downloadAO2D] Parsed runnum line: ${#runnums[@]} entries"
      continue
   fi

   if [[ "$line" == path:* ]]; then
      payload=${line#path:}
      payload=${payload## } # strip leading space if present
      payload=${payload//$'\r'/}
      IFS=',' read -r -a paths <<< "$payload"
      echo "[downloadAO2D] Parsed path line: ${#paths[@]} entries"

      # process immediately when path line appears
      process_runs "$periodName"
      continue
   fi
done < "$infile"

# Process any trailing block (only if it has content)
process_runs "$periodName"