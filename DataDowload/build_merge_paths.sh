#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_merge_paths.sh --period <period> --runlist <file> [options]

Options:
  --base <dir>        Base directory that contains period folders (default: ./period)
  --out-ao2d <file>   Output path for AO2D list (default: merge_path.txt)
  --out-ana <file>    Output path for AnalysisResults list (default: analysis_path.txt)
EOF
}

period=""
runlist=""
base_dir="./"
out_ao2d="merge_path.txt"
out_ana="analysis_path.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --period)
      period="$2"
      shift 2
      ;;
    --runlist)
      runlist="$2"
      shift 2
      ;;
    --base)
      base_dir="$2"
      shift 2
      ;;
    --out-ao2d)
      out_ao2d="$2"
      shift 2
      ;;
    --out-ana)
      out_ana="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$period" || -z "$runlist" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$runlist" ]]; then
  echo "Run list file not found: $runlist" >&2
  exit 1
fi

header_line=$(awk -v p="$period" '$0 ~ "^"p"([[:space:]].*)?:$" {print; exit}' "$runlist")
cbt_tag=$(echo "$header_line" | sed -E "s/^${period}[[:space:]]*//; s/:$//")
if [[ -z "$cbt_tag" ]]; then
  cbt_tag="(none)"
fi

search_root="$base_dir/$period"
if [[ ! -d "$search_root" ]]; then
  echo "Period folder not found: $search_root" >&2
  exit 1
fi

runs_raw=$(awk -v p="$period" '
  $0 ~ "^"p"([[:space:]].*)?:$" {in_block=1; next}
  in_block && $1 ~ /^runnum:/ {sub(/^runnum:[[:space:]]*/, ""); print; next}
  in_block && $0 ~ /:$/ {exit}
' "$runlist" | tr ',' ' ')

runs_raw=$(echo "$runs_raw" | xargs)
if [[ -z "$runs_raw" ]]; then
  echo "No run numbers found for period: $period" >&2
  exit 1
fi

read -r -a runs <<< "$runs_raw"
regex=$(printf "%s|" "${runs[@]}")
regex="${regex%|}"
run_regex="(/|_)(${regex})(/|_)"

all_ao2d=$(mktemp)
all_ana=$(mktemp)
cleanup() {
  rm -f "$all_ao2d" "$all_ana"
}
trap cleanup EXIT

find "$search_root" -name AO2D.root -print | sort -u > "$all_ao2d"
find "$search_root" -name AnalysisResults.root -print | sort -u > "$all_ana"

grep -E "$run_regex" "$all_ao2d" > "$out_ao2d" || true
grep -E "$run_regex" "$all_ana" > "$out_ana" || true

echo "Period: $period"
echo "CBT tag: $cbt_tag"
for run in "${runs[@]}"; do
  ao2d_count=$(grep -E "(/|_)${run}(/|_)" "$out_ao2d" | wc -l || true)
  ana_count=$(grep -E "(/|_)${run}(/|_)" "$out_ana" | wc -l || true)
  echo "Run $run: AO2D=$ao2d_count AnalysisResults=$ana_count"
done

echo "AO2D list: $out_ao2d"
echo "AnalysisResults list: $out_ana"
