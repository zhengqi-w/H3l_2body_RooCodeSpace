#!/usr/bin/env python3
# quick check if the Snapshot process lose candidates after fCentralityFT0C and Pt filtering
# Run command like: python3 ROOTWorkFlow/CodeSpace/Checks/count_candidates.py --pt-min 2 --pt-max 3
"""Compare O2hypcands counts after fCentralityFT0C filtering."""

import argparse
import pathlib
import sys

import numpy as np
import uproot


def count_candidates(
    path: pathlib.Path,
    tree_path: str,
    cen_min: float,
    cen_max: float,
    pt_min: float | None = None,
    pt_max: float | None = None,
) -> int:
    """Return number of entries with fCentralityFT0C range (and optional pt window)."""
    with uproot.open(path) as root_file:
        if tree_path not in root_file:
            raise KeyError(f"Tree '{tree_path}' not found in {path}")
        tree = root_file[tree_path]
        if "fCentralityFT0C" not in tree.keys():
            raise KeyError(f"Branch 'fCentralityFT0C' missing from {tree_path} in {path}")

        branches = ["fCentralityFT0C"]
        pt_calculator = None
        if pt_min is not None or pt_max is not None:
            if "fPt" in tree.keys():
                branches.append("fPt")
                pt_calculator = lambda arrays: arrays["fPt"]
            else:
                required = ["fPtHe3", "fPhiHe3", "fPtPi", "fPhiPi"]
                missing = [br for br in required if br not in tree.keys()]
                if missing:
                    raise KeyError(
                        f"Missing branches for derived pt ({', '.join(missing)}) in {tree_path} of {path}"
                    )
                branches.extend(required)

                def derived_pt(arrays: dict[str, np.ndarray]) -> np.ndarray:
                    px_he3 = arrays["fPtHe3"] * np.cos(arrays["fPhiHe3"])
                    py_he3 = arrays["fPtHe3"] * np.sin(arrays["fPhiHe3"])
                    px_pi = arrays["fPtPi"] * np.cos(arrays["fPhiPi"])
                    py_pi = arrays["fPtPi"] * np.sin(arrays["fPhiPi"])
                    px = px_he3 + px_pi
                    py = py_he3 + py_pi
                    return np.sqrt(px * px + py * py)

                pt_calculator = derived_pt

        count = 0
        for arrays in tree.iterate(branches, library="np", step_size=1_000_000):
            cent = arrays["fCentralityFT0C"]
            mask = (cent >= cen_min) & (cent < cen_max)
            if pt_calculator is not None:
                pt = pt_calculator(arrays)
                if pt_min is not None:
                    mask &= pt >= pt_min
                if pt_max is not None:
                    mask &= pt < pt_max
            count += int(np.count_nonzero(mask))
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Check centrality-filtered O2hypcands counts")
    parser.add_argument(
        "--data-file",
        type=pathlib.Path,
        default=pathlib.Path(
            "/Users/zhengqingwang/alice/data/derived/Hypertriton_2body/LHC23_PbPb_fullTPC/apass5/AO2D_CustomV0s_HadronPID.root"
        ),
        help="Source AO2D ROOT file containing DF and O2hypcands",
    )
    parser.add_argument(
        "--data-tree",
        default="DF_2339954065445376/O2hypcands",
        help="Path to the O2hypcands tree inside the AO2D file",
    )
    parser.add_argument(
        "--snapshot",
        type=pathlib.Path,
        default=pathlib.Path(
            "/Users/zhengqingwang/alice/run3task/H3l_2body_spectrum/ROOTWorkFlow/SnapShotsData/LHC23_PbPb_pass5_CustomV0s_HadronPID/data_cen_0_10_pt_2_3.root"
        ),
        help="Snapshot file to compare",
    )
    parser.add_argument("--snapshot-tree", default="O2hypcands", help="Tree path inside the snapshot")
    parser.add_argument("--cen-min", type=float, default=0.0, help="Lower edge of fCentralityFT0C (inclusive)")
    parser.add_argument("--cen-max", type=float, default=10.0, help="Upper edge of fCentralityFT0C (exclusive)")
    parser.add_argument("--pt-min", type=float, default=None, help="Minimal pt (inclusive) for fPt selection")
    parser.add_argument("--pt-max", type=float, default=None, help="Maximal pt (exclusive) for fPt selection")
    args = parser.parse_args()

    for path in (args.data_file, args.snapshot):
        if not path.exists():
            parser.error(f"File not found: {path}")

    data_count = count_candidates(
        args.data_file, args.data_tree, args.cen_min, args.cen_max, args.pt_min, args.pt_max
    )
    snap_count = count_candidates(
        args.snapshot,
        args.snapshot_tree,
        args.cen_min,
        args.cen_max,
        args.pt_min,
        args.pt_max,
    )

    print(f"{args.data_file}: {data_count} entries with {args.cen_min} <= fCentralityFT0C < {args.cen_max}")
    print(f"{args.snapshot}: {snap_count} entries with {args.cen_min} <= fCentralityFT0C < {args.cen_max}")
    if data_count == snap_count:
        print("Counts match")
        sys.exit(0)
    print("Counts differ", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
