#!/usr/bin/env python3
"""
Resample a TOF NIfTI to **isotropic voxels** using the **smallest** of the
original X/Y/Z spacings. This preserves orientation and FoV in world space
(changes only the sampling grid).

Why this version?
- Avoids ANTs `ResampleImage` "stoi" crashes by using **ResampleImageBySpacing**.
- Requires only `--tof` as input; computes the target isotropic spacing for you.
- Optional `--interp` to choose interpolation (default behavior: let ANTs pick
  its default, typically Linear). If your build is picky, omit `--interp`.

Examples
--------
1) Make isotropic at the smallest native spacing (auto):
   python tof_resampler.py --tof MRI_tof.nii.gz

2) Same as above but be explicit about interpolation:
   python tof_resampler.py --tof MRI_tof.nii.gz --interp Linear

3) Dry-run to see the exact command without executing:
   python tof_resampler.py --tof MRI_tof.nii.gz --dry-run

Requirements
------------
- ANTs installed and `ResampleImageBySpacing` available in PATH
- nibabel (for reading original spacing)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib


# --------------------------- helpers ---------------------------------

def check_dependencies():
    if shutil.which("ResampleImageBySpacing") is None:
        sys.stderr.write("[ERROR] ANTs 'ResampleImageBySpacing' not found in PATH.\n")
        sys.exit(1)


def get_voxel_sizes(nifti_path: Path):
    img = nib.load(str(nifti_path))
    vx, vy, vz = img.header.get_zooms()[:3]
    return float(vx), float(vy), float(vz)


def build_output_name(in_path: Path, iso: float, out_arg: str | None) -> Path:
    if out_arg:
        return Path(out_arg)
    stem = in_path.name.replace(".nii.gz", "").replace(".nii", "")
    return in_path.parent / f"{stem}_iso_{iso:.6f}mm.nii.gz"


# ----------------------------- main ----------------------------------

def main():
    p = argparse.ArgumentParser(description="Resample TOF to isotropic at the smallest native spacing.")
    p.add_argument("--tof", required=True, help="Path to TOF NIfTI.")
    p.add_argument("--out", default=None, help="Output path (.nii.gz). If omitted, auto-generated.")
    p.add_argument("--interp", default=None,
                   choices=["NearestNeighbor", "Linear", "BSpline", "LanczosWindowedSinc"],
                   help="Interpolation method (optional). If omitted, ANTs default is used.")
    p.add_argument("--round", type=int, default=6,
                   help="Round spacing to this many decimals (default: 6).")
    p.add_argument("--dry-run", action="store_true", help="Print the command and exit without running.")

    args = p.parse_args()

    check_dependencies()

    in_path = Path(args.tof)
    if not in_path.exists():
        sys.stderr.write(f"[ERROR] Input not found: {in_path}\n")
        sys.exit(1)

    vx, vy, vz = get_voxel_sizes(in_path)
    iso = round(min(vx, vy, vz), max(0, int(args.round)))

    out_path = build_output_name(in_path, iso, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ANTs command: use spacings in mm (flag '0') and optionally interpolation
    cmd = [
        "ResampleImageBySpacing", "3", str(in_path), str(out_path),
        f"{iso}", f"{iso}", f"{iso}", "0"
    ]
    if args.interp:
        cmd.append(args.interp)

    print("\n--- Resampling TOF to isotropic ---")
    print(f"Input:        {in_path}")
    print(f"Orig spacing: {vx} x {vy} x {vz} mm")
    print(f"Target iso:   {iso} mm (smallest of the three)")
    print(f"Output:       {out_path}")
    if args.interp:
        print(f"Interpolation:{args.interp}")

    if args.dry_run:
        print("\n[DRY RUN] Command:")
        print(" ", " ".join(cmd))
        sys.exit(0)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write("\n[FATAL] ResampleImageBySpacing failed. Check ANTs installation and inputs.\n")
        sys.exit(e.returncode)

    print("\n[SUCCESS] Isotropic TOF saved to:", out_path)


if __name__ == "__main__":
    main()

