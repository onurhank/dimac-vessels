#!/usr/bin/env python3
"""
Vesselness (Frangi / ITK Objectness) for NIfTI volumes with CLI & previews.

- Works on 3D NIfTI. If given 4D (e.g., DIMAC), reduce to 3D via --reduce mean|median|tstd.
- Frangi (skimage) uses pixel sigmas: we either resample to isotropic voxels or
  convert mm->pixels using XY spacing (OK when voxels are near-isotropic).
- ITK Objectness (OOF-like) respects image spacing (mm) and can be preferred on anisotropic data.

Outputs:
  <prefix>_frangi_vesselness.nii.gz
  <prefix>_oof_vesselness.nii.gz
  <prefix>_frangi_mask.nii.gz
  <prefix>_oof_mask.nii.gz
  PNG previews for multiple quantiles.

Author: adapted & extended for CLI and robust defaults.
"""

import argparse
import os
import warnings
import numpy as np
import nibabel as nib

# Optional deps
from skimage.filters import frangi
from skimage.morphology import ball, binary_opening, binary_closing, remove_small_objects
from skimage.filters import threshold_otsu
import itk
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------
# IO helpers
# -----------------------
def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return img, data

def save_like(ref_img, data, out_path, dtype=np.float32):
    img = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    nib.save(img, out_path)
    print(f"Saved: {out_path}")

# -----------------------
# 4D -> 3D reducers (for DIMAC/rsfMRI)
# -----------------------
def reduce_4d(vol4d, how="mean"):
    if how == "mean":
        return np.nanmean(vol4d, axis=-1)
    if how == "median":
        return np.nanmedian(vol4d, axis=-1)
    if how == "tstd":
        return np.nanstd(vol4d, axis=-1)
    raise ValueError(f"Unknown reducer: {how}")

# -----------------------
# Intensity normalization
# -----------------------
def robust_rescale_01(vol, p_lo=1, p_hi=99):
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(v, (p_lo, p_hi))
    den = max(hi - lo, 1e-6)
    out = np.clip((vol - lo) / den, 0, 1).astype(np.float32)
    return out

# -----------------------
# Thresholding utilities
# -----------------------
def threshold_map(vesselness, mode="quantile", value=0.995):
    """
    mode='quantile' -> keep top (1 - q) fraction; value=0.995 => mask = v > Q99.5
    mode='otsu'     -> Otsu on finite voxels
    mode='fixed'    -> absolute threshold
    """
    v = vesselness[np.isfinite(vesselness)]
    if v.size == 0:
        return np.zeros_like(vesselness, dtype=bool)
    if mode == "quantile":
        thr = np.quantile(v, float(value))
        return vesselness > thr, thr
    elif mode == "otsu":
        thr = threshold_otsu(v)
        return vesselness > thr, thr
    elif mode == "fixed":
        thr = float(value)
        return vesselness > thr, thr
    else:
        raise ValueError("mode must be 'quantile', 'otsu', or 'fixed'")

def clean_mask(mask, min_size=50, radius=1):
    se = ball(radius)
    m = binary_opening(mask.astype(bool), se)
    m = binary_closing(m, se)
    m = remove_small_objects(m, min_size=min_size)
    return m.astype(np.uint8)

# -----------------------
# Frangi (skimage)
# -----------------------
def run_frangi(vol, voxel_spacing, scales_mm, frangi_resample_isotropic=False):
    """
    vol: 3D float32 (normalized 0..1 recommended).
    voxel_spacing: (sx, sy, sz) in mm
    scales_mm: array of physical scales in mm (e.g., 1..6 mm)
    frangi_resample_isotropic: if True, warn user to resample externally or accept XY mm->pixel conversion.
    """
    sx, sy, sz = map(float, voxel_spacing[:3])
    if frangi_resample_isotropic:
        # Placeholder: we keep simple conversion using XY spacing average;
        # Proper isotropic resampling can be added if needed.
        pass
    avg_xy = float((sx + sy) / 2.0)
    sigmas_px = (np.array(scales_mm, dtype=np.float32) / max(avg_xy, 1e-6)).astype(np.float32)

    # skimage.frangi expects array in zyx or xyz? It works with ndarray directly; we pass vol as-is.
    v = frangi(vol, sigmas=sigmas_px, black_ridges=False)  # bright tubes
    return np.nan_to_num(v.astype(np.float32))

# -----------------------
# ITK Objectness (OOF-like)
# -----------------------
def run_itk_objectness(vol, voxel_spacing, scales_mm, alpha=0.5, beta=0.5, gamma=5.0, bright=True):
    """
    vol: 3D float32 (normalized 0..1 recommended).
    Returns a 3D float32 vesselness in the SAME axis order as 'vol'.
    Spacing-agnostic: uses physical sigmas if supported, else falls back to pixel sigmas.
    """
    # numpy x,y,z -> ITK z,y,x
    itk_img = itk.image_from_array(np.transpose(vol, (2, 1, 0)))
    itk_img.SetSpacing(tuple(float(s) for s in voxel_spacing[:3]))

    HessianImageType = itk.Image[itk.SymmetricSecondRankTensor[itk.D, 3], 3]
    OutputImageType  = itk.Image[itk.F, 3]
    ObjectnessFilter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, OutputImageType].New(
        BrightObject=bool(bright),
        ObjectDimension=1,
        Alpha=float(alpha),
        Beta=float(beta),
        Gamma=float(gamma),
    )

    MultiScaleType = itk.MultiScaleHessianBasedMeasureImageFilter[type(itk_img), HessianImageType, OutputImageType]
    oof = MultiScaleType.New()
    oof.SetInput(itk_img)
    oof.SetHessianToMeasureFilter(ObjectnessFilter)

    # NormalizeAcrossScale if available
    if hasattr(oof, "SetNormalizeAcrossScale"):
        oof.SetNormalizeAcrossScale(True)

    # Try to use physical sigmas if available; otherwise fall back to pixel sigmas
    have_use_image_spacing = hasattr(oof, "SetUseImageSpacing")
    if have_use_image_spacing:
        # Physical (mm) sigmas
        oof.SetUseImageSpacing(True)
        smin = float(np.min(scales_mm))
        smax = float(np.max(scales_mm))
        nsteps = int(len(scales_mm))
    else:
        # Pixel sigmas: convert mm -> pixels using mean spacing (warn if anisotropic)
        sx, sy, sz = map(float, voxel_spacing[:3])
        anis = np.max([sx, sy, sz]) / max(1e-6, np.min([sx, sy, sz]))
        if anis > 1.2:
            print("[WARN] ITK oof lacks SetUseImageSpacing; using pixel sigmas on anisotropic voxels "
                  f"(spacing={voxel_spacing}). Consider resampling to isotropic for best results.")
        mean_sp = float((sx + sy + sz) / 3.0)
        scales_px = np.array(scales_mm, dtype=np.float32) / max(mean_sp, 1e-6)
        smin = float(np.min(scales_px))
        smax = float(np.max(scales_px))
        nsteps = int(len(scales_px))

    oof.SetSigmaMinimum(smin)
    oof.SetSigmaMaximum(smax)
    oof.SetNumberOfSigmaSteps(nsteps)

    # Some ITK builds expose step-method; if so, prefer equispaced scales
    if hasattr(oof, "SetSigmaStepMethodToEquispaced"):
        oof.SetSigmaStepMethodToEquispaced()

    oof.Update()
    arr_zyx = itk.array_from_image(oof.GetOutput())
    v = np.transpose(arr_zyx, (2, 1, 0)).astype(np.float32)
    return np.nan_to_num(v)

# -----------------------
# Previews
# -----------------------
def save_preview_pngs(vesselness, voxel_spacing, out_prefix, quantiles=(0.99, 0.95, 0.90, 0.85), slice_mode="mid", vmax=None):
    """
    Saves PNGs of the vesselness and masks for given quantiles.
    - slice_mode: 'mid' (axial mid-slice) or 'mip' (axial max-intensity projection)
    """
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # Display base: either mid-slice or MIP
    if slice_mode == "mid":
        z = vesselness.shape[2] // 2
        base = vesselness[:, :, z]
    else:  # 'mip'
        base = np.max(vesselness, axis=2)

    if vmax is None:
        vmax = np.percentile(vesselness[np.isfinite(vesselness)], 99.9)

    # Vesselness image
    plt.figure()
    plt.imshow(base.T, origin="lower", vmin=0, vmax=vmax)
    plt.title("Vesselness")
    plt.axis("off")
    plt.savefig(f"{out_prefix}_vesselness.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Masks at quantiles
    v = vesselness[np.isfinite(vesselness)]
    for q in quantiles:
        thr = np.quantile(v, q)
        if slice_mode == "mid":
            mask_slice = (vesselness[:, :, z] > thr).T
        else:
            mask_slice = (np.max(vesselness > thr, axis=2)).T  # MIP of mask

        plt.figure()
        plt.imshow(base.T, origin="lower", vmin=0, vmax=vmax)
        plt.imshow(mask_slice, origin="lower", alpha=0.35)  # simple overlay
        plt.title(f"Mask at quantile {q:.2f} (thr={thr:.4g})")
        plt.axis("off")
        plt.savefig(f"{out_prefix}_mask_q{q:.2f}.png", dpi=200, bbox_inches="tight")
        plt.close()

# -----------------------
# Main CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Frangi/OOF vesselness for NIfTI with CLI, autothresholding, and previews.")
    ap.add_argument("--input", required=True, help="Path to input NIfTI (3D or 4D). For DIMAC 4D, use --reduce.")
    ap.add_argument("--output-prefix", required=True, help="Prefix for outputs (e.g., out/run1).")
    ap.add_argument("--reduce", choices=["mean", "median", "tstd", "none"], default="mean",
                    help="If input is 4D: reduce across time to 3D. 'none' will error on 4D.")
    ap.add_argument("--normalize", action="store_true", help="Robust 1–99%% rescale to [0,1] before filtering.")
    ap.add_argument("--method", choices=["frangi", "oof", "both"], default="both")
    ap.add_argument("--scales-mm", type=float, nargs="+", default=[1,2,3,4,5,6],
                    help="List of physical scales (mm) for multiscale filters.")
    # Frangi specifics
    ap.add_argument("--frangi-resample-isotropic", action="store_true",
                    help="Assume near-isotropic voxels; convert mm->pixels via XY spacing. (Skips true resampling.)")
    # ITK Objectness specifics
    ap.add_argument("--oof-alpha", type=float, default=0.5)
    ap.add_argument("--oof-beta", type=float, default=0.5)
    ap.add_argument("--oof-gamma", type=float, default=5.0)
    ap.add_argument("--oof-bright", action="store_true", help="Use BrightObject=True (typical for TOF).")
    # Thresholding
    ap.add_argument("--thr-mode", choices=["quantile", "otsu", "fixed"], default="quantile",
                    help="How to threshold vesselness to binary mask.")
    ap.add_argument("--thr-value-frangi", type=float, default=0.995,
                    help="If quantile: e.g., 0.995 keeps top 0.5%%. If fixed: absolute value.")
    ap.add_argument("--thr-value-oof", type=float, default=0.995,
                    help="If quantile: e.g., 0.995 keeps top 0.5%%. If fixed: absolute value.")
    ap.add_argument("--min-size", type=int, default=50, help="Remove components smaller than this (voxels).")
    ap.add_argument("--morph-radius", type=int, default=1, help="Ball radius for opening/closing.")
    # Previews
    ap.add_argument("--previews", action="store_true", help="Save PNG previews at multiple quantiles.")
    ap.add_argument("--preview-quantiles", type=float, nargs="+", default=[0.99, 0.95, 0.90, 0.85])
    ap.add_argument("--preview-mode", choices=["mid", "mip"], default="mid",
                    help="Preview slice: 'mid' axial mid-slice, or 'mip' axial maximum intensity projection.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    # Load
    ref_img, vol = load_nifti(args.input)
    voxel_spacing = ref_img.header.get_zooms()[:3]

    # Reduce 4D if needed
    if vol.ndim == 4:
        if args.reduce == "none":
            raise ValueError("Input is 4D; choose --reduce mean|median|tstd to make it 3D.")
        vol3d = reduce_4d(vol, args.reduce)
    elif vol.ndim == 3:
        vol3d = vol
    else:
        raise ValueError("Unsupported image dimensionality.")

    # Normalize (recommended)
    if args.normalize:
        vol3d = robust_rescale_01(vol3d)

    # ---- Methods ----
    frangi_v = None
    oof_v = None

    if args.method in ("frangi", "both"):
        print("Running Frangi...")
        frangi_v = run_frangi(vol3d, voxel_spacing, np.array(args.scales_mm, dtype=np.float32),
                              frangi_resample_isotropic=args.frangi_resample_isotropic)
        save_like(ref_img, frangi_v, f"{args.output_prefix}_frangi_vesselness.nii.gz")

    if args.method in ("oof", "both"):
        print("Running ITK Objectness (OOF-like)...")
        oof_v = run_itk_objectness(vol3d, voxel_spacing, np.array(args.scales_mm, dtype=np.float32),
                                   alpha=args.oof_alpha, beta=args.oof_beta, gamma=args.oof_gamma,
                                   bright=args.oof_bright)
        save_like(ref_img, oof_v, f"{args.output_prefix}_oof_vesselness.nii.gz")

    # ---- Thresholding & masks ----
    if frangi_v is not None:
        if args.thr_mode == "quantile":
            mask, thr = threshold_map(frangi_v, "quantile", args.thr_value_frangi)
        elif args.thr_mode == "otsu":
            mask, thr = threshold_map(frangi_v, "otsu", None)
        else:
            mask, thr = threshold_map(frangi_v, "fixed", args.thr_value_frangi)
        print(f"Frangi threshold = {thr:.6g}, mask fraction={mask.mean():.6f}")
        mask = clean_mask(mask, min_size=args.min_size, radius=args.morph_radius)
        save_like(ref_img, mask, f"{args.output_prefix}_frangi_mask.nii.gz", dtype=np.uint8)

        if args.previews:
            save_preview_pngs(frangi_v, voxel_spacing, f"{args.output_prefix}_frangi",
                              quantiles=args.preview_quantiles, slice_mode=args.preview_mode)

    if oof_v is not None:
        if args.thr_mode == "quantile":
            mask, thr = threshold_map(oof_v, "quantile", args.thr_value_oof)
        elif args.thr_mode == "otsu":
            mask, thr = threshold_map(oof_v, "otsu", None)
        else:
            mask, thr = threshold_map(oof_v, "fixed", args.thr_value_oof)
        print(f"OOF threshold = {thr:.6g}, mask fraction={mask.mean():.6f}")
        mask = clean_mask(mask, min_size=args.min_size, radius=args.morph_radius)
        save_like(ref_img, mask, f"{args.output_prefix}_oof_mask.nii.gz", dtype=np.uint8)

        if args.previews:
            save_preview_pngs(oof_v, voxel_spacing, f"{args.output_prefix}_oof",
                              quantiles=args.preview_quantiles, slice_mode=args.preview_mode)

    print("Done.")

if __name__ == "__main__":
    main()

