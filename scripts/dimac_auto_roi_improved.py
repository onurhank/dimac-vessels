#!/usr/bin/env python3
"""
Automatic DIMAC ROI extraction via pulse-power ratio (PPR) + k-means (improved).

Outputs:
  <out>_ppr.nii.gz                  -- PPR map (percent, 0..100)
  <out>_ppr_thresh.nii.gz           -- PPR >= threshold (binary)
  <out>_kmeans_k<K>.nii.gz          -- cluster labels in gate (1..K), 0 elsewhere
  <out>_roi.nii.gz                  -- selected arterial ROI (binary, largest 3D CC, small objects removed)
  <out>_roi_timeseries.csv          -- mean DIMAC time series in ROI
  <out>_summary.txt                 -- detailed report (TR, band, threshold, k, sizes, QA metrics)
  <out>_preview.png                 -- middle-slice preview overlay

Key extras in this version:
- --center-frac: restrict search to a centered XY square (anatomical prior for ACA).
- --min-voxels: guaranteed ROI by seeding at max-PPR and dilating until >= N voxels.
- Robust FFT-based PPR + optional Welch PSD refinement inside the gate.
- PCA(+randomized) + KMeans clustering; pick best cluster by PPR (with coherence bonus in auto-band).
- Conservative morphology: largest 3D CC + remove tiny islands + light open/close.

Dependencies: nibabel, numpy, scipy, scikit-learn, scikit-image, matplotlib
"""

import argparse
import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from numpy.fft import rfft, rfftfreq
from scipy.signal import welch, detrend, get_window, find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from skimage.measure import label as cc_label
from skimage.morphology import ball, binary_opening, binary_closing, remove_small_objects, binary_dilation


# ----------------- I/O helpers -----------------

def load_nii(path):
    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    return nii, data

def save_nii_like(ref_nii, data, out_path, dtype=None):
    if dtype is not None:
        data = data.astype(dtype)
    out = nib.Nifti1Image(data, ref_nii.affine, ref_nii.header)
    nib.save(out, out_path)
    # print(f"Saved: {out_path}")


# ----------------- Utilities -----------------

def _nanmean_axis1(x):
    return np.nanmean(x, axis=1, keepdims=True)

def _zscore_safe(x):
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd


# ----------------- Band estimation (auto) -----------------

def estimate_cardio_band_from_global(dimac_4d, tr_sec, f_lo_hz=0.5, f_hi_hz=3.0):
    """
    Estimate cardiac peak frequency and an adaptive half-width from the global signal PSD.
    Returns (f0_hz, df_hz).
    """
    T = dimac_4d.shape[3]
    fs = 1.0 / tr_sec
    freqs = rfftfreq(T, d=tr_sec)
    nyq = freqs[-1]

    # global mean signal
    g = np.nanmean(dimac_4d, axis=(0, 1, 2))
    g = detrend(np.nan_to_num(g, nan=0.0), type="linear")

    # FFT power
    spec = rfft(g)
    pow_g = (spec.real**2 + spec.imag**2)

    # search window, avoid DC
    lo = max(f_lo_hz, freqs[1] if freqs.size > 1 else 0.0)
    hi = min(f_hi_hz, nyq)
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        raise ValueError(f"Auto-band search empty for TR={tr_sec:.4f}s (nyquist={nyq:.2f}Hz).")

    f_m = freqs[mask]
    P_m = pow_g[mask]

    prom_thr = np.percentile(P_m, 75.0)
    peaks, props = find_peaks(P_m, prominence=max(prom_thr, 1e-12))
    if peaks.size == 0:
        peak_idx = int(np.argmax(P_m))
        f0 = float(f_m[peak_idx])
    else:
        best = int(np.argmax(props["prominences"]))
        peak_idx = peaks[best]
        f0 = float(f_m[peak_idx])

    # adaptive half-width
    df = max(0.15, 0.20 * f0)  # ~9–18 bpm typical
    return f0, df


# ----------------- PPR computation -----------------

def compute_ppr_fft(dimac_4d, tr_sec, bpm_lo=40.0, bpm_hi=120.0):
    """
    Fast PPR via rFFT:
    PPR% = 100 * (power in [bpm_lo..bpm_hi]) / (total power excluding DC)
    """
    if dimac_4d.ndim != 4:
        raise ValueError("DIMAC NIfTI must be 4D (X,Y,Z,T).")
    X, Y, Z, T = dimac_4d.shape
    if tr_sec is None or tr_sec <= 0:
        raise ValueError("TR (sec) must be > 0.")

    freqs = rfftfreq(T, d=tr_sec)
    # clip band to realizable range and avoid DC bin
    lo_hz = max(bpm_lo/60.0, freqs[1] if freqs.size > 1 else 0.0)
    hi_hz = min(bpm_hi/60.0, freqs[-1])
    if hi_hz <= lo_hz:
        raise ValueError(
            f"Requested band {bpm_lo:.1f}-{bpm_hi:.1f} bpm not representable at TR={tr_sec:.4f}s "
            f"(Nyquist {60.0*freqs[-1]:.1f} bpm)."
        )

    # reshape to (V, T)
    V = X * Y * Z
    series = dimac_4d.reshape((V, T))
    # NaN-safe demean
    series = series - _nanmean_axis1(series)

    # rFFT along time
    spec = rfft(np.nan_to_num(series, nan=0.0), axis=1)
    power = (spec.real**2 + spec.imag**2)
    # masks
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    non_dc = freqs > 0
    band_power = np.sum(power[:, band], axis=1)
    total_power = np.sum(power[:, non_dc], axis=1) + 1e-12
    ppr = 100.0 * band_power / total_power  # percent
    ppr_3d = ppr.reshape((X, Y, Z))
    return ppr_3d, (lo_hz*60.0, hi_hz*60.0, 60.0*freqs[-1])


def recompute_ppr_welch_in_gate(dimac_4d, gate_mask, tr_sec, lo_hz, hi_hz):
    """
    More robust PPR via Welch PSD, computed only for voxels inside the gate.
    Returns a volume with Welch-based PPR inside the gate and zeros elsewhere.
    """
    X, Y, Z, T = dimac_4d.shape
    fs = 1.0 / tr_sec

    out = np.zeros((X, Y, Z), dtype=np.float32)
    gate_idx = np.flatnonzero(gate_mask.ravel())
    if gate_idx.size == 0:
        return out

    ts = dimac_4d.reshape((-1, T))[gate_idx, :]
    # NaN-safe detrend/prepare
    ts = np.where(np.isfinite(ts), ts, 0.0)
    ts = detrend(ts, axis=1, type='linear')

    # Welch parameters
    nperseg = min(256, T)
    noverlap = int(0.5 * nperseg)
    win = get_window('hann', nperseg)

    band_power = np.zeros(gate_idx.size, dtype=np.float64)
    total_power = np.zeros(gate_idx.size, dtype=np.float64)

    for i in range(gate_idx.size):
        f, Pxx = welch(ts[i, :], fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, detrend=False)
        non_dc = f > 0
        m_band = (f >= lo_hz) & (f <= hi_hz)
        band_power[i] = np.trapz(Pxx[m_band], f[m_band])
        total_power[i] = np.trapz(Pxx[non_dc], f[non_dc]) + 1e-18

    ppr_gate = 100.0 * (band_power / total_power)
    out.reshape(-1)[gate_idx] = ppr_gate.astype(np.float32)
    return out


# ----------------- Clustering inside gate -----------------

def kmeans_in_gate(dimac_4d, gate_mask, k=5, use_pca=True, pca_var=0.9, random_state=0):
    """
    Run PCA(+zscore) -> KMeans over time series inside gate.
    Returns: label_vol (1..k in gate, 0 elsewhere), Xk (features), labels (0..k-1), gate_idx
    """
    X, Y, Z, T = dimac_4d.shape
    gate_idx = np.flatnonzero(gate_mask.ravel())
    if gate_idx.size == 0:
        raise ValueError("Gate mask is empty after thresholding/intersection.")

    ts = dimac_4d.reshape((-1, T))[gate_idx, :]
    # NaN-safe demean and z-score per voxel
    ts = ts - np.nanmean(ts, axis=1, keepdims=True)
    ts = np.where(np.isfinite(ts), ts, 0.0)
    std = np.nanstd(ts, axis=1, keepdims=True) + 1e-12
    ts = ts / std

    if use_pca:
        pca = PCA(n_components=min(T, 100), svd_solver="randomized", iterated_power=3,
                  random_state=random_state)
        Xp = pca.fit_transform(ts)
        if pca_var < 1.0:
            csum = np.cumsum(pca.explained_variance_ratio_)
            ncomp = int(np.searchsorted(csum, pca_var) + 1)
            Xk = Xp[:, :max(2, ncomp)]
        else:
            Xk = Xp
    else:
        Xk = ts

    km = KMeans(n_clusters=k, n_init='auto' if hasattr(KMeans(), 'n_init') else 10, random_state=random_state)
    labels = km.fit_predict(Xk)  # 0..k-1

    label_vol = np.zeros((X, Y, Z), dtype=np.int16)
    label_vol.reshape(-1)[gate_idx] = (labels.astype(np.int16) + 1)  # 1..k
    return label_vol, Xk, labels, gate_idx


# ----------------- ROI cleanup -----------------

def refine_roi_largest_cc(roi_bool, min_cc_size=0, apply_open_close=True):
    """
    Keep the largest 3D connected component; optionally remove tiny islands and lightly open/close.
    """
    lab_cc, ncc = cc_label(roi_bool, connectivity=3, return_num=True)
    if ncc > 1:
        sizes = np.bincount(lab_cc.ravel())
        sizes[0] = 0
        keep_id = int(np.argmax(sizes))
        roi_bool = (lab_cc == keep_id)

    if min_cc_size > 0:
        roi_bool = remove_small_objects(roi_bool, min_size=int(min_cc_size))

    if apply_open_close:
        roi_bool = binary_opening(roi_bool, ball(1))
        roi_bool = binary_closing(roi_bool, ball(1))
    return roi_bool


# ----------------- Preview -----------------

def preview_png(bg_vol, roi_mask, out_png):
    """Middle-slice preview with ROI overlay."""
    Z = bg_vol.shape[2]
    z_counts = [np.count_nonzero(roi_mask[:, :, z]) for z in range(Z)]
    z = int(np.argmax(z_counts)) if (len(z_counts) and max(z_counts) > 0) else Z // 2
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(bg_vol[:, :, z].T, origin='lower', cmap='gray')
    overlay = np.ma.masked_where(~roi_mask[:, :, z].T, roi_mask[:, :, z].T)
    plt.imshow(overlay, origin='lower', alpha=0.5)
    plt.axis('off')
    plt.title(f'ROI overlay (slice z={z})')
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Auto DIMAC ROI via PPR + k-means (improved, no TOF).")
    ap.add_argument("--dimac", required=True, help="4D DIMAC NIfTI (X,Y,Z,T).")
    ap.add_argument("--out", required=True, help="Output prefix (no extension).")
    ap.add_argument("--vessel-mask", help="Optional binary vessel mask NIfTI in SAME grid; intersect with PPR gate.")
    ap.add_argument("--bpm-low", type=float, default=40.0, help="Lower BPM for band (default 40).")
    ap.add_argument("--bpm-high", type=float, default=120.0, help="Upper BPM for band (default 120).")
    ap.add_argument("--ppr-thr", type=float, default=10.0, help="PPR threshold in PERCENT (default 10.0).")
    ap.add_argument("--k", type=int, default=5, help="K-means clusters (default 5).")
    ap.add_argument("--no-pca", action="store_true", help="Disable PCA before k-means (slower).")
    ap.add_argument("--pca-var", type=float, default=0.90, help="Target explained variance if PCA enabled (default 0.90).")
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--auto-band", action="store_true", help="Auto-detect cardiac peak and adaptive band from global PSD.")
    ap.add_argument("--welch-in-gate", action="store_true",
                    help="Recompute PPR with Welch PSD for voxels inside the gate (heavier, more robust).")
    ap.add_argument("--min-cc-size", type=int, default=0, help="Remove connected components smaller than this voxel count (after largest CC).")

    # NEW: Robustness arguments to guarantee an ROI is created
    ap.add_argument("--center-frac", type=float, default=None,
                    help="If set (0<frac<=1), intersect gate with a centered XY square (frac*FOV).")
    ap.add_argument("--min-voxels", type=int, default=0,
                    help="If gate ends empty/small, seed at max-PPR and dilate until >= this many voxels.")

    args = ap.parse_args()

    # load DIMAC
    dimac_nii, X = load_nii(args.dimac)
    if X.ndim != 4:
        print("Error: DIMAC must be 4D (X,Y,Z,T).", file=sys.stderr)
        sys.exit(1)
    Xsz = X.shape
    zooms = dimac_nii.header.get_zooms()
    if len(zooms) < 4 or zooms[3] <= 0:
        print("Error: could not read TR from header (pixdim4).", file=sys.stderr)
        sys.exit(1)
    tr = float(zooms[3])
    fs = 1.0 / tr
    T = X.shape[3]

    # decide band
    if args.auto_band:
        f0_hz, df_hz = estimate_cardio_band_from_global(X, tr)
        lo_hz = max(f0_hz - df_hz, (1.0/T)/tr)  # avoid DC
        hi_hz = min(f0_hz + df_hz, 0.5 * fs)
        bpm_lo_eff, bpm_hi_eff = lo_hz * 60.0, hi_hz * 60.0
    else:
        lo_hz = max(args.bpm_low/60.0, (1.0/T)/tr if T > 1 else 0.0)
        hi_hz = min(args.bpm_high/60.0, 0.5 * fs)
        bpm_lo_eff, bpm_hi_eff = lo_hz * 60.0, hi_hz * 60.0

    # compute PPR (fast FFT)
    ppr, (_, _, _) = compute_ppr_fft(X, tr, bpm_lo=bpm_lo_eff, bpm_hi=bpm_hi_eff)
    save_nii_like(dimac_nii, ppr, f"{args.out}_ppr.nii.gz", dtype=np.float32)

    # gate by PPR >= threshold%
    gate = ppr >= float(args.ppr_thr)
    vm = None  # optional vessel mask

    # optional: intersect with external vessel mask
    if args.vessel_mask:
        vm_nii, vm = load_nii(args.vessel_mask)
        if vm.shape != Xsz[:3]:
            print("Error: vessel mask shape differs from DIMAC 3D shape. Resample externally first.", file=sys.stderr)
            sys.exit(1)
        gate = gate & (vm > 0)

    # NEW: restrict to a centered XY square if requested (anatomical prior for ACA)
    if args.center_frac is not None:
        cf = float(args.center_frac)
        if not (0.0 < cf <= 1.0):
            print("Error: --center-frac must be in (0,1].", file=sys.stderr); sys.exit(1)
        Xd, Yd, Zd = X.shape[:3]
        cx, cy = Xd // 2, Yd // 2
        hx = max(1, int(round(0.5 * cf * Xd)))
        hy = max(1, int(round(0.5 * cf * Yd)))
        xmin, xmax = max(0, cx - hx), min(Xd, cx + hx)
        ymin, ymax = max(0, cy - hy), min(Yd, cy + hy)
        center_mask = np.zeros((Xd, Yd, Zd), dtype=bool)
        center_mask[xmin:xmax, ymin:ymax, :] = True
        gate = gate & center_mask

    # clean small speckles a bit
    gate = binary_opening(gate, ball(1))
    save_nii_like(dimac_nii, gate.astype(np.uint8), f"{args.out}_ppr_thresh.nii.gz", dtype=np.uint8)

    # NEW: force-build a minimal gate if it's too small or empty
    if args.min_voxels > 0 and np.count_nonzero(gate) < int(args.min_voxels):
        print(f"[INFO] Gate has {np.count_nonzero(gate)} voxels < --min-voxels={args.min_voxels}. Forcing ROI seed+dilate.")
        Xd, Yd, Zd = X.shape[:3]

        # seed search domain
        if args.center_frac is not None:
            cf = float(args.center_frac)
            cx, cy = Xd // 2, Yd // 2
            hx = max(1, int(round(0.5 * cf * Xd)))
            hy = max(1, int(round(0.5 * cf * Yd)))
            xmin, xmax = max(0, cx - hx), min(Xd, cx + hx)
            ymin, ymax = max(0, cy - hy), min(Yd, cy + hy)
            seed_domain = np.zeros((Xd, Yd, Zd), dtype=bool)
            seed_domain[xmin:xmax, ymin:ymax, :] = True
        elif vm is not None:
            seed_domain = (vm > 0)
        else:
            seed_domain = np.ones((Xd, Yd, Zd), dtype=bool)

        # find best seed (max PPR) in domain
        ppr_masked = np.where(seed_domain, ppr, -np.inf)
        if not np.any(np.isfinite(ppr_masked)):
            ppr_masked = ppr
        seed_idx = np.unravel_index(int(np.nanargmax(ppr_masked)), ppr.shape)

        # grow by 1-voxel dilations until min_voxels
        roi_force = np.zeros_like(gate, dtype=bool)
        roi_force[seed_idx] = True
        iters, max_iters = 0, 50
        while np.count_nonzero(roi_force) < int(args.min_voxels) and iters < max_iters:
            roi_force = binary_dilation(roi_force, footprint=ball(1))
            iters += 1

        gate = roi_force  # replace gate
        save_nii_like(dimac_nii, gate.astype(np.uint8), f"{args.out}_forced_gate.nii.gz", dtype=np.uint8)

    # optional: recompute PPR inside gate with Welch (heavier, robust)
    if args.welch_in_gate and gate.any():
        # reuse exact lo/hi Hz that were computed above (auto-band or manual)
        ppr_welch_gate = recompute_ppr_welch_in_gate(X, gate, tr, lo_hz, hi_hz)
        ppr = np.where(gate, ppr_welch_gate, ppr)
        save_nii_like(dimac_nii, ppr.astype(np.float32), f"{args.out}_ppr.nii.gz", dtype=np.float32)

    # final safeguard before clustering
    if np.count_nonzero(gate) == 0:
        print("[WARN] Gate is empty after all attempts. Creating single-voxel ROI at global max PPR.")
        seed_idx = np.unravel_index(int(np.nanargmax(ppr)), ppr.shape)
        roi_mask = np.zeros_like(gate, dtype=bool)
        roi_mask[seed_idx] = True
        save_nii_like(dimac_nii, roi_mask.astype(np.uint8), f"{args.out}_roi.nii.gz", dtype=np.uint8)
        ts = X[seed_idx[0], seed_idx[1], seed_idx[2], :]
        np.savetxt(f"{args.out}_roi_timeseries.csv", ts, delimiter=",", fmt="%.6f")
        with open(f"{args.out}_summary.txt", "w") as f:
            f.write("FORCED single-voxel ROI due to empty gate.\n")
        bg = np.nanmean(X, axis=3)
        preview_png(bg, roi_mask, f"{args.out}_preview.png")
        print("Done (forced single-voxel ROI).")
        return

    # clustering
    labels_vol, Xk, labels, gate_idx = kmeans_in_gate(
        X, gate, k=args.k, use_pca=(not args.no_pca),
        pca_var=args.pca_var, random_state=args.random_state
    )
    save_nii_like(dimac_nii, labels_vol, f"{args.out}_kmeans_k{args.k}.nii.gz", dtype=np.int16)

    # pick best cluster by mean PPR (bonus: coherence if auto-band)
    cluster_report = []
    kmax = int(labels_vol.max())
    best_label, best_score = None, -np.inf
    for lab in range(1, kmax + 1):
        M = (labels_vol == lab)
        if not np.any(M):
            cluster_report.append((lab, 0, 0.0))
            continue
        mean_ppr = float(np.nanmean(ppr[M]))
        score = mean_ppr
        if args.auto_band:
            ppr_vals = ppr[M]
            sharp = float(1.0 / (np.nanstd(ppr_vals) + 1e-6))
            score = 0.8 * mean_ppr + 0.2 * (10.0 * sharp)
        cluster_report.append((lab, int(M.sum()), mean_ppr))
        if score > best_score:
            best_score = score
            best_label = lab

    roi_mask = (labels_vol == best_label)
    roi_mask = refine_roi_largest_cc(roi_mask, min_cc_size=args.min_cc_size, apply_open_close=True)
    save_nii_like(dimac_nii, roi_mask.astype(np.uint8), f"{args.out}_roi.nii.gz", dtype=np.uint8)

    # export mean time series
    T = X.shape[3]
    if roi_mask.any():
        ts_mean = X[roi_mask, :].mean(axis=0)
    else:
        ts_mean = np.zeros(T, dtype=np.float32)
    np.savetxt(f"{args.out}_roi_timeseries.csv", ts_mean, delimiter=",", fmt="%.6f")

    # summary
    vx = np.array(dimac_nii.header.get_zooms()[:3], dtype=np.float64)
    roi_vox = int(roi_mask.sum())
    roi_ml = float(roi_vox * np.prod(vx) / 1000.0)
    if roi_vox > 0:
        ijk = np.argwhere(roi_mask)
        com_ijk = ijk.mean(axis=0)
        com_xyz = (dimac_nii.affine @ np.array([com_ijk[0], com_ijk[1], com_ijk[2], 1.0]))[:3]
    else:
        com_ijk = np.array([np.nan, np.nan, np.nan])
        com_xyz = np.array([np.nan, np.nan, np.nan])

    with open(f"{args.out}_summary.txt", "w") as f:
        f.write(f"DIMAC shape: {X.shape}\n")
        f.write(f"TR (s): {tr:.6f}\n")
        f.write(f"Band used (BPM): {bpm_lo_eff:.1f}..{bpm_hi_eff:.1f}\n")
        f.write(f"PPR threshold (%): {args.ppr_thr}\n")
        f.write(f"K-means K: {args.k}\n")
        f.write(f"Best cluster label: {best_label}\n")
        for lab, count, mean_ppr in cluster_report:
            f.write(f"  label {lab}: voxels={count}, mean_PPR%={mean_ppr:.3f}\n")
        try:
            sil = float(silhouette_score(Xk, labels, metric='euclidean')) if len(np.unique(labels)) > 1 else np.nan
        except Exception:
            sil = np.nan
        f.write(f"Silhouette (PCA-space): {sil:.3f}\n")
        f.write(f"ROI voxels: {roi_vox}\n")
        f.write(f"ROI volume (mL): {roi_ml:.3f}\n")
        f.write(f"ROI CoM (ijk): {com_ijk.tolist()}\n")
        f.write(f"ROI CoM (world, mm): {com_xyz.tolist()}\n")
        f.write(f"Auto-band: {bool(args.auto_band)} | Welch-in-gate: {bool(args.welch_in_gate)}\n")
        if args.center_frac is not None:
            f.write(f"center-frac: {args.center_frac}\n")
        if args.min_voxels > 0:
            f.write(f"min-voxels: {args.min_voxels}\n")

    # preview
    bg = np.nanmean(X, axis=3)
    preview_png(bg, roi_mask, f"{args.out}_preview.png")

    print(f"Done.\nROI saved to: {args.out}_roi.nii.gz")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
