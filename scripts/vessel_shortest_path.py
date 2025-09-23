#!/usr/bin/env python3
import argparse, os
import numpy as np
import nibabel as nib
import imageio
from scipy.ndimage import rotate as nd_rotate

from skimage.graph import MCP_Geometric
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_dilation, binary_erosion

import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def load_nii(path, dtype=np.float32):
    img = nib.load(path)
    return img, img.get_fdata(dtype=dtype)

def save_like(ref_img, data, out_path, dtype=np.float32):
    nib.save(nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header), out_path)
    print(f"Saved: {out_path}")

def regrid_roi_to_ref(roi_img, ref_img):
    """Nearest-neighbor resample of a label/ROI image to ref_img grid."""
    if roi_img.shape == ref_img.shape and np.allclose(roi_img.affine, ref_img.affine, atol=1e-5):
        return roi_img
    return resample_from_to(roi_img, (ref_img.shape, ref_img.affine), order=0)

# ---------------- Cost & path ----------------
def robust_norm01(v):
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return np.zeros_like(v, dtype=np.float32)
    lo, hi = np.percentile(vv, (1, 99))
    den = max(hi - lo, 1e-6)
    out = np.clip((v - lo) / den, 0, 1).astype(np.float32)
    return out

def build_cost(vesselness, vessel_mask=None, invert_weight=True, eps=1e-6, min_cost=1e-3):
    v = np.nan_to_num(vesselness.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v = robust_norm01(v)
    if invert_weight:
        c = 1.0 / (eps + v)
    else:
        c = 1.0 - v
    c = np.clip(c, min_cost, None)
    if vessel_mask is not None:
        m = vessel_mask.astype(bool)
        c[~m] = np.inf
    c[~np.isfinite(c)] = np.inf
    return c.astype(np.float32)

def find_path_mcp(cost, starts, goals_mask):
    """
    Multi-source -> multi-goal shortest path on a 3D grid.
    starts: list[(x,y,z)] ; goals_mask: boolean array like cost.
    """
    X, Y, Z = cost.shape
    # keep only in-bounds, finite-cost starts
    starts_in = []
    for s in starts:
        x, y, z = map(int, s)
        if 0 <= x < X and 0 <= y < Y and 0 <= z < Z and np.isfinite(cost[x, y, z]):
            starts_in.append((x, y, z))
    if not starts_in:
        raise RuntimeError("No valid start voxels inside finite-cost region (check ROI/mask alignment).")

    # convert goal mask -> explicit list of endpoints
    goal_pts = [tuple(p) for p in np.argwhere(goals_mask)]
    if not goal_pts:
        raise RuntimeError("Goal mask is empty (after regridding / masking).")

    mcp = MCP_Geometric(cost, fully_connected=True)
    costs, traceback = mcp.find_costs(starts=starts_in, ends=goal_pts)

    # pick best reached goal
    best_goal, best_cost = None, np.inf
    for gx, gy, gz in goal_pts:
        c = costs[gx, gy, gz]
        if np.isfinite(c) and c < best_cost:
            best_cost, best_goal = c, (gx, gy, gz)
    if best_goal is None:
        raise RuntimeError("No path reached any goal (likely blocked by the vessel mask).")

    path = mcp.traceback(best_goal)
    return path, costs

def geometric_path_length_mm(path_xyz, spacing):
    if len(path_xyz) < 2:
        return 0.0
    sx, sy, sz = map(float, spacing[:3])
    diffs = np.diff(np.array(path_xyz, dtype=np.int32), axis=0)
    steps_mm = np.sqrt((diffs[:,0]*sx)**2 + (diffs[:,1]*sy)**2 + (diffs[:,2]*sz)**2)
    return float(np.sum(steps_mm))

# --------------- Rendering (PNG) ---------------
def project_volume(vol, axis="z", mode="mip", slab_bounds=None):
    """
    mode: 'mip' (max), 'mid' (mid-slice), 'slab-mip' (max over slab_bounds)
    axis: 'z'|'y'|'x'
    slab_bounds: (lo, hi) indices along chosen axis (inclusive)
    Returns 2D array.
    """
    ax = {"x":0, "y":1, "z":2}[axis]
    if mode == "mid":
        idx = vol.shape[ax] // 2
        if ax == 0:
            img2d = vol[idx, :, :]
        elif ax == 1:
            img2d = vol[:, idx, :]
        else:
            img2d = vol[:, :, idx]
        return img2d
    elif mode == "mip":
        if ax == 0:
            return np.max(vol, axis=0)
        elif ax == 1:
            return np.max(vol, axis=1)
        else:
            return np.max(vol, axis=2)
    elif mode == "slab-mip":
        if slab_bounds is None:
            return project_volume(vol, axis, "mip")
        lo, hi = slab_bounds
        lo = int(max(0, lo)); hi = int(min(vol.shape[ax]-1, hi))
        if ax == 0:
            return np.max(vol[lo:hi+1, :, :], axis=0)
        elif ax == 1:
            return np.max(vol[:, lo:hi+1, :], axis=1)
        else:
            return np.max(vol[:, :, lo:hi+1], axis=2)
    else:
        raise ValueError("mode must be 'mip', 'mid', or 'slab-mip'")

def outline2d(mask2d, iterations=1):
    if mask2d.dtype != bool:
        mask2d = mask2d.astype(bool)
    if mask2d.sum() == 0:
        return mask2d
    dil = binary_dilation(mask2d, iterations=iterations)
    ero = binary_erosion(mask2d, iterations=iterations)
    return (dil ^ ero)

def render_path_png(vesselness, path_mask, aca_mask, ica_mask, spacing, out_png,
                    axis="z", mode="slab-mip", slab_margin_vox=5, dpi=220,
                    bg_percentiles=(1, 99.7)):
    """
    Save a PNG: background = vesselness projection (realistic), overlays:
    - path (red), ACA outline (green), ICA outline (blue)
    """
    # Determine slab around path along axis
    ax = {"x":0, "y":1, "z":2}[axis]
    path_idx = np.argwhere(path_mask)
    if path_idx.size == 0:
        raise RuntimeError("Empty path_mask; nothing to render.")

    lo = int(np.min(path_idx[:, ax])) - slab_margin_vox
    hi = int(np.max(path_idx[:, ax])) + slab_margin_vox

    # Background vesselness
    bg = project_volume(vesselness, axis=axis, mode=("slab-mip" if mode=="slab-mip" else mode),
                        slab_bounds=(lo, hi) if mode=="slab-mip" else None)
    vfin = bg[np.isfinite(bg)]
    if vfin.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(vfin, bg_percentiles)
        if vmax <= vmin: vmax = vmin + 1e-6

    # Overlays (projected same way)
    path2d = project_volume(path_mask.astype(np.uint8), axis=axis,
                            mode=("slab-mip" if mode=="slab-mip" else mode),
                            slab_bounds=(lo, hi) if mode=="slab-mip" else None) > 0
    aca2d  = project_volume(aca_mask.astype(np.uint8), axis=axis,
                            mode=("slab-mip" if mode=="slab-mip" else mode),
                            slab_bounds=(lo, hi) if mode=="slab-mip" else None) > 0
    ica2d  = project_volume(ica_mask.astype(np.uint8), axis=axis,
                            mode=("slab-mip" if mode=="slab-mip" else mode),
                            slab_bounds=(lo, hi) if mode=="slab-mip" else None) > 0

    # Make thicker path by 2D dilation for visualization
    path2d_thick = binary_dilation(path2d, iterations=1)

    # Outlines for ROIs
    aca_outline = outline2d(aca2d, iterations=1)
    ica_outline = outline2d(ica2d, iterations=1)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(bg.T, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    # path in red
    plt.imshow(path2d_thick.T, origin="lower", alpha=0.8, cmap="Reds")
    # ACA (green) and ICA (blue) outlines
    aca_rgb = np.zeros((*aca_outline.T.shape, 4))
    ica_rgb = np.zeros((*ica_outline.T.shape, 4))
    aca_rgb[aca_outline.T] = (0.0, 1.0, 0.0, 0.9)
    ica_rgb[ica_outline.T] = (0.0, 0.6, 1.0, 0.9)
    plt.imshow(aca_rgb, origin="lower")
    plt.imshow(ica_rgb, origin="lower")

    # Optional simple scalebar (50 mm if feasible)
    sx, sy, _ = spacing
    bar_mm = 50.0
    bar_px = int(round(bar_mm / max(sx, sy)))
    if bar_px > 10 and bg.shape[0] > bar_px + 20:
        y0 = 15; x0 = 15
        plt.plot([x0, x0 + bar_px], [y0, y0], linewidth=6, color="white")
        plt.text(x0, y0 + 8, f"{int(bar_mm)} mm", color="white", fontsize=10, va="bottom")

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

def make_overlay_frame(vesselness2d, path2d, aca2d, ica2d, spacing_xy, scalebar_mm=50.0):
    """
    Compose an RGB uint8 frame:
      - vesselness2d: float 2D (background)
      - path2d: bool 2D (thick red)
      - aca2d / ica2d: bool 2D masks (outlined)
    """
    v = vesselness2d[np.isfinite(vesselness2d)]
    if v.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(v, (1, 99.7))
        if vmax <= vmin: vmax = vmin + 1e-6
    bg = np.clip((vesselness2d - vmin) / (vmax - vmin), 0, 1)
    frame = (np.stack([bg, bg, bg], axis=-1) * 255.0).astype(np.uint8)

    # Thicken path
    path2d = binary_dilation(path2d, iterations=1)

    # Overlays: blend with alpha
    def blend(mask, color, alpha=0.8):
        if not mask.any(): return
        m = mask.astype(bool)
        frame[m] = (alpha * np.array(color, dtype=np.float32) + (1 - alpha) * frame[m]).astype(np.uint8)

    blend(path2d, (255, 0, 0), 0.85)  # red

    # Outlines
    aca_outline = outline2d(aca2d, iterations=1)
    ica_outline = outline2d(ica2d, iterations=1)
    blend(aca_outline, (0, 255, 0), 0.95)  # green
    blend(ica_outline, (0, 170, 255), 0.95)  # blue

    # Simple scale bar (bottom-left)
    sx, sy = spacing_xy
    bar_px = int(round(scalebar_mm / max(sx, sy))) if sx > 0 and sy > 0 else 0
    if bar_px > 10:
        y = frame.shape[0] - 15
        x = 15
        x2 = min(frame.shape[1] - 15, x + bar_px)
        frame[y-2:y+2, x:x2, :] = 255  # white bar

    # NEW: Rotate the final frame 90 degrees counter-clockwise (left)
    frame = np.rot90(frame, k=1)

    return frame

def pad_to_square(img, pad_value=0):
    h, w = img.shape[:2]
    side = int(max(h, w))
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    out = np.full((side, side, img.shape[2]), pad_value, dtype=img.dtype)
    out[pad_y:pad_y+h, pad_x:pad_x+w] = img
    return out

def save_rotation_gif(vesselness, path_mask, aca_mask, ica_mask, spacing, out_gif,
                      axis="z", mode="slab-mip", slab_margin_vox=5,
                      rotate_mode="2d", frames=72, fps=12, pad=True,
                      tilt_deg=0.0, yaw_start_deg=0.0):
    """
    Make a GIF rotating around the chosen axis (default z).
    rotate_mode:
      - "2d": compute one projected frame then rotate in-plane
      - "3d": rotate 3D volumes each frame then re-project (slower)
    tilt_deg (3d): fixed tilt around X (adds parallax); yaw_start_deg: initial yaw.
    """
    ax = {"x":0, "y":1, "z":2}[axis]
    # Determine slab bounds from path
    path_idx = np.argwhere(path_mask)
    if path_idx.size == 0:
        raise RuntimeError("Empty path; nothing to render as GIF.")
    lo = int(np.min(path_idx[:, ax])) - slab_margin_vox
    hi = int(np.max(path_idx[:, ax])) + slab_margin_vox

    def project_all(vess3d, path3d, aca3d, ica3d):
        bg = project_volume(vess3d, axis=axis, mode=("slab-mip" if mode=="slab-mip" else mode),
                            slab_bounds=(lo, hi) if mode=="slab-mip" else None)
        path2d = project_volume(path3d.astype(np.uint8), axis=axis,
                                mode=("slab-mip" if mode=="slab-mip" else mode),
                                slab_bounds=(lo, hi) if mode=="slab-mip" else None) > 0
        aca2d  = project_volume(aca3d.astype(np.uint8), axis=axis,
                                mode=("slab-mip" if mode=="slab-mip" else mode),
                                slab_bounds=(lo, hi) if mode=="slab-mip" else None) > 0
        ica2d  = project_volume(ica3d.astype(np.uint8), axis=axis,
                                mode=("slab-mip" if mode=="slab-mip" else mode),
                                slab_bounds=(lo, hi) if mode=="slab-mip" else None) > 0
        return bg, path2d, aca2d, ica2d

    frames_rgb = []

    if rotate_mode == "2d":
        # Single projection → many 2D rotations
        bg, path2d, aca2d, ica2d = project_all(vesselness, path_mask, aca_mask, ica_mask)
        base = make_overlay_frame(bg, path2d, aca2d, ica2d, spacing_xy=spacing[:2])
        if pad:
            base = pad_to_square(base, pad_value=0)
        step = 360.0 / float(frames)
        for i in range(frames):
            ang = i * step
            rot = nd_rotate(base, angle=ang, reshape=False, order=1, mode='constant', cval=0)
            frames_rgb.append(rot)
    else:
        # True 3D: fixed tilt around X, then yaw around Z each frame (adds parallax)
        step = 360.0 / float(frames)
        tilt = float(tilt_deg)
        yaw0 = float(yaw_start_deg)

        for i in range(frames):
            ang = yaw0 + i * step

            # 1) tilt around X (rotate YZ plane -> axes=(2,1))
            vess_t = nd_rotate(vesselness, angle=tilt, axes=(2,1), reshape=False, order=1, mode='constant', cval=0.0)
            path_t = nd_rotate(path_mask.astype(np.uint8), angle=tilt, axes=(2,1), reshape=False, order=0, mode='constant', cval=0)
            aca_t  = nd_rotate(aca_mask.astype(np.uint8),  angle=tilt, axes=(2,1), reshape=False, order=0, mode='constant', cval=0)
            ica_t  = nd_rotate(ica_mask.astype(np.uint8),  angle=tilt, axes=(2,1), reshape=False, order=0, mode='constant', cval=0)

            # 2) yaw around Z (rotate XY plane -> axes=(1,0))
            vess_r = nd_rotate(vess_t, angle=ang, axes=(1,0), reshape=False, order=1, mode='constant', cval=0.0)
            path_r = nd_rotate(path_t, angle=ang, axes=(1,0), reshape=False, order=0, mode='constant', cval=0)
            aca_r  = nd_rotate(aca_t,  angle=ang, axes=(1,0), reshape=False, order=0, mode='constant', cval=0)
            ica_r  = nd_rotate(ica_t,  angle=ang, axes=(1,0), reshape=False, order=0, mode='constant', cval=0)

            bg, path2d, aca2d, ica2d = project_all(vess_r, path_r>0, aca_r>0, ica_r>0)
            frame = make_overlay_frame(bg, path2d, aca2d, ica2d, spacing_xy=spacing[:2])
            if pad:
                frame = pad_to_square(frame, pad_value=0)
            frames_rgb.append(frame)

    imageio.mimsave(out_gif, frames_rgb, duration=max(1e-3, 1.0/float(fps)), loop=0)
    print(f"Saved GIF: {out_gif}  ({len(frames_rgb)} frames @ {fps} fps)")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Shortest path between ACA and ICA within vessel mask + PNG/GIF rendering.")
    ap.add_argument("--frangi-vesselness", required=True, help="Frangi (or other) vesselness NIfTI (float).")
    ap.add_argument("--frangi-mask", required=False, help="Binary NIfTI of vessels (hard constraint).")
    ap.add_argument("--aca-roi", required=True, help="Binary NIfTI for ACA ROI.")
    ap.add_argument("--ica-roi", required=True, help="Binary NIfTI for ICA ROI.")
    ap.add_argument("--out-prefix", required=True, help="Output prefix (e.g., out/aca_ica).")
    ap.add_argument("--invert-weight", action="store_true", default=True,
                    help="Prefer high vesselness (cost=1/(eps+vesselness)). Use --no-invert-weight to disable.")
    ap.add_argument("--no-invert-weight", dest="invert_weight", action="store_false")
    ap.add_argument("--save-visited-costs", action="store_true", help="Save MCP costs NIfTI.")
    ap.add_argument("--mask-dilate-vox", type=int, default=0, help="Dilate vessel mask by N voxels before search.")
    # Rendering
    ap.add_argument("--render-png", action="store_true", help="Save a PNG rendering of the path.")
    ap.add_argument("--render-mode", choices=["slab-mip","mip","mid"], default="slab-mip")
    ap.add_argument("--render-axis", choices=["z","y","x"], default="z")
    ap.add_argument("--slab-margin-vox", type=int, default=5)
    ap.add_argument("--png-dpi", type=int, default=220)
    ap.add_argument("--render-gif", action="store_true", help="Save an animated GIF rotating around the chosen axis.")
    ap.add_argument("--gif-frames", type=int, default=72, help="Number of frames over 360° (default 72).")
    ap.add_argument("--gif-fps", type=int, default=12, help="GIF playback fps.")
    ap.add_argument("--gif-rotate-mode", choices=["2d","3d"], default="2d",
                    help="2d: rotate projected frame; 3d: rotate volumes then re-project (slower).")
    ap.add_argument("--gif-pad", action="store_true", help="Pad frames to avoid edge clipping during rotation.")
    # NEW: 3D look controls
    ap.add_argument("--gif-tilt-deg", type=float, default=20.0,
                    help="Fixed tilt (degrees) applied before each yaw step in 3D mode.")
    ap.add_argument("--gif-yaw-start-deg", type=float, default=0.0,
                    help="Starting yaw angle (degrees) for the 3D spin.")
    # NEW: path thickness (render-only)
    ap.add_argument("--path-thicken-vox", type=int, default=0,
                    help="Dilate the 3D path by N voxels for rendering only (not for routing).")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    # Load vesselness (reference space)
    ref_img, frangi_v = load_nii(args.frangi_vesselness, dtype=np.float32)
    spacing = ref_img.header.get_zooms()[:3]

    # Load/dilate hard mask if provided
    if args.frangi_mask:
        _, frangi_m = load_nii(args.frangi_mask, dtype=np.float32)
        frangi_m = frangi_m > 0
        if args.mask_dilate_vox > 0:
            frangi_m = binary_dilation(frangi_m, iterations=int(args.mask_dilate_vox))
    else:
        frangi_m = None

    # Load and regrid ROIs to vesselness space
    aca_img = regrid_roi_to_ref(nib.load(args.aca_roi), ref_img)
    ica_img = regrid_roi_to_ref(nib.load(args.ica_roi), ref_img)
    aca = aca_img.get_fdata(dtype=np.float32) > 0.5
    ica = ica_img.get_fdata(dtype=np.float32) > 0.5

    # Build cost and restrict ROIs to finite-cost region
    cost = build_cost(frangi_v, vessel_mask=frangi_m, invert_weight=args.invert_weight)
    finite = np.isfinite(cost)
    aca &= finite
    ica &= finite

    print("ACA voxels inside mask:", int(aca.sum()))
    print("ICA voxels inside mask:", int(ica.sum()))

    if aca.sum() == 0 or ica.sum() == 0:
        raise RuntimeError("ACA or ICA has no voxels inside finite-cost region. Relax/dilate the mask or omit --frangi-mask.")

    # Find path
    starts = [tuple(x) for x in np.argwhere(aca)]
    goals_map = ica
    path_xyz, costs = find_path_mcp(cost, starts, goals_map)

    # Path mask and length
    path_mask = np.zeros_like(frangi_v, dtype=np.uint8)
    for (x, y, z) in path_xyz:
        path_mask[x, y, z] = 1
    save_like(ref_img, path_mask, f"{args.out_prefix}_path_mask.nii.gz", dtype=np.uint8)

    if args.save_visited_costs and isinstance(costs, np.ndarray):
        save_like(ref_img, costs, f"{args.out_prefix}_mcp_costs.nii.gz", dtype=np.float32)

    length_mm = geometric_path_length_mm(path_xyz, spacing)
    print(f"Geometric path length (within constraints): {length_mm:.3f} mm")

    # For rendering only: optionally thicken the 3D path
    path_for_render = path_mask.astype(bool)
    if args.path_thicken_vox > 0:
        path_for_render = binary_dilation(path_for_render, iterations=int(args.path_thicken_vox))

        # Render PNG if requested
    if args.render_png:
        # This first call creates the PNG based on your command-line argument (e.g., 'y')
        render_path_png(
            vesselness=frangi_v,
            path_mask=path_for_render,
            aca_mask=aca.astype(bool),
            ica_mask=ica.astype(bool),
            spacing=spacing,
            out_png=f"{args.out_prefix}_render_{args.render_mode}_{args.render_axis}.png",
            axis=args.render_axis,
            mode=args.render_mode,
            slab_margin_vox=args.slab_margin_vox,
            dpi=args.png_dpi
        )

        # NEW: Add a second call to always create an X-axis view as well
        print("--- Generating additional X-axis PNG ---")
        render_path_png(
            vesselness=frangi_v,
            path_mask=path_for_render,
            aca_mask=aca.astype(bool),
            ica_mask=ica.astype(bool),
            spacing=spacing,
            out_png=f"{args.out_prefix}_render_{args.render_mode}_x.png", # Force filename for x-axis
            axis="x",                                                     # Force render axis to 'x'
            mode=args.render_mode,
            slab_margin_vox=args.slab_margin_vox,
            dpi=args.png_dpi
        )

    # Render GIF if requested
    if args.render_png and args.render_gif:
        save_rotation_gif(
            vesselness=frangi_v,
            path_mask=path_for_render,
            aca_mask=aca.astype(bool),
            ica_mask=ica.astype(bool),
            spacing=spacing,
            out_gif=f"{args.out_prefix}_spin_{args.render_mode}_{args.render_axis}.gif",
            axis=args.render_axis,
            mode=args.render_mode,
            slab_margin_vox=args.slab_margin_vox,
            rotate_mode=args.gif_rotate_mode,
            frames=args.gif_frames,
            fps=args.gif_fps,
            pad=args.gif_pad,
            tilt_deg=args.gif_tilt_deg,
            yaw_start_deg=args.gif_yaw_start_deg
        )

if __name__ == "__main__":
    main()
