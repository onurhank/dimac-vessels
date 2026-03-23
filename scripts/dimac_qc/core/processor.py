import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import binary_dilation, rotate
from scipy.signal import find_peaks, savgol_filter
from skimage.segmentation import watershed
from nibabel.processing import resample_from_to

from .config import QCConfig, ClusterRecord

class DimacProcessor:
    def __init__(self, dimac_fname, vesselness_fname=None, auto_roi_fname=None, config=None):
        self.config = config or QCConfig()
        self.dimac_fname = Path(dimac_fname)
        self.vesselness_fname = Path(vesselness_fname) if vesselness_fname else None
        
        self.dimac_nii = nib.load(str(self.dimac_fname))
        self.dimac_data = np.asarray(self.dimac_nii.dataobj, dtype=np.float32)
        
        zooms = self.dimac_nii.header.get_zooms()
        self.tr = float(zooms[3]) if len(zooms) > 3 and zooms[3] > 0 else 1.0
        self.mean_img = np.nanmean(self.dimac_data, axis=3)
        self.dimac_shape_3d = self.mean_img.shape
        self.dimac_flat = self.dimac_data.reshape(-1, self.dimac_data.shape[3])

        self.has_vess = False
        if self.vesselness_fname and self.vesselness_fname.exists():
            self.vess_nii = nib.load(str(self.vesselness_fname))
            self.vess_data = np.asarray(self.vess_nii.dataobj, dtype=np.float32)
            self.has_vess = True

        self.cluster_records =[]
        
        # Rendering caches
        self.global_base_frames =[]
        self.global_slab_masks =[]
        self.global_frame_coords = np.empty((0, 0, 2))
        self.global_cand_colors_rgb =[] 
        self.slab_vess = None
        self.local_preview_cache = {}

    def compute_pipeline(self, percentile, mode="peak", progress_callback=None):
        if progress_callback: progress_callback(5, "Initializing...")
        self.global_base_frames =[] 
        self.local_preview_cache.clear()
        
        thresh = np.percentile(self.mean_img[np.isfinite(self.mean_img)], percentile)
        masked_img = np.where(self.mean_img > thresh, self.mean_img, 0)
        
        if progress_callback: progress_callback(10, "Detecting candidates...")
        if mode == "peak":
            local_max_mask = ((masked_img == ndimage.maximum_filter(masked_img, size=3)) & (masked_img > 0))
            cluster_map, _ = ndimage.label(local_max_mask)
            slices = ndimage.find_objects(cluster_map)
            struct = np.ones(self.config.dilation_shape, dtype=np.uint8)
        else:
            local_max_mask = ((masked_img == ndimage.maximum_filter(masked_img, size=3)) & (masked_img > 0))
            markers, _ = ndimage.label(local_max_mask)
            inv_img = -self.mean_img
            inv_img[masked_img == 0] = 0 
            cluster_map = watershed(inv_img, markers, mask=(masked_img > 0))
            slices = ndimage.find_objects(cluster_map)
            struct = None 

        self.cluster_records =[]
        for cluster_id, slc in enumerate(slices, start=1):
            if slc is None: continue
            local_voxels = np.argwhere(cluster_map[slc] == cluster_id)
            if local_voxels.size == 0: continue
            
            local_peak = local_voxels[np.argmax(self.mean_img[slc][tuple(local_voxels.T)])]
            peak_coord = np.array([local_peak[i] + slc[i].start for i in range(3)], dtype=int)

            raw_mask = np.zeros_like(cluster_map, dtype=bool)
            raw_mask[slc] = (cluster_map[slc] == cluster_id)
            if struct is not None:
                raw_mask = ndimage.binary_dilation(raw_mask, structure=struct, iterations=self.config.dilation_iterations)

            tc = np.mean(self.dimac_flat[raw_mask.ravel()], axis=0)
            ppr = self.calc_ppr(tc)
            self.cluster_records.append(ClusterRecord(cluster_id=int(cluster_id), peak_coord_dimac=peak_coord, raw_mask_dimac=raw_mask, tc=tc, ppr=ppr))

        self._map_to_vesselness()
        
        total = len(self.cluster_records)
        for i, rec in enumerate(self.cluster_records):
            if progress_callback: progress_callback(20 + int((i/total)*50), f"Gradient QC {i}/{total}...")
            self._apply_gradient_search(rec)
            
        self._compute_hybrid_scores()

        if self.has_vess:
            if progress_callback: progress_callback(75, "Precomputing Slab...")
            self._precompute_slab()
            if progress_callback: progress_callback(85, "Rendering 3D Previews (Rotating)...")
            self._build_global_preview_frames()

        if progress_callback: progress_callback(100, "Done.")

    def calc_ppr(self, tc: np.ndarray) -> float:
        if tc.size < 2: return 0.0
        tc_detrend = tc - np.mean(tc)
        fft_mag = np.abs(np.fft.rfft(tc_detrend)) ** 2
        freqs = np.fft.rfftfreq(len(tc), d=self.tr)
        band = (freqs >= self.config.min_heart_rate_hz) & (freqs <= self.config.max_heart_rate_hz)
        total_power = np.sum(fft_mag)
        return float(np.sum(fft_mag[band]) / total_power) if total_power > 0 else 0.0

    def _map_to_vesselness(self):
        if not self.has_vess: return
        valid_records =[]
        for rec in self.cluster_records:
            phys = nib.affines.apply_affine(self.dimac_nii.affine, rec.peak_coord_dimac.astype(float))
            vess_vox = nib.affines.apply_affine(np.linalg.inv(self.vess_nii.affine), phys)
            vess_vox = np.round(vess_vox).astype(int)
            if all(0 <= vess_vox[i] < self.vess_data.shape[i] for i in range(3)):
                rec.peak_coord_vess = vess_vox
                rec.vesselness_val = float(self.vess_data[tuple(vess_vox)])
                valid_records.append(rec)
        self.cluster_records = valid_records

    def _apply_gradient_search(self, rec: ClusterRecord):
        tc_2d = self.dimac_flat[rec.raw_mask_dimac.ravel()]
        rec.tc_2d = tc_2d
        mean_tc = np.mean(tc_2d, axis=0)
        
        window = max(3, int(3.0 / self.tr))
        if window % 2 == 0: window += 1
        pad_width = window // 2
        padded = np.pad(mean_tc, pad_width, mode='edge')
        x_lp = np.convolve(padded, np.ones(window)/window, mode='valid')
        
        x_hp = mean_tc - x_lp
        if len(x_hp) >= 21: x_hp = savgol_filter(x_hp, window_length=21, polyorder=5)

        thresh = np.nanmean(x_hp) + 0.75 * np.nanstd(x_hp)
        min_dist = max(1, int(0.5 / self.tr))
        peaks, _ = find_peaks(x_hp, height=thresh, distance=min_dist)
        rec.maxind = peaks
        
        Nvox = tc_2d.shape[0]
        if len(peaks) < 3:
            rec.custom_voxel_mask = np.ones(Nvox, dtype=bool)
            rec.grad_mask = rec.raw_mask_dimac.copy()
            rec.tc_grad, rec.ppr_grad = rec.tc, rec.ppr
            return

        minind = []; gradfit = []; gradallfit =[]
        for b in range(1, len(peaks)):
            prev_peak = peaks[b-1]
            curr_peak = peaks[b]
            beat_len = curr_peak - prev_peak

            search_mg_start = max(prev_peak, curr_peak - int(np.ceil(beat_len / 4.0)))
            trough = search_mg_start + np.argmin(mean_tc[search_mg_start:curr_peak]) if curr_peak > search_mg_start else search_mg_start
            minind.append(trough)

            search_grad_start = max(prev_peak, curr_peak - int(np.ceil(beat_len / 2.0)))
            N_loc = trough - search_grad_start + 1
            if N_loc > 1:
                x_loc = np.arange(N_loc)
                x_c = x_loc - np.mean(x_loc)
                ss_xx = np.sum(x_c**2)
                m_loc = np.sum(x_c * tc_2d[:, search_grad_start:trough+1], axis=1) / ss_xx if ss_xx > 0 else np.zeros(Nvox)
            else:
                m_loc = np.zeros(Nvox)
            gradfit.append(m_loc)

            N_all = trough - prev_peak + 1
            if N_all > 1:
                x_all = np.arange(N_all)
                x_c = x_all - np.mean(x_all)
                ss_xx = np.sum(x_c**2)
                m_all = np.sum(x_c * tc_2d[:, prev_peak:trough+1], axis=1) / ss_xx if ss_xx > 0 else np.zeros(Nvox)
            else:
                m_all = np.zeros(Nvox)
            gradallfit.append(m_all)

        rec.minind = np.array(minind)
        rec.gradfit = np.array(gradfit)
        rec.gradallfit = np.array(gradallfit)

        with np.errstate(divide='ignore', invalid='ignore'):
            gradratio = np.where(rec.gradallfit != 0, rec.gradfit / rec.gradallfit, 0)
        rec.gradratio = gradratio

        pass_beat = (gradratio > 0.4) & (gradratio < 2.0) & (rec.gradfit < 0)
        rec.pass_frac = np.mean(pass_beat, axis=0)
        passed_voxels_mask = rec.pass_frac > 0.5
        rec.custom_voxel_mask = passed_voxels_mask.copy()

        rec.grad_pass_fraction = float(np.mean(rec.pass_frac))
        rec.mask_survival_fraction = float(np.sum(passed_voxels_mask) / Nvox)

        if not np.any(passed_voxels_mask):
            rec.grad_mask, rec.tc_grad, rec.ppr_grad = rec.raw_mask_dimac.copy(), rec.tc, rec.ppr
        else:
            flat_mask = rec.raw_mask_dimac.ravel()
            idx_true = np.where(flat_mask)[0]
            new_flat = np.zeros_like(flat_mask)
            new_flat[idx_true[passed_voxels_mask]] = True
            rec.grad_mask = new_flat.reshape(self.dimac_shape_3d)
            rec.tc_grad = np.mean(self.dimac_flat[rec.grad_mask.ravel()], axis=0)
            rec.ppr_grad = self.calc_ppr(rec.tc_grad)

    def _compute_hybrid_scores(self):
        if not self.cluster_records: return
        def norm_arr(arr):
            lo, hi = np.percentile(arr,[5, 95]) if len(arr)>1 else (0,1)
            return np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(arr, 0.5)

        scores = (0.45 * norm_arr(np.array([r.ppr for r in self.cluster_records]))) + \
                 (0.35 * norm_arr(np.array([r.grad_pass_fraction for r in self.cluster_records]))) + \
                 (0.10 * norm_arr(np.array([r.mask_survival_fraction for r in self.cluster_records]))) + \
                 (0.10 * norm_arr(np.array([r.vesselness_val for r in self.cluster_records])))
        
        for rec, s in zip(self.cluster_records, scores): rec.score = float(s)

    # -----------------------------------------------------------------
    # 3D RENDERING Logic
    # -----------------------------------------------------------------
    @staticmethod
    def robust_norm(img: np.ndarray, pmin=1, pmax=99.5) -> np.ndarray:
        vals = img[img > 0]
        if len(vals) == 0: return np.zeros_like(img, dtype=np.uint8)
        lo, hi = np.percentile(vals, (pmin, pmax))
        return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

    def _pad_rgb(self, img, pad=16, value=246):
        h, w, c = img.shape
        out = np.full((h + 2 * pad, w + 2 * pad, c), value, dtype=np.uint8)
        out[pad:pad + h, pad:pad + w] = img
        return out

    def _blend_green_slab(self, rgb_img, mask):
        if np.any(mask):
            patch = rgb_img[mask]
            green_tint = np.array([40, 200, 80], dtype=float)
            rgb_img[mask] = (patch * 0.6 + green_tint * 0.4).astype(np.uint8)
        return rgb_img

    def _burn_spheres_into_image(self, rgb_img, x_coords, y_coords, colors, radius=4):
        h, w, c = rgb_img.shape
        Y, X = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist_sq = X**2 + Y**2
        mask = dist_sq <= radius**2
        alpha_map = np.clip(1.0 - np.sqrt(dist_sq)/radius, 0.3, 1.0) * mask
        
        for x, y, color in zip(x_coords, y_coords, colors):
            x, y = int(x), int(y)
            x0, x1 = max(0, x - radius), min(w, x + radius + 1)
            y0, y1 = max(0, y - radius), min(h, y + radius + 1)
            if x0 >= x1 or y0 >= y1: continue
            kx0, kx1 = x0 - (x - radius), (x0 - (x - radius)) + (x1 - x0)
            ky0, ky1 = y0 - (y - radius), (y0 - (y - radius)) + (y1 - y0)
            patch = rgb_img[y0:y1, x0:x1]
            a_patch = alpha_map[ky0:ky1, kx0:kx1, None]
            rgb_img[y0:y1, x0:x1] = (patch * (1 - a_patch) + np.array(color, dtype=float) * a_patch).astype(np.uint8)
        return rgb_img

    def _create_oblique_slab(self, target_shape, target_affine):
        nx, ny, nz = self.dimac_shape_3d
        corners = np.array([[0,0,0],[nx,0,0], [0,ny,0],[nx,ny,0],[0,0,nz], [nx,0,nz],[0,ny,nz],[nx,ny,nz]])
        c_phys = nib.affines.apply_affine(self.dimac_nii.affine, corners)
        c_vox = nib.affines.apply_affine(np.linalg.inv(target_affine), c_phys)
        
        min_v = np.floor(c_vox.min(axis=0)).astype(int) - 10
        max_v = np.ceil(c_vox.max(axis=0)).astype(int) + 10
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, target_shape)
        
        mask = np.zeros(target_shape, dtype=bool)
        if np.any(max_v <= min_v): return mask
            
        xx, yy, zz = np.meshgrid(np.arange(min_v[0], max_v[0]), np.arange(min_v[1], max_v[1]), np.arange(min_v[2], max_v[2]), indexing='ij')
        pts_vox = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        pts_phys = nib.affines.apply_affine(target_affine, pts_vox)
        pts_dimac = nib.affines.apply_affine(np.linalg.inv(self.dimac_nii.affine), pts_phys)
        
        valid = ((pts_dimac[:, 0] >= -1.0) & (pts_dimac[:, 0] <= nx + 1.0) &
                 (pts_dimac[:, 1] >= -1.0) & (pts_dimac[:, 1] <= ny + 1.0) &
                 (pts_dimac[:, 2] >= -1.0) & (pts_dimac[:, 2] <= nz + 1.0))
        mask[min_v[0]:max_v[0], min_v[1]:max_v[1], min_v[2]:max_v[2]] = valid.reshape(xx.shape)
        return mask

    def _precompute_slab(self):
        self.slab_vess = self._create_oblique_slab(self.vess_data.shape, self.vess_nii.affine)

    def _build_global_preview_frames(self):
        positive = self.vess_data[self.vess_data > 0]
        if len(positive) == 0: return

        thresh = np.percentile(positive, self.config.global_crop_percentile)
        coords = np.array(np.nonzero(self.vess_data > thresh))
        min_c = np.maximum(coords.min(axis=1) - self.config.global_bbox_pad, 0)
        max_c = np.minimum(coords.max(axis=1) + self.config.global_bbox_pad + 1, self.vess_data.shape)
        bbox = tuple(slice(min_c[i], max_c[i]) for i in range(3))
        
        vess_crop = self.vess_data[bbox]
        ds = self.config.global_downsample
        if ds > 1: vess_crop = vess_crop[::ds, ::ds, ::ds]

        slab_crop = self.slab_vess[bbox]
        if ds > 1: slab_crop = slab_crop[::ds, ::ds, ::ds]

        cmap = plt.colormaps["autumn"]
        cand_pts = []
        self.global_cand_colors_rgb =[]
        for rec in self.cluster_records:
            if rec.peak_coord_vess is not None:
                pt = np.array([(rec.peak_coord_vess[i] - bbox[i].start) / ds for i in range(3)])
                cand_pts.append(pt)
                c_rgb = tuple(int(c * 255) for c in cmap(rec.score)[:3])
                rec.color_rgb = c_rgb
                self.global_cand_colors_rgb.append(c_rgb)
                
        cand_pts = np.array(cand_pts) if cand_pts else np.empty((0, 3))
        cx, cy, cz = (np.array(vess_crop.shape) - 1) / 2.0
        cand_centered = cand_pts - np.array([cx, cy, cz])

        self.global_base_frames =[]
        self.global_slab_masks =[]
        angles = np.linspace(0, 360, self.config.global_preview_frames, endpoint=False)
        self.global_frame_coords = np.zeros((len(angles), len(cand_pts), 2))

        for i, ang in enumerate(angles):
            rot_v = rotate(vess_crop, ang, axes=(0, 1), reshape=False, order=0)
            mip_v = np.flipud(np.max(rot_v, axis=1).T)
            
            rot_s = rotate(slab_crop.astype(np.uint8), ang, axes=(0, 1), reshape=False, order=0)
            mip_s = np.flipud(np.max(rot_s, axis=1).T) > 0
            
            pad = self.config.global_frame_pad_px
            self.global_slab_masks.append(np.pad(mip_s, pad_width=pad, mode='constant', constant_values=False))
            
            base = self.robust_norm(mip_v, pmin=0.5, pmax=99.8)
            bg = np.array(self.config.global_bg_rgb, dtype=np.uint8)
            rgb = np.zeros((base.shape[0], base.shape[1], 3), dtype=np.uint8)
            for c in range(3): rgb[..., c] = np.clip(bg[c] + 0.95 * base, 0, 255)

            self.global_base_frames.append(self._pad_rgb(rgb, pad=pad, value=self.config.global_bg_rgb))

            if len(cand_centered) > 0:
                rad = np.radians(ang)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                x_rot = cand_centered[:, 0] * cos_a - cand_centered[:, 1] * sin_a
                z = cand_centered[:, 2]
                self.global_frame_coords[i, :, 0] = x_rot + cx + pad
                self.global_frame_coords[i, :, 1] = ((mip_v.shape[0] - 1) - (z + cz)) + pad

    def _render_eicab_style(self, mip, roi_mask_2d, roi_color_rgb):
        base = self.robust_norm(mip)
        rgb = np.stack([base]*3, axis=-1)
        if np.any(roi_mask_2d): rgb[roi_mask_2d > 0] = roi_color_rgb
        return rgb

    def _render_local_mip(self, mip, roi_mask_2d, roi_color_rgb):
        base = self.robust_norm(mip, pmin=1, pmax=99.7)
        inv = np.clip((255 - base) * 0.92 + 8, 0, 255).astype(np.uint8)
        rgb = np.dstack([inv, inv, inv])
        rgb = ((0.88 * rgb) + (0.12 * np.array([self.config.local_bg_value]*3))).astype(np.uint8)
        
        border = binary_dilation(roi_mask_2d, iterations=self.config.roi_border_dilation_2d) ^ roi_mask_2d
        if np.any(roi_mask_2d):
            rgb[roi_mask_2d] = ((1 - self.config.roi_fill_alpha) * rgb[roi_mask_2d] + self.config.roi_fill_alpha * np.array(roi_color_rgb)).astype(np.uint8)
        if np.any(border): rgb[border] = np.array(roi_color_rgb, dtype=np.uint8)
        return rgb

    def get_local_frames_for_record(self, rec, crop_scale, active_mask, filter_active):
        filt_str = "filt" if filter_active else "raw"
        key = f"{rec.cluster_id}_scale_{crop_scale:.2f}_{filt_str}"

        if key not in self.local_preview_cache:
            if not self.has_vess: return [],[]
            
            roi_vess = resample_from_to(nib.Nifti1Image(active_mask.astype(np.uint8), self.dimac_nii.affine), (self.vess_data.shape, self.vess_nii.affine), order=0).get_fdata() > 0
            
            if not np.any(roi_vess) and rec.peak_coord_vess is not None:
                cx, cy, cz = rec.peak_coord_vess
                xx, yy, zz = np.meshgrid(*[np.arange(s) for s in self.vess_data.shape], indexing="ij")
                roi_vess = ((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2) <= self.config.fallback_sphere_radius_vox**2

            if not np.any(roi_vess): return [],[]
            roi_vess = binary_dilation(roi_vess, iterations=1)
            pad = int(round(self.config.local_crop_pad * crop_scale))
            
            coords = np.array(np.nonzero(roi_vess))
            bbox = [slice(max(0, coords[i].min()-pad), min(self.vess_data.shape[i], coords[i].max()+pad+1)) for i in range(3)]

            vess_crop = self.vess_data[tuple(bbox)]
            roi_crop = roi_vess[tuple(bbox)]
            slab_crop = self.slab_vess[tuple(bbox)]
            
            base_frames = []
            slab_masks =[]
            angles = np.linspace(0, 360, self.config.local_preview_frames, endpoint=False)

            for ang in angles:
                rot_v = rotate(vess_crop, ang, axes=(0, 1), reshape=False, order=0)
                rot_r = rotate(roi_crop.astype(np.uint8), ang, axes=(0, 1), reshape=False, order=0) > 0
                rot_s = rotate(slab_crop.astype(np.uint8), ang, axes=(0, 1), reshape=False, order=0)

                mip_ctx = np.rot90(np.max(rot_v, axis=1), k=1)
                roi_ctx = np.rot90(np.max(rot_r, axis=1), k=1)
                slab_ctx = np.rot90(np.max(rot_s, axis=1), k=1) > 0
                
                mip_a = np.flipud(np.max(rot_v, axis=1).T)
                roi_a = np.flipud(np.max(rot_r, axis=1).T)
                slab_a = np.flipud(np.max(rot_s, axis=1).T) > 0

                frame_eicab = self._render_eicab_style(mip_ctx, roi_ctx, rec.color_rgb)
                frame_a = self._render_local_mip(mip_a, roi_a, rec.color_rgb)

                h = frame_a.shape[0]
                w_e = frame_eicab.shape[1]
                w_a = frame_a.shape[1]
                gap = 16
                
                canvas = np.full((h, w_e + gap + w_a, 3), self.config.local_bg_value, dtype=np.uint8)
                canvas[:, :w_e] = frame_eicab
                canvas[:, w_e + gap:] = frame_a
                canvas[:, w_e + gap//2 - 1 : w_e + gap//2 + 1] = 200
                
                s_mask = np.zeros((h, w_e + gap + w_a), dtype=bool)
                s_mask[:, :w_e] = slab_ctx
                s_mask[:, w_e + gap:] = slab_a

                border_pad = self.config.local_frame_border_px
                base_frames.append(self._pad_rgb(canvas, pad=border_pad, value=self.config.local_bg_value))
                slab_masks.append(np.pad(s_mask, pad_width=border_pad, mode='constant', constant_values=False))
            
            self.local_preview_cache[key] = (base_frames, slab_masks)

        return self.local_preview_cache[key]