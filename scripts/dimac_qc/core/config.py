# core/config.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class QCConfig:
    min_heart_rate_hz: float = 0.66
    max_heart_rate_hz: float = 2.0
    global_crop_percentile: float = 94.0
    global_bbox_pad: int = 28
    global_downsample: int = 2  
    global_preview_frames: int = 30
    global_frame_pad_px: int = 24
    local_crop_pad: int = 36
    local_min_crop_size: Tuple[int, int, int] = (112, 112, 84)
    local_preview_frames: int = 30
    local_frame_border_px: int = 20
    fallback_sphere_radius_vox: int = 3
    local_crop_scales: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)
    local_default_crop_scale_idx: int = 1
    animation_interval_ms: int = 120 
    dimac_click_tolerance_fraction: float = 0.03
    roi_fill_alpha: float = 0.35
    roi_border_dilation_2d: int = 2
    global_bg_rgb: Tuple[int, int, int] = (14, 18, 24)
    local_bg_value: int = 246
    dilation_shape: Tuple[int, int, int] = (5, 5, 1)
    dilation_iterations: int = 1
    mricrogl_exe: str = "MRIcroGL"

@dataclass
class ClusterRecord:
    cluster_id: int
    peak_coord_dimac: np.ndarray
    raw_mask_dimac: np.ndarray
    tc: np.ndarray
    ppr: float
    peak_coord_vess: Optional[np.ndarray] = None
    score: float = 0.0
    color_rgb: Optional[Tuple[int, int, int]] = None
    grad_mask: Optional[np.ndarray] = None
    tc_grad: Optional[np.ndarray] = None
    ppr_grad: float = 0.0
    grad_pass_fraction: float = 0.0
    mask_survival_fraction: float = 0.0
    vesselness_val: float = 0.0
    custom_voxel_mask: Optional[np.ndarray] = None
    tc_2d: Optional[np.ndarray] = None
    maxind: Optional[np.ndarray] = None
    minind: Optional[np.ndarray] = None
    gradfit: Optional[np.ndarray] = None
    gradallfit: Optional[np.ndarray] = None
    gradratio: Optional[np.ndarray] = None
    pass_frac: Optional[np.ndarray] = None