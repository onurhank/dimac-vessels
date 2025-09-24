#!/bin/bash
# ==============================================================================
# DIMAC Vessel Analysis Pipeline - Main Wrapper (idempotent + auto ROI space)
# ==============================================================================

set -Eeuo pipefail
trap 'echo "[FATAL] $(date -Is) line:$LINENO cmd:$BASH_COMMAND" >&2' ERR
export MPLBACKEND=Agg

# --- Defaults ---
TOF_RAW=""
DIMAC_ICA_BOLD=""
DIMAC_ACA_BOLD=""

ACA_ROI_INPUT=""
ACA_ROI_INPUT_SPACE=""   # kept for compatibility (not required)
ICA_ROI_INPUT=""
ICA_ROI_INPUT_SPACE=""   # kept for compatibility (not required)

SUB_ID="sub-default"
SCRIPT_DIR="scripts"
DERIV_DIR="derivatives"
OUT_DIR="analysis_output"

PPR_THR=4
K_CLUSTERS=4
VESSELNESS_SCALES="1 2 3 4 5 6"
PATH_THICKEN_VOX=1
# --- Tunables for clustering (override as needed) ---
K_CLUSTERS=${K_CLUSTERS:-5}

# ACA tends to be smaller; a bit stricter PPR and a midline prior help avoid MCA
PPR_THR_ACA=${PPR_THR_ACA:-8}       # try 8–12; was 3 before (quite low)
MIN_VOX_ACA=${MIN_VOX_ACA:-80}      # ACA size floor
MIDLINE_FRAC=${MIDLINE_FRAC:-0.20}  # ~20% of FOV around midline (assumes X is L–R)
Z_FRAC=${Z_FRAC:-0.50}              # central Z slab (optional)

# ICA can be larger
PPR_THR_ICA=${PPR_THR_ICA:-10}
MIN_VOX_ICA=${MIN_VOX_ICA:-150}

# --- Args ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --tof) TOF_RAW="$2"; shift; shift ;;
    --dimac-aca) DIMAC_ACA_BOLD="$2"; shift; shift ;;
    --dimac-ica) DIMAC_ICA_BOLD="$2"; shift; shift ;;
    --aca-roi) ACA_ROI_INPUT="$2"; shift; shift ;;
    --aca-roi-space) ACA_ROI_INPUT_SPACE="$2"; shift; shift ;;   # optional now
    --ica-roi) ICA_ROI_INPUT="$2"; shift; shift ;;
    --ica-roi-space) ICA_ROI_INPUT_SPACE="$2"; shift; shift ;;   # optional now
    --sub-id) SUB_ID="$2"; shift; shift ;;
    --deriv-dir) DERIV_DIR="$2"; shift; shift ;;
    --out-dir) OUT_DIR="$2"; shift; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "--- Performing pre-flight checks (idempotent + auto ROI space) ---"
[[ -f "$TOF_RAW" ]] || { echo "[ERROR] --tof is required and must exist."; exit 1; }
[[ -f "$DIMAC_ACA_BOLD" ]] || { echo "[ERROR] --dimac-aca is required and must exist."; exit 1; }
[[ -f "$DIMAC_ICA_BOLD" ]] || { echo "[ERROR] --dimac-ica is required and must exist."; exit 1; }
[[ -d "$SCRIPT_DIR" ]] || { echo "[ERROR] Scripts dir not found: $SCRIPT_DIR"; exit 1; }

mkdir -p "$DERIV_DIR" "$DERIV_DIR/resampled_to_tof" "$OUT_DIR"

# --- Helpers ---
vox() { # voxel count; 0 if missing/invalid
  local f="$1"
  [[ -f "$f" ]] || { echo 0; return; }
  fslstats "$f" -V 2>/dev/null | awk '{print $1+0}' || echo 0
}
get_dims() { fslhd "$1" | awk '/^dim1/{d1=$2}/^dim2/{d2=$2}/^dim3/{d3=$2} END{print d1,d2,d3}'; }
get_pix()  { fslhd "$1" | awk '/^pixdim1/{p1=$2}/^pixdim2/{p2=$2}/^pixdim3/{p3=$2} END{print p1,p2,p3}'; }
get_srow() { fslhd "$1" | awk '/^srow_x/{print $2,$3,$4,$5}
/^srow_y/{print $2,$3,$4,$5}
/^srow_z/{print $2,$3,$4,$5}' | tr '\n' ' '; }

vec_close() { # "a1 a2" "b1 b2" tol -> 1/0
  awk -v A="$1" -v B="$2" -v t="$3" '
    BEGIN{split(A,a); split(B,b);
      if (length(a)!=length(b) || length(a)==0) {print 0; exit}
      for(i=1;i<=length(a);i++){d=a[i]-b[i]; if(d<0)d=-d; if(d>t){print 0; exit}}
      print 1
    }'
}

same_grid() { # fileA fileB -> 0 if same, 1 if different
  local A="$1" B="$2"
  local dA pA sA dB pB sB
  dA=$(get_dims "$A"); pA=$(get_pix "$A"); sA=$(get_srow "$A")
  dB=$(get_dims "$B"); pB=$(get_pix "$B"); sB=$(get_srow "$B")
  [[ "$dA" == "$dB" ]] || { echo 1; return; }
  [[ $(vec_close "$pA" "$pB" 1e-5) -eq 1 ]] || { echo 1; return; }
  [[ $(vec_close "$sA" "$sB" 1e-2) -eq 1 ]] || { echo 1; return; }
  echo 0
}

# Will be defined at Step 1
TOF_ISOTROPIC=""

detect_roi_space() { # roi -> TOF_ISO | TOF_RAW | ACA_DIMAC | ICA_DIMAC | UNKNOWN
  local ROI="$1"
  local s_iso=1 s_raw=1 s_aca=1 s_ica=1
  [[ -n "$TOF_ISOTROPIC" && -f "$TOF_ISOTROPIC" ]] && s_iso=$(same_grid "$ROI" "$TOF_ISOTROPIC") || s_iso=1
  s_raw=$(same_grid "$ROI" "$TOF_RAW") || s_raw=1
  s_aca=$(same_grid "$ROI" "$DIMAC_ACA_BOLD") || s_aca=1
  s_ica=$(same_grid "$ROI" "$DIMAC_ICA_BOLD") || s_ica=1
  if [[ $s_iso -eq 0 ]]; then echo "TOF_ISO"; return; fi
  if [[ $s_raw -eq 0 ]]; then echo "TOF_RAW"; return; fi
  if [[ $s_aca -eq 0 ]]; then echo "ACA_DIMAC"; return; fi
  if [[ $s_ica -eq 0 ]]; then echo "ICA_DIMAC"; return; fi
  echo "UNKNOWN"
}

copy_if_exists() { # src dst (only if dst missing or empty)
  local src="$1" dst="$2"
  if [[ -f "$dst" && $(vox "$dst") -gt 0 ]]; then
    echo "[SKIP] $dst exists (vox>0)."
  else
    cp -f "$src" "$dst"
    echo "[COPY] $src -> $dst"
  fi
}

# ==============================================================================
# STEP 1: TOF -> isotropic
# ==============================================================================
echo
echo "================================================="
echo " STEP 1: Resample TOF to isotropic"
echo "================================================="
TOF_ISOTROPIC="${DERIV_DIR}/${SUB_ID}_$(basename "${TOF_RAW}" .nii.gz)_isotropic.nii.gz"
if [[ -f "$TOF_ISOTROPIC" ]]; then
  echo "[SKIP] Isotropic TOF exists: $TOF_ISOTROPIC"
else
  python "${SCRIPT_DIR}/tof_resampler.py" --tof "${TOF_RAW}" --out "${TOF_ISOTROPIC}"
fi

# ==============================================================================
# STEP 2: Frangi vesselness
# ==============================================================================
echo
echo "================================================="
echo " STEP 2: Calculate Vesselness from Isotropic TOF"
echo "================================================="
VESSELNESS_PREFIX="${OUT_DIR}/${SUB_ID}_tof_vesselness"
FRANGI_VESSELNESS="${VESSELNESS_PREFIX}_frangi_vesselness.nii.gz"
FRANGI_MASK="${VESSELNESS_PREFIX}_frangi_mask.nii.gz"

if [[ -f "$FRANGI_VESSELNESS" && -f "$FRANGI_MASK" ]]; then
  echo "[SKIP] Frangi outputs exist."
else
  python "${SCRIPT_DIR}/vessel_cli.py" \
    --input "${TOF_ISOTROPIC}" \
    --output-prefix "${VESSELNESS_PREFIX}" \
    --normalize \
    --method frangi \
    --scales-mm ${VESSELNESS_SCALES}
fi

# ==============================================================================
# STEP 3: Build (or use) ROIs in DIMAC space
# ==============================================================================
echo
echo "================================================="
echo " STEP 3: Generate ACA & ICA ROIs in DIMAC space (NO vessel mask)"
echo "================================================="

ROIS_PROVIDED=false
[[ -n "$ACA_ROI_INPUT" && -f "$ACA_ROI_INPUT" && -n "$ICA_ROI_INPUT" && -f "$ICA_ROI_INPUT" ]] && ROIS_PROVIDED=true

if $ROIS_PROVIDED; then
  CURRENT_ACA_ROI_PATH="$ACA_ROI_INPUT"
  CURRENT_ICA_ROI_PATH="$ICA_ROI_INPUT"
  echo "[INFO] Provided ROIs:"
  echo "  - ACA: $CURRENT_ACA_ROI_PATH (auto-detecting space...)"
  echo "  - ICA: $CURRENT_ICA_ROI_PATH (auto-detecting space...)"
else
  CURRENT_ACA_ROI_PATH="${DERIV_DIR}/${SUB_ID}_ACA_generated_roi.nii.gz"
  CURRENT_ICA_ROI_PATH="${DERIV_DIR}/${SUB_ID}_ICA_generated_roi.nii.gz"

  if [[ -f "$CURRENT_ACA_ROI_PATH" && $(vox "$CURRENT_ACA_ROI_PATH") -gt 0 ]]; then
    echo "[SKIP] ACA ROI already generated (vox>0)."
  else
    python "${SCRIPT_DIR}/dimac_auto_roi_improved.py" \
      --dimac "${DIMAC_ACA_BOLD}" \
      --out "${DERIV_DIR}/${SUB_ID}_ACA_generated" \
      --auto-band --welch-in-gate \
      --k ${K_CLUSTERS} --ppr-thr 3 \
      --center-frac 0.5 \
      --min-voxels 200
  fi

  if [[ -f "$CURRENT_ICA_ROI_PATH" && $(vox "$CURRENT_ICA_ROI_PATH") -gt 0 ]]; then
    echo "[SKIP] ICA ROI already generated (vox>0)."
  else
    python "${SCRIPT_DIR}/dimac_auto_roi_improved.py" \
      --dimac "${DIMAC_ICA_BOLD}" \
      --out "${DERIV_DIR}/${SUB_ID}_ICA_generated" \
      --auto-band --welch-in-gate \
      --k ${K_CLUSTERS} --ppr-thr ${PPR_THR}
  fi
fi

# QC
[[ $(vox "$CURRENT_ACA_ROI_PATH") -gt 0 ]] || { echo "[ERROR] ACA ROI empty."; exit 1; }
[[ $(vox "$CURRENT_ICA_ROI_PATH") -gt 0 ]] || { echo "[ERROR] ICA ROI empty."; exit 1; }

# ==============================================================================
# STEP 4: Regrid ROIs to isotropic TOF
# ==============================================================================
echo
echo "================================================="
echo " STEP 4: Resample ROIs to Isotropic TOF Space"
echo "================================================="
ACA_ROI_FINAL_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_ACA_roi_in_TOF_iso_space.nii.gz"
ICA_ROI_FINAL_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_ICA_roi_in_TOF_iso_space.nii.gz"

# detect spaces (checks against: TOF_ISO, TOF_RAW, ACA_DIMAC, ICA_DIMAC)
ACA_SPACE=$(detect_roi_space "$CURRENT_ACA_ROI_PATH")
ICA_SPACE=$(detect_roi_space "$CURRENT_ICA_ROI_PATH")
echo "  - Detected ACA ROI space: $ACA_SPACE"
echo "  - Detected ICA ROI space: $ICA_SPACE"

# If already in TOF_ISO → copy; else → resample
need_resample=false
if [[ "$ACA_SPACE" == "TOF_ISO" ]]; then
  copy_if_exists "$CURRENT_ACA_ROI_PATH" "$ACA_ROI_FINAL_TOF"
else
  need_resample=true
fi
if [[ "$ICA_SPACE" == "TOF_ISO" ]]; then
  copy_if_exists "$CURRENT_ICA_ROI_PATH" "$ICA_ROI_FINAL_TOF"
else
  need_resample=true
fi

if $need_resample; then
  if [[ ! -f "$ACA_ROI_FINAL_TOF" || $(vox "$ACA_ROI_FINAL_TOF") -eq 0 || \
        ! -f "$ICA_ROI_FINAL_TOF" || $(vox "$ICA_ROI_FINAL_TOF") -eq 0 ]]; then
    python "${SCRIPT_DIR}/resample_dimac_to_tof.py" \
      --tof "${TOF_ISOTROPIC}" \
      --aca-bold "${DIMAC_ACA_BOLD}" \
      --aca-mask "${CURRENT_ACA_ROI_PATH}" \
      --ica-bold "${DIMAC_ICA_BOLD}" \
      --ica-mask "${CURRENT_ICA_ROI_PATH}" \
      --output-dir "${DERIV_DIR}/resampled_to_tof"
    # rename to canonical names
    mv -f "${DERIV_DIR}/resampled_to_tof/$(basename "${CURRENT_ACA_ROI_PATH}" .nii.gz)_in_TOF_space.nii.gz" "$ACA_ROI_FINAL_TOF" || true
    mv -f "${DERIV_DIR}/resampled_to_tof/$(basename "${CURRENT_ICA_ROI_PATH}" .nii.gz)_in_TOF_space.nii.gz" "$ICA_ROI_FINAL_TOF" || true
  else
    echo "[SKIP] Resampled ROIs already present and non-empty."
  fi
fi

[[ $(vox "$ACA_ROI_FINAL_TOF") -gt 0 ]] || { echo "[ERROR] ACA ROI empty in TOF iso space."; exit 1; }
[[ $(vox "$ICA_ROI_FINAL_TOF") -gt 0 ]] || { echo "[ERROR] ICA ROI empty in TOF iso space."; exit 1; }

# ==============================================================================
# STEP 5: Shortest path + renders
# ==============================================================================
echo
echo "================================================="
echo " STEP 5: Find Shortest Path and Render Visuals"
echo "================================================="
PATH_PREFIX="${OUT_DIR}/${SUB_ID}_aca_ica_path_analysis"
MASK_NII="${PATH_PREFIX}_path_mask.nii.gz"
PNG_Y="${PATH_PREFIX}_render_mip_y.png"
GIF_Y="${PATH_PREFIX}_spin_mip_y.gif"

if [[ -f "$MASK_NII" && -f "$PNG_Y" && -f "$GIF_Y" ]]; then
  echo "[SKIP] Path + renders already exist."
else
  python "${SCRIPT_DIR}/vessel_shortest_path.py" \
  --frangi-vesselness "${FRANGI_VESSELNESS}" \
  --aca-roi "${ACA_ROI_FINAL_TOF}" \
  --ica-roi "${ICA_ROI_FINAL_TOF}" \
  --out-prefix "${PATH_PREFIX}" \
  --invert-weight \
  --render-png \
  --render-mode mip \
  --render-axis y \
  --render-gif \
  --gif-frames 60 \
  --gif-fps 12 \
  --gif-rotate-mode 3d \
  --gif-pad \
  --gif-tilt-deg 24 \
  --gif-yaw-start-deg 90 \
  --path-thicken-vox ${PATH_THICKEN_VOX} \
  | tee >(awk -v out="${PATH_PREFIX}_length_mm.txt" '/Geometric path length/{printf "%.3f\n", $(NF-1) > out}')

fi

echo
echo "================================================="
echo " Pipeline finished!"
echo "  ACA ROI (TOF iso): $ACA_ROI_FINAL_TOF  (vox: $(vox "$ACA_ROI_FINAL_TOF"))"
echo "  ICA ROI (TOF iso): $ICA_ROI_FINAL_TOF  (vox: $(vox "$ICA_ROI_FINAL_TOF"))"
echo "  Path/GIF: ${PATH_PREFIX}_path_mask.nii.gz / ${PATH_PREFIX}_spin_mip_y.gif"
echo "================================================="

