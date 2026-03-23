#!/bin/bash
# ==============================================================================
# DIMAC Vessel Analysis Pipeline - PRO UNIFIED WRAPPER (Steps 1-9 Loop)
# ==============================================================================

set -Eeuo pipefail
trap 'echo "[FATAL] $(date -Is) line:$LINENO cmd:$BASH_COMMAND" >&2' ERR

# --- Defaults ---
TOF_RAW=""
DIMAC_ACA_BOLD=""
DIMAC_ICA_BOLD=""
SUB_ID="sub-default"
SCRIPT_DIR="scripts"
DERIV_DIR="derivatives"
OUT_DIR="analysis_output"

MRICROGL_EXE="MRIcroGL"
NO_GUI=false

VESSELNESS_SCALES="1 2 3 4 5 6"
PATH_THICKEN_VOX=1

# --- Args ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --tof) TOF_RAW="$2"; shift; shift ;;
    --dimac-aca) DIMAC_ACA_BOLD="$2"; shift; shift ;;
    --dimac-ica) DIMAC_ICA_BOLD="$2"; shift; shift ;;
    --sub-id) SUB_ID="$2"; shift; shift ;;
    --deriv-dir) DERIV_DIR="$2"; shift; shift ;;
    --out-dir) OUT_DIR="$2"; shift; shift ;;
    --mricrogl-exe) MRICROGL_EXE="$2"; shift; shift ;;
    --no-gui) NO_GUI=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "--- Performing pre-flight checks ---"
[[ -f "$TOF_RAW" ]] || { echo "[ERROR] --tof missing: $TOF_RAW"; exit 1; }
[[ -f "$DIMAC_ACA_BOLD" ]] || { echo "[ERROR] --dimac-aca missing: $DIMAC_ACA_BOLD"; exit 1; }
[[ -f "$DIMAC_ICA_BOLD" ]] || { echo "[ERROR] --dimac-ica missing: $DIMAC_ICA_BOLD"; exit 1; }
[[ -d "$SCRIPT_DIR" ]] || { echo "[ERROR] Scripts dir not found: $SCRIPT_DIR"; exit 1; }

mkdir -p "$DERIV_DIR" "$DERIV_DIR/resampled_to_tof" "$OUT_DIR"

# Helper for voxel counts
vox() { [[ -f "$1" ]] && fslstats "$1" -V 2>/dev/null | awk '{print $1+0}' || echo 0; }

# ==============================================================================
# STEP 1: Resample TOF to Isotropic
# ==============================================================================
echo
echo "================================================="
echo " STEP 1: Resample TOF to Isotropic"
echo "================================================="
TOF_ISOTROPIC="${DERIV_DIR}/${SUB_ID}_isotropic.nii.gz"
if [[ -f "$TOF_ISOTROPIC" && $(vox "$TOF_ISOTROPIC") -gt 0 ]]; then
    echo "[SKIP] Isotropic TOF exists: $TOF_ISOTROPIC"
else
    python3 "${SCRIPT_DIR}/tof_resampler.py" --tof "${TOF_RAW}" --out "${TOF_ISOTROPIC}"
fi

# ==============================================================================
# STEP 2: Vesselness
# ==============================================================================
echo
echo "================================================="
echo " STEP 2: Calculate Vesselness from Isotropic TOF"
echo "================================================="
VESSELNESS_PREFIX="${OUT_DIR}/${SUB_ID}_vesselness"
FRANGI_VESSELNESS="${VESSELNESS_PREFIX}_frangi_vesselness.nii.gz"
if [[ -f "$FRANGI_VESSELNESS" && $(vox "$FRANGI_VESSELNESS") -gt 0 ]]; then
    echo "[SKIP] Frangi outputs exist."
else
    python3 "${SCRIPT_DIR}/vessel_cli.py" --input "${TOF_ISOTROPIC}" --output-prefix "${VESSELNESS_PREFIX}" --normalize --method frangi --scales-mm ${VESSELNESS_SCALES}
fi

# ==============================================================================
# STEP 3: Auto-ROI Generation
# ==============================================================================
echo
echo "================================================="
echo " STEP 3: Generate Auto-ROIs in DIMAC space"
echo "================================================="
ACA_AUTO_DIMAC="${DERIV_DIR}/${SUB_ID}_ACA_generated_roi_roi.nii.gz"
ICA_AUTO_DIMAC="${DERIV_DIR}/${SUB_ID}_ICA_generated_roi_roi.nii.gz"

if [[ -f "$ACA_AUTO_DIMAC" && $(vox "$ACA_AUTO_DIMAC") -gt 0 ]]; then
    echo "[SKIP] ACA Auto ROI exists."
else
    python3 "${SCRIPT_DIR}/dimac_auto_roi_improved.py" --dimac "${DIMAC_ACA_BOLD}" --out "${DERIV_DIR}/${SUB_ID}_ACA_generated_roi" --ppr-thr 3 --min-voxels 80 || echo "[WARNING] ACA Auto failed."
fi

if [[ -f "$ICA_AUTO_DIMAC" && $(vox "$ICA_AUTO_DIMAC") -gt 0 ]]; then
    echo "[SKIP] ICA Auto ROI exists."
else
    python3 "${SCRIPT_DIR}/dimac_auto_roi_improved.py" --dimac "${DIMAC_ICA_BOLD}" --out "${DERIV_DIR}/${SUB_ID}_ICA_generated_roi" --ppr-thr 3 --min-voxels 150 || echo "[WARNING] ICA Auto failed."
fi

# ==============================================================================
# STEP 4: Resample Auto-ROIs to TOF Space
# ==============================================================================
echo
echo "================================================="
echo " STEP 4: Resample Auto-ROIs to Isotropic TOF"
echo "================================================="
ACA_AUTO_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_ACA_roi_in_TOF_iso_space.nii.gz"
ICA_AUTO_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_ICA_roi_in_TOF_iso_space.nii.gz"

if [[ -f "$ACA_AUTO_TOF" && $(vox "$ACA_AUTO_TOF") -gt 0 && -f "$ICA_AUTO_TOF" && $(vox "$ICA_AUTO_TOF") -gt 0 ]]; then
    echo "[SKIP] Resampled Auto-ROIs already present."
else
    python3 "${SCRIPT_DIR}/resample_dimac_to_tof.py" \
      --tof "${TOF_ISOTROPIC}" \
      --aca-bold "${DIMAC_ACA_BOLD}" \
      --aca-mask "${ACA_AUTO_DIMAC}" \
      --ica-bold "${DIMAC_ICA_BOLD}" \
      --ica-mask "${ICA_AUTO_DIMAC}" \
      --output-dir "${DERIV_DIR}/resampled_to_tof"

    mv -f "${DERIV_DIR}/resampled_to_tof/$(basename "${ACA_AUTO_DIMAC}" .nii.gz)_in_TOF_space.nii.gz" "$ACA_AUTO_TOF" || true
    mv -f "${DERIV_DIR}/resampled_to_tof/$(basename "${ICA_AUTO_DIMAC}" .nii.gz)_in_TOF_space.nii.gz" "$ICA_AUTO_TOF" || true
fi

# ==============================================================================
# STEP 5: Auto Shortest Path
# ==============================================================================
echo
echo "================================================="
echo " STEP 5: Find Shortest Path (AUTO ROIs)"
echo "================================================="
AUTO_PATH_PREFIX="${OUT_DIR}/${SUB_ID}_auto_aca_ica_path_analysis"

if [[ -f "${AUTO_PATH_PREFIX}_path_mask.nii.gz" ]]; then
    echo "[SKIP] Auto Path + renders already exist."
else
    python3 "${SCRIPT_DIR}/vessel_shortest_path.py" \
      --frangi-vesselness "${FRANGI_VESSELNESS}" \
      --aca-roi "${ACA_AUTO_TOF}" \
      --ica-roi "${ICA_AUTO_TOF}" \
      --out-prefix "${AUTO_PATH_PREFIX}" \
      --invert-weight \
      --render-png \
      --render-mode mip \
      --render-axis y \
      --render-gif \
      --gif-frames 30 \
      --gif-fps 15 \
      --gif-rotate-mode 3d \
      --gif-pad \
      --gif-tilt-deg 24 \
      --gif-yaw-start-deg 90 \
      --path-thicken-vox ${PATH_THICKEN_VOX} \
      | tee >(awk -v out="${AUTO_PATH_PREFIX}_length_mm.txt" '/Geometric path length/{printf "%.3f\n", $(NF-1) > out}')
fi

# ==============================================================================
# THE INFINITE QC LOOP (Steps 6 -> 7 -> 8 -> Review)
# ==============================================================================

if [ "$NO_GUI" = true ]; then
    echo
    echo ">>>[--no-gui flag active]. Skipping Interactive QC."
    echo ">>> ALL AUTOMATED STEPS COMPLETE!"
    exit 0
fi

export PYTHONPATH="${PYTHONPATH:-}:.:${SCRIPT_DIR}:.."
ACA_MANUAL_PREFIX="${DERIV_DIR}/${SUB_ID}_ACA_manual"
ICA_MANUAL_PREFIX="${DERIV_DIR}/${SUB_ID}_ICA_manual"

while true; do

    echo
    echo "================================================="
    echo " STEP 6: Launching Interactive QC GUI"
    echo "================================================="

    echo "--- Opening ACA QC ---"
    python3 -c "
from dimac_qc import run_qc
side = run_qc(
    dimac_file='${DIMAC_ACA_BOLD}',
    out_prefix='${ACA_MANUAL_PREFIX}',
    vesselness_file='${FRANGI_VESSELNESS}',
    tof_file='${TOF_ISOTROPIC}',
    auto_roi_file=None,
    previous_side=None,
    mricrogl_exe='${MRICROGL_EXE}'
)
with open('${DERIV_DIR}/.prev_side.txt', 'w') as f:
    f.write(side if side else '')
"
    PREV_SIDE=$(cat "${DERIV_DIR}/.prev_side.txt" 2>/dev/null || echo "")

    echo "--- Opening ICA QC ---"
    python3 -c "
from dimac_qc import run_qc
run_qc(
    dimac_file='${DIMAC_ICA_BOLD}',
    out_prefix='${ICA_MANUAL_PREFIX}',
    vesselness_file='${FRANGI_VESSELNESS}',
    tof_file='${TOF_ISOTROPIC}',
    auto_roi_file=None,
    previous_side='${PREV_SIDE}',
    mricrogl_exe='${MRICROGL_EXE}'
)
"

    ACA_MANUAL_ROI="${ACA_MANUAL_PREFIX}_roi.nii.gz"
    ICA_MANUAL_ROI="${ICA_MANUAL_PREFIX}_roi.nii.gz"

    if [[ ! -f "$ACA_MANUAL_ROI" || ! -f "$ICA_MANUAL_ROI" ]]; then
        echo
        echo "[INFO] User did not save both Manual ROIs. Exiting QC Tool."
        break
    fi

    echo
    echo "================================================="
    echo " STEP 7: Resample Manual ROIs to TOF Space"
    echo "================================================="
    
    ACA_MANUAL_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_manual_ACA_roi_in_TOF_iso_space.nii.gz"
    ICA_MANUAL_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_manual_ICA_roi_in_TOF_iso_space.nii.gz"

    python3 "${SCRIPT_DIR}/resample_dimac_to_tof.py" \
      --tof "${TOF_ISOTROPIC}" \
      --aca-bold "${DIMAC_ACA_BOLD}" \
      --aca-mask "${ACA_MANUAL_ROI}" \
      --ica-bold "${DIMAC_ICA_BOLD}" \
      --ica-mask "${ICA_MANUAL_ROI}" \
      --output-dir "${DERIV_DIR}/resampled_to_tof"

    mv -f "${DERIV_DIR}/resampled_to_tof/$(basename "${ACA_MANUAL_ROI}" .nii.gz)_in_TOF_space.nii.gz" "$ACA_MANUAL_TOF" || true
    mv -f "${DERIV_DIR}/resampled_to_tof/$(basename "${ICA_MANUAL_ROI}" .nii.gz)_in_TOF_space.nii.gz" "$ICA_MANUAL_TOF" || true

    echo
    echo "================================================="
    echo " STEP 8: Find Shortest Path for MANUAL ROIs"
    echo "================================================="
    MANUAL_PATH_PREFIX="${OUT_DIR}/${SUB_ID}_manual_aca_ica_path_analysis"

    python3 "${SCRIPT_DIR}/vessel_shortest_path.py" \
      --frangi-vesselness "${FRANGI_VESSELNESS}" \
      --aca-roi "${ACA_MANUAL_TOF}" \
      --ica-roi "${ICA_MANUAL_TOF}" \
      --out-prefix "${MANUAL_PATH_PREFIX}" \
      --invert-weight \
      --render-png \
      --render-mode mip \
      --render-axis y \
      --render-gif \
      --gif-frames 30 \
      --gif-fps 15 \
      --gif-rotate-mode 3d \
      --gif-pad \
      --gif-tilt-deg 24 \
      --gif-yaw-start-deg 90 \
      --path-thicken-vox ${PATH_THICKEN_VOX} \
      | tee >(awk -v out="${MANUAL_PATH_PREFIX}_length_mm.txt" '/Geometric path length/{printf "%.3f\n", $(NF-1) > out}')

echo
    echo "================================================="
    echo " STEP 9: Visual Review (Interactive Player)"
    echo "================================================="

    python3 "${SCRIPT_DIR}/review_path_gui.py" "${MANUAL_PATH_PREFIX}" "${DERIV_DIR}/.review_result.txt"

    REVIEW_RESULT=$(cat "${DERIV_DIR}/.review_result.txt" 2>/dev/null || echo "1")

    if [ "$REVIEW_RESULT" -eq 0 ]; then
        echo "[INFO] User accepted the path. Exiting loop."
        echo "================================================="
        echo " Manual Pipeline finished successfully!"
        echo "  Manual Path/GIF: ${MANUAL_PATH_PREFIX}_path_mask.nii.gz / ${MANUAL_PATH_PREFIX}_spin_mip_y.gif"
        echo "================================================="
        break
    else
        echo "[INFO] User rejected the path. Restarting ROI Selection..."
    fi

done