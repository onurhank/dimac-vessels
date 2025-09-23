#!/bin/bash
# ==============================================================================
#
#           DIMAC Vessel Analysis Pipeline - Main Wrapper Script
#
# This script orchestrates the entire vessel analysis workflow, from initial
# preprocessing and ROI generation to final shortest-path tracking and
# visualization.
#
# Workflow Steps:
#   1. Resample the anatomical TOF scan to be isotropic.
#   2. Calculate a vesselness map (Frangi) from the isotropic TOF.
#   3. (Conditional) Generate territory-constrained ROIs (ACA/ICA) from 4D DIMAC data.
#      - This step is skipped if --aca-roi and --ica-roi are provided.
#   4. Resample all ROIs into the common isotropic TOF space.
#   5. Find the shortest path between the ROIs and render PNG/GIF outputs.
#
# Usage Example (generating ROIs):
#   ./run_vessel_analysis.sh \
#       --tof data/MRI_tof.nii.gz \
#       --dimac-aca data/func/sub-XYZ_acq-dimacACA_bold.nii.gz \
#       --dimac-ica data/func/sub-XYZ_acq-dimacICA_bold.nii.gz \
#       --aca-mask data/masks/ACA_dimac.nii.gz \
#       --ica-mask data/masks/ICA_dimac.nii.gz
#
# Usage Example (with pre-existing DIMAC-space ROIs):
#   ./run_vessel_analysis.sh \
#       --tof data/MRI_tof.nii.gz \
#       --dimac-aca data/func/sub-XYZ_acq-dimacACA_bold.nii.gz \
#       --dimac-ica data/func/sub-XYZ_acq-dimacICA_bold.nii.gz \
#       --aca-roi my_preexisting_aca_roi_dimac.nii.gz --aca-roi-space dimac \
#       --ica-roi my_preexisting_ica_roi_dimac.nii.gz --ica-roi-space dimac
#
# Usage Example (with pre-existing TOF-space ROIs):
#   ./run_vessel_analysis.sh \
#       --tof data/MRI_tof.nii.gz \
#       --dimac-aca data/func/sub-XYZ_acq-dimacACA_bold.nii.gz \
#       --dimac-ica data/func/sub-XYZ_acq-dimacICA_bold.nii.gz \
#       --aca-roi my_preexisting_aca_roi_tof.nii.gz --aca-roi-space tof \
#       --ica-roi my_preexisting_ica_roi_tof.nii.gz --ica-roi-space tof
#
# ==============================================================================

# --- Script Setup ---
# Exit immediately if a command fails
set -e
# Treat unset variables as an error
set -u
# The exit status of a pipeline is the status of the last command to fail
set -o pipefail

# --- Default Configuration ---
# These are default values. Command-line arguments will override them.
TOF_RAW=""
DIMAC_ICA_BOLD=""
DIMAC_ACA_BOLD=""
ACA_TERRITORY_MASK="" # Used for auto-ROI if no --aca-roi
ICA_TERRITORY_MASK="" # Used for auto-ROI if no --ica-roi

ACA_ROI_INPUT=""      # Path to pre-existing ACA ROI (if provided)
ACA_ROI_INPUT_SPACE="" # Space of ACA_ROI_INPUT (dimac or tof)
ICA_ROI_INPUT=""      # Path to pre-existing ICA ROI (if provided)
ICA_ROI_INPUT_SPACE="" # Space of ICA_ROI_INPUT (dimac or tof)

# --- General Parameters ---
SUB_ID="sub-2843808" # Default subject ID, can be overridden or derived
SCRIPT_DIR="scripts"
DERIV_DIR="derivatives"
OUT_DIR="analysis_output"

# --- Algorithm-specific Parameters ---
PPR_THR=4
K_CLUSTERS=4
VESSELNESS_SCALES="1 2 3 4 5 6"
PATH_THICKEN_VOX=1

# ==============================================================================
# --- Command-Line Argument Parsing ---
# ==============================================================================

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --tof)
            TOF_RAW="$2"
            shift # past argument
            shift # past value
            ;;
        --dimac-aca)
            DIMAC_ACA_BOLD="$2"
            shift
            shift
            ;;
        --dimac-ica)
            DIMAC_ICA_BOLD="$2"
            shift
            shift
            ;;
        --aca-mask) # Territory mask for auto-ROI
            ACA_TERRITORY_MASK="$2"
            shift
            shift
            ;;
        --ica-mask) # Territory mask for auto-ROI
            ICA_TERRITORY_MASK="$2"
            shift
            shift
            ;;
        --aca-roi) # Pre-existing ACA ROI
            ACA_ROI_INPUT="$2"
            shift
            shift
            ;;
        --aca-roi-space) # Space of pre-existing ACA ROI
            ACA_ROI_INPUT_SPACE="$2"
            shift
            shift
            ;;
        --ica-roi) # Pre-existing ICA ROI
            ICA_ROI_INPUT="$2"
            shift
            shift
            ;;
        --ica-roi-space) # Space of pre-existing ICA ROI
            ICA_ROI_INPUT_SPACE="$2"
            shift
            shift
            ;;
        --sub-id) # Override default subject ID
            SUB_ID="$2"
            shift
            shift
            ;;
        --scripts-dir) # Override default scripts directory
            SCRIPT_DIR="$2"
            shift
            shift
            ;;
        --deriv-dir) # Override default derivatives directory
            DERIV_DIR="$2"
            shift
            shift
            ;;
        --out-dir) # Override default final output directory
            OUT_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ==============================================================================
# --- Pre-Flight Checks & Setup ---
# ==============================================================================

echo "--- Performing pre-flight checks... ---"

# Check for essential input files
if [ -z "${TOF_RAW}" ]; then echo "[ERROR] --tof is required." >&2; exit 1; fi
if [ -z "${DIMAC_ACA_BOLD}" ]; then echo "[ERROR] --dimac-aca is required." >&2; exit 1; fi
if [ -z "${DIMAC_ICA_BOLD}" ]; then echo "[ERROR] --dimac-ica is required." >&2; exit 1; fi

for f in "${TOF_RAW}" "${DIMAC_ACA_BOLD}" "${DIMAC_ICA_BOLD}"; do
    if [ ! -f "${f}" ]; then
        echo "[ERROR] Required input file not found: ${f}" >&2
        exit 1
    fi
done

# Check scripts directory
if [ ! -d "${SCRIPT_DIR}" ]; then
    echo "[ERROR] Scripts directory not found at: ${SCRIPT_DIR}" >&2
    exit 1
fi

# Determine if ROIs are pre-existing or need generation
ROIS_PROVIDED=false
if [ -n "${ACA_ROI_INPUT}" ] && [ -n "${ICA_ROI_INPUT}" ]; then
    ROIS_PROVIDED=true
    echo "Pre-existing ROIs provided. Skipping auto-ROI generation."

    for f in "${ACA_ROI_INPUT}" "${ICA_ROI_INPUT}"; do
        if [ ! -f "${f}" ]; then echo "[ERROR] Provided ROI file not found: ${f}" >&2; exit 1; fi
    done
    if [ -z "${ACA_ROI_INPUT_SPACE}" ] || [ -z "${ICA_ROI_INPUT_SPACE}" ]; then
        echo "[ERROR] When providing --aca-roi/--ica-roi, --aca-roi-space/--ica-roi-space (dimac or tof) must also be specified." >&2
        exit 1
    fi
    if ! [[ "${ACA_ROI_INPUT_SPACE}" =~ ^(dimac|tof)$ ]] || ! [[ "${ICA_ROI_INPUT_SPACE}" =~ ^(dimac|tof)$ ]]; then
        echo "[ERROR] ROI space must be 'dimac' or 'tof'." >&2
        exit 1
    fi
else
    echo "No pre-existing ROIs provided. Auto-generating ROIs."
    if [ -z "${ACA_TERRITORY_MASK}" ]; then echo "[ERROR] --aca-mask is required for auto-ROI generation." >&2; exit 1; fi
    if [ -z "${ICA_TERRITORY_MASK}" ]; then echo "[ERROR] --ica-mask is required for auto-ROI generation." >&2; exit 1; fi
    for f in "${ACA_TERRITORY_MASK}" "${ICA_TERRITORY_MASK}"; do
        if [ ! -f "${f}" ]; then echo "[ERROR] Required territory mask not found: ${f}" >&2; exit 1; fi
    done
fi

echo "All essential checks passed. Starting pipeline..."

# Create output directories if they don't exist
mkdir -p "${DERIV_DIR}/resampled_to_tof" "${OUT_DIR}"

# ==============================================================================
# --- PIPELINE STEPS ---
# ==============================================================================

echo
echo "================================================="
echo " STEP 1: Resample TOF to be isotropic"
echo "================================================="
TOF_ISOTROPIC="${DERIV_DIR}/$(basename "${TOF_RAW}" .nii.gz)_isotropic.nii.gz"
python "${SCRIPT_DIR}/tof_resampler.py" \
    --tof "${TOF_RAW}" \
    --out "${TOF_ISOTROPIC}"

echo
echo "================================================="
echo " STEP 2: Calculate Vesselness from Isotropic TOF"
echo "================================================="
VESSELNESS_PREFIX="${OUT_DIR}/tof_vesselness"
python "${SCRIPT_DIR}/vessel_cli.py" \
  --input "${TOF_ISOTROPIC}" \
  --output-prefix "${VESSELNESS_PREFIX}" \
  --normalize \
  --method frangi \
  --scales-mm ${VESSELNESS_SCALES} \
  --oof-bright \
  --thr-mode quantile \
  --thr-value-frangi 0.995

# Define the path to the Frangi map needed for the final step
FRANGI_VESSELNESS="${VESSELNESS_PREFIX}_frangi_vesselness.nii.gz"

# --- Conditional ROI Generation / Definition ---
CURRENT_ACA_ROI_PATH=""
CURRENT_ICA_ROI_PATH=""

if [ "${ROIS_PROVIDED}" = true ]; then
    echo
    echo "================================================="
    echo " STEP 3: Using pre-existing ROIs"
    echo "================================================="
    CURRENT_ACA_ROI_PATH="${ACA_ROI_INPUT}"
    CURRENT_ICA_ROI_PATH="${ICA_ROI_INPUT}"
    echo "  - ACA ROI input: ${CURRENT_ACA_ROI_PATH} (space: ${ACA_ROI_INPUT_SPACE})"
    echo "  - ICA ROI input: ${CURRENT_ICA_ROI_PATH} (space: ${ICA_ROI_INPUT_SPACE})"

else # ROIs need to be generated
    echo
    echo "================================================="
    echo " STEP 3: Generate ACA & ICA ROIs in DIMAC space"
    echo "================================================="
    # Generate ACA ROI
    python "${SCRIPT_DIR}/dimac_auto_roi_improved.py" \
      --dimac "${DIMAC_ACA_BOLD}" \
      --out "${DERIV_DIR}/${SUB_ID}_ACA_generated" \
      --vessel-mask "${ACA_TERRITORY_MASK}" \
      --auto-band --welch-in-gate \
      --k ${K_CLUSTERS} --ppr-thr ${PPR_THR}
    CURRENT_ACA_ROI_PATH="${DERIV_DIR}/${SUB_ID}_ACA_generated_roi.nii.gz"
    ACA_ROI_INPUT_SPACE="dimac" # Generated ROIs are always in DIMAC space

    # Generate ICA ROI
    python "${SCRIPT_DIR}/dimac_auto_roi_improved.py" \
      --dimac "${DIMAC_ICA_BOLD}" \
      --out "${DERIV_DIR}/${SUB_ID}_ICA_generated" \
      --vessel-mask "${ICA_TERRITORY_MASK}" \
      --auto-band --welch-in-gate \
      --k ${K_CLUSTERS} --ppr-thr ${PPR_THR}
    CURRENT_ICA_ROI_PATH="${DERIV_DIR}/${SUB_ID}_ICA_generated_roi.nii.gz"
    ICA_ROI_INPUT_SPACE="dimac" # Generated ROIs are always in DIMAC space

    echo "  - Generated ACA ROI: ${CURRENT_ACA_ROI_PATH}"
    echo "  - Generated ICA ROI: ${CURRENT_ICA_ROI_PATH}"
fi


echo
echo "================================================="
echo " STEP 4: Resample ROIs to Isotropic TOF Space"
echo "================================================="

# Define final output paths for the resampled ROIs
ACA_ROI_FINAL_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_ACA_roi_in_TOF_iso_space.nii.gz"
ICA_ROI_FINAL_TOF="${DERIV_DIR}/resampled_to_tof/${SUB_ID}_ICA_roi_in_TOF_iso_space.nii.gz"

# The resample_dimac_to_tof.py script is designed to take DIMAC BOLDs and
# masks and resample them to a TOF reference. It can also handle masks that
# are already in TOF space (it will simply resample them to the *specific*
# TOF reference grid, e.g., isotropic TOF). We pass the original DIMAC bold
# as a placeholder for context, even if the mask is already TOF-space.
python "${SCRIPT_DIR}/resample_dimac_to_tof.py" \
    --tof "${TOF_ISOTROPIC}" \
    --aca-bold "${DIMAC_ACA_BOLD}" \
    --aca-mask "${CURRENT_ACA_ROI_PATH}" \
    --ica-bold "${DIMAC_ICA_BOLD}" \
    --ica-mask "${CURRENT_ICA_ROI_PATH}" \
    --output-dir "${DERIV_DIR}/resampled_to_tof"

# NOTE: The 'resample_dimac_to_tof.py' script currently uses a fixed naming
# convention like "_mean_in_TOF_space.nii.gz" or "_in_TOF_space.nii.gz" for masks.
# We need to correctly identify its output path based on its internal logic.
# Assuming output names are based on input base name with "_in_TOF_space.nii.gz" suffix.
# We need to adjust these paths if 'resample_dimac_to_tof.py' uses different naming.
# Let's verify and override if needed.
TEMP_ACA_OUTPUT_FROM_RESAMPLE="${DERIV_DIR}/resampled_to_tof/$(basename "${CURRENT_ACA_ROI_PATH}" .nii.gz)_in_TOF_space.nii.gz"
TEMP_ICA_OUTPUT_FROM_RESAMPLE="${DERIV_DIR}/resampled_to_tof/$(basename "${CURRENT_ICA_ROI_PATH}" .nii.gz)_in_TOF_space.nii.gz"

# Move and rename for consistency
mv "${TEMP_ACA_OUTPUT_FROM_RESAMPLE}" "${ACA_ROI_FINAL_TOF}"
mv "${TEMP_ICA_OUTPUT_FROM_RESAMPLE}" "${ICA_ROI_FINAL_TOF}"

echo "  - Final ACA ROI (isotropic TOF space): ${ACA_ROI_FINAL_TOF}"
echo "  - Final ICA ROI (isotropic TOF space): ${ICA_ROI_FINAL_TOF}"

echo
echo "================================================="
echo " STEP 5: Find Shortest Path and Render Visuals"
echo "================================================="
PATH_PREFIX="${OUT_DIR}/aca_ica_path_analysis"
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
    --gif-yaw-start_deg 90 \
    --path-thicken-vox ${PATH_THICKEN_VOX}

echo
echo "================================================="
echo " Pipeline finished successfully!"
echo "================================================="
echo " Final path analysis outputs are in: ${OUT_DIR}"
echo " Check the GIF at: ${PATH_PREFIX}_spin_mip_y.gif"
echo "================================================="
