import os
import sys
import subprocess
import shutil
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path

# --- HOW TO USE ---
# This is a command-line tool that automatically calculates mean images.
#
# --- EXAMPLES ---
#
# 1. Process just the 4D ICA BOLD file:
#    python transform_slabs.py \
#      --tof anat/TOF.nii.gz \
#      --ica-bold func/ICA_bold.nii.gz
#
# 2. Process the 4D ICA BOLD and its corresponding mask:
#    python transform_slabs.py \
#      --tof anat/TOF.nii.gz \
#      --ica-bold func/ICA_bold.nii.gz \
#      --ica-mask func/ICA_mask.nii.gz
#
# 3. Process everything for both slabs into a specific output directory:
#    python transform_slabs.py \
#      --tof anat/TOF.nii.gz \
#      --ica-bold func/ICA_bold.nii.gz \
#      --ica-mask func/ICA_mask.nii.gz \
#      --aca-bold func/ACA_bold.nii.gz \
#      --aca-mask func/ACA_mask.nii.gz \
#      --output-dir ./derivatives/resampled_to_tof
# ------------------

def check_dependencies():
    """Checks if required command-line tools are available."""
    print("--- Checking for dependencies... ---")
    if not all(shutil.which(cmd) for cmd in ['antsApplyTransforms', 'fslmaths']):
        print("\n[ERROR] 'antsApplyTransforms' or 'fslmaths' not found.")
        print("Please ensure FSL and ANTs are installed and in your system's PATH.")
        sys.exit(1)
    print("All dependencies found.")


def generate_mean_from_bold(bold_path, output_dir):
    """Calculates the mean from a 4D BOLD file."""
    print(f"\n--- Calculating mean for: {Path(bold_path).name} ---")
    temp_mean_path = output_dir / f"{Path(bold_path).name.replace('_bold.nii.gz', '_temp_mean.nii.gz')}"
    fsl_command = ['fslmaths', str(bold_path), '-Tmean', str(temp_mean_path)]
    subprocess.run(fsl_command, check=True)
    print(f"SUCCESS: Saved temporary mean to: {temp_mean_path.name}")
    return temp_mean_path


def sanitize_nifti_file(input_path, output_path, is_dimac=False):
    """Reads a NIfTI, fixes its header, and saves a clean version."""
    print(f"\n--- Sanitizing: {Path(input_path).name} ---")
    try:
        original_img = nib.load(input_path)
        original_data = original_img.get_fdata()
        true_affine = original_img.header.get_sform()
        
        if is_dimac and original_data.ndim == 2:
            print(f"INFO: Detected 2D slab. Reshaping to 3D.")
            original_data = original_data.reshape(original_data.shape + (1,))

        clean_img = nib.Nifti1Image(original_data.copy(), true_affine, original_img.header)
        clean_img.header.set_sform(true_affine, code=1)
        clean_img.header.set_qform(true_affine, code=1)
        clean_img.to_filename(output_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to sanitize {Path(input_path).name}. Error: {e}")
        return False


def transform_image(input_path, tof_sanitized_path, output_path, interpolation, is_dimac=True):
    """Runs the full sanitizing and resampling workflow for a single image."""
    base_name = Path(input_path).name.replace('.nii.gz', '')
    dir_name = Path(output_path).parent

    input_sanitized = dir_name / f"{base_name}_sanitized.nii.gz"
    resampled_temp = dir_name / f"{base_name}_resampled_temp.nii.gz"

    if not sanitize_nifti_file(input_path, input_sanitized, is_dimac=is_dimac): return None
    
    print(f"\n--- Resampling {Path(input_path).name} using {interpolation}... ---")
    ants_command = [
        'antsApplyTransforms', '-d', '3', '-i', str(input_sanitized),
        '-r', str(tof_sanitized_path), '-o', str(resampled_temp),
        '-t', 'identity', '--default-value', '0',
        '--interpolation', interpolation, '-e', '0', '-v', '0'
    ]
    subprocess.run(ants_command, check=True, capture_output=True)

    if interpolation != 'NearestNeighbor':
        print(f"--- Clamping negative values... ---")
        fsl_command = ['fslmaths', str(resampled_temp), '-thr', '0', str(output_path)]
        subprocess.run(fsl_command, check=True, capture_output=True)
        os.remove(resampled_temp)
    else:
        shutil.move(resampled_temp, output_path)

    os.remove(input_sanitized)
    print(f"[SUCCESS] Final output for {base_name} is ready at: {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A tool to robustly resample 4D DIMAC BOLD slabs and masks into TOF space.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--tof', type=str, required=True, help="Path to the reference Multi-slab TOF NIfTI file.")
    parser.add_argument('--ica-bold', type=str, help="Path to the 4D ICA DIMAC BOLD image.")
    parser.add_argument('--ica-mask', type=str, help="Path to the ICA DIMAC mask. Requires --ica-bold to be set.")
    parser.add_argument('--aca-bold', type=str, help="Path to the 4D ACA DIMAC BOLD image.")
    parser.add_argument('--aca-mask', type=str, help="Path to the ACA DIMAC mask. Requires --aca-bold to be set.")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the output files (default: current directory).")

    args = parser.parse_args()
    check_dependencies()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Validation ---
    if not (args.ica_bold or args.ica_mask or args.aca_bold or args.aca_mask):
        parser.error("At least one DIMAC input (--ica-bold, --aca-bold, etc.) must be provided.")
    if args.ica_mask and not args.ica_bold:
        parser.error("--ica-mask was provided, but the required --ica-bold is missing.")
    if args.aca_mask and not args.aca_bold:
        parser.error("--aca-mask was provided, but the required --aca-bold is missing.")

    for f in [p for p in vars(args).values() if isinstance(p, str)]:
        if not os.path.exists(f):
            print(f"\n[ERROR] Input file not found: {f}"); sys.exit(1)

    # --- Main Workflow ---
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Outputs will be saved to: {output_dir.resolve()} ---")

        temp_files_to_clean = []
        
        # Sanitize TOF
        tof_sanitized_path = output_dir / f"{Path(args.tof).name.replace('.nii.gz', '_sanitized.nii.gz')}"
        if not sanitize_nifti_file(args.tof, tof_sanitized_path, is_dimac=False): sys.exit(1)
        temp_files_to_clean.append(tof_sanitized_path)

        # Process ICA slab
        if args.ica_bold:
            temp_mean = generate_mean_from_bold(args.ica_bold, output_dir)
            temp_files_to_clean.append(temp_mean)
            output_path = output_dir / f"{Path(args.ica_bold).name.replace('_bold.nii.gz', '_mean_in_TOF_space.nii.gz')}"
            transform_image(temp_mean, tof_sanitized_path, output_path, 'BSpline[4]')
        if args.ica_mask:
            output_path = output_dir / f"{Path(args.ica_mask).name.replace('.nii.gz', '_in_TOF_space.nii.gz')}"
            transform_image(args.ica_mask, tof_sanitized_path, output_path, 'NearestNeighbor')

        # Process ACA slab
        if args.aca_bold:
            temp_mean = generate_mean_from_bold(args.aca_bold, output_dir)
            temp_files_to_clean.append(temp_mean)
            output_path = output_dir / f"{Path(args.aca_bold).name.replace('_bold.nii.gz', '_mean_in_TOF_space.nii.gz')}"
            transform_image(temp_mean, tof_sanitized_path, output_path, 'BSpline[4]')
        if args.aca_mask:
            output_path = output_dir / f"{Path(args.aca_mask).name.replace('.nii.gz', '_in_TOF_space.nii.gz')}"
            transform_image(args.aca_mask, tof_sanitized_path, output_path, 'NearestNeighbor')

        # Final Cleanup
        print("\n--- Cleaning up all temporary files... ---")
        for f in temp_files_to_clean:
            if os.path.exists(f):
                print(f"Removing: {f.name}")
                os.remove(f)

        print("\n--- All processing complete! ---")

    except (KeyboardInterrupt, EOFError):
        print("\n\nWorkflow cancelled by user. Exiting.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] A command-line tool failed ({e.args[0]}). Check its error messages.")
        sys.exit(1)
