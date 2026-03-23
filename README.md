# 🧠 dimac-vessels

**dimac-vessels** is a comprehensive processing pipeline for cerebrovascular analysis, combining DIMAC (fMRI) and Time-of-Flight (TOF) angiography data. It provides automated tools to resample, align, extract vessel Regions of Interest (ROIs), and perform shortest-path–based vessel tracking, all integrated with a **custom PyQt5 interactive UI** for quality control.

## ✨ Key Features
* **Resampling & Registration**: Align DIMAC data to TOF angiography using ANTs/ITK.
* **Automatic ROI Selection**: Data-driven scoring system to identify ACA/ICA vessel ROIs.
* **Interactive Quality Control (GUI)**: Native PyQt5 tool featuring 3D rotating MIP views, beat-to-beat gradient filtering, and an interactive 5x5 Voxel Signal Selector.
* **Vessel Path Extraction**: Graph-based shortest path computation for accurate vessel tracing.
* **Interactive Visual Review (GUI)**: PyQt5 playback viewer for 3D spinning GIFs and static 2D MIPs to manually accept/reject extracted paths.
* **End-to-End Pipeline**: One-step execution via `run_vessel_analysis.sh` with a unified CLI.

## 📂 Repository Structure
```text
dimac-vessels/
 ├─ README.md
 ├─ requirements.txt
 ├─ run_vessel_analysis.sh           # Main pipeline wrapper (Steps 1-9)
 └─ scripts/
      ├─ dimac_qc/                   # PyQt5 Package for Interactive QC
      │    ├─ core/                  # Data loading, math, gradient search algorithms
      │    └─ gui/                   # MainWindow UI, multi-threading workers
      ├─ review_path_gui.py          # PyQt5 Path Review & Playback GUI
      ├─ dimac_auto_roi_improved.py  # ROI auto-selection
      ├─ resample_dimac_to_tof.py    # Resampling DIMAC to TOF
      ├─ vessel_cli.py               # Frangi vesselness computation
      └─ vessel_shortest_path.py     # Shortest path extraction



⚙️ Installation
Clone the repository:
code
Bash
git clone https://github.com/onurhank/dimac-vessels.git
cd dimac-vessels
Install Python dependencies:
code
Bash
pip install -r requirements.txt
(Note: This includes PyQt5, numpy, scipy, scikit-image, matplotlib, nibabel)
External Dependencies:
ANTs (for ResampleImageBySpacing, antsApplyTransforms)
FSL (for fslhd, fslstats, fslmaths)
MRIcroGL (Optional, for 3D overlay rendering during QC)
⚠️ HPC / Remote Server Notice: Because Steps 6 and 9 launch native PyQt5 Graphical User Interfaces, if you are running this on a headless HPC cluster, you must connect with X11 forwarding enabled (e.g., ssh -Y user@server or ssh -X user@server).
📥 Inputs
Required:
TOF angiography scan (--tof): A 3D NIfTI (.nii.gz) from Time-of-Flight MRA.
DIMAC ACA (--dimac-aca): The fMRI/DIMAC series covering the Anterior Cerebral Artery.
DIMAC ICA (--dimac-ica): The fMRI/DIMAC series covering the Internal Carotid Artery.
Optional/Directories:
Subject ID (--sub-id): Label used in outputs (default: sub-default).
--deriv-dir (default: derivatives/) → Intermediate files, temp states.
--out-dir (default: analysis_output/) → Final vesselness maps, paths, GIFs.
🚀 Usage
Launch the entire 9-step automated pipeline with a single command. The pipeline will automatically pause and open the PyQt5 Interactive QC window (Step 6) and the Review Player (Step 9) when user verification is needed.
code
Bash
bash run_vessel_analysis.sh \
  --tof /data/sub-001_TOF.nii.gz \
  --dimac-aca /data/sub-001_dimac_ACA_bold.nii.gz \
  --dimac-ica /data/sub-001_dimac_ICA_bold.nii.gz \
  --sub-id sub-001
(Note: To run the pipeline entirely headless without UI interruptions, pass the --no-gui flag).
📊 Outputs
Resampled TOF (*_isotropic.nii.gz)
Vesselness Maps (*_frangi_vesselness.nii.gz)
ROI Masks (*_roi_in_TOF_iso_space.nii.gz)
Path Analysis Results:
Binary Path Mask: *_path_mask.nii.gz
Path length calculations: *_length_mm.txt
Rendered QA Images: Static PNGs (X/Y axes) + Spinning 3D GIFs
🧩 Roadmap

Unit tests for ROI and shortest path modules

Advanced caching (LRU) to minimize memory footprint on large datasets

Package for PyPI (pip install dimac-vessels)

CI/CD with GitHub Actions

Docker Containerization
