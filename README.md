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
