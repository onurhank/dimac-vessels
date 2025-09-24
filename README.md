# 🧠 dimac-vessels

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
**dimac-vessels** is a processing pipeline for vessel analysis combining **DIMAC** and **TOF angiography** data.
It provides tools to resample, align, and automatically extract vessel regions of interest (ROIs), as well as shortest-path–based vessel tracking.

---

## ✨ Features

* **Resampling & registration**
  Align DIMAC data to TOF angiography using ANTs/ITK.
* **Automatic ROI selection**
  Data-driven scoring system to identify vessel ROIs (`dimac_auto_roi_improved.py`).
* **Vessel path extraction**
  Graph-based shortest path computation for vessel tracing.
* **Command line interface (CLI)**
  Unified entry point for running the analysis from the terminal.
* **End-to-end pipeline**
  One-step execution via `run_vessel_analysis.sh`.

---

## 📂 Repository structure

```
dimac-vessels/
 ├─ README.md
 ├─ requirements.txt
 ├─ run_vessel_analysis.sh           # main pipeline wrapper
 └─ scripts/
      ├─ dimac_auto_roi_improved.py  # ROI auto-selection
      ├─ resample_dimac_to_tof.py    # resampling DIMAC to TOF
      ├─ tof_resampler.py            # TOF resampling helper
      ├─ vessel_cli.py               # vesselness computation
      └─ vessel_shortest_path.py     # shortest path extraction
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/onurhank/dimac-vessels.git
cd dimac-vessels
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

You will also need:

* [ANTs](http://stnava.github.io/ANTs/) (with `ResampleImageBySpacing`)
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (for `fslhd`, `fslstats`)

---

## 📥 Inputs

### Required

* **TOF angiography scan** (`--tof`)
  A 3D NIfTI (`.nii.gz`) from Time-of-Flight MRA.
  *Example:* `sub-001_TOF.nii.gz`

* **DIMAC BOLD for ACA** (`--dimac-aca`)
  The fMRI/DIMAC series that covers the **Anterior Cerebral Artery region**.
  *Example:* `sub-001_task-dimac_ACA_bold.nii.gz`

* **DIMAC BOLD for ICA** (`--dimac-ica`)
  The fMRI/DIMAC series that covers the **Internal Carotid Artery region**.
  *Example:* `sub-001_task-dimac_ICA_bold.nii.gz`

### Optional

* **ACA ROI mask** (`--aca-roi`)
  Binary NIfTI mask of ACA ROI. If omitted, the pipeline auto-generates it.
* **ICA ROI mask** (`--ica-roi`)
  Binary NIfTI mask of ICA ROI.
* **Subject ID** (`--sub-id`)
  Label used in outputs (default: `sub-default`).
* **Directories**

  * `--deriv-dir` (default: `derivatives`) → intermediate files
  * `--out-dir` (default: `analysis_output`) → vesselness maps, paths, renders

---

## 🚀 Usage

### Minimal example

```bash
bash run_vessel_analysis.sh \
  --tof /data/sub-001_TOF.nii.gz \
  --dimac-aca /data/sub-001_dimac_ACA_bold.nii.gz \
  --dimac-ica /data/sub-001_dimac_ICA_bold.nii.gz \
  --sub-id sub-001
```

### With pre-made ROIs

```bash
bash run_vessel_analysis.sh \
  --tof /data/sub-001_TOF.nii.gz \
  --dimac-aca /data/sub-001_dimac_ACA_bold.nii.gz \
  --dimac-ica /data/sub-001_dimac_ICA_bold.nii.gz \
  --aca-roi /data/sub-001_ACA_mask.nii.gz \
  --ica-roi /data/sub-001_ICA_mask.nii.gz \
  --sub-id sub-001
```

---

## 📊 Example outputs

* Resampled TOF (`*_isotropic.nii.gz`)
* Vesselness maps (`*_frangi_vesselness.nii.gz`)
* ROI masks (`*_roi_in_TOF_iso_space.nii.gz`)
* Path analysis results:

  * Mask: `*_path_mask.nii.gz`
  * Path length (mm): `*_length_mm.txt`
  * Renders: static PNG + spinning GIF

---
<img width="384" height="580" alt="Ekran Resmi 2025-09-24 14 35 33" src="https://github.com/user-attachments/assets/ecb8f81d-e5e2-4db6-9eee-7e04e391e7ff" />

## 🧩 Roadmap

* [ ] Unit tests for ROI and shortest path modules
* [ ] More documentation & usage examples
* [ ] Package for PyPI (`pip install dimac-vessels`)
* [ ] CI/CD with GitHub Actions

---

## 📜 License

MIT License © 2025 \[Your Name]
