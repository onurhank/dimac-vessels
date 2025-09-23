# dimac-vessels

**dimac-vessels** is a processing pipeline for vessel analysis based on **DIMAC** and **TOF angiography** data.  
It provides tools to resample, align, and automatically extract vessel regions of interest (ROIs), as well as shortest path–based vessel tracking.

---

## ✨ Features
- **Resampling & registration**  
  Align DIMAC data to TOF angiography using ANTs/ITK.
- **Automatic ROI selection**  
  Data-driven scoring system to identify vessel ROIs (`dimac_auto_roi_improved.py`).
- **Vessel path extraction**  
  Graph-based shortest path computation for vessel tracing.
- **Command line interface (CLI)**  
  Unified entry point for running the analysis from the terminal.
- **End-to-end pipeline**  
  One-step execution via `run_vessel_analysis.sh`.

---

## 📂 Repository structure
```
dimac-vessels/
 ├─ .gitignore
 ├─ README.md
 ├─ run_vessel_analysis.sh           # main pipeline script
 └─ scripts/
      ├─ dimac_auto_roi_improved.py  # ROI auto-selection
      ├─ resample_dimac_to_tof.py    # resampling DIMAC to TOF
      ├─ tof_resampler.py            # helper for resampling
      ├─ vessel_cli.py               # CLI wrapper
      └─ vessel_shortest_path.py     # shortest path extraction
```

---

## 🚀 Quick start

### 1. Clone the repository
```bash
git clone https://github.com/onurhank/dimac-vessels.git
cd dimac-vessels
```

### 2. Install requirements
You will need:
- Python 3.8+
- [ANTs](http://stnava.github.io/ANTs/)  
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (optional, depending on workflow)  
- Python packages:
  ```bash
  pip install numpy scipy scikit-image scikit-learn nibabel
  ```

### 3. Run the analysis
```bash
bash run_vessel_analysis.sh \
  --dimac /path/to/dimac.nii.gz \
  --tof /path/to/tof.nii.gz \
  --output ./results
```

---

## 🛠 Scripts overview
- **`dimac_auto_roi_improved.py`** – automatic ROI detection using frequency/temporal features.  
- **`resample_dimac_to_tof.py`** – resample DIMAC to TOF resolution.  
- **`tof_resampler.py`** – utility functions for resampling.  
- **`vessel_cli.py`** – CLI wrapper for modular usage.  
- **`vessel_shortest_path.py`** – vessel path detection using shortest path algorithms.  

---

## 📊 Example output
- Resampled DIMAC aligned to TOF  
- ROI masks (`*_roi.nii.gz`)  
- Vessel path tracings  
- QC figures (optional)

---

## 📖 Roadmap
- [ ] Add unit tests for ROI and shortest path modules  
- [ ] Improve documentation with usage examples  
- [ ] Package for PyPI (`pip install dimac-vessels`)  
- [ ] Add CI/CD with GitHub Actions  

---

## 👤 Author
- **Onurhan Karatay** ([@onurhank](https://github.com/onurhank))  

---

## 📄 License
This project is licensed under the [Apache 2.0 License](LICENSE).
