# DIMAC Vessel Analysis Pipeline

This repository contains a Python-based pipeline for analyzing vessel structure from Time-of-Flight (TOF) and Dynamic Intracranial Magnetic Resonance Angiography (DIMAC) data. The workflow automates vesselness filtering, territory-constrained ROI detection, and shortest-path tracking between arterial territories.

## Features

-   **Automatic ROI Detection:** Uses Pulse-Power Ratio (PPR) and k-means clustering on 4D DIMAC data to identify arterial ROIs within predefined masks.
-   **Vesselness Filtering:** Applies Frangi and ITK-based Objectness filters to TOF scans to enhance vessel-like structures.
-   **Robust Resampling:** Includes scripts to resample data into a common space using ANTs.
-   **Shortest-Path Tracking:** Traces the most likely path between two ROIs (e.g., ACA and ICA) through the vesselness map.
-   **Automated Visualization:** Generates static PNGs and animated GIFs of the final vessel path for easy quality control.

## Analysis Workflow

The pipeline follows these main steps:
Isotropic TOF
|
v
Vesselness Map (Frangi)
|
+------> 5. Shortest Path Analysis ----> 6. Path Mask & Visuals (PNG/GIF)
|
DIMAC ROIs <---+
(ACA & ICA) |
| |
v |
Resample ROIs |
to TOF space --+
code
Code
## Prerequisites

-   ANTs (`ResampleImageBySpacing`, `antsApplyTransforms`)
-   FSL (`fslmaths`)
-   Python 3.8+

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The entire pipeline is orchestrated by the `run_vessel_analysis.sh` script.

1.  **Prepare your data:** Organize your input TOF, DIMAC, and mask files.

2.  **Configure the pipeline:** Open `run_vessel_analysis.sh` and edit the file paths and parameters in the `--- CONFIGURATION ---` section at the top.

3.  **Run the analysis:**
    ```bash
    chmod +x run_vessel_analysis.sh
    ./run_vessel_analysis.sh
    ```

The final outputs, including the path mask and visualizations, will be saved in the directory specified in the wrapper script (e.g., `analysis_output/`).
# dimac-vessels
