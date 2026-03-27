## 🛠️ Environment Preparation & Installation

To run the `dimac-vessels` pipeline smoothly, you need a combination of standard neuroimaging tools and a modern Python environment.

### Step 1: Install External Neuroimaging Dependencies
The bash wrapper relies on two industry-standard neuroimaging suites. Please ensure they are installed and added to your system `$PATH`:
* **[FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)** (Specifically: `fslmaths`, `fslstats`)
* **[ANTs](https://github.com/ANTsX/ANTs)** (Specifically: `ResampleImageBySpacing`, `antsApplyTransforms`)
* **[MRIcroGL](https://www.nitrc.org/projects/mricrogl)** *(Optional)*: For 3D overlay rendering during QC.

### Step 2: Create a Python Environment
It is highly recommended to use **Conda** to manage your environment so the GUI libraries (PyQt5) do not conflict with your system Python.

```bash
# 1. Create a new conda environment named 'dimac_env' with Python 3.10
conda create -n dimac_env python=3.10 -y

# 2. Activate the environment
conda activate dimac_env

# 3. Clone this repository
git clone https://github.com/onurhank/dimac-vessels.git
cd dimac-vessels

# 4. Install the required Python packages
pip install -r requirements.txt
