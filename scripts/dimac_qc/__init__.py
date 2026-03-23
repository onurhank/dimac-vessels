import sys
from PyQt5.QtWidgets import QApplication
from .core.processor import DimacProcessor
from .gui.main_window import MainWindow

def run_qc(dimac_file, out_prefix, vesselness_file=None, tof_file=None, 
           auto_roi_file=None, previous_side=None, mricrogl_exe=None):
    
    # Check if a QApplication already exists (required by PyQt)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    print(f"Loading data for QC: {dimac_file}")
    processor = DimacProcessor(
        dimac_fname=dimac_file,
        vesselness_fname=vesselness_file,
        auto_roi_fname=auto_roi_file
    )
    
    window = MainWindow(processor, out_prefix, previous_side)
    window.show()
    app.exec_() # This blocks the bash script until the window is closed
    
    return window.selected_side