# gui/workers.py
from PyQt5.QtCore import QThread, pyqtSignal

class ComputeWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()

    def __init__(self, processor, percentile, mode):
        super().__init__()
        self.processor = processor
        self.percentile = percentile
        self.mode = mode

    def run(self):
        # Passes the progress signal as a callback to the processor
        self.processor.compute_pipeline(
            self.percentile, 
            self.mode, 
            progress_callback=self.progress.emit
        )
        self.finished.emit()