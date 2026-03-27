#!/usr/bin/env python3
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QPlainTextEdit, QMessageBox, QGroupBox)
from PyQt5.QtCore import QProcess, Qt

class PipelineLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIMAC Vessels - Pipeline Launcher")
        self.resize(900, 700)

        self.process = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Input Form Group ---
        form_group = QGroupBox("Configuration")
        form_layout = QVBoxLayout()

        # Subject ID
        self.sub_id_input = self.create_input_row(form_layout, "Subject ID:", "1151682_test", browse=False)

        # File Inputs
        self.tof_input = self.create_input_row(form_layout, "TOF NIfTI:", "Select TOF .nii.gz file...")
        self.aca_input = self.create_input_row(form_layout, "DIMAC ACA:", "Select ACA .nii.gz file...")
        self.ica_input = self.create_input_row(form_layout, "DIMAC ICA:", "Select ICA .nii.gz file...")
        self.mricrogl_input = self.create_input_row(form_layout, "MRIcroGL Exe:", "Path to MRIcroGL executable (Optional)")

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # --- Run Button ---
        self.btn_run = QPushButton("🚀 START PIPELINE")
        self.btn_run.setStyleSheet("background-color: lightgreen; font-weight: bold; font-size: 16px; padding: 10px;")
        self.btn_run.clicked.connect(self.run_pipeline)
        layout.addWidget(self.btn_run)

        # --- Terminal Output Area ---
        log_group = QGroupBox("Console Output")
        log_layout = QVBoxLayout()
        self.console_output = QPlainTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        log_layout.addWidget(self.console_output)
        log_group.setLayout(log_layout)
        
        layout.addWidget(log_group, stretch=1)

    def create_input_row(self, parent_layout, label_text, placeholder, browse=True):
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(120)
        
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        
        row_layout.addWidget(label)
        row_layout.addWidget(line_edit)

        if browse:
            btn_browse = QPushButton("Browse...")
            btn_browse.clicked.connect(lambda: self.browse_file(line_edit))
            row_layout.addWidget(btn_browse)

        parent_layout.addLayout(row_layout)
        return line_edit

    def browse_file(self, line_edit):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        if filepath:
            line_edit.setText(filepath)

    def run_pipeline(self):
        sub_id = self.sub_id_input.text().strip()
        tof_path = self.tof_input.text().strip()
        aca_path = self.aca_input.text().strip()
        ica_path = self.ica_input.text().strip()
        mri_path = self.mricrogl_input.text().strip()

        if not sub_id or not tof_path or not aca_path or not ica_path:
            QMessageBox.warning(self, "Error", "Subject ID, TOF, ACA, and ICA paths are strictly required!")
            return

        # Check if the bash script exists
        bash_script = "run_vessel_analysis.sh"
        if not os.path.exists(bash_script):
            QMessageBox.critical(self, "Error", f"Cannot find '{bash_script}' in the current directory.\nPlease run this launcher from the dimac-vessels folder.")
            return

        # Build command array
        cmd_args =[
            "bash", bash_script,
            "--sub-id", sub_id,
            "--tof", tof_path,
            "--dimac-aca", aca_path,
            "--dimac-ica", ica_path
        ]
        
        if mri_path:
            cmd_args.extend(["--mricrogl-exe", mri_path])

        self.console_output.clear()
        self.console_output.appendPlainText(f"> Running command: {' '.join(cmd_args)}\n")

        # Disable button while running
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ PIPELINE RUNNING...")
        self.btn_run.setStyleSheet("background-color: lightyellow; font-weight: bold; font-size: 16px; padding: 10px;")

        # Start background process
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels) # Merge Stdout and Stderr
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.process.start(cmd_args[0], cmd_args[1:])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout_text = bytes(data).decode("utf8")
        self.console_output.insertPlainText(stdout_text)
        self.console_output.ensureCursorVisible() # Auto-scroll to bottom

    def process_finished(self, exit_code, exit_status):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("🚀 START PIPELINE")
        self.btn_run.setStyleSheet("background-color: lightgreen; font-weight: bold; font-size: 16px; padding: 10px;")
        
        if exit_code == 0:
            self.console_output.appendPlainText("\n\n[SUCCESS] Pipeline finished perfectly!")
        else:
            self.console_output.appendPlainText(f"\n\n[ERROR] Pipeline exited with code {exit_code}")

def main():
    app = QApplication(sys.argv)
    window = PipelineLauncher()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
