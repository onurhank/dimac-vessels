import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.image as mpimg

from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class ReviewDialog(QDialog):
    def __init__(self, prefix):
        super().__init__()
        self.setWindowTitle("Review Final Path (PyQt5)")
        self.resize(1200, 900)

        # --- Data Loading ---
        self.gif_frames =[]
        try:
            gif = Image.open(prefix + "_spin_mip_y.gif")
            while True:
                self.gif_frames.append(np.array(gif.convert('RGBA')))
                gif.seek(gif.tell()+1)
        except EOFError:
            pass
        except Exception as e:
            print('Error loading GIF:', e)

        try:
            self.png_y = mpimg.imread(prefix + "_render_mip_y.png")
        except:
            self.png_y = np.zeros((10,10,4))
            
        try:
            self.png_x = mpimg.imread(prefix + "_render_mip_x.png")
        except:
            self.png_x = np.zeros((10,10,4))

        if not self.gif_frames:
            self.gif_frames =[self.png_y]

        # --- State ---
        self.mode = 0 # 0: GIF, 1: PNG Front, 2: PNG Side
        self.playing = True
        self.direction = 1
        self.idx = 0
        self.interval = 80
        self.result_code = 1 # Default NO (Retry)

        self.init_ui()

        # Animation Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.interval)

        # Trigger initial load
        self.load_view()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Toolbar & Canvas
        self.canvas = FigureCanvas(Figure(figsize=(10, 8), tight_layout=True))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.axis('off')
        
        self.toolbar = NavigationToolbar(self.canvas, self)

        # View Mode Dropdown
        self.combo_view = QComboBox()
        self.combo_view.addItems(["3D Spinning GIF", "Static PNG (Front View - Y Axis)", "Static PNG (Side View - X Axis)"])
        self.combo_view.currentIndexChanged.connect(self.change_view)
        self.combo_view.setStyleSheet("font-size: 14px; padding: 5px;")

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("<b>Select View:</b>"))
        top_bar.addWidget(self.combo_view)
        top_bar.addStretch()

        # Playback Controls
        playback_layout = QHBoxLayout()
        self.btn_play = QPushButton("⏸ Play/Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_rev = QPushButton("⏪ Reverse Spin")
        self.btn_rev.clicked.connect(self.toggle_reverse)
        self.btn_slow = QPushButton("🐢 Slower")
        self.btn_slow.clicked.connect(self.slower)
        self.btn_fast = QPushButton("🐇 Faster")
        self.btn_fast.clicked.connect(self.faster)

        playback_layout.addWidget(self.btn_play)
        playback_layout.addWidget(self.btn_rev)
        playback_layout.addWidget(self.btn_slow)
        playback_layout.addWidget(self.btn_fast)

        # Accept/Reject Decisions
        decision_layout = QHBoxLayout()
        self.btn_yes = QPushButton("✅ YES (Accept & Finish Pipeline)")
        self.btn_yes.setStyleSheet("background-color: lightgreen; font-weight: bold; font-size: 16px; padding: 15px;")
        self.btn_yes.clicked.connect(self.accept_path)

        self.btn_no = QPushButton("❌ NO (Reject & Retry Selection)")
        self.btn_no.setStyleSheet("background-color: salmon; font-weight: bold; font-size: 16px; padding: 15px;")
        self.btn_no.clicked.connect(self.reject_path)

        decision_layout.addWidget(self.btn_yes)
        decision_layout.addWidget(self.btn_no)

        # Assemble Window
        layout.addWidget(self.toolbar)
        layout.addLayout(top_bar)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(playback_layout)
        layout.addLayout(decision_layout)

    def change_view(self, index):
        self.mode = index
        self.load_view()

    def load_view(self):
        """Clearing the axis completely allows Matplotlib to automatically maximize the new image."""
        self.ax.clear()
        self.ax.axis('off')

        if self.mode == 0:
            self.img_display = self.ax.imshow(self.gif_frames[self.idx])
            self.enable_playback(True)
        elif self.mode == 1:
            self.img_display = self.ax.imshow(self.png_y)
            self.enable_playback(False)
        elif self.mode == 2:
            self.img_display = self.ax.imshow(self.png_x)
            self.enable_playback(False)

        self.canvas.draw_idle()

    def enable_playback(self, state):
        self.btn_play.setEnabled(state)
        self.btn_rev.setEnabled(state)
        self.btn_slow.setEnabled(state)
        self.btn_fast.setEnabled(state)

    def update_frame(self):
        if self.mode == 0 and self.playing:
            self.idx = (self.idx + self.direction) % len(self.gif_frames)
            self.img_display.set_data(self.gif_frames[self.idx])
            self.canvas.draw_idle()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("▶ Play" if not self.playing else "⏸ Pause")

    def toggle_reverse(self):
        self.direction *= -1

    def slower(self):
        self.interval += 20
        self.timer.setInterval(self.interval)

    def faster(self):
        self.interval = max(20, self.interval - 20)
        self.timer.setInterval(self.interval)

    def accept_path(self):
        self.result_code = 0
        self.accept()

    def reject_path(self):
        self.result_code = 1
        self.reject()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Prefix for the rendered path files")
    parser.add_argument("out_result", help="Path to write the 0/1 result")
    args = parser.parse_args()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    dialog = ReviewDialog(args.prefix)
    dialog.exec_()

    # Save the result so the bash wrapper knows what to do!
    with open(args.out_result, 'w') as f:
        f.write(str(dialog.result_code))

if __name__ == "__main__":
    main()
