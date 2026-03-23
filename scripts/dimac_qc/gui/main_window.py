import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QSlider, QLabel, QComboBox, QProgressBar, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import nibabel as nib

from .workers import ComputeWorker

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class VoxelSelectorDialog(QDialog):
    """Interactive 5x5 Grid for Voxel Gradient QC"""
    def __init__(self, main_window, rec):
        super().__init__(main_window)
        self.main_window = main_window
        self.rec = rec
        self.setWindowTitle(f"Interactive Voxel Selector - Cluster {rec.cluster_id}")
        self.resize(1400, 900)
        
        layout = QVBoxLayout(self)
        
        # --- NEW EXPLICIT BUTTON PANEL ---
        btn_layout = QHBoxLayout()
        
        self.btn_auto = QPushButton("🔄 Reset to Ian's Auto-Selection")
        self.btn_auto.setStyleSheet("background-color: lightblue; font-weight: bold; padding: 8px;")
        self.btn_auto.clicked.connect(self.reset_to_auto)
        
        self.btn_all = QPushButton("✅ Force Select ALL Voxels")
        self.btn_all.setStyleSheet("background-color: lightgray; padding: 8px;")
        self.btn_all.clicked.connect(lambda: self.set_all_masks(True))
        
        self.btn_none = QPushButton("❌ Force Reject ALL Voxels")
        self.btn_none.setStyleSheet("background-color: lightgray; padding: 8px;")
        self.btn_none.clicked.connect(lambda: self.set_all_masks(False))
        
        self.btn_done = QPushButton("💾 ACCEPT & CLOSE")
        self.btn_done.setStyleSheet("background-color: lightgreen; font-weight: bold; padding: 8px;")
        self.btn_done.clicked.connect(self.accept) # Closes the popup safely

        btn_layout.addWidget(self.btn_auto)
        btn_layout.addWidget(self.btn_all)
        btn_layout.addWidget(self.btn_none)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_done)
        
        layout.addLayout(btn_layout)
        # ---------------------------------

        self.canvas = MplCanvas(self, width=12, height=8)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.fig = self.canvas.fig
        self.axes = self.fig.subplots(5, 5, sharex=True)
        
        # Updated Title to explain exactly what is happening
        self.fig.suptitle("Algorithm Auto-Selection is applied. GREEN = Passed Algorithm. RED = Failed.\nClick any graph to manually override the algorithm.", fontsize=13, fontweight='bold')
        self.axes_flat = self.axes.ravel()

        self.Nvox = self.rec.tc_2d.shape[0]

        if getattr(self.rec, 'custom_voxel_mask', None) is None:
            self.rec.custom_voxel_mask = (self.rec.pass_frac > 0.5).copy()

        self.draw_grid()
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def reset_to_auto(self):
        """Restores the exact selection chosen by Ian's MATLAB algorithm"""
        self.rec.custom_voxel_mask = (self.rec.pass_frac > 0.5).copy()
        self.update_main_window_state()
        self.draw_grid()

    def set_all_masks(self, state):
        """Forces all voxels to be True or False"""
        self.rec.custom_voxel_mask = np.full(self.Nvox, state, dtype=bool)
        self.update_main_window_state()
        self.draw_grid()

    def draw_grid(self):
        for v in range(25):
            ax = self.axes_flat[v]
            ax.clear()
            if v >= self.Nvox:
                ax.axis('off')
                continue

            y_data = self.rec.tc_2d[v]
            y_plot = y_data - np.mean(y_data)
            
            ax.plot(y_plot, 'b-', alpha=0.8, lw=1.2)

            if self.rec.maxind is not None and len(self.rec.maxind) > 0:
                valid_peaks =[p for p in self.rec.maxind if p < len(y_plot)]
                ax.plot(valid_peaks, y_plot[valid_peaks], 'ro', markersize=4, alpha=0.8)

            vox_ppr = self.main_window.processor.calc_ppr(y_data)
            p_pass = self.rec.pass_frac[v] * 100
            
            is_sel = self.rec.custom_voxel_mask[v]
            title_color = 'darkgreen' if is_sel else 'darkred'
            
            ax.set_title(f"Vox {v} | PPR: {vox_ppr:.2f} | Pass: {p_pass:.0f}%", color=title_color, fontsize=9, fontweight='bold')
            ax.set_facecolor('#eaffea' if is_sel else '#ffeaea')
            ax.set_yticks([]) 

            if v < 20: 
                ax.set_xticks([])

        self.axes_flat[0].set_xlim(0, min(100, self.rec.tc_2d.shape[1]))
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.93])
        self.canvas.draw()

    def on_click(self, event):
        if self.toolbar.mode != '': 
            return

        if event.inaxes in self.axes_flat:
            v = list(self.axes_flat).index(event.inaxes)
            if v < self.Nvox:
                # Toggle Mask manually
                self.rec.custom_voxel_mask[v] = not self.rec.custom_voxel_mask[v]
                self.update_main_window_state()
                self.draw_grid()

    def update_main_window_state(self):
        """Pushes the current voxel selection back to the main window"""
        self.main_window.gradient_filter_active = True
        self.main_window.btn_grad.setChecked(True)
        self.main_window.btn_grad.setText("Beat Filter: CUSTOM")
        self.main_window.btn_grad.setStyleSheet("background-color: gold")

        flat_mask = self.rec.raw_mask_dimac.ravel()
        idx_true = np.where(flat_mask)[0]
        new_flat = np.zeros_like(flat_mask)
        new_flat[idx_true[self.rec.custom_voxel_mask]] = True
        self.rec.grad_mask = new_flat.reshape(self.main_window.processor.dimac_shape_3d)

        if np.any(self.rec.custom_voxel_mask):
            self.rec.tc_grad = np.mean(self.main_window.processor.dimac_flat[self.rec.grad_mask.ravel()], axis=0)
            self.rec.ppr_grad = self.main_window.processor.calc_ppr(self.rec.tc_grad)
        else:
            self.rec.tc_grad = np.zeros_like(self.rec.tc)
            self.rec.ppr_grad = 0.0

        self.main_window.update_selection_view()


class MainWindow(QMainWindow):
    def __init__(self, processor, out_prefix, previous_side=None):
        super().__init__()
        self.processor = processor
        self.out_prefix = out_prefix
        self.previous_side = previous_side
        self.selected_side = None
        
        self.setWindowTitle(f"DIMAC QC - {out_prefix}")
        self.resize(1300, 850)

        self.selected_idx = None
        self.gradient_filter_active = False

        self.display_mode = "global_all"
        self.global_preview_idx = 0
        self.local_preview_frame_idx = 0
        self.spin_active = True
        self.show_slab_active = False
        
        # Lock to prevent timer crashes during computation
        self.is_computing = False

        self.init_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(self.processor.config.animation_interval_ms)

        self.recalculate()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # ---- LEFT PANEL ----
        left_panel = QVBoxLayout()
        
        self.lbl_thresh = QLabel("Threshold %: 90.0")
        self.slider_pc = QSlider(Qt.Horizontal)
        self.slider_pc.setRange(800, 999) 
        self.slider_pc.setValue(900)
        self.slider_pc.valueChanged.connect(lambda v: self.lbl_thresh.setText(f"Threshold %: {v/10.0:.1f}"))
        
        self.btn_apply = QPushButton("Apply Threshold")
        self.btn_apply.clicked.connect(self.recalculate)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["peak", "cluster"])

        self.btn_grad = QPushButton("Beat Filter: OFF")
        self.btn_grad.setCheckable(True)
        self.btn_grad.toggled.connect(self.toggle_filter)

        # ADDED VOXEL SELECTOR BUTTON
        self.btn_qc = QPushButton("Voxel Selector (5x5)")
        self.btn_qc.setStyleSheet("background-color: thistle")
        self.btn_qc.clicked.connect(self.show_gradient_qc)

        self.btn_spin = QPushButton("Pause 3D Spin")
        self.btn_spin.clicked.connect(self.toggle_spin)
        
        self.btn_slab = QPushButton("Toggle DIMAC Slab")
        self.btn_slab.clicked.connect(self.toggle_slab)

        self.btn_save = QPushButton("Confirm & Save ROI")
        self.btn_save.setStyleSheet("background-color: lightgreen; font-weight: bold; padding: 10px;")
        self.btn_save.clicked.connect(self.save_and_close)

        self.progress_bar = QProgressBar()
        self.lbl_status = QLabel("Ready.")

        left_panel.addWidget(self.lbl_thresh)
        left_panel.addWidget(self.slider_pc)
        left_panel.addWidget(self.combo_mode)
        left_panel.addWidget(self.btn_apply)
        left_panel.addWidget(self.btn_grad)
        left_panel.addWidget(self.btn_qc) # Added to layout
        left_panel.addWidget(self.btn_spin)
        left_panel.addWidget(self.btn_slab)
        left_panel.addStretch()
        left_panel.addWidget(self.btn_save) 
        left_panel.addWidget(self.lbl_status)
        left_panel.addWidget(self.progress_bar)

        # ---- RIGHT PANEL ----
        right_panel = QVBoxLayout()
        top_plots = QHBoxLayout()
        self.canvas_dimac = MplCanvas(self, width=5, height=5)
        self.canvas_vess = MplCanvas(self, width=5, height=5)
        top_plots.addWidget(self.canvas_dimac)
        top_plots.addWidget(self.canvas_vess)

        self.canvas_tc = MplCanvas(self, width=10, height=3)

        right_panel.addLayout(top_plots)
        right_panel.addWidget(self.canvas_tc)

        self.canvas_vess.axes.axis("off")
        self.img_vess = self.canvas_vess.axes.imshow(np.zeros((400, 400, 3), dtype=np.uint8))

        self.canvas_dimac.mpl_connect("button_press_event", self.on_dimac_click)
        self.canvas_vess.mpl_connect("button_press_event", self.on_vess_click)

        layout.addLayout(left_panel, 1)  
        layout.addLayout(right_panel, 4) 

    def show_gradient_qc(self):
        """Opens the 5x5 Grid UI"""
        if self.selected_idx is None:
            QMessageBox.warning(self, "Warning", "Please select a cluster first!")
            return
            
        rec = self.processor.cluster_records[self.selected_idx]
        if rec.maxind is None or len(rec.maxind) < 3:
            QMessageBox.information(self, "Info", "Not enough beats detected to run Gradient QC viewer.")
            return

        self.voxel_dialog = VoxelSelectorDialog(self, rec)
        self.voxel_dialog.show()

    def toggle_spin(self):
        self.spin_active = not self.spin_active
        self.btn_spin.setText("Resume 3D Spin" if not self.spin_active else "Pause 3D Spin")

    def toggle_slab(self):
        self.show_slab_active = not self.show_slab_active

    def recalculate(self):
        self.is_computing = True
        percentile = self.slider_pc.value() / 10.0
        mode = self.combo_mode.currentText()
        
        self.btn_apply.setEnabled(False)
        self.worker = ComputeWorker(self.processor, percentile, mode)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_compute_finished)
        self.worker.start()

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(text)

    def on_compute_finished(self):
        self.is_computing = False
        self.global_preview_idx = 0
        self.local_preview_frame_idx = 0
        
        self.btn_apply.setEnabled(True)
        self.lbl_status.setText("Compute Finished.")
        self.progress_bar.setValue(100)
        self.display_mode = "global_all"
        self.selected_idx = None
        self.draw_dimac_panel()

    def update_animation(self):
        if getattr(self, 'is_computing', False): return
        if not self.processor.has_vess or not self.processor.global_base_frames: return

        if self.display_mode.startswith("global"):
            if self.global_preview_idx >= len(self.processor.global_base_frames): self.global_preview_idx = 0
                
            idx = self.global_preview_idx
            base_img = self.processor.global_base_frames[idx].copy()
            coords = self.processor.global_frame_coords[idx]
            
            if self.show_slab_active and self.processor.global_slab_masks:
                base_img = self.processor._blend_green_slab(base_img, self.processor.global_slab_masks[idx])

            if self.display_mode == "global_all":
                img = self.processor._burn_spheres_into_image(base_img, coords[:, 0], coords[:, 1], self.processor.global_cand_colors_rgb, radius=3)
                self.canvas_vess.axes.set_title("Global Vessels (All)")
            elif self.display_mode == "global_single" and self.selected_idx is not None:
                img = self.processor._burn_spheres_into_image(base_img, [coords[self.selected_idx, 0]], [coords[self.selected_idx, 1]],[self.processor.global_cand_colors_rgb[self.selected_idx]], radius=6)
                self.canvas_vess.axes.set_title("Global Vessels (Selected)")
            else:
                img = base_img
                
            self.img_vess.set_data(img)
            if self.spin_active: 
                self.global_preview_idx = (self.global_preview_idx + 1) % len(self.processor.global_base_frames)

        elif self.display_mode == "local" and self.selected_idx is not None:
            rec = self.processor.cluster_records[self.selected_idx]
            active_mask = rec.grad_mask if self.gradient_filter_active and rec.grad_mask is not None else rec.raw_mask_dimac
            scale_idx = self.processor.config.local_default_crop_scale_idx
            crop_scale = self.processor.config.local_crop_scales[scale_idx]
            
            frames, masks = self.processor.get_local_frames_for_record(rec, crop_scale, active_mask, self.gradient_filter_active)
            
            if frames:
                if self.local_preview_frame_idx >= len(frames): self.local_preview_frame_idx = 0
                    
                idx = self.local_preview_frame_idx
                base_img = frames[idx].copy()
                if self.show_slab_active and masks:
                    base_img = self.processor._blend_green_slab(base_img, masks[idx])

                self.img_vess.set_data(base_img)
                self.canvas_vess.axes.set_title(f"Local ROI: Cluster {rec.cluster_id}")
                
                if self.spin_active: 
                    self.local_preview_frame_idx = (self.local_preview_frame_idx + 1) % len(frames)

        self.canvas_vess.draw_idle()

    def draw_dimac_panel(self):
        ax = self.canvas_dimac.axes
        ax.clear()
        ax.imshow(np.max(self.processor.mean_img, axis=2).T, origin="lower", cmap="gray")
        
        if self.processor.cluster_records:
            peak_coords = np.array([r.peak_coord_dimac for r in self.processor.cluster_records])
            ax.scatter(
                peak_coords[:, 0], peak_coords[:, 1], 
                c=[r.score for r in self.processor.cluster_records],
                cmap="autumn", vmin=0, vmax=1, s=20, edgecolors="black"
            )
        ax.set_title("DIMAC Axial (1 Click = Select | 2 Clicks = Local View)")
        ax.axis("off")
        
        if self.previous_side:
            ax.text(0.5, 0.96, f"REMINDER: Pick the {self.previous_side} side!", 
                    transform=ax.transAxes, color="white", fontsize=12, fontweight="bold", 
                    ha="center", va="top", bbox=dict(boxstyle="round,pad=0.4", fc="red", ec="none", alpha=0.9))
                    
        self.canvas_dimac.draw()

    def toggle_filter(self, checked):
        self.gradient_filter_active = checked
        if checked:
            self.btn_grad.setText("Beat Filter: ON")
            self.btn_grad.setStyleSheet("background-color: gold")
        else:
            self.btn_grad.setText("Beat Filter: OFF")
            self.btn_grad.setStyleSheet("")
        self.update_selection_view()

    def on_dimac_click(self, event):
        if not event.inaxes or not self.processor.cluster_records: return
        
        peak_coords = np.array([r.peak_coord_dimac for r in self.processor.cluster_records])
        dist2 = (peak_coords[:, 0] - event.xdata)**2 + (peak_coords[:, 1] - event.ydata)**2
        idx = int(np.argmin(dist2))

        if np.sqrt(dist2[idx]) > 20: return 

        self.selected_idx = idx
        self.draw_dimac_panel() 
        
        rec = self.processor.cluster_records[idx]
        self.canvas_dimac.axes.plot(rec.peak_coord_dimac[0], rec.peak_coord_dimac[1], 
                                    marker="o", markersize=15, fillstyle="none", 
                                    markeredgewidth=2.4, markeredgecolor="cyan")
        self.canvas_dimac.draw()
        
        if event.dblclick:
            self.display_mode = "local"
        else:
            self.display_mode = "global_single"
            
        self.update_selection_view()

    def on_vess_click(self, event):
        if event.dblclick:
            if self.display_mode == "local":
                self.display_mode = "global_single"
            elif self.display_mode.startswith("global") and self.selected_idx is not None:
                self.display_mode = "local"
        else:
            self.toggle_spin()

    def update_selection_view(self):
        if self.selected_idx is None: return
        rec = self.processor.cluster_records[self.selected_idx]
        
        ax = self.canvas_tc.axes
        ax.clear()
        
        tc = rec.tc_grad if self.gradient_filter_active else rec.tc
        ppr = rec.ppr_grad if self.gradient_filter_active else rec.ppr
        
        ax.plot(np.arange(len(tc)) * self.processor.tr, tc, 'k-')
        filt_tag = "[FILTERED]" if self.gradient_filter_active else ""
        ax.set_title(f"Cluster {rec.cluster_id} | Score: {rec.score:.2f} | PPR: {ppr:.3f} {filt_tag}")
        ax.grid(True)
        self.canvas_tc.draw()

    def _determine_laterality(self, x_coord):
        mid_x = self.processor.dimac_shape_3d[0] / 2.0
        try: 
            x_code = nib.aff2axcodes(self.processor.dimac_nii.affine)[0]
        except Exception: 
            x_code = "R" 
        if x_code == "R": return "LEFT" if x_coord < mid_x else "RIGHT"
        elif x_code == "L": return "RIGHT" if x_coord < mid_x else "LEFT"
        return "LEFT" if x_coord < mid_x else "RIGHT"

    def save_and_close(self):
        if self.selected_idx is None:
            QMessageBox.warning(self, "Warning", "Please select a cluster first!")
            return
            
        rec = self.processor.cluster_records[self.selected_idx]
        self.selected_side = self._determine_laterality(rec.peak_coord_dimac[0])

        if self.gradient_filter_active and getattr(rec, 'grad_mask', None) is not None:
            mask_to_use = rec.grad_mask
        else:
            mask_to_use = rec.raw_mask_dimac
            
        mask = mask_to_use.astype(np.uint8)
        
        header = self.processor.dimac_nii.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_data_shape(mask.shape)
        
        out_path = f"{self.out_prefix}_roi.nii.gz"
        nib.save(nib.Nifti1Image(mask, self.processor.dimac_nii.affine, header), out_path)
        
        print(f"\nSUCCESS! ROI saved to: {out_path}")
        self.close()
