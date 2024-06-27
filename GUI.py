import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QComboBox,
    QCheckBox,
    QWidget,
    QHBoxLayout,
    QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dataclasses import dataclass
from typing import Union
import numpy.typing as npt


from src.data_operations.load_dicom import load_patient_scan


# Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.patient_scan = None

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QVBoxLayout(central_widget)
        control_layout = QHBoxLayout()

        # Load button
        load_button = QPushButton("Load Volume")
        load_button.clicked.connect(self.load_volume)
        control_layout.addWidget(load_button)

        # Slice slider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.update_slice)
        control_layout.addWidget(self.slice_slider)

        # Plane dropdown
        self.plane_dropdown = QComboBox()
        self.plane_dropdown.addItems(["Axial", "Sagittal", "Coronal"])
        self.plane_dropdown.setEnabled(False)
        self.plane_dropdown.currentTextChanged.connect(self.update_plane)
        control_layout.addWidget(self.plane_dropdown)

        # Mask toggle
        self.mask_toggle = QCheckBox("Toggle Mask")
        self.mask_toggle.setEnabled(False)
        self.mask_toggle.stateChanged.connect(self.update_mask)
        control_layout.addWidget(self.mask_toggle)

        # Add control layout to the main layout
        main_layout.addLayout(control_layout)

        # Canvas for displaying the image
        self.canvas = FigureCanvas(plt.Figure())
        main_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        self.ax.axis("off")

        self.current_slice = 0
        self.current_plane = "Axial"
        self.show_mask = False

    def load_volume(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.patient_scan = load_patient_scan(directory, preprocess=True)
            self.slice_slider.setMaximum(self.patient_scan.volume.shape[0] - 1)
            self.slice_slider.setEnabled(True)
            self.plane_dropdown.setEnabled(True)
            self.mask_toggle.setEnabled(True)
            self.update_slice(0)

    def update_slice(self, slice_index):
        self.current_slice = slice_index
        self.display_slice()

    def update_plane(self, plane):
        self.current_plane = plane
        self.update_slice(0)

    def update_mask(self, state):
        self.show_mask = state == Qt.Checked
        self.display_slice()

    def display_slice(self):
        if self.patient_scan is None:
            return

        plane_index = {"Axial": 0, "Sagittal": 1, "Coronal": 2}
        axis = plane_index[self.current_plane]
        self.slice_slider.setMaximum(self.patient_scan.volume.shape[axis])

        if axis == 0:
            slice_image = self.patient_scan.volume[self.current_slice, :, :]
            mask_image = (
                self.patient_scan.mask[self.current_slice, :, :]
                if self.show_mask
                else None
            )
        elif axis == 1:
            slice_image = self.patient_scan.volume[:, self.current_slice, :]
            mask_image = (
                self.patient_scan.mask[:, self.current_slice, :]
                if self.show_mask
                else None
            )
        else:
            slice_image = self.patient_scan.volume[:, :, self.current_slice]
            mask_image = (
                self.patient_scan.mask[:, :, self.current_slice]
                if self.show_mask
                else None
            )

        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(slice_image, cmap="gray")

        if mask_image is not None:
            self.ax.imshow(
                np.ma.masked_where(mask_image == 0, mask_image), cmap="jet", alpha=0.5
            )

        self.ax.set_title(f"Slice {self.current_slice} - {self.current_plane}")
        self.canvas.draw()


# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
