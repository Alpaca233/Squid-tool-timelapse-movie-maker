#!/usr/bin/env python3
"""
GUI for creating timelapse movies from calcium imaging data.

Features:
- Drag and drop for baseline and response folders
- Per-channel contrast control via integrated NDV viewer
- Adjustable frame interval (defaults to dt from acquisition parameters)
- Max intensity projection preview
- Batch processing of multiple folder pairs
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QSpinBox,
    QDoubleSpinBox, QGroupBox, QMessageBox, QProgressDialog, QFileDialog,
    QAbstractItemView, QScrollArea, QFrame, QSizePolicy, QTabWidget,
    QComboBox, QCheckBox, QSlider, QSplitter, QStyleFactory
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap

import numpy as np

# Try to import superqt range slider
try:
    from superqt import QRangeSlider
    RANGE_SLIDER_AVAILABLE = True
except ImportError:
    RANGE_SLIDER_AVAILABLE = False

from movie_maker_backend import (
    AcquisitionFolder, AcquisitionParams, ChannelSettings,
    FolderPairSettings, discover_channels_for_pair,
    compute_auto_contrast_for_pair, generate_movie_pair,
    load_preview_frame, get_max_intensity_projection,
    wavelength_to_rgb, apply_contrast, colorize_channel
)


def find_matching_response_folder(baseline_path: str) -> Optional[str]:
    """
    Auto-detect matching response folder for a baseline folder.

    Folder naming convention: {prefix}_{name}_{date}_{time}
    - Baseline prefixes: "before_", "baseline_"
    - Response prefixes: "after_", "response_"

    Strips the last two underscore-separated parts (date and time) to get the core name.

    Returns the matching response folder path if exactly one match is found,
    otherwise returns None.
    """
    baseline_dir = Path(baseline_path)
    if not baseline_dir.exists():
        return None

    parent_dir = baseline_dir.parent
    folder_name = baseline_dir.name

    baseline_prefixes = ["before_", "baseline_"]
    response_prefixes = ["after_", "response_"]

    core_name = folder_name
    matched_prefix = None

    # Strip baseline prefix
    for prefix in baseline_prefixes:
        if folder_name.lower().startswith(prefix):
            core_name = folder_name[len(prefix):]
            matched_prefix = prefix
            break

    if matched_prefix is None:
        # Not a baseline folder pattern
        return None

    # Strip last two underscore-separated parts (date and time)
    parts = core_name.rsplit('_', 2)
    if len(parts) >= 3:
        core_name = parts[0]

    # Now search for matching response folders
    matching_responses = []

    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue

        item_name = item.name

        # Check if it starts with a response prefix
        for resp_prefix in response_prefixes:
            if item_name.lower().startswith(resp_prefix):
                # Extract core name from response folder
                resp_core = item_name[len(resp_prefix):]

                # Strip last two parts from response
                resp_parts = resp_core.rsplit('_', 2)
                if len(resp_parts) >= 3:
                    resp_core = resp_parts[0]

                # Check if core names match (case-insensitive)
                if resp_core.lower() == core_name.lower():
                    matching_responses.append(str(item))
                break

    # Return only if exactly one match
    if len(matching_responses) == 1:
        return matching_responses[0]

    return None


class DropListWidget(QListWidget):
    """List widget that accepts drag and drop of folders and supports reordering."""

    items_changed = pyqtSignal()
    folders_dropped = pyqtSignal(list)  # Emits list of newly dropped folder paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.model().rowsInserted.connect(self.items_changed.emit)
        self.model().rowsRemoved.connect(self.items_changed.emit)
        self.model().rowsMoved.connect(self.items_changed.emit)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        elif event.source() == self:
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        elif event.source() == self:
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            dropped_paths = []
            for url in urls:
                path = url.toLocalFile()
                if os.path.isdir(path):
                    self.addItem(path)
                    dropped_paths.append(path)
            event.acceptProposedAction()
            if dropped_paths:
                self.folders_dropped.emit(dropped_paths)
        elif event.source() == self:
            super().dropEvent(event)
            self.items_changed.emit()


class ChannelContrastWidget(QWidget):
    """Compact widget for controlling contrast of a single channel with range slider."""

    contrast_changed = pyqtSignal()

    def __init__(self, channel_name: str, wavelength: Optional[int], parent=None):
        super().__init__(parent)
        self.channel_name = channel_name
        self.wavelength = wavelength
        self._updating = False  # Prevent circular updates

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(8)

        # Enable checkbox
        self.enabled_cb = QCheckBox()
        self.enabled_cb.setChecked(True)
        self.enabled_cb.stateChanged.connect(self.contrast_changed.emit)
        layout.addWidget(self.enabled_cb)

        # Channel color indicator
        rgb = wavelength_to_rgb(wavelength)
        color_style = f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border-radius: 3px;"
        color_label = QLabel()
        color_label.setFixedSize(12, 12)
        color_label.setStyleSheet(color_style)
        layout.addWidget(color_label)

        # Channel name (truncate if too long)
        display_name = channel_name if len(channel_name) <= 25 else channel_name[:22] + "..."
        name_label = QLabel(display_name)
        name_label.setToolTip(channel_name)
        name_label.setFixedWidth(160)
        layout.addWidget(name_label)

        # Min spinbox
        self.min_spin = QSpinBox()
        self.min_spin.setRange(0, 65535)
        self.min_spin.setValue(0)
        self.min_spin.setFixedWidth(60)
        self.min_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.min_spin.valueChanged.connect(self._on_spin_changed)
        layout.addWidget(self.min_spin)

        # Range slider (if available) or fallback to two sliders
        if RANGE_SLIDER_AVAILABLE:
            self.range_slider = QRangeSlider(Qt.Horizontal)
            self.range_slider.setRange(0, 65535)
            self.range_slider.setValue((0, 65535))
            self.range_slider.valueChanged.connect(self._on_range_slider_changed)
            layout.addWidget(self.range_slider, 1)
            self.min_slider = None
            self.max_slider = None
        else:
            # Fallback: use two regular sliders side by side
            self.range_slider = None
            slider_container = QWidget()
            slider_layout = QHBoxLayout(slider_container)
            slider_layout.setContentsMargins(0, 0, 0, 0)
            slider_layout.setSpacing(2)

            self.min_slider = QSlider(Qt.Horizontal)
            self.min_slider.setRange(0, 65535)
            self.min_slider.setValue(0)
            self.min_slider.valueChanged.connect(self._on_min_slider_changed)
            slider_layout.addWidget(self.min_slider, 1)

            self.max_slider = QSlider(Qt.Horizontal)
            self.max_slider.setRange(0, 65535)
            self.max_slider.setValue(65535)
            self.max_slider.valueChanged.connect(self._on_max_slider_changed)
            slider_layout.addWidget(self.max_slider, 1)

            layout.addWidget(slider_container, 1)

        # Max spinbox
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 65535)
        self.max_spin.setValue(65535)
        self.max_spin.setFixedWidth(60)
        self.max_spin.setButtonSymbols(QSpinBox.NoButtons)
        self.max_spin.valueChanged.connect(self._on_spin_changed)
        layout.addWidget(self.max_spin)

        # Auto button
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setFixedWidth(45)
        layout.addWidget(self.auto_btn)

    def _on_spin_changed(self):
        """Handle spinbox value change."""
        if self._updating:
            return
        self._updating = True
        if self.range_slider:
            self.range_slider.setValue((self.min_spin.value(), self.max_spin.value()))
        else:
            if self.min_slider:
                self.min_slider.setValue(self.min_spin.value())
            if self.max_slider:
                self.max_slider.setValue(self.max_spin.value())
        self._updating = False
        self.contrast_changed.emit()

    def _on_range_slider_changed(self, value):
        """Handle range slider value change."""
        if self._updating:
            return
        self._updating = True
        self.min_spin.setValue(value[0])
        self.max_spin.setValue(value[1])
        self._updating = False
        self.contrast_changed.emit()

    def _on_min_slider_changed(self, value):
        """Handle min slider change (fallback mode)."""
        if self._updating:
            return
        self._updating = True
        self.min_spin.setValue(value)
        self._updating = False
        self.contrast_changed.emit()

    def _on_max_slider_changed(self, value):
        """Handle max slider change (fallback mode)."""
        if self._updating:
            return
        self._updating = True
        self.max_spin.setValue(value)
        self._updating = False
        self.contrast_changed.emit()

    def get_settings(self) -> ChannelSettings:
        """Get current channel settings."""
        return ChannelSettings(
            name=self.channel_name,
            wavelength=self.wavelength,
            contrast_min=self.min_spin.value(),
            contrast_max=self.max_spin.value(),
            enabled=self.enabled_cb.isChecked()
        )

    def set_contrast(self, vmin: float, vmax: float):
        """Set contrast values."""
        self._updating = True
        self.min_spin.setValue(int(vmin))
        self.max_spin.setValue(int(vmax))
        if self.range_slider:
            self.range_slider.setValue((int(vmin), int(vmax)))
        else:
            if self.min_slider:
                self.min_slider.setValue(int(vmin))
            if self.max_slider:
                self.max_slider.setValue(int(vmax))
        self._updating = False
        self.contrast_changed.emit()


class PreviewWidget(QWidget):
    """Widget for displaying side-by-side preview of baseline and response."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pixmap = None  # Store original pixmap for rescaling

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.image_label.setMinimumSize(200, 150)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label, 1)

        # Timepoint sliders
        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 5, 0, 0)
        sliders_layout.setSpacing(5)

        # Baseline timepoint slider
        baseline_tp_layout = QHBoxLayout()
        baseline_tp_label = QLabel("Baseline Timepoint:")
        baseline_tp_label.setStyleSheet("color: #2196F3;")
        baseline_tp_label.setFixedWidth(150)
        baseline_tp_layout.addWidget(baseline_tp_label)
        self.baseline_tp_slider = QSlider(Qt.Horizontal)
        self.baseline_tp_slider.setMinimum(0)
        self.baseline_tp_slider.setMaximum(0)
        self.baseline_tp_slider.valueChanged.connect(self._on_baseline_tp_changed)
        baseline_tp_layout.addWidget(self.baseline_tp_slider, 1)
        self.baseline_tp_label = QLabel("0")
        self.baseline_tp_label.setFixedWidth(40)
        baseline_tp_layout.addWidget(self.baseline_tp_label)
        sliders_layout.addLayout(baseline_tp_layout)

        # Response timepoint slider
        response_tp_layout = QHBoxLayout()
        response_tp_label = QLabel("Response Timepoint:")
        response_tp_label.setStyleSheet("color: #FF9800;")
        response_tp_label.setFixedWidth(150)
        response_tp_layout.addWidget(response_tp_label)
        self.response_tp_slider = QSlider(Qt.Horizontal)
        self.response_tp_slider.setMinimum(0)
        self.response_tp_slider.setMaximum(0)
        self.response_tp_slider.valueChanged.connect(self._on_response_tp_changed)
        response_tp_layout.addWidget(self.response_tp_slider, 1)
        self.response_tp_label = QLabel("0")
        self.response_tp_label.setFixedWidth(40)
        response_tp_layout.addWidget(self.response_tp_label)
        sliders_layout.addLayout(response_tp_layout)

        layout.addWidget(sliders_widget)

        # State
        self.baseline_folder = None
        self.response_folder = None
        self.channel_settings = {}

    def set_folders(self, baseline_path: Optional[str], response_path: Optional[str]):
        """Set the folders to preview."""
        self.baseline_folder = AcquisitionFolder(baseline_path) if baseline_path else None
        self.response_folder = AcquisitionFolder(response_path) if response_path else None

        # Update timepoint sliders
        baseline_max = 0
        response_max = 0
        if self.baseline_folder and self.baseline_folder.timepoints:
            baseline_max = len(self.baseline_folder.timepoints) - 1
        if self.response_folder and self.response_folder.timepoints:
            response_max = len(self.response_folder.timepoints) - 1

        self.baseline_tp_slider.setMaximum(baseline_max)
        self.baseline_tp_slider.setValue(0)
        self.response_tp_slider.setMaximum(response_max)
        self.response_tp_slider.setValue(0)

    def set_channel_settings(self, settings: Dict[str, ChannelSettings]):
        """Set channel settings for preview."""
        self.channel_settings = settings
        self._update_preview()

    def _on_baseline_tp_changed(self, value):
        """Handle baseline timepoint change."""
        self.baseline_tp_label.setText(str(value))
        self._update_preview()

    def _on_response_tp_changed(self, value):
        """Handle response timepoint change."""
        self.response_tp_label.setText(str(value))
        self._update_preview()

    def _update_preview(self):
        """Update the preview image (side-by-side view)."""
        baseline_frame = None
        response_frame = None

        baseline_tp = self.baseline_tp_slider.value()
        response_tp = self.response_tp_slider.value()

        # Load baseline frame
        if self.baseline_folder and self.baseline_folder.timepoints:
            tp_idx = min(baseline_tp, len(self.baseline_folder.timepoints) - 1)
            actual_tp = self.baseline_folder.timepoints[tp_idx]
            baseline_frame = self._create_composite_frame(self.baseline_folder, actual_tp)

        # Load response frame
        if self.response_folder and self.response_folder.timepoints:
            tp_idx = min(response_tp, len(self.response_folder.timepoints) - 1)
            actual_tp = self.response_folder.timepoints[tp_idx]
            response_frame = self._create_composite_frame(self.response_folder, actual_tp)

        # Compose side by side image
        if baseline_frame is not None and response_frame is not None:
            # Make sure both have same height
            h1, h2 = baseline_frame.shape[0], response_frame.shape[0]
            if h1 != h2:
                max_h = max(h1, h2)
                if h1 < max_h:
                    pad = np.zeros((max_h - h1, baseline_frame.shape[1], 3), dtype=np.uint8)
                    baseline_frame = np.vstack([baseline_frame, pad])
                if h2 < max_h:
                    pad = np.zeros((max_h - h2, response_frame.shape[1], 3), dtype=np.uint8)
                    response_frame = np.vstack([response_frame, pad])
            # Add separator
            sep = np.full((baseline_frame.shape[0], 2, 3), 128, dtype=np.uint8)
            frame = np.hstack([baseline_frame, sep, response_frame])
        elif baseline_frame is not None:
            frame = baseline_frame
        elif response_frame is not None:
            frame = response_frame
        else:
            frame = None

        if frame is not None:
            self._display_frame(frame)
        else:
            self.image_label.setText("No data to display")

    def _create_composite_frame(self, folder: AcquisitionFolder, timepoint: int) -> Optional[np.ndarray]:
        """Create a composite frame from folder at given timepoint."""
        images = folder.load_timepoint_all_channels(timepoint)
        if not images:
            return None

        first_img = next(iter(images.values()))
        h, w = first_img.shape[:2]
        composite = np.zeros((h, w, 3), dtype=np.float32)

        for ch_name, img in images.items():
            if ch_name not in self.channel_settings:
                continue

            settings = self.channel_settings[ch_name]
            if not settings.enabled:
                continue

            img_contrast = apply_contrast(img, settings.contrast_min, settings.contrast_max)
            rgb = wavelength_to_rgb(settings.wavelength)
            colored = colorize_channel(img_contrast, rgb)
            composite += colored.astype(np.float32)

        return np.clip(composite, 0, 255).astype(np.uint8)

    def _display_frame(self, frame: np.ndarray):
        """Display a frame in the image label."""
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        # Need to copy the data since QImage doesn't own it
        frame_copy = frame.copy()
        qimg = QImage(frame_copy.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._scale_and_display()

    def _scale_and_display(self):
        """Scale the current pixmap to fit the label."""
        if self._current_pixmap is None:
            return

        label_size = self.image_label.size()
        if label_size.width() < 10 or label_size.height() < 10:
            return

        scaled = self._current_pixmap.scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        """Handle resize to rescale the image."""
        super().resizeEvent(event)
        self._scale_and_display()


class PairRowWidget(QWidget):
    """Widget for a single folder pair row."""

    selection_changed = pyqtSignal(int)  # Emits row index when selected

    def __init__(self, row_index: int, baseline_path: str, response_path: str, parent=None):
        super().__init__(parent)
        self.row_index = row_index
        self.baseline_path = baseline_path
        self.response_path = response_path

        # Load acquisition params for dt
        self.baseline_folder = AcquisitionFolder(baseline_path)
        self.dt_seconds = self.baseline_folder.params.dt_seconds

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Selection indicator
        self.select_btn = QPushButton()
        self.select_btn.setFixedSize(24, 24)
        self.select_btn.setCheckable(True)
        self.select_btn.setStyleSheet("""
            QPushButton { background: #444; border-radius: 12px; }
            QPushButton:checked { background: #2196F3; }
            QPushButton:hover:!checked { background: #555; }
            QPushButton:checked:hover { background: #1976D2; }
        """)
        self.select_btn.clicked.connect(lambda: self.selection_changed.emit(self.row_index))
        layout.addWidget(self.select_btn)

        # Row number
        row_label = QLabel(f"{row_index + 1}.")
        row_label.setFixedWidth(25)
        layout.addWidget(row_label)

        # Paths
        paths_widget = QWidget()
        paths_layout = QVBoxLayout(paths_widget)
        paths_layout.setContentsMargins(0, 0, 0, 0)
        paths_layout.setSpacing(2)

        baseline_label = QLabel(os.path.basename(baseline_path))
        baseline_label.setToolTip(baseline_path)
        baseline_label.setStyleSheet("color: #2196F3;")
        paths_layout.addWidget(baseline_label)

        response_label = QLabel(os.path.basename(response_path))
        response_label.setToolTip(response_path)
        response_label.setStyleSheet("color: #FF9800;")
        paths_layout.addWidget(response_label)

        paths_widget.setFixedWidth(250)
        layout.addWidget(paths_widget)

        # Frame interval display (read-only)
        dt_label = QLabel(f"dt: {self.dt_seconds:.2f}s")
        dt_label.setStyleSheet("color: #888;")
        dt_label.setToolTip("Frame interval (seconds) from acquisition parameters")
        layout.addWidget(dt_label)

        # Output FPS
        layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(5)
        self.fps_spin.setFixedWidth(45)
        self.fps_spin.setToolTip("Output video frames per second")
        layout.addWidget(self.fps_spin)

        # Channel info
        channels = self.baseline_folder.channels
        ch_label = QLabel(f"{len(channels)}ch")
        ch_label.setStyleSheet("color: #888;")
        ch_label.setFixedWidth(30)
        layout.addWidget(ch_label)

        layout.addStretch()

    def set_selected(self, selected: bool):
        """Set selection state."""
        self.select_btn.setChecked(selected)

    def get_settings(self, channel_settings: Dict[str, ChannelSettings]) -> FolderPairSettings:
        """Get settings for this pair."""
        return FolderPairSettings(
            baseline_path=self.baseline_path,
            response_path=self.response_path,
            channels=channel_settings,
            frame_interval_seconds=self.dt_seconds,
            output_fps=self.fps_spin.value()
        )


class MovieWorker(QThread):
    """Worker thread for creating movies."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, pairs: List[Tuple[FolderPairSettings, str]], output_dir: str):
        super().__init__()
        self.pairs = pairs  # List of (settings, name)
        self.output_dir = output_dir

    def run(self):
        total = len(self.pairs)
        success_count = 0
        errors = []

        for i, (settings, name) in enumerate(self.pairs):
            self.progress.emit(i, f"Processing {name}...")
            try:
                success = generate_movie_pair(
                    settings, self.output_dir, name
                )
                if success:
                    success_count += 1
                else:
                    errors.append(f"{name}: Failed to create movie")
            except Exception as e:
                errors.append(f"{name}: {str(e)}")

        self.progress.emit(total, "Done")

        if errors:
            self.finished.emit(False, f"Completed {success_count}/{total}.\nErrors:\n" + "\n".join(errors))
        else:
            self.finished.emit(True, f"Successfully created {success_count} movies.")


class MovieMakerGUI(QMainWindow):
    """Main GUI window for movie maker."""

    def __init__(self):
        super().__init__()
        self.pair_rows = []
        self.output_dir = None
        self.channel_widgets = {}
        self.current_channel_settings = {}
        self.per_pair_settings = {}  # Store settings per pair: {row_index: {ch_name: ChannelSettings}}
        self.selected_row_index = -1
        self._set_dark_theme()
        self.init_ui()

    def _set_dark_theme(self):
        """Apply dark theme."""
        self.setStyle(QStyleFactory.create("Fusion"))
        p = self.palette()
        p.setColor(QPalette.Window, QColor(53, 53, 53))
        p.setColor(QPalette.WindowText, QColor(255, 255, 255))
        p.setColor(QPalette.Base, QColor(35, 35, 35))
        p.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        p.setColor(QPalette.Text, QColor(255, 255, 255))
        p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        p.setColor(QPalette.Highlight, QColor(42, 130, 218))
        p.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        self.setPalette(p)

    def init_ui(self):
        self.setWindowTitle("Timelapse Movie Maker")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - folder selection and pair list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(600)

        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)

        self.output_dir_label = QLabel("Not selected")
        self.output_dir_label.setStyleSheet("color: #ff6666;")
        output_layout.addWidget(self.output_dir_label, 1)

        self.output_dir_btn = QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_dir_btn)

        left_layout.addWidget(output_group)

        # Drop zones
        drop_group = QGroupBox("Acquisition Folders (Drag Here)")
        drop_group_layout = QVBoxLayout(drop_group)

        # Auto-detection checkbox
        auto_detect_layout = QHBoxLayout()
        self.auto_detect_cb = QCheckBox("Auto Detection")
        self.auto_detect_cb.setChecked(True)
        self.auto_detect_cb.setToolTip(
            "Automatically find matching response folder when baseline is dropped.\n"
            "Baseline folders should be named: before_* or baseline_*\n"
            "Response folders should be named: after_* or response_*"
        )
        auto_detect_layout.addWidget(self.auto_detect_cb)
        auto_detect_layout.addStretch()
        drop_group_layout.addLayout(auto_detect_layout)

        # Folder columns
        drop_layout = QHBoxLayout()

        # Baseline
        baseline_layout = QVBoxLayout()
        baseline_label = QLabel("Baseline")
        baseline_label.setAlignment(Qt.AlignCenter)
        baseline_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        baseline_layout.addWidget(baseline_label)

        self.baseline_list = DropListWidget()
        self.baseline_list.setMaximumHeight(100)
        self.baseline_list.items_changed.connect(self.on_lists_changed)
        self.baseline_list.folders_dropped.connect(self.on_baseline_folders_dropped)
        baseline_layout.addWidget(self.baseline_list)

        remove_baseline_btn = QPushButton("Remove Selected")
        remove_baseline_btn.clicked.connect(lambda: self.remove_selected(self.baseline_list))
        baseline_layout.addWidget(remove_baseline_btn)

        drop_layout.addLayout(baseline_layout)

        # Response
        response_layout = QVBoxLayout()
        response_label = QLabel("Response")
        response_label.setAlignment(Qt.AlignCenter)
        response_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        response_layout.addWidget(response_label)

        self.response_list = DropListWidget()
        self.response_list.setMaximumHeight(100)
        self.response_list.items_changed.connect(self.on_lists_changed)
        response_layout.addWidget(self.response_list)

        remove_response_btn = QPushButton("Remove Selected")
        remove_response_btn.clicked.connect(lambda: self.remove_selected(self.response_list))
        response_layout.addWidget(remove_response_btn)

        drop_layout.addLayout(response_layout)

        drop_group_layout.addLayout(drop_layout)
        left_layout.addWidget(drop_group)

        # Pair list
        pairs_group = QGroupBox("Folder Pairs (click to select for contrast adjustment)")
        pairs_layout = QVBoxLayout(pairs_group)

        self.pairs_scroll = QScrollArea()
        self.pairs_scroll.setWidgetResizable(True)
        self.pairs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.pairs_container = QWidget()
        self.pairs_container_layout = QVBoxLayout(self.pairs_container)
        self.pairs_container_layout.setAlignment(Qt.AlignTop)
        self.pairs_container_layout.setSpacing(5)

        self.no_pairs_label = QLabel("Drop folders above to create pairs")
        self.no_pairs_label.setAlignment(Qt.AlignCenter)
        self.no_pairs_label.setStyleSheet("color: #888; padding: 20px;")
        self.pairs_container_layout.addWidget(self.no_pairs_label)

        self.pairs_scroll.setWidget(self.pairs_container)
        pairs_layout.addWidget(self.pairs_scroll)

        left_layout.addWidget(pairs_group, 1)

        # Make Movie button
        self.make_movie_btn = QPushButton("Generate Movies")
        self.make_movie_btn.setFixedHeight(45)
        self.make_movie_btn.setStyleSheet("font-size: 14px; font-weight: bold; background: #2196F3;")
        self.make_movie_btn.clicked.connect(self.make_movies)
        left_layout.addWidget(self.make_movie_btn)

        main_layout.addWidget(left_panel)

        # Right panel - preview and contrast control
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Channel contrast controls
        contrast_group = QGroupBox("Channel Contrast")
        contrast_layout = QVBoxLayout(contrast_group)

        # Auto all button
        auto_layout = QHBoxLayout()
        auto_layout.addStretch()
        self.auto_all_btn = QPushButton("Auto Contrast All Channels")
        self.auto_all_btn.clicked.connect(self.auto_contrast_all)
        auto_layout.addWidget(self.auto_all_btn)
        contrast_layout.addLayout(auto_layout)

        # Channel controls container (no scroll, show all channels)
        self.channel_container = QWidget()
        self.channel_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.channel_container_layout = QVBoxLayout(self.channel_container)
        self.channel_container_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_container_layout.setSpacing(2)
        self.channel_container_layout.setAlignment(Qt.AlignTop)

        self.no_channels_label = QLabel("Select a pair to adjust contrast")
        self.no_channels_label.setAlignment(Qt.AlignCenter)
        self.no_channels_label.setStyleSheet("color: #888;")
        self.channel_container_layout.addWidget(self.no_channels_label)

        contrast_layout.addWidget(self.channel_container)

        right_layout.addWidget(contrast_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_widget = PreviewWidget()
        preview_layout.addWidget(self.preview_widget, 1)

        right_layout.addWidget(preview_group, 1)

        main_layout.addWidget(right_panel, 1)

    def remove_selected(self, list_widget):
        """Remove selected items from a list widget."""
        for item in list_widget.selectedItems():
            list_widget.takeItem(list_widget.row(item))

    def on_baseline_folders_dropped(self, dropped_paths: List[str]):
        """Handle baseline folders being dropped - auto-detect matching response folders."""
        if not self.auto_detect_cb.isChecked():
            return

        for baseline_path in dropped_paths:
            response_path = find_matching_response_folder(baseline_path)
            if response_path:
                # Check if response is not already in the list
                existing_responses = [
                    self.response_list.item(i).text()
                    for i in range(self.response_list.count())
                ]
                if response_path not in existing_responses:
                    self.response_list.addItem(response_path)

    def on_lists_changed(self):
        """Update pair rows when lists change."""
        # Clear existing rows and per-pair settings
        for row in self.pair_rows:
            row.setParent(None)
            row.deleteLater()
        self.pair_rows.clear()
        self.per_pair_settings.clear()
        self.selected_row_index = -1

        # Clear existing channel widgets
        for widget in self.channel_widgets.values():
            widget.setParent(None)
            widget.deleteLater()
        self.channel_widgets.clear()
        self.current_channel_settings.clear()

        baseline_count = self.baseline_list.count()
        response_count = self.response_list.count()
        pair_count = min(baseline_count, response_count)

        self.no_pairs_label.setVisible(pair_count == 0)
        self.no_channels_label.setVisible(pair_count == 0)

        # Create pair rows
        for i in range(pair_count):
            baseline_path = self.baseline_list.item(i).text()
            response_path = self.response_list.item(i).text()

            row = PairRowWidget(i, baseline_path, response_path)
            row.selection_changed.connect(self.on_row_selected)
            self.pair_rows.append(row)
            self.pairs_container_layout.addWidget(row)

        # Select first row if available and update preview
        if self.pair_rows:
            self.on_row_selected(0)
        else:
            # Clear preview if no pairs
            self.preview_widget.set_folders(None, None)
            self.preview_widget.image_label.clear()
            self.preview_widget.image_label.setText("No data to display")

        # Warning for mismatched counts
        if baseline_count != response_count and baseline_count > 0 and response_count > 0:
            self.statusBar().showMessage(
                f"Warning: {baseline_count} baseline vs {response_count} response folders. "
                f"Only {pair_count} pairs will be processed.", 5000
            )

    def on_row_selected(self, row_index: int):
        """Handle row selection for contrast adjustment."""
        # Save current pair's settings before switching
        if self.selected_row_index >= 0 and self.channel_widgets:
            self._save_current_pair_settings()

        # Update selection state
        for i, row in enumerate(self.pair_rows):
            row.set_selected(i == row_index)

        self.selected_row_index = row_index

        if row_index < 0 or row_index >= len(self.pair_rows):
            return

        row = self.pair_rows[row_index]

        # Discover channels for this pair
        channels = discover_channels_for_pair(row.baseline_path, row.response_path)

        # Clear existing channel widgets
        for widget in self.channel_widgets.values():
            widget.setParent(None)
            widget.deleteLater()
        self.channel_widgets.clear()

        self.no_channels_label.setVisible(not channels)

        # Check if we have saved settings for this pair
        pair_settings = self.per_pair_settings.get(row_index, {})

        # Create channel control widgets
        for ch_name, ch_settings in sorted(channels.items()):
            widget = ChannelContrastWidget(ch_name, ch_settings.wavelength)
            widget.contrast_changed.connect(self.on_contrast_changed)
            widget.auto_btn.clicked.connect(lambda _, c=ch_name: self.auto_contrast_channel(c))

            # Use saved settings if available, otherwise auto-contrast
            if ch_name in pair_settings:
                s = pair_settings[ch_name]
                widget.set_contrast(s.contrast_min, s.contrast_max)
            else:
                # Auto contrast by default for new pairs
                lo, hi = compute_auto_contrast_for_pair(row.baseline_path, row.response_path, ch_name)
                widget.set_contrast(lo, hi)

            self.channel_widgets[ch_name] = widget
            self.channel_container_layout.addWidget(widget)

        # Update current settings
        self._update_channel_settings()

        # Save settings for this pair (initializes with auto-contrast values)
        if row_index not in self.per_pair_settings:
            self._save_current_pair_settings()

        # Update preview
        self.preview_widget.set_folders(row.baseline_path, row.response_path)
        self.preview_widget.set_channel_settings(self.current_channel_settings)

    def _save_current_pair_settings(self):
        """Save current channel settings to per-pair storage."""
        if self.selected_row_index >= 0:
            self._update_channel_settings()
            self.per_pair_settings[self.selected_row_index] = dict(self.current_channel_settings)

    def _update_channel_settings(self):
        """Update current channel settings from widgets."""
        self.current_channel_settings = {}
        for ch_name, widget in self.channel_widgets.items():
            self.current_channel_settings[ch_name] = widget.get_settings()

    def on_contrast_changed(self):
        """Handle contrast change from any channel."""
        self._update_channel_settings()
        self._save_current_pair_settings()
        self.preview_widget.set_channel_settings(self.current_channel_settings)

    def auto_contrast_channel(self, channel_name: str):
        """Auto-contrast a single channel."""
        if self.selected_row_index < 0:
            return

        row = self.pair_rows[self.selected_row_index]
        lo, hi = compute_auto_contrast_for_pair(row.baseline_path, row.response_path, channel_name)

        if channel_name in self.channel_widgets:
            self.channel_widgets[channel_name].set_contrast(lo, hi)

    def auto_contrast_all(self):
        """Auto-contrast all channels."""
        if self.selected_row_index < 0:
            return

        row = self.pair_rows[self.selected_row_index]

        for ch_name in self.channel_widgets:
            lo, hi = compute_auto_contrast_for_pair(row.baseline_path, row.response_path, ch_name)
            self.channel_widgets[ch_name].set_contrast(lo, hi)

    def select_output_dir(self):
        """Select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.output_dir_label.setStyleSheet("color: #66ff66;")

    def make_movies(self):
        """Generate movies for all pairs."""
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory first.")
            return

        if not self.pair_rows:
            QMessageBox.warning(self, "Warning", "No folder pairs to process.")
            return

        # Save current pair's settings before generating
        self._save_current_pair_settings()

        # Collect pairs with their individual settings
        pairs = []
        for i, row in enumerate(self.pair_rows):
            # Get settings for this specific pair (use auto-contrast if not set)
            pair_channel_settings = self.per_pair_settings.get(i)
            if not pair_channel_settings:
                # Auto-contrast for pairs that were never selected
                channels = discover_channels_for_pair(row.baseline_path, row.response_path)
                pair_channel_settings = {}
                for ch_name, ch_settings in channels.items():
                    lo, hi = compute_auto_contrast_for_pair(row.baseline_path, row.response_path, ch_name)
                    ch_settings.contrast_min = lo
                    ch_settings.contrast_max = hi
                    pair_channel_settings[ch_name] = ch_settings
                self.per_pair_settings[i] = pair_channel_settings

            settings = row.get_settings(pair_channel_settings)
            name = f"movie_{os.path.basename(row.baseline_path)}"
            pairs.append((settings, name))

        # Create progress dialog
        progress = QProgressDialog("Creating movies...", "Cancel", 0, len(pairs), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Create worker
        self.worker = MovieWorker(pairs, self.output_dir)

        def on_progress(idx, msg):
            progress.setValue(idx)
            progress.setLabelText(msg)

        def on_finished(success, msg):
            progress.close()
            if success:
                QMessageBox.information(self, "Success", msg)
            else:
                QMessageBox.warning(self, "Completed with Errors", msg)

        self.worker.progress.connect(on_progress)
        self.worker.finished.connect(on_finished)
        self.worker.start()


def main():
    app = QApplication(sys.argv)
    window = MovieMakerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
