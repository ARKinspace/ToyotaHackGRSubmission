"""
Complete Render3D implementation with Telemetry Data Visualization
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QFileDialog, QComboBox, QSlider, QFrame, QApplication
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QVector3D, QFont, QColor, QPalette
import numpy as np
import math
import pandas as pd
import os
import traceback
from pathlib import Path

from Code.Core.TelemetryEngine import TelemetryLoader, SessionManager, StateProcessor
from Code.Core.TelemetryParsing import TelemetryParser
from scipy.spatial import KDTree
import numpy as np

try:
    import pyqtgraph.opengl as gl
    HAS_3D = True
except ImportError:
    HAS_3D = False

class Render3D(QWidget):
    # Signals for telemetry tab
    telemetryDataLoaded = pyqtSignal(object)  # Emits full telemetry DataFrame
    playbackPositionChanged = pyqtSignal(int)  # Emits current playback index
    
    def __init__(self):
        super().__init__()
        self.track_data = None
        self.current_telemetry_df = None  # Store current vehicle's telemetry
        self.session_manager = SessionManager()
        self.telemetry_loader = TelemetryLoader()
        self.state_processor = StateProcessor()
        self.current_state_history = []
        self.car_pos = {'index': 0, 'x': 0, 'y': 0, 'z': 0, 'angle': 0}
        self.playback_index = 0
        self.playback_speed = 1.0
        self.is_playing = False
        self.playback_start_time = None  # Track when playback started
        self.playback_start_index = 0    # Track which frame we started from
        self.current_lap = 1             # Track current lap for trail reset
        self.lap_start_index = 0         # Index where current lap started
        self.camera_mode = 'third_person'
        self.loaded_folder = None
        self.slider_is_dragging = False  # Track if user is dragging slider
        self.was_playing_before_drag = False  # Track playback state before drag
        self.turn_data = {}  # Store turn information
        self.sector_data = {}  # Store sector information
        self.turn_markers = []  # Store turn marker items for interaction
        self.sector_markers = []  # Store sector marker items
        self.lap_telemetry_markers = []  # Store per-lap telemetry text markers
        self.optimal_line_mesh = None  # Store optimal racing line visualization
        self.current_lap_analytics = {}  # Cache for current lap analytics
        
        # Sector timing tracking
        self.current_sector = 0  # 0=before S1, 1=in S1, 2=in S2, 3=in S3
        self.sector_start_time = None  # Timestamp when current sector started
        self.sector_times = {'S1': None, 'S2': None, 'S3': None}  # Current lap sector times
        self.lap_start_time = None  # Timestamp when lap started
        self.sector_crossed = False  # Flag to prevent multiple crossings
        
        # Sector timing tracking
        self.current_sector = 0  # 0=before S1, 1=in S1, 2=in S2, 3=in S3
        self.sector_start_time = None  # Timestamp when current sector started
        self.sector_times = {'S1': None, 'S2': None, 'S3': None}  # Current lap sector times
        self.lap_start_time = None  # Timestamp when lap started
        self.sector_crossed = False  # Flag to prevent multiple crossings
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_playback)
        self.timer.setInterval(16)  # Base interval in ms

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        if not HAS_3D:
            layout.addWidget(QLabel("Error: pyqtgraph and PyOpenGL required"))
            return
        
        # Single compact control bar
        control_bar = QFrame()
        control_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e293b, stop:0.5 #0f172a, stop:1 #1e293b);
                padding: 8px;
                border-bottom: 3px solid #3b82f6;
                border-radius: 0px;
            }
        """)
        control_bar.setMaximumHeight(60)
        bar_layout = QHBoxLayout(control_bar)
        bar_layout.setContentsMargins(12, 6, 12, 6)
        bar_layout.setSpacing(10)
        
        # Load folder button
        self.load_btn = QPushButton("📁 Folder")
        self.load_btn.clicked.connect(self.load_telemetry_folder)
        self.load_btn.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        self.load_btn.setMaximumWidth(100)
        self.load_btn.setToolTip("Load telemetry data folder")
        
        # Race selector
        self.race_selector = QComboBox()
        self.race_selector.currentIndexChanged.connect(self.on_race_changed)
        self.race_selector.setEnabled(False)
        self.race_selector.setMinimumWidth(120)
        self.race_selector.setMaximumWidth(150)
        self.race_selector.setToolTip("Select race session")
        self.race_selector.setStyleSheet("""
            QComboBox {
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
        # Vehicle selector - wider for long names
        self.vehicle_selector = QComboBox()
        self.vehicle_selector.currentIndexChanged.connect(self.on_vehicle_changed)
        self.vehicle_selector.setEnabled(False)
        self.vehicle_selector.setMinimumWidth(180)
        self.vehicle_selector.setMaximumWidth(250)
        self.vehicle_selector.setToolTip("Select vehicle to view")
        self.vehicle_selector.setStyleSheet("""
            QComboBox {
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
        # Playback controls
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        self.play_btn.setMaximumWidth(40)
        self.play_btn.setMaximumHeight(35)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: 2px solid #2563eb;
                border-radius: 6px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
                border: 2px solid #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #475569;
                border: 2px solid #64748b;
            }
        """)
        
        self.reset_btn = QPushButton("⏮")
        self.reset_btn.clicked.connect(self.reset_playback)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setMaximumWidth(40)
        self.reset_btn.setMaximumHeight(35)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: 2px solid #4f46e5;
                border-radius: 6px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4f46e5;
                border: 2px solid #4338ca;
            }
            QPushButton:disabled {
                background-color: #475569;
                border: 2px solid #64748b;
            }
        """)
        
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.valueChanged.connect(self.on_slider_changed)
        self.playback_slider.sliderPressed.connect(self.on_slider_pressed)
        self.playback_slider.sliderReleased.connect(self.on_slider_released)
        self.playback_slider.setEnabled(False)
        self.playback_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e293b, stop:1 #334155);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
                border: 2px solid #2563eb;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #93c5fd, stop:1 #60a5fa);
                border: 2px solid #3b82f6;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 4px;
            }
        """)
        
        # Speed controls
        self.speed_1x = QPushButton("1x")
        self.speed_1x.clicked.connect(lambda: self.set_speed(1.0))
        self.speed_1x.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border: 2px solid #1d4ed8;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }
        """)
        self.speed_1x.setMaximumWidth(45)
        self.speed_1x.setMaximumHeight(30)
        
        self.speed_10x = QPushButton("10x")
        self.speed_10x.clicked.connect(lambda: self.set_speed(10.0))
        self.speed_10x.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: 2px solid #64748b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
        """)
        self.speed_10x.setMaximumWidth(45)
        self.speed_10x.setMaximumHeight(30)
        
        self.speed_100x = QPushButton("100x")
        self.speed_100x.clicked.connect(lambda: self.set_speed(100.0))
        self.speed_100x.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: 2px solid #64748b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
        """)
        self.speed_100x.setMaximumWidth(50)
        self.speed_100x.setMaximumHeight(30)
        
        self.speed_1000x = QPushButton("1000x")
        self.speed_1000x.clicked.connect(lambda: self.set_speed(1000.0))
        self.speed_1000x.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: 2px solid #64748b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
        """)
        self.speed_1000x.setMaximumWidth(55)
        self.speed_1000x.setMaximumHeight(30)
        
        # Camera controls
        self.third_person_btn = QPushButton("3rd")
        self.third_person_btn.clicked.connect(lambda: self.set_camera_mode('third_person'))
        self.third_person_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border: 2px solid #1d4ed8;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }
        """)
        self.third_person_btn.setMaximumWidth(50)
        self.third_person_btn.setMaximumHeight(30)
        
        self.top_down_btn = QPushButton("Top")
        self.top_down_btn.clicked.connect(lambda: self.set_camera_mode('top_down'))
        self.top_down_btn.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: 2px solid #64748b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
        """)
        self.top_down_btn.setMaximumWidth(50)
        self.top_down_btn.setMaximumHeight(30)
        
        self.free_view_btn = QPushButton("Free")
        self.free_view_btn.clicked.connect(lambda: self.set_camera_mode('free'))
        self.free_view_btn.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: 2px solid #64748b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
        """)
        self.free_view_btn.setMaximumWidth(50)
        self.free_view_btn.setMaximumHeight(30)
        
        # RPM and Gear (inline in control bar)
        rpm_label = QLabel("RPM:")
        rpm_label.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11pt;")
        
        self.rpm_value = QLabel("0")
        self.rpm_value.setStyleSheet("""
            QLabel {
                color: #10b981;
                font-weight: bold;
                font-size: 18pt;
                background-color: rgba(16, 185, 129, 0.1);
                border: 2px solid #10b981;
                border-radius: 6px;
                padding: 2px 8px;
            }
        """)
        self.rpm_value.setMinimumWidth(70)
        
        gear_label = QLabel("G:")
        gear_label.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11pt;")
        
        self.gear_value = QLabel("N")
        self.gear_value.setStyleSheet("""
            QLabel {
                color: #3b82f6;
                font-weight: bold;
                font-size: 18pt;
                background-color: rgba(59, 130, 246, 0.1);
                border: 2px solid #3b82f6;
                border-radius: 6px;
                padding: 2px 8px;
            }
        """)
        self.gear_value.setMinimumWidth(35)
        
        # Assemble control bar
        bar_layout.addWidget(self.load_btn)
        bar_layout.addWidget(QLabel("Race:"))
        bar_layout.addWidget(self.race_selector)
        bar_layout.addWidget(QLabel("Vehicle:"))
        bar_layout.addWidget(self.vehicle_selector)
        
        # Add vertical separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setStyleSheet("background-color: #475569;")
        bar_layout.addWidget(separator1)
        
        bar_layout.addWidget(self.reset_btn)
        bar_layout.addWidget(self.play_btn)
        bar_layout.addWidget(self.playback_slider, stretch=1)
        
        # Add vertical separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setStyleSheet("background-color: #475569;")
        bar_layout.addWidget(separator2)
        
        bar_layout.addWidget(self.speed_1x)
        bar_layout.addWidget(self.speed_10x)
        bar_layout.addWidget(self.speed_100x)
        bar_layout.addWidget(self.speed_1000x)
        
        # Add vertical separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setStyleSheet("background-color: #475569;")
        bar_layout.addWidget(separator3)
        
        bar_layout.addWidget(self.third_person_btn)
        bar_layout.addWidget(self.top_down_btn)
        bar_layout.addWidget(self.free_view_btn)
        
        # Add vertical separator
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.Shape.VLine)
        separator4.setStyleSheet("background-color: #475569;")
        bar_layout.addWidget(separator4)
        
        bar_layout.addWidget(rpm_label)
        bar_layout.addWidget(self.rpm_value)
        bar_layout.addWidget(gear_label)
        bar_layout.addWidget(self.gear_value)
        
        layout.addWidget(control_bar)

        # 3D View (takes up all remaining space)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=100, elevation=30, azimuth=0)
        self.view.setBackgroundColor('#0f172a')
        self.view.opts['fov'] = 60
        
        g = gl.GLGridItem()
        g.setSize(200, 200)
        g.setSpacing(10, 10)
        g.setColor((255, 255, 255, 80))
        self.view.addItem(g)
        
        # Remove car mesh - no longer showing red box
        # self.car_mesh = gl.GLBoxItem(color=(255, 0, 0, 255))
        # self.car_mesh.setSize(1.77, 4.26, 1.31)
        # self.view.addItem(self.car_mesh)
        
        self.road_mesh = None
        self.pit_mesh = None
        self.trail_mesh = None  # Blue trail for current lap
        self.path_mesh = None   # Green reference path for current lap
        
        layout.addWidget(self.view)
        
        # Add HUD overlay for sector times and turn info
        self.create_hud_overlay()
        
        # Add legend for markers
        self.create_legend()

    def create_hud_overlay(self):
        """Create HUD overlay to display sector times, turn info, and analytics"""
        hud_frame = QFrame(self)
        hud_frame.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        hud_frame.setFixedWidth(300)
        hud_frame.setFixedHeight(400)
        hud_frame.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        hud_layout = QVBoxLayout(hud_frame)
        hud_layout.setContentsMargins(10, 10, 10, 10)
        hud_layout.setSpacing(8)
        
        # Title
        title = QLabel("📊 TELEMETRY", hud_frame)
        title.setStyleSheet("color: #60a5fa; font-size: 16pt; font-weight: bold; background-color: rgba(0,0,0,70); padding: 8px; border-radius: 6px;")
        hud_layout.addWidget(title)
        hud_layout.addSpacing(10)
        
        # Distance
        self.distance_label = QLabel("Distance: 0 ft", hud_frame)
        self.distance_label.setStyleSheet("color: white; font-size: 13pt; background-color: rgba(0,0,0,70); padding: 6px; border-radius: 5px;")
        hud_layout.addWidget(self.distance_label)
        hud_layout.addSpacing(6)
        
        # Lap progress
        self.lap_progress_label = QLabel("Lap Progress: 0%", hud_frame)
        self.lap_progress_label.setStyleSheet("color: white; font-size: 13pt; background-color: rgba(0,0,0,70); padding: 6px; border-radius: 5px;")
        hud_layout.addWidget(self.lap_progress_label)
        hud_layout.addSpacing(6)
        
        # Current sector
        self.sector_label = QLabel("Sector: S1", hud_frame)
        self.sector_label.setStyleSheet("color: #60a5fa; font-size: 13pt; font-weight: bold; background-color: rgba(0,0,0,70); padding: 6px; border-radius: 5px;")
        hud_layout.addWidget(self.sector_label)
        hud_layout.addSpacing(15)
        
        # Sector times header
        sector_header = QLabel("Lap Sector Times:", hud_frame)
        sector_header.setStyleSheet("color: #94a3b8; font-size: 11pt; background-color: rgba(0,0,0,60); padding: 5px; border-radius: 4px;")
        hud_layout.addWidget(sector_header)
        hud_layout.addSpacing(6)
        
        # Best sector times in a compact format
        self.s1_best_label = QLabel("S1: --.-s", hud_frame)
        self.s1_best_label.setStyleSheet("color: #60a5fa; font-size: 12pt; background-color: rgba(0,0,0,70); padding: 5px; border-radius: 4px;")
        hud_layout.addWidget(self.s1_best_label)
        hud_layout.addSpacing(4)
        
        self.s2_best_label = QLabel("S2: --.-s", hud_frame)
        self.s2_best_label.setStyleSheet("color: #fbbf24; font-size: 12pt; background-color: rgba(0,0,0,70); padding: 5px; border-radius: 4px;")
        hud_layout.addWidget(self.s2_best_label)
        hud_layout.addSpacing(4)
        
        self.s3_best_label = QLabel("S3: --.-s", hud_frame)
        self.s3_best_label.setStyleSheet("color: #f87171; font-size: 12pt; background-color: rgba(0,0,0,70); padding: 5px; border-radius: 4px;")
        hud_layout.addWidget(self.s3_best_label)
        
        hud_layout.addStretch()
        
        print(f"HUD: Created {len([w for w in hud_frame.children() if isinstance(w, QLabel)])} labels")
        
        hud_layout.addStretch()
        
        # Position overlay in top-right corner
        hud_frame.setParent(self)
        hud_frame.move(self.width() - 320, 70)
        hud_frame.show()
        hud_frame.raise_()
        
        # Force update
        hud_frame.update()
        
        self.hud_overlay = hud_frame
        print(f"HUD overlay created and positioned at ({self.width() - 320}, 70)")
    
    def create_legend(self):
        """Create legend showing what markers mean"""
        legend_frame = QFrame(self)
        legend_frame.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
            }
        """)
        legend_frame.setFixedWidth(250)
        legend_frame.setFixedHeight(400)
        legend_frame.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        legend_layout = QVBoxLayout(legend_frame)
        legend_layout.setContentsMargins(10, 10, 10, 10)
        legend_layout.setSpacing(8)
        
        # Title
        title = QLabel("🗺️ LEGEND", legend_frame)
        title.setStyleSheet("color: #a78bfa; font-size: 16pt; font-weight: bold; background-color: rgba(0,0,0,70); padding: 8px; border-radius: 6px;")
        legend_layout.addWidget(title)
        print("Legend: Added title")
        legend_layout.addSpacing(10)
        
        # Legend items
        items = [
            "🏁 START/FINISH",
            "🛑 Peak brake",
            "⚡ Peak G-force",
        ]
        
        for item_text in items:
            if not item_text:
                legend_layout.addSpacing(6)
                continue
            
            item_label = QLabel(item_text, legend_frame)
            # Ensure emoji fonts are used
            item_label.setStyleSheet("color: white; font-size: 11pt; font-family: 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif; background-color: rgba(0,0,0,70); padding: 6px; border-radius: 4px;")
            legend_layout.addWidget(item_label)
            legend_layout.addSpacing(4)
        
        print(f"Legend: Created {len(items)} items")
        
        legend_layout.addStretch()
        
        # Position overlay in top-left corner
        legend_frame.setParent(self)
        legend_frame.move(20, 70)
        legend_frame.show()
        legend_frame.raise_()
        
        # Force update
        legend_frame.update()
        
        self.legend_overlay = legend_frame
        print(f"Legend overlay created and positioned at (20, 70)")
    
    def resizeEvent(self, event):
        """Reposition HUD and legend on resize"""
        super().resizeEvent(event)
        if hasattr(self, 'hud_overlay') and self.hud_overlay:
            self.hud_overlay.move(self.width() - 320, 70)
        if hasattr(self, 'legend_overlay') and self.legend_overlay:
            self.legend_overlay.move(20, 70)

    def load_telemetry_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Race Data Folder", "Non_Code/dataSets"
        )
        
        if not folder_path:
            return
        
        try:
            print(f"Selected folder: {folder_path}")
            self.loaded_folder = Path(folder_path)
            
            # Check if folder already has parsed data
            parsed_files = list(self.loaded_folder.glob("vehicle_*.csv"))
            
            if parsed_files:
                print("Found parsed vehicle files. Loading directly...")
                self.load_parsed_data(self.loaded_folder)
            else:
                # Check for raw telemetry files
                raw_files = list(self.loaded_folder.glob("*telemetry_data.csv"))
                if raw_files:
                    print("Found raw telemetry files. Asking for output location...")
                    
                    # Ask for output directory first
                    output_dir = QFileDialog.getExistingDirectory(
                        self, "Select Output Directory for Parsed Files", "outputs"
                    )
                    
                    if output_dir:
                        self.parse_and_load(folder_path, output_dir)
                    else:
                        # User cancelled - ask if they want to load raw anyway
                        reply = QMessageBox.question(
                            self, "Load Raw?",
                            "Load raw telemetry data without parsing?\n(This will be slower)",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.load_raw_session(folder_path)
                else:
                    QMessageBox.warning(self, "No Data", "No telemetry data found in this folder.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load folder: {e}\n{traceback.format_exc()}")

    def parse_and_load(self, input_folder, output_folder):
        try:
            parser = TelemetryParser()
            
            self.setCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            
            count = parser.parse_folder(input_folder, output_folder)
            
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            if count > 0:
                QMessageBox.information(self, "Success", f"Successfully parsed {count} vehicles.")
                
                session_info = {
                    "input_folder": str(input_folder),
                    "output_folder": str(output_folder),
                    "parsed_count": count
                }
                import json
                with open(Path(output_folder) / "session_info.json", "w") as f:
                    json.dump(session_info, f, indent=2)
                
                self.load_parsed_data(output_folder)
            else:
                QMessageBox.warning(self, "Warning", "No vehicles were parsed.")
                
        except Exception as e:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            QMessageBox.critical(self, "Error", f"Parsing failed: {e}")
            import traceback
            traceback.print_exc()

    def load_parsed_data(self, folder_path):
        try:
            self.telemetry_loader.load_from_parsed_folder(folder_path)
            
            # Populate race selector
            races = self.telemetry_loader.get_races()
            self.race_selector.clear()
            self.race_selector.addItems(races)
            self.race_selector.setEnabled(True)
            
            # Trigger vehicle population for first race
            if races:
                self.on_race_changed(0)
            else:
                # Fallback if no races found
                vehicles = self.telemetry_loader.get_vehicles()
                self.vehicle_selector.clear()
                self.vehicle_selector.addItems([str(v) for v in vehicles])
                self.vehicle_selector.setEnabled(True)
            
            QMessageBox.information(self, "Success", f"Loaded data from parsed folder.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parsed data: {e}")

    def load_raw_session(self, folder_path):
        sessions = self.session_manager.load_folder(folder_path)
        
        if not sessions:
            QMessageBox.warning(self, "No Data", "No race data files found in folder.")
            return
        
        self.loaded_folder = folder_path
        
        self.race_selector.clear()
        self.race_selector.addItems([f"Race {race_num}" for race_num in sorted(sessions.keys())])
        self.race_selector.setEnabled(True)
        
        QMessageBox.information(self, "Success", 
            f"Loaded {len(sessions)} race session(s) from folder (Raw Mode)")
    
    def on_race_changed(self, index):
        if index < 0:
            return
            
        try:
            race_text = self.race_selector.currentText()
            print(f"Selected race: {race_text}")
            
            if self.telemetry_loader.mode == 'parsed':
                # Parsed mode: text is the race name
                vehicles = self.telemetry_loader.get_vehicles(race_text)
                print(f"Vehicles found for {race_text}: {vehicles}")
                self.vehicle_selector.clear()
                for vehicle in vehicles:
                    self.vehicle_selector.addItem(vehicle)
                    # Set tooltip to show full name
                    self.vehicle_selector.setItemData(self.vehicle_selector.count() - 1, vehicle, Qt.ItemDataRole.ToolTipRole)
                self.vehicle_selector.setEnabled(True)
            else:
                # Raw mode
                if not self.session_manager.sessions:
                    return
                race_num = int(race_text.split()[1])
                session_files = self.session_manager.get_session_files(race_num)
                
                if not session_files or 'telemetry' not in session_files:
                    QMessageBox.warning(self, "No Data", f"No telemetry file found for Race {race_num}")
                    return
                
                self.telemetry_loader.load_telemetry_file(session_files['telemetry'])
                vehicles = self.telemetry_loader.get_vehicles()
                print(f"Vehicles found for Race {race_num}: {vehicles}")
                
                self.vehicle_selector.clear()
                for vehicle in vehicles:
                    vehicle_str = str(vehicle)
                    self.vehicle_selector.addItem(vehicle_str)
                    # Set tooltip to show full name
                    self.vehicle_selector.setItemData(self.vehicle_selector.count() - 1, vehicle_str, Qt.ItemDataRole.ToolTipRole)
                self.vehicle_selector.setEnabled(True)
            
        except Exception as e:
            import traceback
            print(f"Error loading race: {e}\n{traceback.format_exc()}")
    
    def on_vehicle_changed(self, index):
        if index < 0:
            return
            
        try:
            selected_vehicle_id = self.vehicle_selector.currentText()
            print(f"Processing telemetry for vehicle {selected_vehicle_id}...")
            
            # Store current playback state if playing
            was_playing = self.is_playing
            current_elapsed_time = None
            
            # Calculate current elapsed time if we have existing data
            if self.current_state_history and self.playback_index < len(self.current_state_history):
                current_elapsed_time = self.current_state_history[self.playback_index].get('timestamp', None)
                print(f"Current playback time: {current_elapsed_time}s at index {self.playback_index}")
            
            # Pause playback while switching
            if was_playing:
                self.is_playing = False
                self.timer.stop()
            
            # Show busy cursor while processing
            self.setCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()  # Update UI
            
            race_id = self.race_selector.currentText() if self.telemetry_loader.mode == 'parsed' else None
            vehicle_telemetry = self.telemetry_loader.get_vehicle_data(selected_vehicle_id, race_id=race_id)
            
            if len(vehicle_telemetry) == 0:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                QMessageBox.warning(self, "No Data", f"No telemetry data found for {selected_vehicle_id}.")
                return
            
            print(f"Processing {len(vehicle_telemetry)} time points...")
            
            # Store full telemetry DataFrame and emit signal
            self.current_telemetry_df = vehicle_telemetry.copy()
            self.telemetryDataLoaded.emit(self.current_telemetry_df)
            print(f"Loaded telemetry DataFrame with {len(self.current_telemetry_df)} rows, columns: {list(self.current_telemetry_df.columns)[:10]}")
            
            # Process telemetry (this is now vectorized and much faster)
            self.current_state_history = self.state_processor.process_telemetry(vehicle_telemetry)
            
            # Debug: Check alignment
            if self.current_state_history:
                first_pos = self.current_state_history[0]['position']
                print(f"Vehicle Start Pos: {first_pos}")
                if self.track_data and 'center' in self.track_data:
                    print(f"Track Center: {self.track_data['center']}")
                
                # Verify path matches state history
                if hasattr(self, 'path_mesh') and self.path_mesh:
                    print(f"Path visualization exists with {len(self.current_state_history)} points")
            
            # Align states to track elevation
            self.align_states_to_track()
            
            if len(self.current_state_history) > 0:
                self.playback_slider.setMaximum(len(self.current_state_history) - 1)
                
                # Sync to the same elapsed time if we were already playing
                new_playback_index = 0
                if current_elapsed_time is not None:
                    # Find the closest timestamp in the new vehicle's data
                    best_diff = float('inf')
                    for i, state in enumerate(self.current_state_history):
                        timestamp = state.get('timestamp', 0)
                        diff = abs(timestamp - current_elapsed_time)
                        if diff < best_diff:
                            best_diff = diff
                            new_playback_index = i
                        if timestamp > current_elapsed_time:
                            break
                    print(f"Synced to index {new_playback_index} (time: {self.current_state_history[new_playback_index].get('timestamp', 0)}s)")
                    
                self.playback_index = new_playback_index
                
                # Update lap tracking for new vehicle at current position
                if new_playback_index < len(self.current_state_history):
                    self.current_lap = self.current_state_history[new_playback_index].get('lap', 1)
                else:
                    self.current_lap = 1
                self.lap_start_index = 0
                
                # Reset sector timing
                self.current_sector = 0
                self.sector_start_time = None
                self.sector_times = {'S1': None, 'S2': None, 'S3': None}
                self.lap_start_time = None
                self.sector_crossed = False
                
                self.play_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                self.playback_slider.setEnabled(True)
                
                print(f"Loaded {len(self.current_state_history)} state points")
                
                # Update visualization to current position
                self.playback_slider.setValue(self.playback_index)
                self.update_car_from_state(self.playback_index)
                self.update_path_visualization(lap=self.current_lap)
                
                # Trigger initial marker update if we have turn data
                if self.turn_data and 'turns' in self.turn_data:
                    print(f"Updating initial markers for lap {self.current_lap}")
                    self.update_lap_telemetry_markers(self.current_lap)
                else:
                    print(f"No turn data available yet - markers will appear after Fine-Tuning")
                
                # Emit playback position to update telemetry tab
                self.playbackPositionChanged.emit(self.playback_index)
                
                # Resume playback if it was playing
                if was_playing:
                    self.is_playing = True
                    import time
                    self.playback_start_time = time.time()
                    self.playback_start_index = self.playback_index
                    self.timer.start()
                    self.play_btn.setText("⏸")
                
                # Restore cursor
                self.setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                QMessageBox.warning(self, "No Data", "No telemetry data found for this vehicle.")
                
        except Exception as e:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to process vehicle: {e}\n{traceback.format_exc()}")

    def catmull_rom_spline(self, p0, p1, p2, p3, num_points=5):
        """Generate smooth curve points using Catmull-Rom spline"""
        points = []
        for i in range(num_points):
            t = i / float(num_points)
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom spline formula
            x = 0.5 * ((2 * p1['x']) +
                      (-p0['x'] + p2['x']) * t +
                      (2*p0['x'] - 5*p1['x'] + 4*p2['x'] - p3['x']) * t2 +
                      (-p0['x'] + 3*p1['x'] - 3*p2['x'] + p3['x']) * t3)
            
            y = 0.5 * ((2 * p1['y']) +
                      (-p0['y'] + p2['y']) * t +
                      (2*p0['y'] - 5*p1['y'] + 4*p2['y'] - p3['y']) * t2 +
                      (-p0['y'] + 3*p1['y'] - 3*p2['y'] + p3['y']) * t3)
            
            z = p1.get('z', 0) + (p2.get('z', 0) - p1.get('z', 0)) * t
            width = p1.get('width', 10) + (p2.get('width', 10) - p1.get('width', 10)) * t
            
            points.append({'x': x, 'y': y, 'z': z, 'width': width})
        
        return points
    
    def build_road_mesh(self):
        if not self.track_data or 'splinePoints' not in self.track_data:
            return
            
        all_points = self.track_data['splinePoints']
        if len(all_points) < 2:
            return
        
        # Separate track points from pit lane points
        # Check if any points have 'pit_lane' attribute
        track_end_idx = len(all_points)
        for i, p in enumerate(all_points):
            if 'pit_lane' in p:
                track_end_idx = i
                break
        
        points = all_points[:track_end_idx]  # Only main track points
        pit_points_data = all_points[track_end_idx:]  # Pit lane points

        sectors = self.track_data.get('splineData', {}).get('sectors', {})
        s1_end = sectors.get('s1EndIndex', -1)
        s2_end = sectors.get('s2EndIndex', -1)

        def get_color(idx):
            if s1_end > 0 and idx <= s1_end: return [0.23, 0.51, 0.96, 1.0]
            if s2_end > 0 and idx <= s2_end: return [0.92, 0.7, 0.03, 1.0]
            if s1_end > 0 and s2_end > 0: return [0.94, 0.27, 0.27, 1.0]
            return [0.06, 0.73, 0.51, 1.0]

        verts = []
        faces = []
        vertex_colors = []
        
        # Use interpolation for smoother curves
        for i in range(len(points)):
            # Get surrounding points for Catmull-Rom spline
            p0 = points[i-1] if i > 0 else points[-1]
            p1 = points[i]
            p2 = points[(i+1) % len(points)]
            p3 = points[(i+2) % len(points)]
            
            # Generate smooth intermediate points (more points in curves)
            interp_points = self.catmull_rom_spline(p0, p1, p2, p3, num_points=8)
            
            for j in range(len(interp_points) - 1):
                pt1 = interp_points[j]
                pt2 = interp_points[j+1]
                
                dx = pt2['x'] - pt1['x']
                dy = pt2['y'] - pt1['y']
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0: continue
                
                nx = -dy / length
                ny = dx / length
                
                w1 = pt1.get('width', 10) / 2
                w2 = pt2.get('width', 10) / 2
                z1 = pt1.get('z', 0)
                z2 = pt2.get('z', 0)
                
                v1 = [pt1['x'] + nx*w1, pt1['y'] + ny*w1, z1]
                v2 = [pt1['x'] - nx*w1, pt1['y'] - ny*w1, z1]
                v3 = [pt2['x'] - nx*w2, pt2['y'] - ny*w2, z2]
                v4 = [pt2['x'] + nx*w2, pt2['y'] + ny*w2, z2]
                
                base_idx = len(verts)
                verts.extend([v1, v2, v3, v4])
                
                faces.append([base_idx, base_idx+1, base_idx+2])
                faces.append([base_idx, base_idx+2, base_idx+3])
                
                c = get_color(i)
                vertex_colors.extend([c, c, c, c])
        
        if self.road_mesh:
            self.view.removeItem(self.road_mesh)
            
        mesh_data = gl.MeshData(vertexes=np.array(verts), faces=np.array(faces), 
                               vertexColors=np.array(vertex_colors))
        self.road_mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=True, 
                                      shader='shaded', glOptions='opaque')
        self.view.addItem(self.road_mesh)
        
        # Build pit lane mesh from integrated pit points
        if pit_points_data:
            self._build_pit_mesh_from_points(pit_points_data)
        elif 'visualPaths' in self.track_data:
            self._build_pit_mesh()
        
        # Add optimal racing line visualization
        self._build_optimal_line()
            
        self._add_markers()

    def _build_pit_mesh(self):
        if self.pit_mesh:
            self.view.removeItem(self.pit_mesh)
            self.pit_mesh = None

        verts = []
        faces = []
        vertex_colors = []
        
        for path in self.track_data['visualPaths']:
            if path['id'] in ['s1', 's2', 's3', 'track_full']:
                continue
            
            if 'points' in path and path['points']:
                points = path['points']
            else:
                d_str = path['d']
                parts = d_str.split(' ')
                points = []
                if len(parts) >= 3:
                    points.append({'x': float(parts[1]), 'y': float(parts[2]), 'z': 0})
                    i = 3
                    while i < len(parts):
                        if parts[i] == 'L':
                            points.append({'x': float(parts[i+1]), 'y': float(parts[i+2]), 'z': 0})
                            i += 3
                        else:
                            i += 1
            
            if len(points) < 2:
                continue
            
            width = path.get('width', 10)
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                z1 = p1.get('z', 0)
                z2 = p2.get('z', 0)
                
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0: continue
                
                nx = -dy / length
                ny = dx / length
                
                w = width / 2
                
                v1 = [p1['x'] + nx*w, p1['y'] + ny*w, z1]
                v2 = [p1['x'] - nx*w, p1['y'] - ny*w, z1]
                v3 = [p2['x'] - nx*w, p2['y'] - ny*w, z2]
                v4 = [p2['x'] + nx*w, p2['y'] + ny*w, z2]
                
                base_idx = len(verts)
                verts.extend([v1, v2, v3, v4])
                
                faces.append([base_idx, base_idx+1, base_idx+2])
                faces.append([base_idx, base_idx+2, base_idx+3])
                
                c = [0.97, 0.45, 0.09, 1.0]
                vertex_colors.extend([c, c, c, c])

        if verts:
            mesh_data = gl.MeshData(vertexes=np.array(verts), faces=np.array(faces), 
                                   vertexColors=np.array(vertex_colors))
            self.pit_mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=True, 
                                         shader='shaded', glOptions='opaque')
            self.view.addItem(self.pit_mesh)
    
    def _build_pit_mesh_from_points(self, pit_points):
        """Build pit lane mesh from integrated splinePoints with pit_lane attribute"""
        if self.pit_mesh:
            self.view.removeItem(self.pit_mesh)
            self.pit_mesh = None
        
        # Group pit points by pit_lane ID
        pit_groups = {}
        for p in pit_points:
            pit_id = p.get('pit_lane')
            if pit_id:
                if pit_id not in pit_groups:
                    pit_groups[pit_id] = []
                pit_groups[pit_id].append(p)
        
        verts = []
        faces = []
        vertex_colors = []
        
        # Build mesh for each pit lane group
        for pit_id, points in pit_groups.items():
            if len(points) < 2:
                continue
            
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0: continue
                
                nx = -dy / length
                ny = dx / length
                
                w1 = p1.get('width', 10) / 2
                w2 = p2.get('width', 10) / 2
                z1 = p1.get('z', 0)
                z2 = p2.get('z', 0)
                
                v1 = [p1['x'] + nx*w1, p1['y'] + ny*w1, z1]
                v2 = [p1['x'] - nx*w1, p1['y'] - ny*w1, z1]
                v3 = [p2['x'] - nx*w2, p2['y'] - ny*w2, z2]
                v4 = [p2['x'] + nx*w2, p2['y'] + ny*w2, z2]
                
                base_idx = len(verts)
                verts.extend([v1, v2, v3, v4])
                
                faces.append([base_idx, base_idx+1, base_idx+2])
                faces.append([base_idx, base_idx+2, base_idx+3])
                
                c = [0.97, 0.45, 0.09, 1.0]  # Orange for pit lanes
                vertex_colors.extend([c, c, c, c])
        
        if verts:
            mesh_data = gl.MeshData(vertexes=np.array(verts), faces=np.array(faces), 
                                   vertexColors=np.array(vertex_colors))
            self.pit_mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=True, 
                                         shader='shaded', glOptions='opaque')
            self.view.addItem(self.pit_mesh)
    
    def _build_optimal_line(self):
        """
        Build golden optimal racing line visualization.
        Draws the optimal line as a bright golden line following track elevation.
        """
        # Remove old optimal line if it exists
        if self.optimal_line_mesh:
            self.view.removeItem(self.optimal_line_mesh)
            self.optimal_line_mesh = None
        
        # Check if optimal line data exists
        if not self.track_data or 'optimalLine' not in self.track_data:
            return
        
        optimal_data = self.track_data['optimalLine']
        
        # Extract coordinates
        x_coords = optimal_data.get('x', [])
        y_coords = optimal_data.get('y', [])
        
        if not x_coords or not y_coords or len(x_coords) != len(y_coords):
            print("[Render3D] Invalid optimal line data")
            return
        
        # Get elevation from spline points by matching closest points
        spline_points = self.track_data.get('splinePoints', [])
        
        if spline_points and len(spline_points) > 0:
            # Build KD-tree for fast nearest neighbor lookup
            spline_coords = np.array([[p['x'], p['y']] for p in spline_points])
            spline_elevations = np.array([p.get('z', 0) for p in spline_points])
            
            from scipy.spatial import KDTree
            tree = KDTree(spline_coords)
            
            # Find nearest spline point for each optimal line point
            line_coords = np.array([[x, y] for x, y in zip(x_coords, y_coords)])
            distances, indices = tree.query(line_coords)
            
            # Get elevations from matched spline points and add small offset
            z_coords = spline_elevations[indices] + 2.0  # 2 meters above track
        else:
            # Fallback: use constant elevation
            z_coords = np.full(len(x_coords), 2.0)
        
        # Build line points (close the loop)
        line_points = np.array([[x, y, z] for x, y, z in zip(x_coords, y_coords, z_coords)])
        
        # Close the loop by adding first point at the end
        line_points = np.vstack([line_points, line_points[0:1]])
        
        # Create golden line (bright yellow/gold color)
        # RGB: (1.0, 0.84, 0.0) = Gold
        # Make it thick and bright
        self.optimal_line_mesh = gl.GLLinePlotItem(
            pos=line_points,
            color=(1.0, 0.84, 0.0, 1.0),  # Bright gold
            width=4.0,  # Thick line
            antialias=True,
            mode='line_strip'
        )
        
        self.view.addItem(self.optimal_line_mesh)
        
        # Log information
        lap_time = optimal_data.get('lap_time', 0)
        grip = optimal_data.get('grip_coefficient', 0)
        elevation_range = f"{z_coords.min():.1f}m to {z_coords.max():.1f}m" if len(z_coords) > 0 else "N/A"
        print(f"[Render3D] ✨ Optimal line displayed - Lap time: {lap_time:.2f}s, Grip: {grip:.2f}, Elevation: {elevation_range}")

    def _add_markers(self):
        # Clear old markers (but keep sector and turn markers separate)
        items_to_remove = []
        for item in self.view.items:
            if isinstance(item, gl.GLTextItem):
                # Don't remove sector markers if they're already set
                if item not in getattr(self, 'sector_markers', []):
                    items_to_remove.append(item)
        
        for item in items_to_remove:
            self.view.removeItem(item)
        
        # Only clear turn markers, keep sector markers
        self.turn_markers = []
        if not hasattr(self, 'sector_markers'):
            self.sector_markers = []
        
        # Add Start/Finish marker
        if self.track_data.get('sfMarker'):
            sf = self.track_data['sfMarker']
            sf_text = gl.GLTextItem(pos=(sf['x'], sf['y'], 8), text='🏁 START/FINISH', 
                                   color=(1, 1, 1, 1), font=QFont('Arial', 12, QFont.Weight.Bold))
            self.view.addItem(sf_text)
        
        # Add sector boundary markers (only if not already added)
        if not self.sector_markers:
            sectors = self.track_data.get('splineData', {}).get('sectors', {})
            if sectors:
                spline_points = self.track_data.get('splinePoints', [])
            
            # Sector 1 end
            s1_end_idx = sectors.get('s1EndIndex', -1)
            if s1_end_idx > 0 and s1_end_idx < len(spline_points):
                pt = spline_points[s1_end_idx]
                font = QFont('Arial', 12, QFont.Weight.Bold)
                marker = gl.GLTextItem(pos=(pt['x'], pt['y'], 10), text='S1->S2', 
                                      color=(0.4, 0.7, 1.0, 1), font=font)  # Brighter blue
                self.view.addItem(marker)
                self.sector_markers.append(marker)
                print(f"✓ Added S1▸S2 marker at ({pt['x']:.1f}, {pt['y']:.1f}, 10.0)")
            
            # Sector 2 end
            s2_end_idx = sectors.get('s2EndIndex', -1)
            if s2_end_idx > 0 and s2_end_idx < len(spline_points):
                pt = spline_points[s2_end_idx]
                font = QFont('Arial', 12, QFont.Weight.Bold)
                marker = gl.GLTextItem(pos=(pt['x'], pt['y'], 10), text='S2->S3', 
                                      color=(1.0, 0.9, 0.1, 1), font=font)  # Brighter yellow
                self.view.addItem(marker)
                self.sector_markers.append(marker)
                print(f"✓ Added S2▸S3 marker at ({pt['x']:.1f}, {pt['y']:.1f}, 10.0)")
        
        # Add turn markers if turn data is available
        if hasattr(self, 'turn_data') and self.turn_data and 'turns' in self.turn_data:
            spline_points = self.track_data.get('splinePoints', [])
            
            for turn_id, turn_info in self.turn_data['turns'].items():
                if 'apex' in turn_info and turn_info['apex']:
                    apex_idx = turn_info['apex']
                    if apex_idx < len(spline_points):
                        pt = spline_points[apex_idx]
                        
                        # Color based on turn direction
                        direction = turn_info.get('direction', 'unknown')
                        if direction == 'left':
                            color = (0.06, 0.73, 0.51, 1)  # Green
                            arrow = '↰'
                        elif direction == 'right':
                            color = (0.94, 0.27, 0.27, 1)  # Red
                            arrow = '↱'
                        else:
                            color = (0.8, 0.8, 0.8, 1)  # Gray
                            arrow = '⟳'
                        
                        # Add turn number marker
                        marker = gl.GLTextItem(
                            pos=(pt['x'], pt['y'], 7), 
                            text=f'{arrow}T{turn_id}', 
                            color=color,
                            font=QFont('Arial', 11, QFont.Weight.Bold)
                        )
                        self.view.addItem(marker)
                        self.turn_markers.append({'item': marker, 'turn_id': turn_id, 'info': turn_info})
    
    def update_lap_telemetry_markers(self, lap):
        """Update 3D markers with lap-specific telemetry data"""
        # Remove old lap markers
        for marker in self.lap_telemetry_markers:
            if marker in self.view.items:
                self.view.removeItem(marker)
        self.lap_telemetry_markers.clear()
        
        if not self.track_data or 'splinePoints' not in self.track_data:
            return
        
        # Calculate lap analytics
        if lap not in self.current_lap_analytics:
            self.current_lap_analytics[lap] = self._calculate_lap_analytics(lap)
        
        lap_analytics = self.current_lap_analytics.get(lap, {})
        spline_points = self.track_data.get('splinePoints', [])
        
        # Add turn telemetry markers
        if self.turn_data and 'turns' in self.turn_data:
            for turn_id, turn_info in self.turn_data['turns'].items():
                turn_analytics = lap_analytics.get('turns', {}).get(str(turn_id), {})
                if turn_analytics and 'apex' in turn_info:
                    apex_idx = turn_info['apex']
                    if apex_idx < len(spline_points):
                        pt = spline_points[apex_idx]
                        
                        # Build telemetry text
                        apex_speed = turn_analytics.get('apex_speed', 0)
                        max_g = turn_analytics.get('max_lateral_g', 0)
                        turn_time = turn_analytics.get('turn_time', 0)
                        
                        direction = turn_info.get('direction', 'unknown')
                        color = (0.06, 0.73, 0.51, 1) if direction == 'left' else (0.94, 0.27, 0.27, 1)
                        
                        text = f"{apex_speed:.0f}mph\n{max_g:.1f}g\n{turn_time:.1f}s"
                        
                        marker = gl.GLTextItem(
                            pos=(pt['x'] + 5, pt['y'] + 5, 5),  # Offset from turn marker
                            text=text,
                            color=color,
                            font=QFont('Consolas', 8, QFont.Weight.Bold)
                        )
                        self.view.addItem(marker)
                        self.lap_telemetry_markers.append(marker)
        
        # Add sector time markers
        sectors = self.track_data.get('splineData', {}).get('sectors', {})
        if sectors:
            s1_end_idx = sectors.get('s1EndIndex', -1)
            s2_end_idx = sectors.get('s2EndIndex', -1)
            
            # Sector 1 marker
            if s1_end_idx > 0 and s1_end_idx < len(spline_points):
                pt = spline_points[s1_end_idx]
                s1_time = lap_analytics.get('sector1_time', None)
                if s1_time:
                    marker = gl.GLTextItem(
                        pos=(pt['x'] + 5, pt['y'] + 5, 4),
                        text=f"{s1_time:.2f}s",
                        color=(0.23, 0.51, 0.96, 1),
                        font=QFont('Consolas', 9, QFont.Weight.Bold)
                    )
                    self.view.addItem(marker)
                    self.lap_telemetry_markers.append(marker)
            
            # Sector 2 marker
            if s2_end_idx > 0 and s2_end_idx < len(spline_points):
                pt = spline_points[s2_end_idx]
                s2_time = lap_analytics.get('sector2_time', None)
                if s2_time:
                    marker = gl.GLTextItem(
                        pos=(pt['x'] + 5, pt['y'] + 5, 4),
                        text=f"{s2_time:.2f}s",
                        color=(0.92, 0.7, 0.03, 1),
                        font=QFont('Consolas', 9, QFont.Weight.Bold)
                    )
                    self.view.addItem(marker)
                    self.lap_telemetry_markers.append(marker)
        
        # Add peak brake pressure marker
        peak_brake_idx = lap_analytics.get('peak_brake_location')
        if peak_brake_idx is not None and peak_brake_idx < len(self.current_state_history):
            # Get position from state history (not spline points)
            pos = self.current_state_history[peak_brake_idx]['position']
            peak_brake = lap_analytics.get('peak_brake_pressure', 0)
            font = QFont('Segoe UI Emoji', 9, QFont.Weight.Bold)
            font.setStyleHint(QFont.StyleHint.SansSerif)
            marker = gl.GLTextItem(
                pos=(pos[0], pos[1], pos[2] + 8),  # Lift above track
                text=f"🛑{peak_brake:.0f}bar",
                color=(1, 0, 0, 1),
                font=font
            )
            self.view.addItem(marker)
            self.lap_telemetry_markers.append(marker)
            print(f"✓ Added peak brake marker at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2] + 8:.1f}): {peak_brake:.0f}bar")
        else:
            print(f"⚠️ Peak brake marker not added: idx={peak_brake_idx}, state_history_len={len(self.current_state_history) if hasattr(self, 'current_state_history') else 0}")
        
        # Add peak G-force marker
        peak_g_idx = lap_analytics.get('peak_g_location')
        if peak_g_idx is not None and peak_g_idx < len(self.current_state_history):
            # Get position from state history (not spline points)
            pos = self.current_state_history[peak_g_idx]['position']
            peak_g = lap_analytics.get('peak_lateral_g', 0)
            marker = gl.GLTextItem(
                pos=(pos[0], pos[1], pos[2] + 8),  # Lift above track
                text=f"⚡{peak_g:.2f}g",
                color=(1, 1, 0, 1),
                font=QFont('Arial', 9, QFont.Weight.Bold)
            )
            self.view.addItem(marker)
            self.lap_telemetry_markers.append(marker)

    def align_states_to_track(self):
        """Snap vehicle states to track elevation."""
        if not self.current_state_history or not self.track_data or 'splinePoints' not in self.track_data:
            return
            
        try:
            # Extract track points (x, y, z)
            # splinePoints is a list of dicts, so we need to extract values manually
            spline_data = self.track_data['splinePoints']
            track_points = np.array([[p['x'], p['y'], p.get('z', 0)] for p in spline_data])
            
            if track_points.shape[0] < 3:
                return
                
            # Build KDTree for 2D lookup (x, y)
            tree = KDTree(track_points[:, :2])
            
            # Extract state positions
            state_positions = np.array([s['position'] for s in self.current_state_history])
            
            # Query closest points
            dists, idxs = tree.query(state_positions[:, :2])
            
            # Update Z coordinates
            for i, state in enumerate(self.current_state_history):
                track_z = track_points[idxs[i], 2]
                state['position'][2] = track_z
                
            print("Aligned vehicle states to track elevation.")
            
        except Exception as e:
            print(f"Failed to align states to track: {e}")

    def update_path_visualization(self, lap=None):
        """
        Update the reference path visualization.
        If lap is specified, only show the path for that lap (green).
        Otherwise, show all laps.
        """
        if not self.current_state_history:
            return
        
        # Extract positions for the specified lap or all laps
        path_points = []
        for state in self.current_state_history:
            # Filter by lap if specified
            if lap is not None and state.get('lap', 1) != lap:
                continue
            
            pos = state['position']
            # Lift path higher above track to ensure visibility
            path_points.append([pos[0], pos[1], pos[2] + 0.15])
        
        if not path_points:
            # No points for this lap, hide the path
            if hasattr(self, 'path_mesh') and self.path_mesh:
                self.view.removeItem(self.path_mesh)
                self.path_mesh = None
            return
        
        # Remove old path
        if hasattr(self, 'path_mesh') and self.path_mesh:
            self.view.removeItem(self.path_mesh)
        
        # Create new path (green reference line for the lap)
        self.path_mesh = gl.GLLinePlotItem(
            pos=np.array(path_points), 
            color=(0, 1, 0, 0.6),  # Green with transparency
            width=2, 
            antialias=True,
            glOptions='translucent'  # Ensure it renders above track
        )
        # Set depth value to render above track surface
        self.path_mesh.setGLOptions('translucent')
        self.view.addItem(self.path_mesh)
    
    def update_car_from_state(self, index):
        if len(self.current_state_history) == 0 or index >= len(self.current_state_history):
            return
        
        state = self.current_state_history[index]
        
        self.car_pos['x'] = state['position'][0]
        self.car_pos['y'] = state['position'][1]
        self.car_pos['z'] = state['position'][2]
        self.car_pos['angle'] = state['rotation'][2]
        
        # Check for lap change and reset trail
        current_lap = state.get('lap', 1)
        if current_lap != self.current_lap:
            self.current_lap = current_lap
            self.lap_start_index = index
            print(f"Lap changed to {current_lap}, resetting trail")
            # Update reference path to show new lap
            self.update_path_visualization(lap=current_lap)
            # Update lap-specific telemetry markers
            if hasattr(self, 'current_telemetry_df') and self.current_telemetry_df is not None:
                print(f"Updating telemetry markers for lap {current_lap}")
                self.update_lap_telemetry_markers(current_lap)
            else:
                print(f"No telemetry data available for markers")
        
        # Update Tron-style trail (from lap start to current position)
        self.update_trail(index)
        
        if index % 100 == 0:
            print(f"Frame {index}: pos=({self.car_pos['x']:.1f}, {self.car_pos['y']:.1f}, {self.car_pos['z']:.1f}), yaw={math.degrees(self.car_pos['angle']):.1f}°")
        
        speed = state.get('speed', 0) * 3.6
        rpm = state.get('rpm', 0)
        gear = state.get('gear', 0)
        
        self.rpm_value.setText(f"{int(rpm)}")
        gear_display = "N" if gear == 0 else ("R" if gear < 0 else str(int(gear)))
        self.gear_value.setText(gear_display)
        
        # Update RPM color based on value
        if rpm > 6500:
            rpm_color = "#ef4444"
            rpm_bg = "rgba(239, 68, 68, 0.15)"
        elif rpm > 5000:
            rpm_color = "#f59e0b"
            rpm_bg = "rgba(245, 158, 11, 0.15)"
        else:
            rpm_color = "#10b981"
            rpm_bg = "rgba(16, 185, 129, 0.1)"
            
        self.rpm_value.setStyleSheet(f"""
            QLabel {{
                color: {rpm_color};
                font-weight: bold;
                font-size: 18pt;
                background-color: {rpm_bg};
                border: 2px solid {rpm_color};
                border-radius: 6px;
                padding: 2px 8px;
            }}
        """)
        
        # Update HUD overlay with current telemetry data
        self.update_hud(state, index)
        
        self.update_car_transform()
    
    def update_hud(self, state, index):
        """Update HUD overlay with sector times, turn info, and analytics"""
        if not hasattr(self, 'hud_overlay'):
            return
        
        # Debug once
        if index == 50 and hasattr(self, 'current_telemetry_df'):
            print(f"DEBUG: HUD update at index {index}")
            print(f"  - Has telemetry_df: {self.current_telemetry_df is not None}")
            if self.current_telemetry_df is not None:
                print(f"  - DF shape: {self.current_telemetry_df.shape}")
                print(f"  - Has timestamp column: {'timestamp' in self.current_telemetry_df.columns}")
                print(f"  - Has elapsed_seconds: {'elapsed_seconds' in self.current_telemetry_df.columns}")
                print(f"  - Has track_data: {self.track_data is not None}")
                if self.track_data:
                    print(f"  - Has splineData: {'splineData' in self.track_data}")
                    if 'splineData' in self.track_data:
                        spline_count = len(self.track_data.get('splinePoints', []))
                        print(f"  - Spline points count: {spline_count}")
                        print(f"  - Telemetry rows: {len(self.current_telemetry_df)}")
                        print(f"  - NOTE: Sector indices are spline indices, not telemetry indices!")
        
        # Get current telemetry values
        lap = state.get('lap', 1)
        
        # Get distance from telemetry dataframe (same as Tab 4)
        distance = 0
        if hasattr(self, 'current_telemetry_df') and self.current_telemetry_df is not None:
            if not self.current_telemetry_df.empty and index < len(self.current_telemetry_df):
                if 'Laptrigger_lapdist_dls' in self.current_telemetry_df.columns:
                    distance = self.current_telemetry_df.iloc[index]['Laptrigger_lapdist_dls']
                elif 'distance' in self.current_telemetry_df.columns:
                    distance = self.current_telemetry_df.iloc[index]['distance']
                
                # If distance is NaN, use the last valid distance
                import math
                if math.isnan(distance) or distance is None:
                    if hasattr(self, '_last_valid_distance'):
                        distance = self._last_valid_distance
                    else:
                        distance = 0
                else:
                    # Detect SF line crossing (distance wraps around from end to start)
                    # This happens when distance drops significantly (> 1000m)
                    if hasattr(self, '_last_valid_distance') and self._last_valid_distance is not None:
                        distance_drop = self._last_valid_distance - distance
                        # If we wrapped around (e.g., 3900m -> 50m), reset sector times
                        if distance_drop > 1000:
                            print(f"🏁 SF line crossed - sector times reset (distance: {self._last_valid_distance:.1f}m → {distance:.1f}m)")
                            self.sector_times = {'S1': None, 'S2': None, 'S3': None}
                            # Get current timestamp to initialize sector start time
                            if 'elapsed_seconds' in self.current_telemetry_df.columns:
                                current_ts = self.current_telemetry_df.iloc[index]['elapsed_seconds']
                                self.sector_start_time = current_ts
                            else:
                                self.sector_start_time = None
                            self.current_sector = 0
                            self.sector_crossed = False
                    
                    # Store this as the last valid distance
                    self._last_valid_distance = distance
        
        # Update distance (convert meters to feet)
        if hasattr(self, 'distance_label'):
            distance_ft = distance * 3.28084  # Convert meters to feet
            self.distance_label.setText(f"Distance: {distance_ft:.0f} ft")
        
        # Update lap progress
        if hasattr(self, 'lap_progress_label') and len(self.current_state_history) > 0:
            lap_data = [s for s in self.current_state_history if s.get('lap', 1) == lap]
            if lap_data:
                progress = (index - self.lap_start_index) / max(1, len(lap_data)) * 100
                self.lap_progress_label.setText(f"Lap Progress: {progress:.1f}%")
        
        # Get sector boundaries based on distance, not index
        s1_distance = -1
        s2_distance = -1
        if self.track_data and 'splineData' in self.track_data:
            sectors = self.track_data.get('splineData', {}).get('sectors', {})
            s1_end_idx = sectors.get('s1EndIndex', -1)
            s2_end_idx = sectors.get('s2EndIndex', -1)
            
            # Convert spline indices to distances
            spline_points = self.track_data.get('splinePoints', [])
            if spline_points and s1_end_idx > 0 and s1_end_idx < len(spline_points):
                s1_distance = spline_points[s1_end_idx].get('dist', -1)
            if spline_points and s2_end_idx > 0 and s2_end_idx < len(spline_points):
                s2_distance = spline_points[s2_end_idx].get('dist', -1)
            
            # Debug output once
            if index == 100:
                print(f"Sector boundaries: S1 distance={s1_distance:.1f}m, S2 distance={s2_distance:.1f}m")
        else:
            if index == 100:
                print(f"No sector data - track_data exists: {self.track_data is not None}, has splineData: {'splineData' in self.track_data if self.track_data else False}")
        
        # Get current timestamp
        current_time = None
        if hasattr(self, 'current_telemetry_df') and self.current_telemetry_df is not None:
            if not self.current_telemetry_df.empty:
                if index < len(self.current_telemetry_df):
                    # Try different timestamp column names
                    if 'timestamp' in self.current_telemetry_df.columns:
                        current_time = self.current_telemetry_df.iloc[index]['timestamp']
                    elif 'elapsed_seconds' in self.current_telemetry_df.columns:
                        current_time = self.current_telemetry_df.iloc[index]['elapsed_seconds']
                    elif 'meta_time' in self.current_telemetry_df.columns:
                        current_time = self.current_telemetry_df.iloc[index]['meta_time']
                    
                    # Debug output once
                    if index == 100:
                        print(f"Timestamp at index {index}: {current_time:.2f}s (column: {'timestamp' if 'timestamp' in self.current_telemetry_df.columns else 'elapsed_seconds' if 'elapsed_seconds' in self.current_telemetry_df.columns else 'meta_time'})")
        
        # Determine which sector we're in based on distance with hysteresis
        # Add 10-meter buffer zones to prevent flickering at boundaries
        buffer = 10.0  # meters
        new_sector = 1  # Default to sector 1
        
        if s1_distance > 0 and s2_distance > 0:
            # We have all sector data
            if distance < s1_distance - buffer:
                new_sector = 1
            elif distance < s1_distance + buffer:
                # In buffer zone around S1 end - keep current sector
                new_sector = self.current_sector if hasattr(self, 'current_sector') and self.current_sector else 1
            elif distance < s2_distance - buffer:
                new_sector = 2
            elif distance < s2_distance + buffer:
                # In buffer zone around S2 end - keep current sector
                new_sector = self.current_sector if hasattr(self, 'current_sector') and self.current_sector else 2
            else:
                new_sector = 3
        
        # If no sectors defined, show message once
        if s1_distance <= 0 and s2_distance <= 0 and index == 100:
            print("⚠️ No sector boundaries found - run Tab 1: Scan to define sectors")
            print("   Sector timing will work once sectors are defined")
        
        # Detect sector crossings and calculate times
        if current_time is not None:
            # Initialize on first run
            if self.lap_start_time is None:
                # Find the actual start time of this lap from telemetry data
                if hasattr(self, 'current_telemetry_df') and self.current_telemetry_df is not None:
                    lap_data = self.current_telemetry_df[self.current_telemetry_df['lap'] == lap]
                    if not lap_data.empty:
                        # Use the first timestamp of this lap as the lap start time
                        if 'elapsed_seconds' in lap_data.columns:
                            self.lap_start_time = lap_data.iloc[0]['elapsed_seconds']
                        elif 'timestamp' in lap_data.columns:
                            self.lap_start_time = lap_data.iloc[0]['timestamp']
                        elif 'meta_time' in lap_data.columns:
                            self.lap_start_time = lap_data.iloc[0]['meta_time']
                
                # Reset sector timing for new lap
                if self.lap_start_time is None:
                    self.lap_start_time = current_time
                    
                self.sector_start_time = self.lap_start_time
                self.current_sector = new_sector
                if not hasattr(self, 'sector_times') or self.sector_times is None:
                    self.sector_times = {'S1': None, 'S2': None, 'S3': None}
                self.sector_crossed = False
                
                print(f"🔄 Sector timing initialized (lap starts at {self.lap_start_time:.2f}s)")
                self.current_lap = lap
            
            # Update lap number tracking but don't reset sector times
            # (sector times now reset on SF line crossing, not lap change)
            if not hasattr(self, 'current_lap'):
                self.current_lap = lap
            
            # Detect sector crossing
            elif new_sector != self.current_sector:
                # Only trigger if we haven't already crossed into this sector
                if not self.sector_crossed or self.current_sector != new_sector:
                    # Calculate the time for the sector we just left (check for None)
                    if self.sector_start_time is not None:
                        sector_time = current_time - self.sector_start_time
                    else:
                        sector_time = 0.0
                    
                    # Only record if the time seems reasonable (> 5 seconds)
                    if sector_time > 5.0:
                        if self.current_sector == 1:
                            self.sector_times['S1'] = sector_time
                            print(f"✓ Sector 1: {sector_time:.2f}s (crossed at index {index})")
                        elif self.current_sector == 2:
                            self.sector_times['S2'] = sector_time
                            print(f"✓ Sector 2: {sector_time:.2f}s (crossed at index {index})")
                        elif self.current_sector == 3:
                            self.sector_times['S3'] = sector_time
                            print(f"✓ Sector 3: {sector_time:.2f}s (crossed at index {index})")
                    
                    # Update for new sector
                    self.current_sector = new_sector
                    self.sector_start_time = current_time
                    self.sector_crossed = True
            else:
                # We're stable in the same sector, clear the crossed flag
                if self.sector_crossed:
                    self.sector_crossed = False
        
        # Update sector label with color coding
        if hasattr(self, 'sector_label'):
            colors = {'S1': '#60a5fa', 'S2': '#fbbf24', 'S3': '#f87171'}
            sector_name = f"S{new_sector}"
            color = colors.get(sector_name, '#10b981')
            self.sector_label.setText(f"Sector: {sector_name}")
            self.sector_label.setStyleSheet(f"color: {color}; font-size: 13pt; font-weight: bold; background-color: rgba(0,0,0,70); padding: 6px; border-radius: 5px;")
        
        # Update sector time displays
        if hasattr(self, 's1_best_label'):
            s1_time = self.sector_times.get('S1')
            if s1_distance <= 0:
                self.s1_best_label.setText("S1: (run Tab 1: Scan)")
            else:
                # Show live timing for current sector, completed time for finished sectors
                if new_sector == 1 and current_time and self.sector_start_time:
                    live_time = current_time - self.sector_start_time
                    self.s1_best_label.setText(f"S1: {live_time:.2f}s ⏱")
                else:
                    self.s1_best_label.setText(f"S1: {s1_time:.2f}s" if s1_time else "S1: --.-s")
        if hasattr(self, 's2_best_label'):
            s2_time = self.sector_times.get('S2')
            if s2_distance <= 0:
                self.s2_best_label.setText("S2: (run Tab 1: Scan)")
            else:
                # Show live timing for current sector
                if new_sector == 2 and current_time and self.sector_start_time:
                    live_time = current_time - self.sector_start_time
                    self.s2_best_label.setText(f"S2: {live_time:.2f}s ⏱")
                else:
                    self.s2_best_label.setText(f"S2: {s2_time:.2f}s" if s2_time else "S2: --.-s")
        if hasattr(self, 's3_best_label'):
            s3_time = self.sector_times.get('S3')
            if s1_distance <= 0 or s2_distance <= 0:
                self.s3_best_label.setText("S3: (run Tab 1: Scan)")
            else:
                # Show live timing for current sector
                if new_sector == 3 and current_time and self.sector_start_time:
                    live_time = current_time - self.sector_start_time
                    self.s3_best_label.setText(f"S3: {live_time:.2f}s ⏱")
                else:
                    self.s3_best_label.setText(f"S3: {s3_time:.2f}s" if s3_time else "S3: --.-s")
    
    def _calculate_lap_analytics(self, lap):
        """Calculate comprehensive analytics for a specific lap"""
        analytics = {
            'sector1_time': None,
            'sector2_time': None,
            'sector3_time': None,
            'turns': {},
            'peak_brake_pressure': 0,
            'peak_brake_location': None,
            'peak_lateral_g': 0,
            'peak_g_location': None
        }
        
        if not hasattr(self, 'current_telemetry_df') or self.current_telemetry_df.empty:
            return analytics
        
        try:
            # Get lap data
            lap_df = self.current_telemetry_df[self.current_telemetry_df['lap'] == lap]
            if lap_df.empty:
                return analytics
            
            # Calculate sector times if sector data is available
            if self.track_data and 'splineData' in self.track_data:
                sectors = self.track_data.get('splineData', {}).get('sectors', {})
                s1_end = sectors.get('s1EndIndex', -1)
                s2_end = sectors.get('s2EndIndex', -1)
                
                if 'timestamp' in lap_df.columns and len(lap_df) > 0:
                    lap_start_time = lap_df['timestamp'].iloc[0]
                    
                    # Find sector times by looking at rows near sector boundaries
                    if s1_end > 0 and len(lap_df) > s1_end:
                        analytics['sector1_time'] = lap_df['timestamp'].iloc[min(s1_end, len(lap_df)-1)] - lap_start_time
                    
                    if s2_end > 0 and s1_end > 0 and len(lap_df) > s2_end:
                        analytics['sector2_time'] = lap_df['timestamp'].iloc[min(s2_end, len(lap_df)-1)] - lap_df['timestamp'].iloc[min(s1_end, len(lap_df)-1)]
                    
                    if s2_end > 0 and len(lap_df) > s2_end:
                        analytics['sector3_time'] = lap_df['timestamp'].iloc[-1] - lap_df['timestamp'].iloc[min(s2_end, len(lap_df)-1)]
            
            # Calculate turn analytics
            if self.turn_data and 'turns' in self.turn_data:
                for turn_id, turn_info in self.turn_data['turns'].items():
                    turn_indices = turn_info.get('indices', [])
                    if turn_indices:
                        turn_lap_data = lap_df[lap_df.index.isin(turn_indices)]
                        if not turn_lap_data.empty:
                            turn_analytics = {}
                            
                            # Min speed (apex speed)
                            if 'speed' in turn_lap_data.columns:
                                min_speed_idx = turn_lap_data['speed'].idxmin()
                                turn_analytics['apex_speed'] = turn_lap_data.loc[min_speed_idx, 'speed'] * 2.237  # mph
                            
                            # Max lateral G-force
                            if 'accy_can' in turn_lap_data.columns:
                                turn_analytics['max_lateral_g'] = abs(turn_lap_data['accy_can']).max()
                            
                            # Turn time
                            if 'timestamp' in turn_lap_data.columns:
                                turn_analytics['turn_time'] = turn_lap_data['timestamp'].iloc[-1] - turn_lap_data['timestamp'].iloc[0]
                            
                            analytics['turns'][str(turn_id)] = turn_analytics
            
            # Peak brake pressure
            if 'pbrake_f' in lap_df.columns and 'pbrake_r' in lap_df.columns:
                # Use combined front + rear brake pressure
                lap_df['brake_total'] = lap_df['pbrake_f'] + lap_df['pbrake_r']
                max_brake_idx = lap_df['brake_total'].idxmax()
                analytics['peak_brake_pressure'] = lap_df.loc[max_brake_idx, 'brake_total']
                analytics['peak_brake_location'] = max_brake_idx
                print(f"✓ Lap {lap} peak brake: {analytics['peak_brake_pressure']:.0f} bar at index {max_brake_idx}")
            elif 'pbrake_f' in lap_df.columns:
                # Fall back to front only if rear not available
                max_brake_idx = lap_df['pbrake_f'].idxmax()
                analytics['peak_brake_pressure'] = lap_df.loc[max_brake_idx, 'pbrake_f']
                analytics['peak_brake_location'] = max_brake_idx
                print(f"✓ Lap {lap} peak brake (front only): {analytics['peak_brake_pressure']:.0f} bar at index {max_brake_idx}")
            else:
                print(f"⚠️ Brake pressure columns not found in telemetry. Available columns: {list(lap_df.columns)[:10]}...")
            
            # Peak lateral G
            if 'accy_can' in lap_df.columns:
                max_g_idx = lap_df['accy_can'].abs().idxmax()
                analytics['peak_lateral_g'] = abs(lap_df.loc[max_g_idx, 'accy_can'])
                analytics['peak_g_location'] = max_g_idx
        
        except Exception as e:
            print(f"Error calculating lap analytics: {e}")
        
        return analytics
    
    def update_trail(self, current_index):
        """Update Tron-style trail from lap start to current position."""
        if not self.current_state_history or current_index < self.lap_start_index:
            return
        
        # Extract positions from lap start to current position
        trail_points = []
        for i in range(self.lap_start_index, current_index + 1):
            if i < len(self.current_state_history):
                pos = self.current_state_history[i]['position']
                # Lift trail higher above track to ensure visibility
                trail_points.append([pos[0], pos[1], pos[2] + 0.2])
        
        if len(trail_points) < 2:
            # Not enough points for a trail yet
            if hasattr(self, 'trail_mesh') and self.trail_mesh:
                self.view.removeItem(self.trail_mesh)
                self.trail_mesh = None
            return
        
        # Remove old trail
        if hasattr(self, 'trail_mesh') and self.trail_mesh:
            self.view.removeItem(self.trail_mesh)
        
        # Create new trail with blue highlight (instead of cyan)
        self.trail_mesh = gl.GLLinePlotItem(
            pos=np.array(trail_points), 
            color=(0.23, 0.51, 0.96, 0.9),  # Blue with high opacity
            width=4, 
            antialias=True,
            glOptions='translucent'  # Ensure it renders above track
        )
        # Set depth value to render above track surface
        self.trail_mesh.setGLOptions('translucent')
        self.view.addItem(self.trail_mesh)
    
    def toggle_playback(self):
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_btn.setText("⏸")
            # Record start time and index for timestamp-based playback
            import time
            self.playback_start_time = time.time()
            self.playback_start_index = self.playback_index
            self.timer.start()
        else:
            self.play_btn.setText("▶")
            self.timer.stop()
            # Keep playback_start_time so we can resume from same position
    
    def reset_playback(self):
        self.playback_index = 0
        self.playback_slider.setValue(0)
        self.is_playing = False
        self.play_btn.setText("▶")
        self.timer.stop()
        self.playback_start_time = None
        self.playback_start_index = 0
        # Reset lap tracking
        self.current_lap = 1
        self.lap_start_index = 0
        self.update_car_from_state(0)
    
    def set_speed(self, speed):
        self.playback_speed = speed
        
        # Active style (gradient blue)
        active_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border: 2px solid #1d4ed8;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #60a5fa, stop:1 #3b82f6);
            }
        """
        
        # Inactive style (gray)
        inactive_style = """
            QPushButton {
                background-color: #475569;
                color: white;
                border: 2px solid #64748b;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
        """
        
        self.speed_1x.setStyleSheet(active_style if speed == 1.0 else inactive_style)
        self.speed_10x.setStyleSheet(active_style if speed == 10.0 else inactive_style)
        self.speed_100x.setStyleSheet(active_style if speed == 100.0 else inactive_style)
        self.speed_1000x.setStyleSheet(active_style if speed == 1000.0 else inactive_style)
    
    def on_slider_changed(self, value):
        """Handle slider value changes - lightweight preview during drag"""
        # Always update the index for visual feedback
        self.playback_index = value
        
        # During dragging, only update 3D visualization (lightweight)
        if self.slider_is_dragging:
            self.update_car_from_state(value)
        # When not dragging and not playing, update everything
        elif not self.is_playing:
            self.update_car_from_state(value)
            self.playbackPositionChanged.emit(value)  # Emit signal for manual seek
    
    def on_slider_pressed(self):
        """Handle slider press - pause playback during drag"""
        self.slider_is_dragging = True
        self.was_playing_before_drag = self.is_playing
        
        # Pause playback while dragging
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
    
    def on_slider_released(self):
        """Handle slider release - update all data and optionally resume playback"""
        self.slider_is_dragging = False
        
        # Get the final value
        final_value = self.playback_slider.value()
        self.playback_index = final_value
        
        # Reset sector timing when slider is moved
        self.current_sector = 0
        self.sector_start_time = None
        self.sector_times = {'S1': None, 'S2': None, 'S3': None}
        self.lap_start_time = None
        self.sector_crossed = False
        
        # Full update with all visualizations
        self.update_car_from_state(final_value)
        self.playbackPositionChanged.emit(final_value)  # Update telemetry tab
        
        # Resume playback if it was playing before drag
        if self.was_playing_before_drag:
            import time
            self.is_playing = True
            self.playback_start_time = time.time()
            self.playback_start_index = self.playback_index
            self.timer.start()
            self.play_btn.setText("⏸")
            self.was_playing_before_drag = False
    
    def update_playback(self):
        if len(self.current_state_history) == 0:
            return
        
        # Time-based playback: find the frame that matches the current simulation time
        # Never skip frames - display every frame as we pass through time
        
        if self.playback_start_time is None:
            import time
            self.playback_start_time = time.time()
            self.playback_start_index = self.playback_index
        
        import time
        # Calculate elapsed real time since playback started
        elapsed_real_time = time.time() - self.playback_start_time
        
        # Convert to simulation time based on playback speed
        elapsed_sim_time = elapsed_real_time * self.playback_speed
        
        # Get the timestamp where we started playback
        start_timestamp = self.current_state_history[self.playback_start_index].get('timestamp', 0)
        
        # Calculate the target timestamp we should be at now
        target_timestamp = start_timestamp + elapsed_sim_time
        
        # Find the frame that matches this timestamp (or just passed it)
        # Start searching from current position to avoid going backwards
        found_frame = self.playback_index
        
        for i in range(self.playback_index, len(self.current_state_history)):
            frame_timestamp = self.current_state_history[i].get('timestamp', 0)
            if frame_timestamp >= target_timestamp:
                found_frame = i
                break
        else:
            # Reached end of data, loop back to start
            self.playback_start_time = time.time()
            self.playback_start_index = 0
            self.playback_index = 0
            self.playback_slider.setValue(0)
            self.update_car_from_state(0)
            self.playbackPositionChanged.emit(0)  # Emit signal
            return
        
        # Update to the found frame (only if we've advanced)
        if found_frame != self.playback_index:
            self.playback_index = found_frame
            self.playback_slider.setValue(self.playback_index)
            self.update_car_from_state(self.playback_index)
            self.playbackPositionChanged.emit(self.playback_index)  # Emit signal

    def set_turn_data(self, turn_data):
        """Receive turn data from fine-tuning tab"""
        self.turn_data = turn_data if turn_data else {}
        # Rebuild markers with turn information
        if self.track_data:
            self._add_markers()
    
    def set_data(self, track_data):
        if not HAS_3D:
            return
            
        self.track_data = track_data
        
        # Debug: Check sector data
        if 'splineData' in track_data and 'sectors' in track_data['splineData']:
            sectors = track_data['splineData']['sectors']
            print(f"✓ Sector data loaded: S1 ends at {sectors.get('s1EndIndex', -1)}, S2 ends at {sectors.get('s2EndIndex', -1)}")
        else:
            print(f"⚠️ No sector data in track_data. Keys: {list(track_data.keys())}")
        
        # Extract turn data if available
        if 'turns' in track_data:
            self.turn_data = track_data
        self.view.setBackgroundColor('k')
        
        # Set geo reference for telemetry alignment
        if 'center' in track_data:
            center = track_data['center']
            scaling = 1.0
            if 'splineData' in track_data and 'meta' in track_data['splineData']:
                try:
                    scaling = float(track_data['splineData']['meta'].get('scalingFactorApplied', 1.0))
                except:
                    pass
            
            self.state_processor.set_geo_reference(center['lat'], center['lon'], scaling)
            
        self.build_road_mesh()
        self.reset_car()
        self.update_grid_size()

    def showEvent(self, event):
        if self.track_data:
            self.build_road_mesh()
        
        # Restore trail and path visualization after rebuild
        if hasattr(self, 'current_state_history') and self.current_state_history:
            # Restore path for current lap
            if hasattr(self, 'current_lap'):
                self.update_path_visualization(lap=self.current_lap)
            
            # Restore trail up to current playback position
            if hasattr(self, 'playback_index'):
                self.update_trail(self.playback_index)
        
        # Ensure HUD and legend are visible and on top
        if hasattr(self, 'hud_overlay') and self.hud_overlay:
            self.hud_overlay.raise_()
            self.hud_overlay.show()
        if hasattr(self, 'legend_overlay') and self.legend_overlay:
            self.legend_overlay.raise_()
            self.legend_overlay.show()
        
        super().showEvent(event)

    def reset_car(self):
        if not self.track_data:
            return
        p = self.track_data['splinePoints'][0]
        self.car_pos = {'index': 0, 'x': p['x'], 'y': p['y'], 
                       'z': p.get('z', 0) + 0.5, 'angle': 0}
        self.update_car_transform()

    def set_camera_mode(self, mode):
        self.camera_mode = mode
        
        if HAS_3D:
            # Active style (gradient blue)
            active_style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3b82f6, stop:1 #2563eb);
                    color: white;
                    border: 2px solid #1d4ed8;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #60a5fa, stop:1 #3b82f6);
                }
            """
            
            # Inactive style (gray)
            inactive_style = """
                QPushButton {
                    background-color: #475569;
                    color: white;
                    border: 2px solid #64748b;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #64748b;
                    border: 2px solid #94a3b8;
                }
            """
            
            self.third_person_btn.setStyleSheet(active_style if mode == 'third_person' else inactive_style)
            self.top_down_btn.setStyleSheet(active_style if mode == 'top_down' else inactive_style)
            self.free_view_btn.setStyleSheet(active_style if mode == 'free' else inactive_style)
        
        self.update_camera_transform()
    
    def update_car_transform(self):
        if not HAS_3D:
            return
        
        # No car mesh to transform - only update camera to follow position
        self.update_camera_transform()
    
    def update_camera_transform(self):
        if not HAS_3D or not hasattr(self, 'view'):
            return
        
        if self.camera_mode == 'third_person':
            # Follow car from behind and above
            distance = 50
            elevation_angle = 20
            
            # Calculate camera position behind the car
            cam_x = self.car_pos['x'] - math.cos(self.car_pos['angle']) * distance * math.cos(math.radians(elevation_angle))
            cam_y = self.car_pos['y'] - math.sin(self.car_pos['angle']) * distance * math.cos(math.radians(elevation_angle))
            cam_z = self.car_pos['z'] + distance * math.sin(math.radians(elevation_angle))
            
            # Look at the car
            self.view.opts['center'] = QVector3D(self.car_pos['x'], self.car_pos['y'], self.car_pos['z'])
            self.view.opts['distance'] = distance
            self.view.opts['elevation'] = elevation_angle
            self.view.opts['azimuth'] = -self.car_pos['angle'] * 180 / math.pi + 90
        
        elif self.camera_mode == 'top_down':
            # Top-down view following car
            height = 200
            self.view.opts['center'] = QVector3D(self.car_pos['x'], self.car_pos['y'], self.car_pos['z'])
            self.view.opts['distance'] = height
            self.view.opts['elevation'] = 90
            self.view.opts['azimuth'] = -self.car_pos['angle'] * 180 / math.pi + 90
        
        elif self.camera_mode == 'free':
            # Free view - don't move camera, just update if needed
            pass

    def update_grid_size(self):
        if not self.track_data:
            return
            
        points = self.track_data.get('splinePoints', [])
        if not points:
            return
            
        # Calculate bounds from points directly
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p.get('z', 0) for p in points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z = min(zs)
        
        w = max_x - min_x
        h = max_y - min_y
        max_dim = max(w, h) * 1.5
        
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        items_to_remove = []
        for item in self.view.items:
            # Only remove grid items, preserve trail and path
            if isinstance(item, gl.GLGridItem):
                items_to_remove.append(item)
        for item in items_to_remove:
            self.view.removeItem(item)
        
        # Create a simple grid item
        self.grid = gl.GLGridItem()
        self.grid.setSize(x=max_dim, y=max_dim, z=1)
        self.grid.setSpacing(x=max_dim/20, y=max_dim/20, z=1)
        
        # Position grid
        self.grid.resetTransform()
        self.grid.translate(cx, cy, min_z)
        self.view.addItem(self.grid)
        
        # Center camera
        self.view.setCameraPosition(pos=QVector3D(cx, cy, min_z), distance=max_dim, elevation=45)

    def get_session_state(self):
        """
        Get all telemetry session state for saving to .sark file.
        Returns a dict with all necessary data to restore the session.
        Includes ALL parsed telemetry data so file is self-contained.
        """
        state = {
            'loaded_folder': str(self.loaded_folder) if self.loaded_folder else None,
            'telemetry_mode': self.telemetry_loader.mode,
            'selected_race': self.race_selector.currentText() if self.race_selector.isEnabled() else None,
            'selected_vehicle': self.vehicle_selector.currentText() if self.vehicle_selector.isEnabled() else None,
        }
        
        # Save ALL parsed telemetry data directly in the SARK file
        if self.telemetry_loader.mode == 'parsed':
            state['parsed_sessions'] = dict(self.telemetry_loader.parsed_sessions)
            state['parsed_folder'] = str(self.telemetry_loader.parsed_folder) if self.telemetry_loader.parsed_folder else None
            
            # Store all the actual telemetry data
            state['telemetry_data'] = {}
            
            try:
                for race_name, vehicles in self.telemetry_loader.parsed_sessions.items():
                    state['telemetry_data'][race_name] = {}
                    
                    for vehicle_id in vehicles:
                        # Load the telemetry data for this vehicle
                        df = self.telemetry_loader.get_vehicle_data(vehicle_id, race_id=race_name)
                        if not df.empty:
                            # Convert to dict for pickling (more efficient than DataFrame)
                            state['telemetry_data'][race_name][vehicle_id] = df.to_dict('list')
                            
                print(f"Saved telemetry data for {len(state['telemetry_data'])} races")
            except Exception as e:
                print(f"Warning: Could not save all telemetry data: {e}")
                state['telemetry_data'] = {}
        
        return state
    
    def restore_session_state(self, state):
        """
        Restore telemetry session state from .sark file data.
        Loads all telemetry data from the SARK file itself (self-contained).
        """
        if not state:
            return
            
        try:
            telemetry_mode = state.get('telemetry_mode')
            
            # Check if we have embedded telemetry data in the SARK file
            if telemetry_mode == 'parsed' and 'telemetry_data' in state:
                # Load from embedded data (preferred - self-contained)
                parsed_sessions = state.get('parsed_sessions', {})
                telemetry_data = state.get('telemetry_data', {})
                
                if parsed_sessions and telemetry_data:
                    # Restore telemetry loader state
                    self.telemetry_loader.mode = 'parsed'
                    self.telemetry_loader.parsed_sessions = parsed_sessions
                    self.telemetry_loader.parsed_folder = None  # Data is embedded, no folder needed
                    self.telemetry_loader.parsed_data_cache = {}
                    
                    # Restore cached data from embedded telemetry
                    for race_name, vehicles_data in telemetry_data.items():
                        for vehicle_id, df_dict in vehicles_data.items():
                            # Convert back to DataFrame
                            df = pd.DataFrame(df_dict)
                            
                            # Parse datetime if needed
                            if 'meta_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['meta_time']):
                                try:
                                    df['meta_time'] = pd.to_datetime(df['meta_time'])
                                except:
                                    pass
                            
                            cache_key = f"{race_name}_{vehicle_id}"
                            self.telemetry_loader.parsed_data_cache[cache_key] = df
                    
                    # Populate race selector
                    self.race_selector.clear()
                    self.race_selector.addItems(sorted(list(parsed_sessions.keys())))
                    self.race_selector.setEnabled(True)
                    
                    # Restore race selection
                    selected_race = state.get('selected_race')
                    if selected_race:
                        index = self.race_selector.findText(selected_race)
                        if index >= 0:
                            self.race_selector.setCurrentIndex(index)
                            
                    # Restore vehicle selection
                    selected_vehicle = state.get('selected_vehicle')
                    if selected_vehicle:
                        index = self.vehicle_selector.findText(selected_vehicle)
                        if index >= 0:
                            self.vehicle_selector.setCurrentIndex(index)
                    
                    print(f"Restored telemetry data from SARK file: {len(telemetry_data)} races")
                    QMessageBox.information(self, "Telemetry Restored", 
                        f"Loaded telemetry data for {len(telemetry_data)} race(s) from SARK file.\\n" +
                        f"Race: {selected_race or 'None'}\\n" +
                        f"Vehicle: {selected_vehicle or 'None'}")
                    return
            
            # Fallback: Try to load from folder paths (if data not embedded or folders still exist)
            parsed_folder = state.get('parsed_folder')
            loaded_folder = state.get('loaded_folder')
            
            if parsed_folder and Path(parsed_folder).exists():
                # Load from parsed folder
                self.loaded_folder = Path(parsed_folder)
                self.load_parsed_data(parsed_folder)
                
                # Restore race selection
                selected_race = state.get('selected_race')
                if selected_race:
                    index = self.race_selector.findText(selected_race)
                    if index >= 0:
                        self.race_selector.setCurrentIndex(index)
                        
                # Restore vehicle selection
                selected_vehicle = state.get('selected_vehicle')
                if selected_vehicle:
                    index = self.vehicle_selector.findText(selected_vehicle)
                    if index >= 0:
                        self.vehicle_selector.setCurrentIndex(index)
                        
            elif loaded_folder and Path(loaded_folder).exists():
                # Try to load from original folder
                self.loaded_folder = Path(loaded_folder)
                
                # Check if it has parsed data
                parsed_files = list(self.loaded_folder.glob("vehicle_*.csv"))
                if parsed_files:
                    self.load_parsed_data(self.loaded_folder)
                else:
                    # Load raw session
                    sessions = self.session_manager.load_folder(str(self.loaded_folder))
                    if sessions:
                        self.race_selector.clear()
                        self.race_selector.addItems([f"Race {race_num}" for race_num in sorted(sessions.keys())])
                        self.race_selector.setEnabled(True)
            else:
                QMessageBox.warning(self, "Telemetry Not Found",
                    "Telemetry data folders not found. You may need to browse for the data again.")
                        
            print("Session state restored from folder paths")
            
        except Exception as e:
            print(f"Failed to restore session state: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Restore Warning", 
                f"Could not fully restore telemetry state: {e}\\n\\n" +
                "You may need to browse for the telemetry folder again.")
