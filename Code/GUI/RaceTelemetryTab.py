"""
Race Telemetry Analysis Tab - Real-time telemetry visualization and analysis
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QPushButton, QScrollArea, QFrame, QGridLayout, QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import uniform_filter1d

class FocusablePlotWidget(pg.PlotWidget):
    """
    A PlotWidget that requires a click to enable mouse interactions (zoom/pan).
    Shows a visual indicator and crosshair cursor when focused.
    """
    sigFocused = pyqtSignal(object)  # Emit self when focused

    def __init__(self, parent=None, background='default', **kargs):
        super().__init__(parent, background, **kargs)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.is_focused = False
        
        # Crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#cbd5e1', width=1, style=Qt.PenStyle.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#cbd5e1', width=1, style=Qt.PenStyle.DashLine))
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)
        self.vLine.hide()
        self.hLine.hide()
        
        self.scene().sigMouseMoved.connect(self.mouseMoved)

    def mousePressEvent(self, ev):
        if not self.is_focused:
            self.sigFocused.emit(self)
            self.set_focus(True)
            ev.accept()  # Consume event so we don't zoom on the first click
        else:
            super().mousePressEvent(ev)

    def set_focus(self, focused: bool):
        self.is_focused = focused
        self.setMouseEnabled(x=focused, y=focused)
        
        if focused:
            self.getViewBox().setBorder(pg.mkPen(color='#3b82f6', width=3))
            self.vLine.show()
            self.hLine.show()
        else:
            self.getViewBox().setBorder(pg.mkPen(None))
            self.vLine.hide()
            self.hLine.hide()

    def mouseMoved(self, evt):
        if self.is_focused and self.sceneBoundingRect().contains(evt):
            mousePoint = self.getViewBox().mapSceneToView(evt)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())



class TelemetryAnalytics:
    """Helper class for calculating derived metrics and statistics"""
    
    @staticmethod
    def calculate_lap_stats(df: pd.DataFrame) -> Dict:
        """Calculate lap statistics"""
        if df.empty:
            return {}
        
        stats = {
            'top_speed': df['speed'].max() if 'speed' in df else 0,
            'max_lat_g': abs(df['accy_can']).max() if 'accy_can' in df else 0,
            'peak_brake': df['pbrake_f'].max() if 'pbrake_f' in df else 0,
            'avg_speed': df['speed'].mean() if 'speed' in df else 0,
        }
        
        # Coasting percentage
        if 'aps' in df and 'pbrake_f' in df:
            throttle = df['aps'].fillna(0)
            brake = df['pbrake_f'].fillna(0)
            coasting = ((throttle < 5) & (brake < 1)).sum()
            stats['coasting_pct'] = (coasting / len(df)) * 100 if len(df) > 0 else 0
        else:
            stats['coasting_pct'] = 0
        
        # Brake bias (only when braking hard)
        if 'pbrake_f' in df and 'pbrake_r' in df:
            braking_mask = df['pbrake_f'] > 10
            if braking_mask.any():
                front = df.loc[braking_mask, 'pbrake_f']
                rear = df.loc[braking_mask, 'pbrake_r']
                total = front + rear
                bias = (front / total * 100).mean()
                stats['brake_bias'] = bias if not np.isnan(bias) else 50
            else:
                stats['brake_bias'] = 50
        else:
            stats['brake_bias'] = 50
        
        # Lap time (if available)
        if 'elapsed_seconds' in df:
            stats['lap_time'] = df['elapsed_seconds'].max() - df['elapsed_seconds'].min()
        else:
            stats['lap_time'] = 0
        
        # Distance
        if 'Laptrigger_lapdist_dls' in df:
            stats['distance'] = df['Laptrigger_lapdist_dls'].max() - df['Laptrigger_lapdist_dls'].min()
        else:
            stats['distance'] = 0
        
        return stats
    
    @staticmethod
    def calculate_instability_index(df: pd.DataFrame) -> np.ndarray:
        """
        Calculate dynamic instability index.
        Formula: |Steering Angle| * |Lateral G| * (Speed / 100)
        Apply 5-point moving average smoothing.
        """
        if df.empty or 'Steering_Angle' not in df or 'accy_can' not in df or 'speed' not in df:
            return np.array([])
        
        steering = df['Steering_Angle'].fillna(0).abs()
        lat_g = df['accy_can'].fillna(0).abs()
        speed = df['speed'].fillna(0)
        
        raw_score = steering * lat_g * (speed / 100)
        
        # 5-point moving average
        if len(raw_score) >= 5:
            smoothed = uniform_filter1d(raw_score, size=5, mode='nearest')
        else:
            smoothed = raw_score.values
        
        return smoothed
    
    @staticmethod
    def detect_gear_shifts(df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Detect upshifts and downshifts.
        Returns dict with 'upshifts' and 'downshifts' lists.
        Each entry contains: distance, rpm, speed, from_gear, to_gear
        """
        if df.empty or 'gear' not in df:
            return {'upshifts': [], 'downshifts': []}
        
        upshifts = []
        downshifts = []
        
        gear = df['gear'].fillna(0).astype(int)
        
        for i in range(1, len(df)):
            prev_gear = gear.iloc[i-1]
            curr_gear = gear.iloc[i]
            
            if curr_gear > prev_gear:  # Upshift
                shift = {
                    'distance': df.iloc[i].get('Laptrigger_lapdist_dls', i),
                    'rpm_before': df.iloc[i-1].get('nmot', 0),
                    'speed': df.iloc[i].get('speed', 0),
                    'from_gear': prev_gear,
                    'to_gear': curr_gear,
                    'index': i
                }
                upshifts.append(shift)
                
            elif curr_gear < prev_gear and curr_gear > 0:  # Downshift (ignore to neutral)
                shift = {
                    'distance': df.iloc[i].get('Laptrigger_lapdist_dls', i),
                    'rpm_after': df.iloc[i].get('nmot', 0),
                    'speed': df.iloc[i].get('speed', 0),
                    'from_gear': prev_gear,
                    'to_gear': curr_gear,
                    'index': i
                }
                downshifts.append(shift)
        
        return {'upshifts': upshifts, 'downshifts': downshifts}
    
    @staticmethod
    def calculate_time_delta(active_df: pd.DataFrame, reference_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate time delta between two laps, synced by distance.
        Returns: (distance_array, delta_array)
        Delta is positive when active lap is slower.
        """
        if active_df.empty or reference_df.empty:
            return np.array([]), np.array([])
        
        # Need distance and time columns
        if 'Laptrigger_lapdist_dls' not in active_df or 'Laptrigger_lapdist_dls' not in reference_df:
            return np.array([]), np.array([])
        
        if 'elapsed_seconds' not in active_df or 'elapsed_seconds' not in reference_df:
            return np.array([]), np.array([])
        
        # Normalize time to lap start
        active_time = active_df['elapsed_seconds'] - active_df['elapsed_seconds'].iloc[0]
        ref_time = reference_df['elapsed_seconds'] - reference_df['elapsed_seconds'].iloc[0]
        
        active_dist = active_df['Laptrigger_lapdist_dls'].values
        ref_dist = reference_df['Laptrigger_lapdist_dls'].values
        
        # For each point in active lap, find closest point in reference lap
        deltas = []
        distances = []
        
        for i, dist in enumerate(active_dist):
            # Find closest distance in reference (within 33ft tolerance)
            diff = np.abs(ref_dist - dist)
            min_idx = np.argmin(diff)
            
            if diff[min_idx] < 33:  # Within tolerance (~10m)
                delta = active_time.iloc[i] - ref_time.iloc[min_idx]
                deltas.append(delta)
                distances.append(dist)
        
        return np.array(distances), np.array(deltas)
    
    @staticmethod
    def identify_apexes(df: pd.DataFrame, min_lat_g: float = 0.6) -> List[Dict]:
        """
        Identify corner apexes as local maxima of lateral G > threshold.
        Returns list of apex points with distance, speed, lat_g
        """
        if df.empty or 'accy_can' not in df:
            return []
        
        lat_g = df['accy_can'].fillna(0).abs()
        
        apexes = []
        for i in range(1, len(lat_g) - 1):
            if lat_g.iloc[i] > min_lat_g:
                # Check if local maximum
                if lat_g.iloc[i] > lat_g.iloc[i-1] and lat_g.iloc[i] > lat_g.iloc[i+1]:
                    apex = {
                        'distance': df.iloc[i].get('Laptrigger_lapdist_dls', i),
                        'speed': df.iloc[i].get('speed', 0),
                        'lat_g': lat_g.iloc[i],
                        'index': i
                    }
                    apexes.append(apex)
        
        return apexes


class RaceTelemetryTab(QWidget):
    """
    Fourth tab: Real-time telemetry analysis and visualization.
    Updates as the race plays in tab 3.
    """
    
    # Signal to request current telemetry data
    requestTelemetryUpdate = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Data
        self.current_telemetry_df = pd.DataFrame()  # Full telemetry for current vehicle
        self.active_lap = 'All'  # Current selected lap
        self.compare_lap = None  # Reference lap for comparison
        self.selected_turn = 'All Turns'  # Selected turn for filtering
        self.turn_data = {}  # Turn definitions from Tab 2
        
        # Current playback index (updated from Render3D)
        self.playback_index = 0
        self.current_lap_number = 1
        self.previous_lap_number = 0
        self.lap_start_index = 0
        self.lap_start_time = 0
        self.previous_lap_data = pd.DataFrame()
        
        # Real-time data accumulation
        self.accumulated_data = pd.DataFrame()
        
        # Update debouncing with QTimer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_debounce_ms = 16  # ~60fps max update rate
        
        # Analytics helper
        self.analytics = TelemetryAnalytics()
        
        # Focus management
        self.plots = []
        self.current_focused_plot = None
        
        self.init_ui()
        
        # Initialize stats with default values
        self.initialize_default_stats()

    def init_ui(self):
        """Initialize the UI with controls and charts"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Set main tab background
        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
            }
        """)
        
        # === TOP CONTROL BAR ===
        control_bar = self.create_control_bar()
        layout.addWidget(control_bar)
        
        # Add subtle separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.5 #3b82f6, stop:1 transparent);
                max-height: 2px;
                border: none;
            }
        """)
        layout.addWidget(separator)
        
        # === STATS GRID (Fixed at top) ===
        self.stats_frame = self.create_stats_grid()
        layout.addWidget(self.stats_frame)
        
        # === SCROLLABLE CHARTS AREA ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #0f172a;
            }
            QScrollBar:vertical {
                background-color: #1e293b;
                width: 14px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 7px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #60a5fa, stop:1 #a78bfa);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        charts_layout.setContentsMargins(20, 20, 20, 20)
        charts_layout.setSpacing(25)
        
        # Create all charts with unique designs
        self.create_all_charts(charts_layout)
        
        scroll.setWidget(charts_widget)
        layout.addWidget(scroll)

    
    def create_control_bar(self) -> QFrame:
        """Create top control bar with lap selectors and mode toggles"""
        bar = QFrame()
        bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e293b, stop:0.5 #0f172a, stop:1 #1e293b);
                padding: 10px;
                border-bottom: 3px solid #3b82f6;
            }
            QComboBox {
                background-color: #334155;
                color: #f1f5f9;
                border: 2px solid #475569;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11pt;
                font-weight: bold;
            }
            QComboBox:hover {
                border: 2px solid #3b82f6;
                background-color: #3b4252;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #3b82f6;
                width: 0;
                height: 0;
                margin-right: 8px;
            }
            QPushButton {
                background-color: #475569;
                color: #f1f5f9;
                border: 2px solid #64748b;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ef4444;
                border: 2px solid #dc2626;
            }
            QLabel {
                color: #cbd5e1;
                font-size: 11pt;
                font-weight: bold;
            }
        """)
        bar.setMaximumHeight(70)
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(15)
        
        # Active Lap selector
        active_label = QLabel("📊 Active Lap:")
        layout.addWidget(active_label)
        
        self.active_lap_combo = QComboBox()
        self.active_lap_combo.addItem("All")
        self.active_lap_combo.currentTextChanged.connect(self.on_active_lap_changed)
        self.active_lap_combo.setMinimumWidth(180) # Increased width
        self.active_lap_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.active_lap_combo)
        
        # Compare Lap selector
        compare_label = QLabel("🔄 Compare to:")
        layout.addWidget(compare_label)
        
        self.compare_lap_combo = QComboBox()
        self.compare_lap_combo.addItem("None")
        self.compare_lap_combo.currentTextChanged.connect(self.on_compare_lap_changed)
        self.compare_lap_combo.setMinimumWidth(180) # Increased width
        self.compare_lap_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.compare_lap_combo)
        
        # Turn selector
        turn_label = QLabel("🏁 Turn:")
        layout.addWidget(turn_label)
        
        self.turn_combo = QComboBox()
        self.turn_combo.addItem("All Turns")
        self.turn_combo.currentTextChanged.connect(self.on_turn_changed)
        self.turn_combo.setMinimumWidth(180) # Increased width
        self.turn_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.turn_combo)
        
        layout.addStretch()
        
        return bar
    
    def _format_stat_html(self, key: str, value: str, color: str = None) -> str:
        """Format stat value with unit using HTML"""
        if color is None:
            color = self.stat_colors.get(key, '#ffffff')
        unit = self.stat_unit_names.get(key, '')
        return f'<span style="color: {color}; font-size: 18pt; font-weight: bold;">{value}</span> <span style="color: #64748b; font-size: 11pt;">{unit}</span>'
    
    def initialize_default_stats(self):
        """Initialize stats with default values"""
        if hasattr(self, 'stat_labels'):
            self.stat_labels['lap_number'].setText(self._format_stat_html('lap_number', '1'))
            self.stat_labels['top_speed'].setText(self._format_stat_html('top_speed', '0.0'))
            self.stat_labels['max_lat_g'].setText(self._format_stat_html('max_lat_g', '0.00'))
            self.stat_labels['peak_brake'].setText(self._format_stat_html('peak_brake', '0.0'))
            self.stat_labels['brake_bias'].setText(self._format_stat_html('brake_bias', '50.0'))
            self.stat_labels['coasting_pct'].setText(self._format_stat_html('coasting_pct', '0.0'))
            self.stat_labels['lap_time'].setText(self._format_stat_html('lap_time', '0.00'))
            self.stat_labels['total_time'].setText(self._format_stat_html('total_time', '0.00'))
            self.stat_labels['time_delta'].setText(self._format_stat_html('time_delta', '+0.00'))
            
            # Force all labels and their parents to be visible
            for label in self.stat_labels.values():
                label.show()
            self.stats_frame.show()
            self.stats_frame.update()

    def create_stats_grid(self) -> QFrame:
        """Create statistics display grid with enhanced styling"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1e293b, stop:0.5 #0f172a, stop:1 #1e293b);
                padding: 10px;
                border-radius: 12px;
                border: 2px solid #334155;
            }
        """)
        frame.setMinimumHeight(280)  # Reduced height slightly
        frame.setMaximumHeight(350)
        
        layout = QGridLayout(frame)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Stat labels and units (will be populated dynamically)
        self.stat_labels = {}
        self.stat_colors = {}
        self.stat_unit_names = {}
        
        stat_names = [
            ('lap_number', '📍 Current Lap', '', '#06b6d4'),
            ('top_speed', '🏁 Top Speed', 'mph', '#10b981'),
            ('max_lat_g', '🌀 Max Lat G', 'g', '#3b82f6'),
            ('peak_brake', '🛑 Peak Brake', 'bar', '#ef4444'),
            ('brake_bias', '⚖️ Brake Bias', '%', '#f59e0b'),
            ('coasting_pct', '⚡ Coasting', '%', '#8b5cf6'),
            ('lap_time', '⏱️ Lap Time', 's', '#ec4899'),
            ('total_time', '⏲️ Total Time', 's', '#a855f7'),
            ('time_delta', '△ Time Delta', 's', '#f97316'),
        ]
        
        for i, (key, label, unit, color) in enumerate(stat_names):
            self.stat_colors[key] = color
            self.stat_unit_names[key] = unit
            row = i // 3
            col = i % 3
            
            # Create container for each stat
            stat_container = QFrame()
            stat_container.setMinimumHeight(80)
            stat_container.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #0f172a, stop:1 #1e293b);
                    border-radius: 8px;
                    border: 1px solid #334155;
                }}
                QFrame:hover {{
                    border: 1px solid {color};
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1a1f2e, stop:1 #243044);
                }}
            """)
            
            stat_layout = QHBoxLayout(stat_container)
            stat_layout.setSpacing(10)
            stat_layout.setContentsMargins(15, 5, 15, 5)
            
            # Label (Left side)
            name_label = QLabel(label)
            name_label.setStyleSheet(f"""
                color: #94a3b8;
                font-size: 10pt;
                font-weight: 600;
                background: transparent;
                border: none;
            """)
            name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            stat_layout.addWidget(name_label)
            
            stat_layout.addStretch()
            
            # Value (Right side)
            value_text = "0.0" if key != 'brake_bias' else "50.0"
            combined_label = QLabel(f'<span style="color: {color}; font-size: 16pt; font-weight: bold;">{value_text}</span> <span style="color: #64748b; font-size: 10pt;">{unit}</span>')
            combined_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            combined_label.setStyleSheet("background: transparent; border: none;")
            stat_layout.addWidget(combined_label)
            
            self.stat_labels[key] = combined_label
            
            layout.addWidget(stat_container, row, col)
        
        return frame
    
    def on_plot_focused(self, plot_widget):
        """Handle focus switching between plots"""
        if self.current_focused_plot and self.current_focused_plot != plot_widget:
            self.current_focused_plot.set_focus(False)
        self.current_focused_plot = plot_widget

    def create_all_charts(self, parent_layout: QVBoxLayout):
        """Create all visualization charts with unique designs in a grid layout"""
        
        # Configure PyQtGraph
        pg.setConfigOptions(antialias=True, background='#0f172a', foreground='#e2e8f0')
        
        # Helper to add plot to grid
        def add_plot_to_grid(container, row, col, colspan=1):
            grid_layout.addWidget(container, row, col, 1, colspan)
            
        # Main grid layout for charts
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        parent_layout.addWidget(grid_widget)
        
        # === ROW 1: Track Map + Friction Circle ===
        
        # 1. TRACK MAP
        # 1. TRACK MAP
        track_map_container = self.create_chart_container("🗺️ Track Map", "GPS Trace with Speed/Brake Overlay")
        track_map_layout = track_map_container.layout()
        
        self.track_map_plot = FocusablePlotWidget()
        self.track_map_plot.sigFocused.connect(self.on_plot_focused)
        self.track_map_plot.setMinimumHeight(400)
        self.track_map_plot.setBackground('#0a0e1a')
        self.track_map_plot.setLabel('left', 'Latitude', **{'color': '#ec4899', 'font-size': '12pt', 'font-weight': 'bold'})
        self.track_map_plot.setLabel('bottom', 'Longitude', **{'color': '#ec4899', 'font-size': '12pt', 'font-weight': 'bold'})
        self.track_map_plot.setAspectLocked(True)
        self.track_map_plot.showGrid(x=False, y=False)
        self.track_map_plot.getAxis('left').setTextPen('#ec4899')
        self.track_map_plot.getAxis('bottom').setTextPen('#ec4899')
        
        self.track_map_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
        self.track_map_plot.addItem(self.track_map_scatter)
        track_map_layout.addWidget(self.track_map_plot)
        
        add_plot_to_grid(track_map_container, 0, 0)
        
        # 2. FRICTION CIRCLE
        friction_container = self.create_chart_container("🎯 Friction Circle", "G-G Diagram showing tire grip utilization")
        friction_layout = friction_container.layout()
        
        self.friction_circle_plot = FocusablePlotWidget()
        self.friction_circle_plot.sigFocused.connect(self.on_plot_focused)
        self.friction_circle_plot.setMinimumHeight(400)
        self.friction_circle_plot.setBackground('#1a0a1a')
        self.friction_circle_plot.setLabel('left', 'Longitudinal G', units='g', **{'color': '#a78bfa', 'font-size': '13pt', 'font-weight': 'bold'})
        self.friction_circle_plot.setLabel('bottom', 'Lateral G', units='g', **{'color': '#a78bfa', 'font-size': '13pt', 'font-weight': 'bold'})
        self.friction_circle_plot.setAspectLocked(True)
        self.friction_circle_plot.showGrid(x=True, y=True, alpha=0.2)
        self.friction_circle_plot.getAxis('left').setTextPen('#a78bfa')
        self.friction_circle_plot.getAxis('bottom').setTextPen('#a78bfa')
        
        # Add reference circles
        for radius, alpha in [(0.5, 0.3), (1.0, 0.4), (1.5, 0.5)]:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = radius * np.cos(theta)
            circle_y = radius * np.sin(theta)
            self.friction_circle_plot.plot(circle_x, circle_y, 
                pen=pg.mkPen(color=(147, 51, 234, int(alpha * 255)), width=2, style=Qt.PenStyle.DashLine))
        
        self.friction_scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None))
        self.friction_circle_plot.addItem(self.friction_scatter)
        friction_layout.addWidget(self.friction_circle_plot)
        
        add_plot_to_grid(friction_container, 0, 1)
        
        # === ROW 2: Braking & Speed + G-Force ===
        
        # 3. BRAKING & SPEED
        brake_speed_container = self.create_chart_container("🏎️ Braking & Speed Profile", "Compare speed and braking points across laps")
        brake_speed_layout = brake_speed_container.layout()
        
        self.brake_speed_plot = FocusablePlotWidget()
        self.brake_speed_plot.sigFocused.connect(self.on_plot_focused)
        self.brake_speed_plot.setMinimumHeight(350)
        self.brake_speed_plot.setBackground('#0f1a0f')
        self.brake_speed_plot.setLabel('left', 'Speed (mph) / Brake (bar)', **{'color': '#10b981', 'font-size': '12pt'})
        self.brake_speed_plot.setLabel('bottom', 'Distance', units='ft', **{'color': '#3b82f6', 'font-size': '11pt'})
        self.brake_speed_plot.showGrid(x=True, y=True, alpha=0.25)
        
        self.speed_curve = self.brake_speed_plot.plot(pen=pg.mkPen(color=(16, 185, 129), width=3), name='Speed')
        self.brake_curve = self.brake_speed_plot.plot(pen=pg.mkPen(color=(239, 68, 68), width=3), name='Brake')
        
        # Ghost curves
        self.speed_ghost_curve = self.brake_speed_plot.plot(
            pen=pg.mkPen(color=(147, 51, 234), width=3, style=Qt.PenStyle.DashLine), name='Ref Speed')
        self.brake_ghost_curve = self.brake_speed_plot.plot(
            pen=pg.mkPen(color=(236, 72, 153), width=3, style=Qt.PenStyle.DashLine), name='Ref Brake')
            
        # Real-time line
        self.brake_speed_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.brake_speed_plot.addItem(self.brake_speed_line)
        
        legend = self.brake_speed_plot.addLegend(offset=(10, 10))
        legend.setParentItem(self.brake_speed_plot.getPlotItem())
        brake_speed_layout.addWidget(self.brake_speed_plot)
        
        add_plot_to_grid(brake_speed_container, 1, 0)
        
        # 4. G-FORCE PROFILE
        g_force_container = self.create_chart_container("💪 G-Force Profile", "Lateral (cornering) and Longitudinal (accel/brake) forces")
        g_force_layout = g_force_container.layout()
        
        self.g_force_plot = FocusablePlotWidget()
        self.g_force_plot.sigFocused.connect(self.on_plot_focused)
        self.g_force_plot.setMinimumHeight(350)
        self.g_force_plot.setBackground('#1a0f1a')
        self.g_force_plot.setLabel('left', 'G-Force', units='g', **{'color': '#a78bfa', 'font-size': '12pt'})
        self.g_force_plot.setLabel('bottom', 'Distance', units='ft', **{'color': '#3b82f6', 'font-size': '11pt'})
        self.g_force_plot.showGrid(x=True, y=True, alpha=0.25)
        self.g_force_plot.addLine(y=0, pen=pg.mkPen(color=(128, 128, 128), width=1))
        
        self.lat_g_curve = self.g_force_plot.plot(pen=pg.mkPen(color=(147, 51, 234), width=3), name='Lateral G')
        self.long_g_curve = self.g_force_plot.plot(pen=pg.mkPen(color=(59, 130, 246), width=3), name='Longitudinal G')
        
        # Real-time line
        self.g_force_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.g_force_plot.addItem(self.g_force_line)
        
        legend = self.g_force_plot.addLegend(offset=(10, 10))
        legend.setParentItem(self.g_force_plot.getPlotItem())
        g_force_layout.addWidget(self.g_force_plot)
        
        add_plot_to_grid(g_force_container, 1, 1)
        
        # === ROW 3: Driver Demand + Gear Shift ===
        
        # 5. DRIVER DEMAND
        demand_container = self.create_chart_container("🎮 Driver Demand", "Throttle application vs RPM")
        demand_layout = demand_container.layout()
        
        self.driver_demand_plot = FocusablePlotWidget()
        self.driver_demand_plot.sigFocused.connect(self.on_plot_focused)
        self.driver_demand_plot.setMinimumHeight(350)
        self.driver_demand_plot.setBackground('#0f0f1a')
        self.driver_demand_plot.setLabel('left', 'Throttle', units='%', **{'color': '#22c55e', 'font-size': '12pt'})
        self.driver_demand_plot.setLabel('bottom', 'RPM', **{'color': '#f59e0b', 'font-size': '12pt'})
        self.driver_demand_plot.showGrid(x=True, y=True, alpha=0.2)
        
        self.driver_demand_scatter = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
        self.driver_demand_plot.addItem(self.driver_demand_scatter)
        demand_layout.addWidget(self.driver_demand_plot)
        
        add_plot_to_grid(demand_container, 2, 0)
        
        # 6. GEAR SHIFT ANALYSIS
        gear_container = self.create_chart_container("⚙️ Gear Shift Analysis", "Green=upshift, Orange=downshift")
        gear_layout = gear_container.layout()
        
        self.gear_shift_plot = FocusablePlotWidget()
        self.gear_shift_plot.sigFocused.connect(self.on_plot_focused)
        self.gear_shift_plot.setMinimumHeight(350)
        self.gear_shift_plot.setBackground('#0a0a1a')
        self.gear_shift_plot.setLabel('left', 'Gear / RPM (÷1000)', **{'color': '#e2e8f0', 'font-size': '12pt'})
        self.gear_shift_plot.setLabel('bottom', 'Distance', units='ft', **{'color': '#3b82f6', 'font-size': '11pt'})
        self.gear_shift_plot.showGrid(x=True, y=True, alpha=0.25)
        
        self.rpm_curve = self.gear_shift_plot.plot(pen=None, brush=(59, 130, 246, 80), fillLevel=0)
        self.gear_curve = self.gear_shift_plot.plot(pen=pg.mkPen(color=(255, 255, 255), width=4), stepMode='right')
        
        self.upshift_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(16, 185, 129, 255), 
                                                   symbol='t', pen=pg.mkPen(color=(255, 255, 255), width=2))
        self.downshift_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(251, 146, 60, 255),
                                                     symbol='t1', pen=pg.mkPen(color=(255, 255, 255), width=2))
        self.gear_shift_plot.addItem(self.upshift_scatter)
        self.gear_shift_plot.addItem(self.downshift_scatter)
        
        # Real-time line
        self.gear_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.gear_shift_plot.addItem(self.gear_line)
        
        gear_layout.addWidget(self.gear_shift_plot)
        
        add_plot_to_grid(gear_container, 2, 1)
        
        # === ROW 4: Dynamic Instability + Brake Bias ===
        
        # 7. DYNAMIC INSTABILITY
        instability_container = self.create_chart_container("⚠️ Dynamic Instability Index", "Driver corrections and oversteer detection")
        instability_layout = instability_container.layout()
        
        self.instability_plot = FocusablePlotWidget()
        self.instability_plot.sigFocused.connect(self.on_plot_focused)
        self.instability_plot.setMinimumHeight(350)
        self.instability_plot.setBackground('#1a0f0f')
        self.instability_plot.setLabel('left', 'Instability Score', **{'color': '#ef4444', 'font-size': '13pt', 'font-weight': 'bold'})
        self.instability_plot.setLabel('bottom', 'Distance', units='ft', **{'color': '#f59e0b', 'font-size': '12pt', 'font-weight': 'bold'})
        self.instability_plot.showGrid(x=True, y=True, alpha=0.25)
        self.instability_plot.getAxis('left').setTextPen('#ef4444')
        self.instability_plot.getAxis('bottom').setTextPen('#f59e0b')
        
        self.instability_curve = self.instability_plot.plot(pen=None, brush=(239, 68, 68, 120), fillLevel=0)
        self.steering_curve = self.instability_plot.plot(pen=pg.mkPen(color=(6, 182, 212), width=3))
        
        self.instability_plot.addLine(y=100, pen=pg.mkPen(color=(255, 255, 255), width=2, style=Qt.PenStyle.DashLine))
        
        # Real-time line
        self.instability_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.instability_plot.addItem(self.instability_line)
        
        instability_layout.addWidget(self.instability_plot)
        
        add_plot_to_grid(instability_container, 3, 0)
        
        # 8. BRAKE BIAS
        bias_container = self.create_chart_container("⚖️ Brake Bias Trend", "Front brake percentage")
        bias_layout = bias_container.layout()
        
        self.brake_bias_plot = FocusablePlotWidget()
        self.brake_bias_plot.sigFocused.connect(self.on_plot_focused)
        self.brake_bias_plot.setMinimumHeight(350)
        self.brake_bias_plot.setBackground('#1a0f0a')
        self.brake_bias_plot.setLabel('left', 'Front Brake Bias', units='%', **{'color': '#fb923c', 'font-size': '12pt'})
        self.brake_bias_plot.setLabel('bottom', 'Distance', units='ft', **{'color': '#3b82f6', 'font-size': '11pt'})
        self.brake_bias_plot.showGrid(x=True, y=True, alpha=0.25)
        self.brake_bias_plot.setYRange(40, 80)
        
        self.brake_bias_plot.addLine(y=50, pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.PenStyle.DashLine))
        self.brake_bias_plot.addLine(y=60, pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.PenStyle.DotLine))
        
        self.brake_bias_curve = self.brake_bias_plot.plot(pen=pg.mkPen(color=(251, 146, 60), width=3), connect='finite')
        
        # Real-time line
        self.bias_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.brake_bias_plot.addItem(self.bias_line)
        
        bias_layout.addWidget(self.brake_bias_plot)
        
        add_plot_to_grid(bias_container, 3, 1)
        
        # === ROW 5: Time Delta (Full Width) ===
        
        # 9. TIME DELTA
        time_delta_container = self.create_chart_container("⏱️ Time Delta Analysis", "Compare lap times at every point on track")
        time_delta_layout = time_delta_container.layout()
        
        self.time_delta_plot = FocusablePlotWidget()
        self.time_delta_plot.sigFocused.connect(self.on_plot_focused)
        self.time_delta_plot.setMinimumHeight(350)
        self.time_delta_plot.setBackground('#1e293b')
        self.time_delta_plot.setLabel('left', 'Time Delta', units='s', **{'color': '#10b981', 'font-size': '13pt', 'font-weight': 'bold'})
        self.time_delta_plot.setLabel('bottom', 'Distance', units='ft', **{'color': '#3b82f6', 'font-size': '13pt', 'font-weight': 'bold'})
        self.time_delta_plot.showGrid(x=True, y=True, alpha=0.25)
        self.time_delta_plot.getAxis('left').setTextPen('#10b981')
        self.time_delta_plot.getAxis('bottom').setTextPen('#3b82f6')
        self.time_delta_plot.addLine(y=0, pen=pg.mkPen(color=(255, 255, 255), width=2, style=Qt.PenStyle.DashLine))
        
        self.time_delta_curve = self.time_delta_plot.plot(pen=None, brush=(59, 130, 246, 120), fillLevel=0)
        
        # Real-time line
        self.time_delta_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.time_delta_plot.addItem(self.time_delta_line)
        
        time_delta_layout.addWidget(self.time_delta_plot)
        time_delta_container.setVisible(False)
        self.time_delta_container = time_delta_container
        
        add_plot_to_grid(time_delta_container, 4, 0, 2)  # Span 2 columns
        
        # Setup hover cursor for all plots
        self.setup_hover_cursor()

    
    def set_telemetry_data(self, data):
        """Set telemetry data from loaded session"""
        if isinstance(data, list) and len(data) > 0:
            # Use the first dataset for now
            self.current_telemetry_df = data[0]
        elif isinstance(data, pd.DataFrame):
            self.current_telemetry_df = data
        else:
            return

        # Store current playback index to maintain position across vehicle switches
        # Don't reset to 0 - let update_from_playback handle the position
        # Only reset if this is truly a new session (not a vehicle switch)
        if not hasattr(self, 'current_telemetry_df') or self.current_telemetry_df is None:
            self.playback_index = 0
            self.current_lap_number = 1
        
        self._last_data_id = None  # Invalidate cache
        
        # Reset caches - force recalculation for new vehicle
        if hasattr(self, '_cache_track_brushes'): del self._cache_track_brushes
        if hasattr(self, '_cache_friction_brushes'): del self._cache_friction_brushes
        if hasattr(self, '_cache_demand_brushes'): del self._cache_demand_brushes
        
        # Update UI with new vehicle data
        self.populate_lap_selectors()
        
        # Force immediate update of all visualizations with new vehicle data
        self.update_visualizations()

    @pyqtSlot(int)
    def update_from_playback(self, current_index: int):
        """
        Called when playback position changes in Render3D.
        Updates the visualizations to show data up to current_index.
        Uses throttling to limit update rate to ~60fps.
        """
        self.playback_index = current_index
        
        # Update current lap number from the data if available
        if not self.current_telemetry_df.empty and 'lap' in self.current_telemetry_df:
            if current_index < len(self.current_telemetry_df):
                self.current_lap_number = int(self.current_telemetry_df.iat[current_index, self.current_telemetry_df.columns.get_loc('lap')])
        
        # Throttle updates - only start timer if not already running
        if hasattr(self, 'update_timer'):
            if not self.update_timer.isActive():
                self.update_timer.start(self.update_debounce_ms)
    
    def populate_lap_selectors(self):
        """Populate lap combo boxes based on available laps in telemetry"""
        self.active_lap_combo.blockSignals(True)
        self.compare_lap_combo.blockSignals(True)
        
        self.active_lap_combo.clear()
        self.compare_lap_combo.clear()
        
        # Add modes
        self.active_lap_combo.addItem("Current Lap")  # Accumulating current lap
        self.active_lap_combo.addItem("All Data")     # All data up to current point
        
        self.compare_lap_combo.addItem("None")
        
        if 'lap' in self.current_telemetry_df:
            laps = sorted(self.current_telemetry_df['lap'].unique())
            for lap in laps:
                lap_str = f"Lap {int(lap)}"
                self.active_lap_combo.addItem(lap_str)
                self.compare_lap_combo.addItem(lap_str)
        
        self.active_lap_combo.blockSignals(False)
        self.compare_lap_combo.blockSignals(False)
        
        # Default to Current Lap
        self.active_lap = "Current Lap"
        self.active_lap_combo.setCurrentText("Current Lap")

    def on_active_lap_changed(self, text: str):
        """Handle active lap selection change"""
        self.active_lap = text
        self.update_visualizations()
    
    def on_compare_lap_changed(self, text: str):
        """Handle compare lap selection change"""
        if text == "None":
            self.compare_lap = None
            self.time_delta_container.setVisible(False)
        else:
            self.compare_lap = text
            self.time_delta_container.setVisible(True)
        
        self.update_visualizations()
    
    def on_turn_changed(self, text: str):
        """Handle turn selection change"""
        self.selected_turn = text
        self.update_visualizations()
    
    def set_turn_data(self, turn_data: dict):
        """Set turn data from Tab 2 fine-tuning"""
        self.turn_data = turn_data if turn_data else {}
        
        # Update turn combo box
        self.turn_combo.blockSignals(True)
        self.turn_combo.clear()
        self.turn_combo.addItem("All Turns")
        
        if self.turn_data and 'turns' in self.turn_data:
            turn_numbers = sorted([int(k) for k in self.turn_data['turns'].keys()])
            for turn_num in turn_numbers:
                self.turn_combo.addItem(f"Turn {turn_num}")
        
        self.turn_combo.blockSignals(False)
    
    def create_chart_container(self, title: str, subtitle: str = "") -> QFrame:
        """Create a styled container for a chart with title and subtitle"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e293b, stop:0.7 #0f172a, stop:1 #1e293b);
                border-radius: 14px;
                border: 2px solid #334155;
            }
            QFrame:hover {
                border: 2px solid #3b82f6;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #243044, stop:0.7 #1a1f2e, stop:1 #243044);
            }
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        
        # Header container with gradient
        header = QWidget()
        header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(59, 130, 246, 0.15), stop:1 rgba(139, 92, 246, 0.15));
            border-radius: 8px;
            padding: 10px;
        """)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        header_layout.setSpacing(4)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #f1f5f9;
                font-size: 17pt;
                font-weight: bold;
                background: transparent;
                border: none;
                letter-spacing: 0.5px;
            }
        """)
        header_layout.addWidget(title_label)
        
        # Subtitle
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("""
                QLabel {
                    color: #94a3b8;
                    font-size: 10pt;
                    font-style: italic;
                    background: transparent;
                    border: none;
                }
            """)
            header_layout.addWidget(subtitle_label)
        
        layout.addWidget(header)
        
        return container

    def get_filtered_data(self) -> pd.DataFrame:
        """Get data filtered by active lap and turn selection - Optimized"""
        if self.current_telemetry_df.empty:
            return pd.DataFrame()
            
        # Get current lap number from telemetry at playback position
        # We access the column directly to avoid overhead
        if 'lap' in self.current_telemetry_df and self.playback_index < len(self.current_telemetry_df):
            self.current_lap_number = int(self.current_telemetry_df.iat[self.playback_index, self.current_telemetry_df.columns.get_loc('lap')])
        
        # Start with the full dataframe (reference, no copy yet)
        df = self.current_telemetry_df
        
        # Filter based on mode
        if self.active_lap == 'Current Lap':
            # Show current lap data up to playback position
            if 'lap' in df:
                # Find the start index of the current lap
                # We assume data is sorted by time/distance, so laps are contiguous
                # Optimization: Cache lap start indices if possible, but for now just find it efficiently
                
                # Fast search for lap start
                # We know the current row has self.current_lap_number
                # We can search backwards or just use boolean indexing (which is fast enough for <100k rows)
                
                # Better approach: Use the fact that we are at playback_index
                # We want [start_of_current_lap : playback_index + 1]
                
                # Find start of this lap
                # We can assume the first occurrence of this lap number is the start
                # But searching the whole array every frame is slow.
                # Let's trust the 'lap' column is monotonic increasing usually.
                
                # Optimization: Only search if lap changed or first run
                # But simpler: just use boolean mask on the 'lap' column, it's vectorized
                
                # Create a view, not a copy yet
                lap_mask = df['lap'] == self.current_lap_number
                if lap_mask.any():
                    start_pos = np.argmax(lap_mask.values) # First True
                    end_pos = min(self.playback_index, len(df) - 1)
                    
                    if end_pos >= start_pos:
                        # Return a copy only of the slice we need
                        return df.iloc[start_pos : end_pos + 1].copy()
                    else:
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()
                    
        elif self.active_lap == 'All Data':
            # Show all data up to playback position
            return df.iloc[:self.playback_index + 1].copy()
            
        elif self.active_lap.startswith('Lap '):
            # Static view of a specific lap
            lap_num = int(self.active_lap.split()[1])
            if 'lap' in df:
                return df[df['lap'] == lap_num].copy()
        
        # If we are here, we might need turn filtering on the result
        # But wait, the above returns early.
        # Logic for turn filtering needs to be applied to the result.
        
        # Let's restructure:
        # 1. Slice by Lap/Mode
        # 2. Slice by Turn
        
        result_df = pd.DataFrame()
        
        if self.active_lap == 'Current Lap':
             if 'lap' in df:
                lap_mask = df['lap'] == self.current_lap_number
                if lap_mask.any():
                    start_pos = np.argmax(lap_mask.values)
                    end_pos = min(self.playback_index, len(df) - 1)
                    if end_pos >= start_pos:
                        result_df = df.iloc[start_pos : end_pos + 1].copy()
        
        elif self.active_lap == 'All Data':
            result_df = df.iloc[:self.playback_index + 1].copy()
            
        elif self.active_lap.startswith('Lap '):
            lap_num = int(self.active_lap.split()[1])
            if 'lap' in df:
                result_df = df[df['lap'] == lap_num].copy()
        
        # Filter by turn
        if not result_df.empty and self.selected_turn != 'All Turns' and self.turn_data and 'turns' in self.turn_data:
            turn_num = int(self.selected_turn.split()[1])
            turn_key = str(turn_num)
            
            if turn_key in self.turn_data['turns']:
                turn_info = self.turn_data['turns'][turn_key]
                turn_indices = turn_info.get('indices', [])
                
                if turn_indices:
                    # Filter to only rows whose index is in the turn indices
                    result_df = result_df[result_df.index.isin(turn_indices)]
        
        return result_df

    def get_reference_lap_data(self) -> pd.DataFrame:
        """Get data for the reference/compare lap"""
        # If explicit comparison selected
        if self.compare_lap and self.compare_lap != "None":
            df = self.current_telemetry_df.copy()
            if 'lap' in df:
                try:
                    lap_num = int(self.compare_lap.split()[1])
                    df = df[df['lap'] == lap_num].copy()
                except (ValueError, IndexError):
                    return pd.DataFrame()
        else:
            # Default to best lap SO FAR (completed laps only)
            if self.current_telemetry_df.empty or 'lap' not in self.current_telemetry_df or 'elapsed_seconds' not in self.current_telemetry_df:
                return pd.DataFrame()
                
            # Find best lap among COMPLETED laps (less than current lap)
            # If we are in Lap 1, there is no best lap so far.
            current_lap = self.current_lap_number
            completed_laps = self.current_telemetry_df[self.current_telemetry_df['lap'] < current_lap]['lap'].unique()
            
            best_lap = None
            min_time = float('inf')
            
            for lap in completed_laps:
                lap_data = self.current_telemetry_df[self.current_telemetry_df['lap'] == lap]
                if not lap_data.empty:
                    time = lap_data['elapsed_seconds'].max() - lap_data['elapsed_seconds'].min()
                    if time < min_time and time > 10: # Ignore invalid short laps
                        min_time = time
                        best_lap = lap
            
            if best_lap is not None:
                df = self.current_telemetry_df[self.current_telemetry_df['lap'] == best_lap].copy()
            else:
                return pd.DataFrame()

        # Apply same turn filter if active
        if self.selected_turn != 'All Turns' and self.turn_data and 'turns' in self.turn_data:
            turn_num = int(self.selected_turn.split()[1])
            turn_key = str(turn_num)
            
            if turn_key in self.turn_data['turns']:
                turn_info = self.turn_data['turns'][turn_key]
                turn_indices = turn_info.get('indices', [])
                
                if turn_indices and not df.empty:
                    df = df[df.index.isin(turn_indices)].copy()
        
        return df

    def _check_and_update_cache(self):
        """Check if main data has changed and update static caches"""
        if self.current_telemetry_df.empty:
            return

        # Check if data changed by comparing ID or length
        current_id = id(self.current_telemetry_df)
        if hasattr(self, '_last_data_id') and self._last_data_id == current_id:
            return
            
        self._last_data_id = current_id
        df = self.current_telemetry_df
        
        # 1. Track Map Brushes
        self._cache_track_brushes = np.array([pg.mkBrush(59, 130, 246, 180)] * len(df), dtype=object)
        if 'pbrake_f' in df and 'aps' in df:
            brake = df['pbrake_f'].fillna(0).values
            throttle = df['aps'].fillna(0).values
            brake_norm = np.clip(brake / 100, 0, 1)
            throttle_norm = np.clip(throttle / 100, 0, 1)
            
            brushes = []
            for b, t in zip(brake_norm, throttle_norm):
                if b > 0.1: brushes.append(pg.mkBrush(int(255 * b), 0, 0, 180))
                elif t > 0.1: brushes.append(pg.mkBrush(0, int(255 * t), 0, 180))
                else: brushes.append(pg.mkBrush(100, 100, 255, 180))
            self._cache_track_brushes = np.array(brushes, dtype=object)
            
        # 2. Friction Circle Brushes
        self._cache_friction_brushes = np.array([pg.mkBrush(59, 130, 246, 180)] * len(df), dtype=object)
        if 'speed' in df:
            speed = df['speed'].fillna(0).values
            max_speed = speed.max() if speed.max() > 0 else 200
            brushes = []
            for s in speed:
                norm = s / max_speed
                if norm < 0.33:
                    brushes.append(pg.mkBrush(0, int(norm * 3 * 255), 255, 200))
                elif norm < 0.67:
                    brushes.append(pg.mkBrush(int((norm - 0.33) * 3 * 255), 255, int(255 - (norm - 0.33) * 3 * 255), 200))
                else:
                    brushes.append(pg.mkBrush(255, int(255 - (norm - 0.67) * 3 * 255), 0, 200))
            self._cache_friction_brushes = np.array(brushes, dtype=object)

        # 3. Driver Demand Brushes
        self._cache_demand_brushes = np.array([pg.mkBrush(59, 130, 246, 180)] * len(df), dtype=object)
        if 'accx_can' in df:
            long_g = df['accx_can'].fillna(0).values
            brushes = []
            for g in long_g:
                intensity = min(abs(g), 1.0)
                if g > 0: brushes.append(pg.mkBrush(0, int(255 * intensity), 0, 180))
                else: brushes.append(pg.mkBrush(int(255 * intensity), 0, 0, 180))
            self._cache_demand_brushes = np.array(brushes, dtype=object)

    def update_visualizations(self):
        """Update all charts and statistics"""
        self._check_and_update_cache()
        
        df = self.get_filtered_data()
        
        if df.empty:
            return
        
        self.update_statistics(df)
        
        self.update_track_map(df)
        self.update_friction_circle(df)
        self.update_instability_chart(df)
        self.update_brake_speed_chart(df)
        self.update_driver_demand(df)
        self.update_g_force_profile(df)
        self.update_brake_bias_chart(df)
        self.update_gear_shift_chart(df)
        
        # Always try to get reference data (explicit or best lap)
        ref_df = self.get_reference_lap_data()
        if not ref_df.empty:
            self.update_time_delta_chart(df, ref_df)
            
            # Only show ghost plots if explicit comparison is selected
            if self.compare_lap and self.compare_lap != "None":
                self.update_comparison_visualizations(df, ref_df)
            else:
                self.clear_ghost_plots()
        else:
            # Clear ghost plots if no reference
            self.clear_ghost_plots()
            
        # Update sector markers for distance plots
        for plot in self.distance_plots:
            self.add_sector_markers(plot)
            
        current_x = 0
        if 'Laptrigger_lapdist_dls' in df and not df.empty:
            val = df.iloc[-1]['Laptrigger_lapdist_dls']
            if pd.notna(val):
                current_x = val
        lines = [
            getattr(self, 'brake_speed_line', None),
            getattr(self, 'g_force_line', None),
            getattr(self, 'gear_line', None),
            getattr(self, 'instability_line', None),
            getattr(self, 'bias_line', None),
            getattr(self, 'time_delta_line', None)
        ]
        
        for line in lines:
            if line:
                line.setValue(current_x)
                line.show()
    
    def update_statistics(self, df: pd.DataFrame):
        """Update the statistics display with color coding"""
        if df.empty:
            # Clear stats
            for key in self.stat_labels:
                val = "50.0" if key == 'brake_bias' else "0.0"
                if key == 'max_lat_g' or key == 'lap_time' or key == 'total_time' or key == 'time_delta':
                    val = "0.00"
                if key == 'time_delta': val = "+0.00"
                if key == 'lap_number': val = "1"
                self.stat_labels[key].setText(self._format_stat_html(key, val))
            return
        
        # Check for lap change
        if 'lap' in df and not df.empty:
            new_lap = int(df.iloc[-1]['lap'])
            if new_lap != self.current_lap_number:
                # Lap changed - save previous lap data
                self.previous_lap_data = self.current_telemetry_df[
                    self.current_telemetry_df['lap'] == self.current_lap_number
                ].copy()
                self.current_lap_number = new_lap
        
        stats = self.analytics.calculate_lap_stats(df)
        
        self.stat_labels['lap_number'].setText(self._format_stat_html('lap_number', str(self.current_lap_number)))
        self.stat_labels['top_speed'].setText(self._format_stat_html('top_speed', f"{stats.get('top_speed', 0):.1f}"))
        self.stat_labels['max_lat_g'].setText(self._format_stat_html('max_lat_g', f"{stats.get('max_lat_g', 0):.2f}"))
        self.stat_labels['peak_brake'].setText(self._format_stat_html('peak_brake', f"{stats.get('peak_brake', 0):.1f}"))
        self.stat_labels['brake_bias'].setText(self._format_stat_html('brake_bias', f"{stats.get('brake_bias', 50):.1f}"))
        self.stat_labels['coasting_pct'].setText(self._format_stat_html('coasting_pct', f"{stats.get('coasting_pct', 0):.1f}"))
        
        # Lap time
        current_lap_df = self.current_telemetry_df[
            self.current_telemetry_df['lap'] == self.current_lap_number
        ].copy()
        
        if not current_lap_df.empty and 'elapsed_seconds' in current_lap_df:
            lap_start_time = current_lap_df.iloc[0]['elapsed_seconds']
            current_time = df.iloc[-1]['elapsed_seconds'] if 'elapsed_seconds' in df else 0
            lap_time = current_time - lap_start_time
        else:
            lap_time = stats.get('lap_time', 0)
        
        if lap_time >= 60:
            minutes = int(lap_time // 60)
            seconds = lap_time % 60
            lap_time_str = f"{minutes}:{seconds:05.2f}"
        else:
            lap_time_str = f"{lap_time:.2f}"
        self.stat_labels['lap_time'].setText(self._format_stat_html('lap_time', lap_time_str))
        
        # Total time
        if 'elapsed_seconds' in df and not df.empty:
            total_time = df.iloc[-1]['elapsed_seconds']
            if total_time >= 60:
                minutes = int(total_time // 60)
                seconds = total_time % 60
                total_time_str = f"{minutes}:{seconds:05.2f}"
            else:
                total_time_str = f"{total_time:.2f}"
            self.stat_labels['total_time'].setText(self._format_stat_html('total_time', total_time_str))
        else:
            self.stat_labels['total_time'].setText(self._format_stat_html('total_time', '0.00'))
        
        # Time delta
        delta_str = '+0.00'
        delta_color = self.stat_colors.get('time_delta', '#f97316')
        
        if not self.previous_lap_data.empty and 'elapsed_seconds' in df and not df.empty:
            current_lap_progress = len(current_lap_df[current_lap_df.index <= df.index[-1]])
            if current_lap_progress <= len(self.previous_lap_data):
                prev_lap_at_position = self.previous_lap_data.iloc[:current_lap_progress]
                if not prev_lap_at_position.empty and 'elapsed_seconds' in prev_lap_at_position:
                    prev_lap_start = self.previous_lap_data.iloc[0]['elapsed_seconds']
                    prev_time_at_position = prev_lap_at_position.iloc[-1]['elapsed_seconds'] - prev_lap_start
                    time_delta = lap_time - prev_time_at_position
                    
                    if abs(time_delta) < 0.01:
                        delta_str = '0.00'
                    elif time_delta > 0:
                        delta_str = f'+{time_delta:.2f}'
                        delta_color = '#dc2626'
                    else:
                        delta_str = f'{time_delta:.2f}'
                        delta_color = '#22c55e'
        
        self.stat_labels['time_delta'].setText(self._format_stat_html('time_delta', delta_str, delta_color))
        
        for label in self.stat_labels.values():
            label.show()
            label.update()
        
        # Comparison logic
        if self.compare_lap and self.compare_lap != "None":
            ref_df = self.get_reference_lap_data()
            ref_stats = self.analytics.calculate_lap_stats(ref_df)
            
            better_color = "#22c55e"
            worse_color = "#dc2626"
            
            for key in ['top_speed', 'max_lat_g']:
                value = stats.get(key, 0)
                ref_value = ref_stats.get(key, 0)
                if value > ref_value: color = better_color
                elif value < ref_value: color = worse_color
                else: color = self.stat_colors[key]
                formatted_value = f"{value:.1f}" if key == 'top_speed' else f"{value:.2f}"
                self.stat_labels[key].setText(self._format_stat_html(key, formatted_value, color))
            
            lap_time_val = stats.get('lap_time', 999)
            ref_lap_time = ref_stats.get('lap_time', 999)
            if lap_time_val < ref_lap_time: color = better_color
            elif lap_time_val > ref_lap_time: color = worse_color
            else: color = self.stat_colors['lap_time']
            self.stat_labels['lap_time'].setText(self._format_stat_html('lap_time', lap_time_str, color))
    
    def update_track_map(self, df: pd.DataFrame):
        """Update track map with cached brushes"""
        if 'VBOX_Long_Minutes' not in df or 'VBOX_Lat_Min' not in df:
            return
        
        x = df['VBOX_Long_Minutes'].values
        y = df['VBOX_Lat_Min'].values
        
        if hasattr(self, '_cache_track_brushes'):
            try:
                brushes = self._cache_track_brushes[df.index]
            except IndexError:
                brushes = [pg.mkBrush(59, 130, 246, 180)] * len(x)
        else:
            brushes = [pg.mkBrush(59, 130, 246, 180)] * len(x)
        
        sizes = np.ones(len(x)) * 5
        if hasattr(self, 'playback_index') and not df.empty:
            if df.index[-1] == self.playback_index:
                 sizes[-1] = 15
                 brushes = brushes.copy()
                 brushes[-1] = pg.mkBrush(255, 255, 0, 255)
        
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.all():
            x = x[mask]
            y = y[mask]
            sizes = sizes[mask]
            brushes = brushes[mask]
        
        if len(x) > 0:
            self.track_map_scatter.setData(x=x, y=y, brush=brushes, size=sizes)
        else:
            self.track_map_scatter.setData(x=[], y=[])
    
    def update_friction_circle(self, df: pd.DataFrame):
        """Update G-G diagram with cached brushes"""
        if 'accy_can' not in df or 'accx_can' not in df:
            return
        
        lat_g = df['accy_can'].fillna(0).values
        long_g = df['accx_can'].fillna(0).values
        
        if hasattr(self, '_cache_friction_brushes'):
            try:
                brushes = self._cache_friction_brushes[df.index]
            except IndexError:
                brushes = [pg.mkBrush(59, 130, 246, 180)] * len(lat_g)
        else:
            brushes = [pg.mkBrush(59, 130, 246, 180)] * len(lat_g)
        
        sizes = np.ones(len(lat_g)) * 4
        if hasattr(self, 'playback_index') and not df.empty:
            if df.index[-1] == self.playback_index:
                sizes[-1] = 20
                brushes = brushes.copy()
                brushes[-1] = pg.mkBrush(255, 255, 255, 255)
        
        mask = np.isfinite(lat_g) & np.isfinite(long_g)
        if not mask.all():
            lat_g = lat_g[mask]
            long_g = long_g[mask]
            sizes = sizes[mask]
            brushes = brushes[mask]
                
        if len(lat_g) > 0:
            self.friction_scatter.setData(x=lat_g, y=long_g, brush=brushes, size=sizes)
        else:
            self.friction_scatter.setData(x=[], y=[])
    
    def update_instability_chart(self, df: pd.DataFrame):
        """Update dynamic instability chart"""
        instability = self.analytics.calculate_instability_index(df)
        
        if len(instability) == 0:
            return
        
        if 'Laptrigger_lapdist_dls' in df:
            x = df['Laptrigger_lapdist_dls'].values[:len(instability)]
        else:
            x = np.arange(len(instability))
        
        self.instability_curve.setData(x=x, y=instability)
        
        if 'Steering_Angle' in df:
            steering = df['Steering_Angle'].fillna(0).abs().values[:len(x)]
            steering_scaled = steering * (instability.max() / steering.max()) if steering.max() > 0 else steering
            self.steering_curve.setData(x=x, y=steering_scaled)
    
    def update_brake_speed_chart(self, df: pd.DataFrame):
        """Update braking and speed chart"""
        if 'Laptrigger_lapdist_dls' in df:
            x = df['Laptrigger_lapdist_dls'].values
        else:
            x = np.arange(len(df))
        
        if 'speed' in df:
            self.speed_curve.setData(x=x, y=df['speed'].fillna(0).values)
        
        if 'pbrake_f' in df:
            self.brake_curve.setData(x=x, y=df['pbrake_f'].fillna(0).values)
    
    def update_brake_speed_comparison(self, active_df: pd.DataFrame, ref_df: pd.DataFrame):
        """Update ghost curves for comparison"""
        if ref_df.empty:
            self.speed_ghost_curve.setData(x=[], y=[])
            self.brake_ghost_curve.setData(x=[], y=[])
            return
        
        if 'Laptrigger_lapdist_dls' in ref_df:
            x = ref_df['Laptrigger_lapdist_dls'].values
        else:
            x = np.arange(len(ref_df))
        
        if 'speed' in ref_df:
            y_speed = ref_df['speed'].fillna(0).values
            if len(x) == len(y_speed):
                self.speed_ghost_curve.setData(x=x, y=y_speed)
        
        if 'pbrake_f' in ref_df:
            y_brake = ref_df['pbrake_f'].fillna(0).values
            if len(x) == len(y_brake):
                self.brake_ghost_curve.setData(x=x, y=y_brake)
    
    def update_driver_demand(self, df: pd.DataFrame):
        """Update driver demand scatter plot with cached brushes"""
        if 'nmot' not in df or 'aps' not in df:
            return
        
        rpm = df['nmot'].fillna(0).values
        throttle = df['aps'].fillna(0).values
        
        if hasattr(self, '_cache_demand_brushes'):
            try:
                brushes = self._cache_demand_brushes[df.index]
            except IndexError:
                brushes = [pg.mkBrush(59, 130, 246, 180)] * len(rpm)
        else:
            brushes = [pg.mkBrush(59, 130, 246, 180)] * len(rpm)
            
        sizes = np.ones(len(rpm)) * 6
        
        mask = np.isfinite(rpm) & np.isfinite(throttle)
        if not mask.all():
            rpm = rpm[mask]
            throttle = throttle[mask]
            sizes = sizes[mask]
            brushes = brushes[mask]
                
        if len(rpm) > 0:
            self.driver_demand_scatter.setData(x=rpm, y=throttle, brush=brushes, size=sizes)
        else:
            self.driver_demand_scatter.setData(x=[], y=[])
    
    def update_g_force_profile(self, df: pd.DataFrame):
        """Update G-force profile over distance"""
        if 'Laptrigger_lapdist_dls' in df:
            x = df['Laptrigger_lapdist_dls'].values
        else:
            x = np.arange(len(df))
        
        if 'accy_can' in df:
            self.lat_g_curve.setData(x=x, y=df['accy_can'].fillna(0).values)
        
        if 'accx_can' in df:
            self.long_g_curve.setData(x=x, y=df['accx_can'].fillna(0).values)
    
    def update_brake_bias_chart(self, df: pd.DataFrame):
        """Update brake bias trend (only when braking > 5 bar)"""
        if 'pbrake_f' not in df or 'pbrake_r' not in df:
            return
        
        if 'Laptrigger_lapdist_dls' in df:
            x = df['Laptrigger_lapdist_dls'].values
        else:
            x = np.arange(len(df))
        
        front = df['pbrake_f'].fillna(0).values
        rear = df['pbrake_r'].fillna(0).values
        total = front + rear
        
        bias = np.where(front > 5, (front / (total + 1e-6)) * 100, np.nan)
        
        self.brake_bias_curve.setData(x=x, y=bias)
    
    def update_gear_shift_chart(self, df: pd.DataFrame):
        """Update gear shift analysis"""
        if 'gear' not in df:
            return
        
        if 'Laptrigger_lapdist_dls' in df:
            x = df['Laptrigger_lapdist_dls'].values
        else:
            x = np.arange(len(df))
        
        # Fix: Replace 0s with previous known gear (forward fill)
        gear_series = df['gear'].replace(0, np.nan).ffill().fillna(0)
        gears = gear_series.values
        
        self.gear_curve.setData(x=x, y=gears)
        
        if 'nmot' in df:
            rpm = df['nmot'].fillna(0).values / 1000.0
            self.rpm_curve.setData(x=x, y=rpm, fillLevel=0)
            
        upshifts_x, upshifts_y = [], []
        downshifts_x, downshifts_y = [], []
        
        for i in range(1, len(gears)):
            diff = gears[i] - gears[i-1]
            if diff > 0 and diff < 3: 
                upshifts_x.append(x[i])
                upshifts_y.append(gears[i])
            elif diff < 0 and diff > -3:
                downshifts_x.append(x[i])
                downshifts_y.append(gears[i])
                
        self.upshift_scatter.setData(x=upshifts_x, y=upshifts_y)
        self.downshift_scatter.setData(x=downshifts_x, y=downshifts_y)

    def update_comparison_visualizations(self, active_df: pd.DataFrame, ref_df: pd.DataFrame):
        """Update ghost curves for all plots"""
        if ref_df.empty:
            return

        # Helper to get X and Y for ghost plots
        def get_xy(df, col):
            if 'Laptrigger_lapdist_dls' in df and col in df:
                return df['Laptrigger_lapdist_dls'].values, df[col].fillna(0).values
            return [], []

        # 1. Brake/Speed
        x, y = get_xy(ref_df, 'speed')
        if len(x) > 0: self.speed_ghost_curve.setData(x=x, y=y)
        x, y = get_xy(ref_df, 'pbrake_f')
        if len(x) > 0: self.brake_ghost_curve.setData(x=x, y=y)

        # 2. G-Force
        # Need to add ghost curves to g_force_plot first if they don't exist
        if not hasattr(self, 'lat_g_ghost'):
            self.lat_g_ghost = self.g_force_plot.plot(pen=pg.mkPen('#1e3a8a', width=1, style=Qt.PenStyle.DashLine))
            self.long_g_ghost = self.g_force_plot.plot(pen=pg.mkPen('#3f2e10', width=1, style=Qt.PenStyle.DashLine))
        
        x, y = get_xy(ref_df, 'accy_can')
        if len(x) > 0: self.lat_g_ghost.setData(x=x, y=y)
        x, y = get_xy(ref_df, 'accx_can')
        if len(x) > 0: self.long_g_ghost.setData(x=x, y=y)

        # 3. Gear
        if not hasattr(self, 'gear_ghost'):
            self.gear_ghost = self.gear_shift_plot.plot(pen=pg.mkPen('#475569', width=1, style=Qt.PenStyle.DashLine))
        
        # Fix gear 0 for ghost
        if 'gear' in ref_df:
            gear_series = ref_df['gear'].replace(0, np.nan).ffill().fillna(0)
            if 'Laptrigger_lapdist_dls' in ref_df:
                self.gear_ghost.setData(x=ref_df['Laptrigger_lapdist_dls'].values, y=gear_series.values)

        # 4. Instability
        if not hasattr(self, 'instability_ghost'):
            self.instability_ghost = self.instability_plot.plot(pen=pg.mkPen('#450a0a', width=1, style=Qt.PenStyle.DashLine))
        
        instability = self.analytics.calculate_instability_index(ref_df)
        if len(instability) > 0 and 'Laptrigger_lapdist_dls' in ref_df:
            self.instability_ghost.setData(x=ref_df['Laptrigger_lapdist_dls'].values[:len(instability)], y=instability)

    def clear_ghost_plots(self):
        """Clear all ghost plots"""
        self.speed_ghost_curve.setData(x=[], y=[])
        self.brake_ghost_curve.setData(x=[], y=[])
        if hasattr(self, 'lat_g_ghost'): self.lat_g_ghost.setData(x=[], y=[])
        if hasattr(self, 'long_g_ghost'): self.long_g_ghost.setData(x=[], y=[])
        if hasattr(self, 'gear_ghost'): self.gear_ghost.setData(x=[], y=[])
        if hasattr(self, 'instability_ghost'): self.instability_ghost.setData(x=[], y=[])

    def update_time_delta_chart(self, active_df: pd.DataFrame, ref_df: pd.DataFrame):
        """Update time delta chart (comparison mode)"""
        distances, deltas = self.analytics.calculate_time_delta(active_df, ref_df)
        
        if len(distances) == 0:
            self.time_delta_curve.setData(x=[], y=[])
            return
        
        # Create gradient fill based on delta sign
        # Green where negative (faster), red where positive (slower)
        # For area chart, use fillLevel=0
        # Create gradient fill based on delta sign
        # Green where negative (faster), red where positive (slower)
        # For area chart, use fillLevel=0
        
        # Sanitize data
        mask = np.isfinite(distances) & np.isfinite(deltas)
        if not mask.all():
            distances = distances[mask]
            deltas = deltas[mask]
            
        if len(distances) > 0:
            self.time_delta_curve.setData(x=distances, y=deltas, fillLevel=0)
        else:
            self.time_delta_curve.setData(x=[], y=[])
        
        # TODO: Add color gradient based on sign (requires custom implementation)

    def setup_hover_cursor(self):
        """Setup shared hover cursor for all distance-based plots"""
        self.hover_lines = {}
        self.hover_labels = {}
        
        # Plots that share the distance X-axis
        self.distance_plots = [
            self.brake_speed_plot,
            self.g_force_plot,
            # self.driver_demand_plot, # Exclude: RPM vs Throttle
            self.gear_shift_plot,
            self.instability_plot,
            self.brake_bias_plot,
            self.time_delta_plot
        ]
        
        # Driver demand is special (RPM x Throttle), Track Map is GPS, Friction is G-G
        # We only link distance-based plots
        
        for plot in self.distance_plots:
            # Vertical line
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#cbd5e1', width=1, style=Qt.PenStyle.DashLine))
            plot.addItem(v_line)
            v_line.hide()
            self.hover_lines[plot] = v_line
            
            # Connect hover event
            plot.scene().sigMouseMoved.connect(self.on_mouse_hover)
            
            # Add label for value display (TextItem with ignoreBounds to prevent expansion)
            label = pg.TextItem(anchor=(0, 1), color='#cbd5e1', fill=pg.mkBrush(15, 23, 42, 200))
            plot.addItem(label, ignoreBounds=True)
            label.hide()
            self.hover_labels[plot] = label

    def on_mouse_hover(self, pos):
        """Handle mouse hover to update cursor across all plots"""
        sender_scene = self.sender()
        
        # Find which plot triggered the event
        active_plot = None
        for plot in self.distance_plots:
            if plot.scene() == sender_scene:
                active_plot = plot
                break
        
        if not active_plot:
            return

        if active_plot.sceneBoundingRect().contains(pos):
            mouse_point = active_plot.plotItem.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            
            # Find nearest data point in current dataframe
            df = self.get_filtered_data()
            if df.empty or 'Laptrigger_lapdist_dls' not in df:
                return
                
            # Find index of nearest distance
            distances = df['Laptrigger_lapdist_dls'].values
            idx = (np.abs(distances - x_val)).argmin()
            
            # Get values at this index
            row = df.iloc[idx]
            
            # Get reference values if available
            ref_row = None
            ref_df = self.get_reference_lap_data()
            if not ref_df.empty and 'Laptrigger_lapdist_dls' in ref_df:
                ref_distances = ref_df['Laptrigger_lapdist_dls'].values
                ref_idx = (np.abs(ref_distances - x_val)).argmin()
                if abs(ref_distances[ref_idx] - x_val) < 20: # Only if close enough
                    ref_row = ref_df.iloc[ref_idx]

            # Update all lines and labels
            for p in self.distance_plots:
                if p in self.hover_lines:
                    self.hover_lines[p].setPos(x_val)
                    self.hover_lines[p].show()
                
                if p in self.hover_labels:
                    label = self.hover_labels[p]
                    label.setPos(x_val, p.getAxis('left').range[1]) # Top of plot
                    label.setZValue(100) # Ensure on top
                    
                    # Determine what text to show based on plot type
                    text = ""
                    
                    def fmt_val(name, val, unit, ref_val=None):
                        s = f"{name}: {val}{unit}"
                        if ref_val is not None:
                            diff = val - ref_val
                            sign = "+" if diff > 0 else ""
                            s += f" ({sign}{diff:.1f})"
                        return s

                    if p == self.brake_speed_plot:
                        speed = row.get('speed', 0)
                        brake = row.get('pbrake_f', 0)
                        ref_speed = ref_row.get('speed', 0) if ref_row is not None else None
                        text = fmt_val("Speed", f"{speed:.1f}", "mph", ref_speed) + "\n"
                        text += f"Brake: {brake:.1f} bar"
                    elif p == self.g_force_plot:
                        lat = row.get('accy_can', 0)
                        long = row.get('accx_can', 0)
                        text = f"Lat G: {lat:.2f}\nLong G: {long:.2f}"
                    elif p == self.gear_shift_plot:
                        gear = row.get('gear', 0)
                        rpm = row.get('nmot', 0)
                        # Handle NaN values
                        if pd.isna(gear): gear = 0
                        if pd.isna(rpm): rpm = 0
                        text = f"Gear: {int(gear)}\nRPM: {int(rpm)}"
                    elif p == self.instability_plot:
                        steer = row.get('Steering_Angle', 0)
                        text = f"Steer: {steer:.1f}°"
                    elif p == self.brake_bias_plot:
                        front = row.get('pbrake_f', 0)
                        rear = row.get('pbrake_r', 0)
                        total = front + rear
                        bias = (front / total * 100) if total > 5 else 50
                        text = f"Bias: {bias:.1f}%"
                    elif p == self.time_delta_plot:
                        text = f"Dist: {x_val:.0f}m"
                        if ref_row is not None:
                             t1 = row.get('elapsed_seconds', 0)
                             t2 = ref_row.get('elapsed_seconds', 0)
                             t1 -= df.iloc[0].get('elapsed_seconds', 0)
                             t2 -= ref_df.iloc[0].get('elapsed_seconds', 0)
                             delta = t1 - t2
                             sign = "+" if delta > 0 else ""
                             text += f"\nDelta: {sign}{delta:.2f}s"
                        
                    label.setText(text)
                    label.show()

    def add_sector_markers(self, plot_widget):
        """Add vertical lines/regions for turns/sectors"""
        # Check if we have turn data
        if not self.turn_data or 'turns' not in self.turn_data:
            return

        # We need a way to track items to remove them later
        if not hasattr(plot_widget, '_sector_items'):
            plot_widget._sector_items = []
        
        # Remove old items
        for item in plot_widget._sector_items:
            plot_widget.removeItem(item)
        plot_widget._sector_items = []
        
        # Add new markers
        for turn_id, turn_info in self.turn_data['turns'].items():
            # We need distance for start/end of turn
            if 'indices' in turn_info and not self.current_telemetry_df.empty:
                indices = turn_info['indices']
                if not indices: continue
                
                # Get min/max index
                start_idx = min(indices)
                end_idx = max(indices)
                
                # Get distance at these indices
                if 'Laptrigger_lapdist_dls' in self.current_telemetry_df:
                    try:
                        # Ensure indices are valid
                        if start_idx in self.current_telemetry_df.index and end_idx in self.current_telemetry_df.index:
                            start_dist = self.current_telemetry_df.at[start_idx, 'Laptrigger_lapdist_dls']
                            end_dist = self.current_telemetry_df.at[end_idx, 'Laptrigger_lapdist_dls']
                            
                            # Add region
                            region = pg.LinearRegionItem([start_dist, end_dist], brush=pg.mkBrush(59, 130, 246, 20), movable=False)
                            # Set pen for the boundary lines
                            for line in region.lines:
                                line.setPen(pg.mkPen(59, 130, 246, 50))
                            plot_widget.addItem(region)
                            plot_widget._sector_items.append(region)
                            
                            # Add label (text item)
                            # Note: TextItem position is x, y. Y needs to be within view range.
                            # We'll place it at the top.
                            text = pg.TextItem(f"T{turn_id}", color='#64748b', anchor=(0.5, 0))
                            # Use a fixed Y or try to get view range. 
                            # Since view range changes, maybe just put it at 0 if we can't determine.
                            # Better: Use viewbox limits if available, or just don't add text for now if it's tricky.
                            # Let's try adding it and setting a reasonable Y or using view coordinates.
                            # Actually, let's just stick to the region for now to avoid clutter/positioning issues.
                            
                    except (KeyError, IndexError):
                        pass
