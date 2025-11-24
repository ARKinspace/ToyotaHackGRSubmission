from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QGroupBox, QMessageBox, QFormLayout, QScrollArea, QApplication, QCheckBox)
from PyQt6.QtCore import pyqtSignal, Qt
from Code.Core.MapCreator.track_fetcher import TrackFetcher
from Code.Core.MapCreator.track_processor import TrackProcessor
from Code.GUI.TrackViewer import TrackViewer
from Code.Core.OptimalLine import OptimalLineGenerator, WeatherParser
import pandas as pd
import json
from pathlib import Path

class TrackScanner(QWidget):
    trackFinalized = pyqtSignal(dict)  # Emits finalized data
    trackLoaded = pyqtSignal(dict)     # Emits raw track data when scanned

    def __init__(self):
        super().__init__()
        self.fetcher = TrackFetcher()
        self.processor = TrackProcessor()
        self.current_track_data = None
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Sidebar (Inputs)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setFixedWidth(300)
        
        # Search Section
        search_group = QGroupBox("Coordinate Scan")
        search_layout = QFormLayout()
        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("Latitude")
        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("Longitude")
        self.scan_btn = QPushButton("🔍 Scan")
        self.scan_btn.clicked.connect(self.scan_track)
        
        search_layout.addRow("Lat:", self.lat_input)
        search_layout.addRow("Lon:", self.lon_input)
        search_layout.addRow(self.scan_btn)
        search_group.setLayout(search_layout)
        sidebar_layout.addWidget(search_group)
        
        # Technical Data Section
        tech_group = QGroupBox("Technical Data (Mandatory)")
        tech_layout = QFormLayout()
        
        self.sector1_input = QLineEdit()
        self.sector2_input = QLineEdit()
        self.sector3_input = QLineEdit()
        self.length_input = QLineEdit()
        
        tech_layout.addRow("Sector 1 (in):", self.sector1_input)
        tech_layout.addRow("Sector 2 (in):", self.sector2_input)
        tech_layout.addRow("Sector 3 (in):", self.sector3_input)
        tech_layout.addRow("Total Length (mi):", self.length_input)
        
        # Flat track option
        self.flat_track_checkbox = QCheckBox("Flat Track (Skip Elevation)")
        self.flat_track_checkbox.setToolTip("Check this to create a flat track without fetching elevation data from OSM")
        tech_layout.addRow("", self.flat_track_checkbox)
        
        tech_group.setLayout(tech_layout)
        sidebar_layout.addWidget(tech_group)
        
        # Pit GPS (Optional)
        pit_group = QGroupBox("Pit Lane GPS (Optional)")
        pit_layout = QFormLayout()
        self.pit_in_lat = QLineEdit()
        self.pit_in_lon = QLineEdit()
        self.pit_out_lat = QLineEdit()
        self.pit_out_lon = QLineEdit()
        pit_layout.addRow("Pit In Lat:", self.pit_in_lat)
        pit_layout.addRow("Pit In Lon:", self.pit_in_lon)
        pit_layout.addRow("Pit Out Lat:", self.pit_out_lat)
        pit_layout.addRow("Pit Out Lon:", self.pit_out_lon)
        pit_group.setLayout(pit_layout)
        sidebar_layout.addWidget(pit_group)
        
        # Editor Actions
        edit_group = QGroupBox("Edit Segment")
        edit_layout = QVBoxLayout()
        
        width_layout = QHBoxLayout()
        self.edit_width_input = QLineEdit()
        self.edit_width_input.setPlaceholderText("Width (m)")
        self.set_width_btn = QPushButton("Set")
        self.set_width_btn.clicked.connect(self.update_segment_width)
        width_layout.addWidget(self.edit_width_input)
        width_layout.addWidget(self.set_width_btn)
        
        self.toggle_type_btn = QPushButton("Toggle Type (Track/Pit)")
        self.toggle_type_btn.clicked.connect(self.toggle_segment_type)
        
        self.delete_seg_btn = QPushButton("Delete Segment")
        self.delete_seg_btn.setStyleSheet("background-color: #ef4444; color: white;")
        self.delete_seg_btn.clicked.connect(self.delete_segment)
        
        edit_layout.addLayout(width_layout)
        edit_layout.addWidget(self.toggle_type_btn)
        edit_layout.addWidget(self.delete_seg_btn)
        edit_group.setLayout(edit_layout)
        sidebar_layout.addWidget(edit_group)
        
        # Disable edit buttons initially
        self.set_edit_enabled(False)
        
        # Actions
        action_layout = QHBoxLayout()
        self.finalize_btn = QPushButton("✓ Finalize Track")
        self.finalize_btn.clicked.connect(self.finalize_track)
        self.finalize_btn.setEnabled(False)
        self.finalize_btn.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 8px;")
        
        self.edit_raw_btn = QPushButton("✎ Edit Raw")
        self.edit_raw_btn.clicked.connect(self.edit_raw_track)
        self.edit_raw_btn.setEnabled(False)
        
        action_layout.addWidget(self.edit_raw_btn)
        action_layout.addWidget(self.finalize_btn)
        sidebar_layout.addLayout(action_layout)
        
        sidebar_layout.addStretch()
        
        # Main Area (Viewer)
        self.viewer = TrackViewer()
        self.viewer.segmentSelected.connect(self.on_segment_selected)
        self.viewer.segmentDeleted.connect(self.delete_segment)
        
        layout.addWidget(sidebar)
        layout.addWidget(self.viewer)

    def scan_track(self):
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid coordinates.")
            return

        self.scan_btn.setText("Scanning...")
        self.scan_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            data = self.fetcher.fetch_by_coords(lat, lon)
            if not data:
                raise Exception("No track data found nearby.")
            
            pit_anchor = None
            if self.pit_in_lat.text() and self.pit_in_lon.text():
                try:
                    pit_anchor = {'lat': float(self.pit_in_lat.text()), 'lon': float(self.pit_in_lon.text())}
                except:
                    pass

            processed = self.processor.process_osm_data(data, "Scanned Track", pit_anchor)
            if not processed:
                raise Exception("Failed to process track geometry.")
            
            self.current_track_data = processed
            self.viewer.set_track_data(processed)
            self.finalize_btn.setEnabled(True)
            self.trackLoaded.emit(processed)
            
        except Exception as e:
            QMessageBox.critical(self, "Scan Error", str(e))
        finally:
            self.scan_btn.setText("🔍 Scan")
            self.scan_btn.setEnabled(True)

    def finalize_track(self):
        if not self.current_track_data:
            return

        # Validate Mandatory Inputs
        try:
            s1 = float(self.sector1_input.text())
            s2 = float(self.sector2_input.text())
            s3 = float(self.sector3_input.text())
            length = float(self.length_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "All Technical Data fields (Sectors, Length) are MANDATORY.")
            return

        try:
            sf_lat = float(self.lat_input.text())
            sf_lon = float(self.lon_input.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Start/Finish coordinates required for anchoring.")
            return

        # Check if user wants flat track (no elevation)
        use_flat_track = self.flat_track_checkbox.isChecked()
        
        finalized = self.processor.finalize_track(
            self.current_track_data,
            sf_lat, sf_lon,
            s1, s2, s3, length,
            fetcher=None if use_flat_track else self.fetcher  # Pass None to skip elevation
        )
        
        if finalized:
            # Calculate optimal racing line
            optimal_line_data = self._calculate_optimal_line(finalized)
            if optimal_line_data is not None:
                finalized['optimalLine'] = optimal_line_data
                print(f"[TrackScanner] Optimal line calculated successfully")
            else:
                print(f"[TrackScanner] Optimal line calculation skipped or failed")
            
            self.viewer.set_finalized_data(finalized)
            self.trackFinalized.emit(finalized)
            self.edit_raw_btn.setEnabled(True)
            QMessageBox.information(self, "Success", "Track Finalized Successfully!")
        else:
            QMessageBox.critical(self, "Error", "Failed to finalize track.")
    
    def _calculate_optimal_line(self, finalized_data):
        """
        Calculate optimal racing line using weather-adjusted physics.
        
        Args:
            finalized_data (dict): Finalized track data with splinePoints
        
        Returns:
            dict: Optimal line data with coordinates and metadata, or None if failed
        """
        try:
            # Extract spline points
            spline_points = finalized_data.get('splinePoints', [])
            if not spline_points or len(spline_points) < 10:
                print("[TrackScanner] Insufficient spline points for optimal line")
                return None
            
            # Create DataFrame from spline points
            centerline_df = pd.DataFrame({
                'x': [p['x'] for p in spline_points],
                'y': [p['y'] for p in spline_points],
                'distance': [p['dist'] for p in spline_points]
            })
            
            # Estimate average track width
            track_width = 14.0  # Default
            if spline_points[0].get('width'):
                widths = [p.get('width', 14.0) for p in spline_points if p.get('width')]
                if widths:
                    track_width = sum(widths) / len(widths)
            
            # Load vehicle configuration
            vehicle_config = self._load_vehicle_config()
            
            # Try to get weather data
            weather_config = self._get_weather_data()
            
            # Create optimal line generator
            generator = OptimalLineGenerator(
                track_centerline=centerline_df,
                track_width=track_width,
                vehicle_config=vehicle_config,
                weather_config=weather_config
            )
            
            # Generate optimal line (2000 points for smooth visualization)
            optimal_line_df = generator.generate_optimal_line(n_points=2000)
            
            # Convert to dict format for storage
            optimal_line_dict = {
                'x': optimal_line_df['x'].tolist(),
                'y': optimal_line_df['y'].tolist(),
                'distance': optimal_line_df['distance'].tolist(),
                'speed': optimal_line_df['speed'].tolist(),
                'curvature': optimal_line_df['curvature'].tolist(),
                'grip_coefficient': float(optimal_line_df['grip_coefficient'].iloc[0]),
                'lap_time': float(optimal_line_df['lap_time'].iloc[0]),
                'track_width': track_width,
                'weather': weather_config,
                'vehicle': vehicle_config['name']
            }
            
            return optimal_line_dict
            
        except Exception as e:
            print(f"[TrackScanner] Error calculating optimal line: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_vehicle_config(self):
        """Load vehicle configuration from reference folder."""
        try:
            config_path = Path(__file__).parent.parent.parent / "reference" / "vehicle_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    # Flatten the config structure
                    config = {}
                    config.update(data.get('vehicle', {}))
                    config.update(data.get('tires', {}))
                    config.update(data.get('performance_limits', {}))
                    return config
        except Exception as e:
            print(f"[TrackScanner] Could not load vehicle_config.json: {e}")
        
        # Return default config
        return {
            'name': 'Toyota GR86 Cup Car',
            'mass_kg': 1270,
            'power_hp': 228,
            'top_speed_ms': 67.0,
            'tire_friction_dry': 1.40,
            'tire_friction_wet': 0.70,
            'tire_friction_intermediate': 1.00,
            'optimal_tire_temp': 85,
            'max_lateral_g': 1.35,
            'max_brake_g': 1.50,
            'max_accel_g': 0.85,
            'corner_speed_factor': 0.92
        }
    
    def _get_weather_data(self):
        """Try to get weather data for the current track."""
        try:
            # Try to determine track name from the track data
            track_name = self.current_track_data.get('name', '')
            
            if track_name:
                parser = WeatherParser()
                weather = parser.parse_weather_for_track(track_name)
                if weather:
                    return weather
        except Exception as e:
            print(f"[TrackScanner] Could not load weather data: {e}")
        
        # Return default ideal conditions
        return {
            'track_temp': 85.0,  # Optimal
            'air_temp': 25.0,
            'humidity': 50,
            'rainfall': 0.0,
            'wind_speed': 0.0,
            'wind_direction': 0,
            'pressure': 1013.25
        }

    def edit_raw_track(self):
        if self.current_track_data:
            self.viewer.set_track_data(self.current_track_data)
            self.edit_raw_btn.setEnabled(False)
            
    def set_data(self, track_data, inputs):
        """Restore state from saved session"""
        self.current_track_data = track_data
        self.viewer.set_track_data(track_data)
        
        if inputs:
            self.lat_input.setText(str(inputs.get('lat', '')))
            self.lon_input.setText(str(inputs.get('lon', '')))
            self.sector1_input.setText(str(inputs.get('sector1', '')))
            self.sector2_input.setText(str(inputs.get('sector2', '')))
            self.sector3_input.setText(str(inputs.get('sector3', '')))
            self.length_input.setText(str(inputs.get('length', '')))
            self.pit_in_lat.setText(str(inputs.get('pit_in_lat', '')))
            self.pit_in_lon.setText(str(inputs.get('pit_in_lon', '')))
            self.pit_out_lat.setText(str(inputs.get('pit_out_lat', '')))
            self.pit_out_lon.setText(str(inputs.get('pit_out_lon', '')))
            
        if track_data:
            self.finalize_btn.setEnabled(True)

    def set_edit_enabled(self, enabled):
        self.edit_width_input.setEnabled(enabled)
        self.set_width_btn.setEnabled(enabled)
        self.toggle_type_btn.setEnabled(enabled)
        self.delete_seg_btn.setEnabled(enabled)

    def on_segment_selected(self, seg_id):
        self.set_edit_enabled(True)
        # Get current width
        if self.current_track_data:
            for p in self.current_track_data['paths']:
                if p['id'] == seg_id:
                    self.edit_width_input.setText(str(p.get('widthValue', 12)))
                    break

    def update_segment_width(self):
        if not self.viewer.selected_segment_id: return
        try:
            new_width = float(self.edit_width_input.text())
            self.viewer.update_segment_width(self.viewer.selected_segment_id, new_width)
            # Update data model
            for p in self.current_track_data['paths']:
                if p['id'] == self.viewer.selected_segment_id:
                    p['widthValue'] = new_width
                    break
        except ValueError:
            pass

    def toggle_segment_type(self):
        if not self.viewer.selected_segment_id: return
        self.viewer.toggle_segment_type(self.viewer.selected_segment_id)
        # Update data model
        for p in self.current_track_data['paths']:
            if p['id'] == self.viewer.selected_segment_id:
                p['type'] = 'pit' if p['type'] == 'track' else 'track'
                break

    def delete_segment(self):
        if not self.viewer.selected_segment_id: return
        seg_id = self.viewer.selected_segment_id
        self.viewer.delete_segment(seg_id)
        # Update data model
        self.current_track_data['paths'] = [p for p in self.current_track_data['paths'] if p['id'] != seg_id]
        self.set_edit_enabled(False)

    def get_inputs(self):
        return {
            'lat': self.lat_input.text(),
            'lon': self.lon_input.text(),
            'sector1': self.sector1_input.text(),
            'sector2': self.sector2_input.text(),
            'sector3': self.sector3_input.text(),
            'length': self.length_input.text(),
            'pit_in_lat': self.pit_in_lat.text(),
            'pit_in_lon': self.pit_in_lon.text(),
            'pit_out_lat': self.pit_out_lat.text(),
            'pit_out_lon': self.pit_out_lon.text()
        }
