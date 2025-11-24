import json
import os
import pickle
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QFileDialog, QMessageBox, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel)
from PyQt6.QtGui import QAction
from Code.GUI.TrackScanner import TrackScanner
from Code.GUI.FineTuner import FineTuner
from Code.GUI.Render3D import Render3D
from Code.GUI.RaceTelemetryTab import RaceTelemetryTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RaceTrack Studio")
        self.resize(1200, 800)
        
        self.init_ui()
        self.track_data = None
        self.finalized_data = None
        self.unsaved_changes = False

    def init_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        save_action = QAction("Save Session", self)
        save_action.triggered.connect(self.save_session)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Session", self)
        load_action.triggered.connect(self.load_session)
        file_menu.addAction(load_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.scanner_tab = TrackScanner()
        self.scanner_tab.trackFinalized.connect(self.on_track_finalized)
        self.scanner_tab.trackLoaded.connect(self.on_track_loaded)
        
        self.fine_tuner_tab = FineTuner()
        self.fine_tuner_tab.fineTuneFinalized.connect(self.on_fine_tune_finalized)
        
        self.render_tab = Render3D()
        
        self.telemetry_tab = RaceTelemetryTab()
        
        # Connect signals from render tab to telemetry tab
        self.render_tab.telemetryDataLoaded.connect(self.telemetry_tab.set_telemetry_data)
        self.render_tab.playbackPositionChanged.connect(self.telemetry_tab.update_from_playback)
        
        self.tabs.addTab(self.scanner_tab, "1. Scan & Finalize")
        self.tabs.addTab(self.fine_tuner_tab, "2. Fine-Tune")
        self.tabs.addTab(self.render_tab, "3. 3D Render")
        self.tabs.addTab(self.telemetry_tab, "4. Race Telemetry")
        
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)

    def on_track_loaded(self, data):
        self.track_data = data
        self.unsaved_changes = True
        self.setWindowTitle("RaceTrack Studio *")
        # Reset other tabs
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setTabEnabled(3, False)

    def on_track_finalized(self, data):
        self.finalized_data = data
        self.unsaved_changes = True
        self.setWindowTitle("RaceTrack Studio *")
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.tabs.setTabEnabled(3, True)
        
        self.fine_tuner_tab.set_data(data)
        self.render_tab.set_data(data)
        
        # Pass turn data to telemetry tab if available
        if hasattr(self, 'telemetry_tab') and self.telemetry_tab is not None:
            self.telemetry_tab.set_turn_data(data)

    def on_fine_tune_finalized(self, data):
        self.finalized_data = data
        self.unsaved_changes = True
        self.setWindowTitle("RaceTrack Studio *")
        self.render_tab.set_data(data)
        self.render_tab.set_turn_data(data)  # Pass turn data to 3D render
        self.scanner_tab.viewer.set_finalized_data(data)
        
        # Pass turn data to telemetry tab
        if hasattr(self, 'telemetry_tab') and self.telemetry_tab is not None:
            self.telemetry_tab.set_turn_data(data)

    def save_session(self):
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Session", "", 
            "SARK Files (*.sark);;JSON Files (*.json)"
        )
        if not file_path:
            return
        
        # Determine format based on selection or extension
        is_sark = selected_filter.startswith("SARK") or file_path.endswith('.sark')
        
        session = {
            'track_data': self.track_data,
            'finalized_data': self.finalized_data,
            'inputs': self.scanner_tab.get_inputs(),
            'telemetry_state': self.render_tab.get_session_state()  # Include telemetry data
        }
        
        try:
            if is_sark:
                # Save as pickle with .sark extension
                with open(file_path, 'wb') as f:
                    pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                file_size = Path(file_path).stat().st_size
                
                # Check if telemetry data was embedded
                telemetry_state = session.get('telemetry_state', {})
                has_telemetry = 'telemetry_data' in telemetry_state and telemetry_state['telemetry_data']
                
                self.unsaved_changes = False
                self.setWindowTitle("RaceTrack Studio")
                
                msg = f"Session saved successfully as SARK file.\n\n"
                msg += f"File: {file_path}\n"
                msg += f"Size: {file_size:,} bytes\n"
                if has_telemetry:
                    num_races = len(telemetry_state['telemetry_data'])
                    msg += f"\n✓ Includes all telemetry data ({num_races} race(s))"
                
                QMessageBox.information(self, "Saved", msg)
            else:
                # Save as JSON (legacy format - without telemetry state)
                json_session = {
                    'track_data': self.track_data,
                    'finalized_data': self.finalized_data,
                    'inputs': self.scanner_tab.get_inputs()
                }
                with open(file_path, 'w') as f:
                    json.dump(json_session, f, indent=2)
                self.unsaved_changes = False
                self.setWindowTitle("RaceTrack Studio")
                QMessageBox.information(self, "Saved", "Session saved successfully as JSON.\n(Note: Telemetry data not included in JSON format)")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to save: {e}\n\n{traceback.format_exc()}")

    def load_session(self):
        if self.unsaved_changes:
            reply = QMessageBox.question(self, 'Unsaved Changes', 
                                         "You have unsaved changes. Do you want to save before loading?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self.save_session()
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", 
            "All Supported (*.sark *.json);;SARK Files (*.sark);;JSON Files (*.json)"
        )
        if not file_path:
            return
            
        try:
            # Detect file format
            if file_path.endswith('.sark'):
                # Load pickled SARK file
                with open(file_path, 'rb') as f:
                    session = pickle.load(f)
                format_name = "SARK"
            else:
                # Load JSON file
                with open(file_path, 'r') as f:
                    session = json.load(f)
                format_name = "JSON"
            
            self.track_data = session.get('track_data')
            self.finalized_data = session.get('finalized_data')
            inputs = session.get('inputs')
            telemetry_state = session.get('telemetry_state')
            
            # Restore scanner tab
            self.scanner_tab.set_data(self.track_data, inputs)
            
            # Restore finalized data to all tabs
            if self.finalized_data:
                self.on_track_finalized(self.finalized_data)
            
            # Restore telemetry state (only available in SARK files)
            if telemetry_state:
                self.render_tab.restore_session_state(telemetry_state)
                info_msg = f"Session loaded successfully from {format_name} file.\nTelemetry data and selections restored."
            else:
                info_msg = f"Session loaded successfully from {format_name} file."
            
            self.unsaved_changes = False
            self.setWindowTitle("RaceTrack Studio")
            QMessageBox.information(self, "Loaded", info_msg)
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Failed to load: {e}\n\n{traceback.format_exc()}")

    def closeEvent(self, event):
        if self.unsaved_changes:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                         "You have unsaved changes. Do you want to save before exiting?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)

            if reply == QMessageBox.StandardButton.Yes:
                self.save_session()
                event.accept()
            elif reply == QMessageBox.StandardButton.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
