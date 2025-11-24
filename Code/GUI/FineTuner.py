from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QGroupBox, QMessageBox, QFormLayout)
from PyQt6.QtCore import pyqtSignal
from Code.GUI.TrackViewer import TrackViewer

class FineTuner(QWidget):
    fineTuneFinalized = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.current_data = None
        self.node_a_index = -1
        self.node_b_index = -1
        self.selected_turn_num = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("Width (m)")
        
        self.update_btn = QPushButton("Update Width Between Nodes")
        self.update_btn.clicked.connect(self.update_selected_width)
        self.update_btn.setEnabled(False)
        
        self.invert_btn = QPushButton("⇄ Invert Selection")
        self.invert_btn.clicked.connect(self.invert_selection)
        self.invert_btn.setEnabled(False)
        self.invert_btn.setStyleSheet("background-color: #6366f1; color: white; font-weight: bold; padding: 6px;")
        
        self.create_turn_btn = QPushButton("📍 Create Turn")
        self.create_turn_btn.clicked.connect(self.create_turn)
        self.create_turn_btn.setEnabled(False)
        self.create_turn_btn.setStyleSheet("background-color: #f59e0b; color: white; font-weight: bold; padding: 6px;")
        
        self.create_turn_btn = QPushButton("📍 Create Turn")
        self.create_turn_btn.clicked.connect(self.create_turn)
        self.create_turn_btn.setEnabled(False)
        self.create_turn_btn.setStyleSheet("background-color: #f59e0b; color: white; font-weight: bold; padding: 6px;")
        
        self.delete_turn_btn = QPushButton("🗑️ Delete Turn")
        self.delete_turn_btn.clicked.connect(self.delete_turn)
        self.delete_turn_btn.setEnabled(False)
        self.delete_turn_btn.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold; padding: 6px;")
        
        # Add to main sidebar layout directly to ensure visibility
        sidebar_layout.addWidget(QLabel("<b>Edit Controls</b>"))
        sidebar_layout.addWidget(self.width_input)
        sidebar_layout.addWidget(self.update_btn)
        sidebar_layout.addWidget(self.invert_btn)
        sidebar_layout.addWidget(self.create_turn_btn)
        sidebar_layout.addWidget(self.delete_turn_btn)
        sidebar_layout.addWidget(self.delete_turn_btn)
        sidebar_layout.addWidget(self.create_turn_btn)
        
        # Remove old groupbox code
        # edit_group = QGroupBox("Edit Node Width")
        # control_layout = QVBoxLayout()
        # ...
        # edit_group.setLayout(control_layout)
        # sidebar_layout.addWidget(edit_group)
        
        self.info_label = QLabel("Left-click: Place Node A (Blue)\nCtrl+Click: Place Node B (Orange)\nESC: Clear both nodes")
        self.info_label.setWordWrap(True)
        sidebar_layout.addWidget(self.info_label)

        self.finalize_btn = QPushButton("✓ Finalize Fine-Tuning")
        self.finalize_btn.clicked.connect(self.finalize_tuning)
        self.finalize_btn.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 8px;")
        sidebar_layout.addWidget(self.finalize_btn)
        
        sidebar_layout.addStretch()
        
        # Viewer
        self.viewer = TrackViewer()
        self.viewer.set_fine_tune_mode(True)
        self.viewer.twoNodesPlaced.connect(self.on_two_nodes_placed)
        self.viewer.turnSelected.connect(self.on_turn_selected)
        
        layout.addWidget(sidebar)
        layout.addWidget(self.viewer)

    def set_data(self, finalized_data):
        self.current_data = finalized_data
        self.viewer.set_finalized_data(finalized_data)
        self.viewer.set_fine_tune_mode(True)
        self.width_input.clear()
        self.node_a_index = -1
        self.node_b_index = -1
        self.info_label.setText("Left-click: Place Node A (Blue)\nCtrl+Click: Place Node B (Orange)\nESC: Clear both nodes")

    def on_two_nodes_placed(self, node_a_idx, node_b_idx):
        self.node_a_index = node_a_idx
        self.node_b_index = node_b_idx
        
        if node_a_idx == -1 and node_b_idx == -1:
            # Both nodes cleared
            self.info_label.setText("Left-click: Place Node A (Blue)\nCtrl+Click: Place Node B (Orange)\nESC: Clear both nodes")
            self.update_btn.setEnabled(False)
            self.invert_btn.setEnabled(False)
            self.create_turn_btn.setEnabled(False)
        elif node_a_idx >= 0 and node_b_idx >= 0:
            # Both nodes placed
            start = min(node_a_idx, node_b_idx)
            end = max(node_a_idx, node_b_idx)
            count = end - start + 1
            self.info_label.setText(f"Node A: {node_a_idx}\nNode B: {node_b_idx}\nRange: {count} nodes")
            self.update_btn.setEnabled(True)
            self.invert_btn.setEnabled(True)
            self.create_turn_btn.setEnabled(True)
        elif node_a_idx >= 0:
            # Only Node A placed
            self.info_label.setText(f"Node A: {node_a_idx}\nCtrl+Click to place Node B")
            self.update_btn.setEnabled(False)
            self.invert_btn.setEnabled(False)
            self.create_turn_btn.setEnabled(False)
        elif node_b_idx >= 0:
            # Only Node B placed
            self.info_label.setText(f"Node B: {node_b_idx}\nLeft-click to place Node A")
            self.update_btn.setEnabled(False)
            self.invert_btn.setEnabled(False)
            self.create_turn_btn.setEnabled(False)

    def update_selected_width(self):
        if self.node_a_index == -1 or self.node_b_index == -1 or not self.current_data:
            return
            
        try:
            new_width = float(self.width_input.text())
            if new_width <= 0:
                QMessageBox.warning(self, "Invalid Width", "Width must be positive.")
                return
            
            # Get the actual selected indices from the viewer (which handles inversion)
            indices_to_update = [self.node_a_index, self.node_b_index] + self.viewer.between_nodes_indices
            
            # All nodes are now in splinePoints (including pit lane nodes)
            points = self.current_data['splinePoints']
            pit_lanes_updated = set()
            
            # Apply width to selected nodes
            for idx in indices_to_update:
                if 0 <= idx < len(points):
                    points[idx]['width'] = new_width
                    # Track which pit lanes are affected
                    if 'pit_lane' in points[idx]:
                        pit_lanes_updated.add(points[idx]['pit_lane'])
            
            # Sync updated pit lane widths back to visualPaths
            if pit_lanes_updated and 'visualPaths' in self.current_data:
                for path in self.current_data['visualPaths']:
                    if path['id'] in pit_lanes_updated:
                        path['widthValue'] = new_width
                        path['width'] = new_width
            
            count = len(indices_to_update)
            self.viewer.set_finalized_data(self.current_data)
            
            if pit_lanes_updated:
                pit_names = ', '.join(pit_lanes_updated)
                self.info_label.setText(f"Updated {count} nodes to {new_width}m width (pit lanes: {pit_names}).")
            else:
                self.info_label.setText(f"Updated {count} track nodes to {new_width}m width.")
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for width.")

    def finalize_tuning(self):
        if self.current_data:
            self.fineTuneFinalized.emit(self.current_data)
            QMessageBox.information(self, "Success", "Fine-tuning finalized!")
    
    def invert_selection(self):
        """Invert the selection between Node A and Node B"""
        if self.node_a_index >= 0 and self.node_b_index >= 0:
            self.viewer.invert_between_nodes_selection()
    
    def create_turn(self):
        """Create a turn marker for the selected nodes"""
        if self.node_a_index == -1 or self.node_b_index == -1 or not self.current_data:
            return
        
        from PyQt6.QtWidgets import QInputDialog
        
        # Get existing turn numbers
        existing_turns = []
        if 'turns' in self.current_data:
            existing_turns = [int(k) for k in self.current_data['turns'].keys()]
        
        # Get turn number from user
        turn_num, ok = QInputDialog.getInt(
            self, 
            "Create Turn", 
            "Enter turn number:",
            min=1,
            max=99
        )
        
        if not ok:
            return
        
        # Check for duplicate
        if turn_num in existing_turns:
            QMessageBox.warning(self, "Duplicate Turn", f"Turn {turn_num} already exists. Please choose a different number.")
            return
        
        # Get the selected indices (including in-between nodes)
        indices_to_mark = [self.node_a_index, self.node_b_index] + self.viewer.between_nodes_indices
        
        # Store turn data in the viewer
        self.viewer.create_turn(turn_num, indices_to_mark)
        
        # Mark the nodes with turn number
        if 'turns' not in self.current_data:
            self.current_data['turns'] = {}
        
        self.current_data['turns'][str(turn_num)] = {
            'indices': indices_to_mark,
            'inverted': self.viewer.selection_inverted
        }
        
        self.info_label.setText(f"Turn {turn_num} created with {len(indices_to_mark)} nodes")
    
    def on_turn_selected(self, turn_num):
        """Handle turn selection from viewer"""
        self.selected_turn_num = turn_num
        self.delete_turn_btn.setEnabled(True)
        self.info_label.setText(f"Turn {turn_num} selected\nClick 'Delete Turn' to remove")
    
    def delete_turn(self):
        """Delete the currently selected turn"""
        if self.selected_turn_num is None or not self.current_data:
            return
        
        turn_num = self.selected_turn_num
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, 
            "Delete Turn", 
            f"Are you sure you want to delete Turn {turn_num}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Remove from viewer
        self.viewer.delete_turn(turn_num)
        
        # Remove from data
        if 'turns' in self.current_data and str(turn_num) in self.current_data['turns']:
            del self.current_data['turns'][str(turn_num)]
        
        # Reset selection
        self.selected_turn_num = None
        self.delete_turn_btn.setEnabled(False)
        self.info_label.setText(f"Turn {turn_num} deleted")
