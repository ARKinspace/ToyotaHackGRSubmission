from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPathItem, QGraphicsEllipseItem, QGraphicsItemGroup
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer
from PyQt6.QtGui import QPen, QColor, QPainter, QPainterPath, QBrush, QTransform, QPainterPathStroker
import math

class TrackSegmentItem(QGraphicsPathItem):
    def __init__(self, path, color, width, seg_id, data):
        super().__init__(path)
        self.setAcceptHoverEvents(True)
        self.default_color = color
        self.default_width = width
        self.seg_id = seg_id
        self.data_payload = data
        
        self.setData(0, seg_id)
        self.setData(1, data)
        
        self.update_pen()
        
    def shape(self):
        # Create a wider hit area for easier selection
        path = self.path()
        stroker = QPainterPathStroker()
        # Minimum hit width of 20 units, or the actual width + 10
        hit_width = max(self.default_width + 10, 20)
        stroker.setWidth(hit_width)
        return stroker.createStroke(path)

    def update_pen(self, highlighted=False, hover=False):
        width = self.default_width
        color = self.default_color
        
        if highlighted:
            color = QColor("#ffff00")
        elif hover:
            color = self.default_color.lighter(150)
            width += 2
            
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def hoverEnterEvent(self, event):
        if self.pen().color().name() != "#ffff00": # Not selected
            self.update_pen(hover=True)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        if self.pen().color().name() != "#ffff00": # Not selected
            self.update_pen(hover=False)
        super().hoverLeaveEvent(event)


class TrackViewer(QGraphicsView):
    segmentSelected = pyqtSignal(str)  # Signal emitting segment ID
    nodeSelected = pyqtSignal(int, float) # Signal emitting node index and current width
    nodesSelected = pyqtSignal(list) # Signal emitting list of node indices
    twoNodesPlaced = pyqtSignal(int, int) # Signal emitting Node A and Node B indices
    turnSelected = pyqtSignal(int)  # Signal emitting selected turn number
    segmentDeleted = pyqtSignal(str) # Signal emitting deleted segment ID

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor("#0f172a")))  # Slate 900
        
        self.track_items = {}
        self.selected_segment_id = None
        self.finalized = False
        self.fine_tune_mode = False
        self.finalized_data = None
        self.is_middle_dragging = False  # Track middle mouse drag state
        self.original_track_node_count = 0  # Track count before pit lanes added
        self.original_track_node_count = 0  # Track count before pit lanes added
        
        # Two-node placement system for Fine Tuner
        self.node_a_index = -1
        self.node_a_marker = None
        self.node_b_index = -1
        self.node_b_marker = None
        
        # In-between nodes selection
        self.between_nodes_indices = []
        self.between_nodes_markers = []
        self.selection_inverted = False  # Track if we're showing the inverted (longer) path
        
        # Turn markers and labels
        self.turns = {}  # {turn_num: {'indices': [], 'label': QGraphicsTextItem}}
        self.turn_labels = []  # List of QGraphicsTextItem for turn numbers
        
        # Pit lane boundary points for selection
        self.pit_boundary_points = []  # List of {'x', 'y', 'side': 'left'/'right', 'path_id': '...'}
        self.pit_boundary_markers = []
        self.pit_boundary_markers = []
        
        # Pulse animation for selected nodes
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse_animation_step)
        self.pulse_value = 0
        self.pulse_direction = 1

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.fine_tune_mode:
                # Clear both nodes in fine tune mode
                self.clear_node_markers()
            else:
                self.deselect_all()
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.selected_segment_id:
                self.segmentDeleted.emit(self.selected_segment_id)
                self.delete_segment(self.selected_segment_id)
        super().keyPressEvent(event)

    def deselect_all(self):
        # Deselect segment
        if self.selected_segment_id:
            self.select_segment(None) # Pass None to deselect
            
        # Deselect nodes
        self.clear_node_selection()
        self.nodeSelected.emit(-1, 0)
        self.nodesSelected.emit([])

    def mouseMoveEvent(self, event):
        # Handle middle mouse drag
        if self.is_middle_dragging:
            from PyQt6.QtGui import QMouseEvent
            fake_event = QMouseEvent(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mouseMoveEvent(fake_event)
            return
        super().mouseMoveEvent(event)
    
    def wheelEvent(self, event):
        zoomInFactor = 1.1
        zoomOutFactor = 1 / zoomInFactor

        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor

        self.scale(zoomFactor, zoomFactor)

    def clear(self):
        self.scene.clear()
        self.track_items = {}
        self.selected_segment_id = None
        self.finalized = False
        self.finalized_data = None
        self.selected_node_marker = None
        self.selected_node_index = -1
        self.multi_selection_markers = []
        self.pit_boundary_points = []
        self.pit_boundary_markers = []

    def set_track_data(self, track_data):
        self.clear()
        if not track_data or 'paths' not in track_data:
            return

        for path in track_data['paths']:
            self._add_path_item(path)
            
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _add_path_item(self, path_data):
        points = path_data['points']
        if not points:
            return

        qpath = QPainterPath()
        qpath.moveTo(points[0]['x'], points[0]['y'])
        for p in points[1:]:
            qpath.lineTo(p['x'], p['y'])

        color = QColor("#f97316") if path_data['type'] == 'pit' else QColor("#34d399")
        width = path_data.get('widthValue', 12)
        
        item = TrackSegmentItem(qpath, color, width, path_data['id'], path_data)
        
        self.scene.addItem(item)
        self.track_items[path_data['id']] = item

    def mousePressEvent(self, event):
        # Enable panning with middle mouse button
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_middle_dragging = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            # Simulate left button press for dragging
            from PyQt6.QtGui import QMouseEvent
            fake_event = QMouseEvent(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
            return
        
        super().mousePressEvent(event)
        
        if self.fine_tune_mode and self.finalized_data:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = self.mapToScene(event.pos())
                ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                
                # Check if clicking on a turn label
                item = self.itemAt(event.pos())
                if item and item.data(0) and str(item.data(0)).startswith("turn_"):
                    turn_data = item.data(1)
                    if turn_data:
                        self._handle_turn_label_click(turn_data['turn_num'], turn_data['indices'])
                        return
                
                # Always try to place/move nodes (works everywhere including on pit lanes)
                self._place_or_move_node(pos.x(), pos.y(), is_node_b=ctrl_pressed)
            return

        # if self.finalized:
        #    return
            
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, TrackSegmentItem):
                seg_id = item.data(0)
                if seg_id:
                    self.select_segment(seg_id)
            elif isinstance(item, QGraphicsPathItem):
                # Fallback for non-TrackSegmentItem path items
                seg_id = item.data(0)
                if seg_id:
                    self.select_segment(seg_id)

    def _place_or_move_node(self, x, y, is_node_b=False):
        """Place or move Node A (left-click) or Node B (ctrl+click)"""
        if not self.finalized_data or 'splinePoints' not in self.finalized_data:
            return
            
        points = self.finalized_data['splinePoints']
        min_dist = float('inf')
        nearest_idx = -1
        
        # Find nearest node (works for both track and pit lane nodes)
        for i, p in enumerate(points):
            dist = math.sqrt((p['x'] - x)**2 + (p['y'] - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        if min_dist > 5:  # Click tolerance - must click on track itself
            return
        
        # Place/move the appropriate node
        if is_node_b:
            self.node_b_index = nearest_idx
            self._update_node_b_marker()
        else:
            self.node_a_index = nearest_idx
            self._update_node_a_marker()
        
        # Emit signal with both node indices
        self.twoNodesPlaced.emit(self.node_a_index, self.node_b_index)
    
    def _update_node_a_marker(self):
        """Update visual marker for Node A"""
        if self.node_a_marker:
            self.scene.removeItem(self.node_a_marker)
            self.node_a_marker = None
        
        if self.node_a_index >= 0 and self.finalized_data:
            points = self.finalized_data['splinePoints']
            if self.node_a_index < len(points):
                p = points[self.node_a_index]
                self.node_a_marker = QGraphicsEllipseItem(p['x'] - 3, p['y'] - 3, 6, 6)
                self.node_a_marker.setBrush(QBrush(QColor("#3b82f6")))  # Blue for Node A
                self.node_a_marker.setPen(QPen(QColor("#1e40af"), 2))
                self.node_a_marker.setZValue(100)
                self.scene.addItem(self.node_a_marker)
        
        # Update in-between nodes when Node A changes
        self._update_between_nodes_markers()
    
    def _update_node_b_marker(self):
        """Update visual marker for Node B"""
        if self.node_b_marker:
            self.scene.removeItem(self.node_b_marker)
            self.node_b_marker = None
        
        if self.node_b_index >= 0 and self.finalized_data:
            points = self.finalized_data['splinePoints']
            if self.node_b_index < len(points):
                p = points[self.node_b_index]
                self.node_b_marker = QGraphicsEllipseItem(p['x'] - 3, p['y'] - 3, 6, 6)
                self.node_b_marker.setBrush(QBrush(QColor("#f59e0b")))  # Orange for Node B
                self.node_b_marker.setPen(QPen(QColor("#d97706"), 2))
                self.node_b_marker.setZValue(100)
                self.scene.addItem(self.node_b_marker)
        
        # Update in-between nodes
        self._update_between_nodes_markers()
    
    def clear_node_markers(self):
        """Clear both Node A and Node B markers"""
        if self.node_a_marker:
            self.scene.removeItem(self.node_a_marker)
            self.node_a_marker = None
        if self.node_b_marker:
            self.scene.removeItem(self.node_b_marker)
            self.node_b_marker = None
        
        self.node_a_index = -1
        self.node_b_index = -1
        self.selection_inverted = False  # Reset inversion state
        
        # Clear in-between nodes
        self._clear_between_nodes_markers()
        
        # Emit signal to notify FineTuner
        self.twoNodesPlaced.emit(-1, -1)

    def select_segment(self, seg_id):
        # Reset previous selection
        if self.selected_segment_id and self.selected_segment_id in self.track_items:
            prev_item = self.track_items[self.selected_segment_id]
            if isinstance(prev_item, TrackSegmentItem):
                prev_item.update_pen(highlighted=False)
            else:
                # Fallback
                data = prev_item.data(1)
                color = QColor("#f97316") if data['type'] == 'pit' else QColor("#34d399")
                prev_item.setPen(QPen(color, data.get('widthValue', 12)))

        self.selected_segment_id = seg_id
        
        # Highlight new selection
        if seg_id and seg_id in self.track_items:
            item = self.track_items[seg_id]
            if isinstance(item, TrackSegmentItem):
                item.update_pen(highlighted=True)
            else:
                data = item.data(1)
                pen = QPen(QColor("#ffff00"), data.get('widthValue', 12)) # Yellow highlight
                item.setPen(pen)
            self.segmentSelected.emit(str(seg_id))

    def update_segment_width(self, seg_id, new_width):
        if seg_id in self.track_items:
            item = self.track_items[seg_id]
            if isinstance(item, TrackSegmentItem):
                item.default_width = new_width
                item.update_pen(highlighted=(seg_id == self.selected_segment_id))
            else:
                data = item.data(1)
                data['widthValue'] = new_width
                item.setData(1, data)
                color = QColor("#ffff00") if seg_id == self.selected_segment_id else (QColor("#f97316") if data['type'] == 'pit' else QColor("#34d399"))
                item.setPen(QPen(color, new_width))

    def toggle_segment_type(self, seg_id):
        if seg_id in self.track_items:
            item = self.track_items[seg_id]
            if isinstance(item, TrackSegmentItem):
                data = item.data(1)
                data['type'] = 'pit' if data['type'] == 'track' else 'track'
                item.default_color = QColor("#f97316") if data['type'] == 'pit' else QColor("#34d399")
                item.update_pen(highlighted=(seg_id == self.selected_segment_id))
            else:
                data = item.data(1)
                data['type'] = 'pit' if data['type'] == 'track' else 'track'
                item.setData(1, data)
                color = QColor("#ffff00") if seg_id == self.selected_segment_id else (QColor("#f97316") if data['type'] == 'pit' else QColor("#34d399"))
                item.setPen(QPen(color, data.get('widthValue', 12)))

    def delete_segment(self, seg_id):
        if seg_id in self.track_items:
            item = self.track_items[seg_id]
            self.scene.removeItem(item)
            del self.track_items[seg_id]
            if self.selected_segment_id == seg_id:
                self.selected_segment_id = None

    def select_node(self, index, multi=False):
        if not self.finalized_data:
            return
            
        points = self.finalized_data['splinePoints']
        if index < 0 or index >= len(points):
            return
            
        if self.range_select_mode:
            if self.range_start_index == -1:
                # First click: Set start
                self.range_start_index = index
                self.clear_node_selection()
                
                # Visual feedback for start node
                p = points[index]
                self.selected_node_marker = QGraphicsEllipseItem(p['x'] - 2, p['y'] - 2, 4, 4)
                self.selected_node_marker.setBrush(QBrush(Qt.GlobalColor.green)) # Green for start
                self.selected_node_marker.setPen(QPen(Qt.GlobalColor.black, 1))
                self.selected_node_marker.setZValue(100)
                self.scene.addItem(self.selected_node_marker)
                
                self.nodeSelected.emit(index, p['width']) # Emit for info
            else:
                # Second click: Set end and select path
                self.selected_node_index = self.range_start_index # Set start for path calc
                self._select_path_to(index)
                self.range_start_index = -1 # Reset
        elif not multi:
            self.clear_node_selection()
            self.selected_node_index = index
            p = points[index]
            
            self.selected_node_marker = QGraphicsEllipseItem(p['x'] - 2, p['y'] - 2, 4, 4)
            self.selected_node_marker.setBrush(QBrush(Qt.GlobalColor.yellow))
            self.selected_node_marker.setPen(QPen(Qt.GlobalColor.red, 1))
            self.selected_node_marker.setZValue(100)
            self.scene.addItem(self.selected_node_marker)
            
            self.nodeSelected.emit(index, p['width'])
            self.nodesSelected.emit([index])
        else:
            # Multi-selection logic (Shortest Path)
            self._select_path_to(index)

    def _select_path_to(self, index):
        if self.selected_node_index == -1:
            self.select_node(index, multi=False)
            return
            
        start = self.selected_node_index
        end = index
        points = self.finalized_data['splinePoints']
        N = len(points)
        
        # Calculate distances
        forward_dist = (end - start) % N
        backward_dist = (start - end) % N
        
        selected_indices = []
        if forward_dist <= backward_dist:
            # Forward path
            curr = start
            while curr != end:
                selected_indices.append(curr)
                curr = (curr + 1) % N
            selected_indices.append(end)
        else:
            # Backward path
            curr = start
            while curr != end:
                selected_indices.append(curr)
                curr = (curr - 1 + N) % N
            selected_indices.append(end)
        
        self.highlight_nodes(selected_indices)
        self.nodesSelected.emit(selected_indices)

    def highlight_nodes(self, indices):
        self.clear_node_selection()
        if not self.finalized_data: return
        points = self.finalized_data['splinePoints']
        
        for i in indices:
            p = points[i]
            marker = QGraphicsEllipseItem(p['x'] - 2, p['y'] - 2, 4, 4)
            marker.setBrush(QBrush(Qt.GlobalColor.yellow))
            self.scene.addItem(marker)
            self.multi_selection_markers.append(marker)
            
    def invert_selection(self, current_indices):
        if not self.finalized_data: return
        N = len(self.finalized_data['splinePoints'])
        all_indices = set(range(N))
        current_set = set(current_indices)
        inverted = list(all_indices - current_set)
        self.highlight_nodes(inverted)
        self.nodesSelected.emit(inverted)

    def clear_node_selection(self):
        if self.selected_node_marker:
            self.scene.removeItem(self.selected_node_marker)
            self.selected_node_marker = None
        
        for m in self.multi_selection_markers:
            self.scene.removeItem(m)
        self.multi_selection_markers = []
        self.selected_node_index = -1

    def set_finalized_data(self, finalized_data):
        self.clear()
        self.finalized = True
        self.finalized_data = finalized_data
        
        # Integrate pit lanes into splinePoints if in fine tune mode
        if self.fine_tune_mode and finalized_data:
            self._integrate_pit_lanes_into_spline(finalized_data)
        
        if not finalized_data:
            return

        # Draw main track from splinePoints if available (supports variable width)
        if 'splinePoints' in finalized_data and finalized_data['splinePoints']:
            self._draw_from_spline_points(finalized_data)
            
            # Only draw pit lanes from visualPaths if NOT in fine tune mode
            # (in fine tune mode, pit lanes are integrated into splinePoints)
            if not self.fine_tune_mode and 'visualPaths' in finalized_data:
                for path in finalized_data['visualPaths']:
                    # Skip main track paths as they are covered by splinePoints
                    if path['id'] in ['s1', 's2', 's3', 'track_full']:
                        continue
                    self._add_finalized_path(path)
                    
                    # Add Pit Markers (Green Triangles) at start/end of pit paths
                    d_str = path['d']
                    parts = d_str.split(' ')
                    if len(parts) >= 3:
                        start_x, start_y = float(parts[1]), float(parts[2])
                        self._add_marker(start_x, start_y, QColor("#00ff00"), "triangle")
                        
                        # Find end
                        end_x, end_y = start_x, start_y
                        i = 3
                        while i < len(parts):
                            if parts[i] == 'L':
                                end_x, end_y = float(parts[i+1]), float(parts[i+2])
                                i += 3
                            else:
                                i += 1
                        self._add_marker(end_x, end_y, QColor("#00ff00"), "triangle")

        elif 'visualPaths' in finalized_data:
            # Fallback for legacy data
            for path in finalized_data['visualPaths']:
                self._add_finalized_path(path)
            
        # Add SF Marker
        if finalized_data.get('sfMarker'):
            sf = finalized_data['sfMarker']
            self._add_marker(sf['x'], sf['y'], QColor("#ffffff"), "circle")
        
        # Load and display turns if in fine tune mode
        if self.fine_tune_mode and 'turns' in finalized_data:
            for turn_num_str, turn_data in finalized_data['turns'].items():
                turn_num = int(turn_num_str)
                indices = turn_data['indices']
                self.create_turn(turn_num, indices)

        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._draw_legend()

    def _add_marker(self, x, y, color, shape):
        if shape == "triangle":
            path = QPainterPath()
            path.moveTo(x, y - 6)
            path.lineTo(x - 5, y + 4)
            path.lineTo(x + 5, y + 4)
            path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(color))
            item.setPen(QPen(Qt.GlobalColor.black, 1))
            item.setZValue(50)
            self.scene.addItem(item)
        else:
            marker = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
            marker.setBrush(QBrush(color))
            marker.setPen(QPen(Qt.GlobalColor.black, 2))
            marker.setZValue(50)
            self.scene.addItem(marker)
    
    def _integrate_pit_lanes_into_spline(self, finalized_data):
        """Add pit lane boundary points to splinePoints with pit_lane attribute"""
        if 'visualPaths' not in finalized_data or 'splinePoints' not in finalized_data:
            return
        
        spline_points = finalized_data['splinePoints']
        # Remember where the original track ends before adding pit lanes
        self.original_track_node_count = len(spline_points)
        # Remember where the original track ends before adding pit lanes
        self.original_track_node_count = len(spline_points)
        # Remember where the original track ends before adding pit lanes
        self.original_track_node_count = len(spline_points)
        
        for path in finalized_data['visualPaths']:
            # Skip main track paths
            if path['id'] in ['s1', 's2', 's3', 'track_full']:
                continue
            
            # Get centerline points (with z-coordinates if available)
            if 'points' in path and path['points']:
                # Points already have x, y, z from OSM data
                centerline = path['points']
            else:
                d_str = path['d']
                parts = d_str.split(' ')
                centerline = []
                if len(parts) >= 3:
                    centerline.append({'x': float(parts[1]), 'y': float(parts[2]), 'z': 0})
                    i = 3
                    while i < len(parts):
                        if parts[i] == 'L':
                            centerline.append({'x': float(parts[i+1]), 'y': float(parts[i+2]), 'z': 0})
                            i += 3
                        else:
                            i += 1
            
            if len(centerline) < 2:
                continue
            
            # Resample centerline to 1-meter spacing like the main track
            resampled_centerline = self._resample_path(centerline, spacing=1.0)
            
            width = path.get('widthValue', path.get('width', 12)) / 2
            
            # Add left and right boundary points to splinePoints
            for i, p in enumerate(resampled_centerline):
                # Calculate perpendicular direction
                if i < len(resampled_centerline) - 1:
                    dx = resampled_centerline[i+1]['x'] - p['x']
                    dy = resampled_centerline[i+1]['y'] - p['y']
                elif i > 0:
                    dx = p['x'] - resampled_centerline[i-1]['x']
                    dy = p['y'] - resampled_centerline[i-1]['y']
                else:
                    continue
                
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0:
                    continue
                dx /= length
                dy /= length
                
                # Perpendicular
                perp_x = -dy
                perp_y = dx
                
                # Get elevation from centerline point
                z = p.get('z', 0)
                
                # Add left boundary point
                spline_points.append({
                    'x': p['x'] + perp_x * width,
                    'y': p['y'] + perp_y * width,
                    'z': z,
                    'width': path.get('widthValue', path.get('width', 12)),
                    'pit_lane': path['id'],
                    'pit_side': 'left'
                })
                
                # Add right boundary point
                spline_points.append({
                    'x': p['x'] - perp_x * width,
                    'y': p['y'] - perp_y * width,
                    'z': z,
                    'width': path.get('widthValue', path.get('width', 12)),
                    'pit_lane': path['id'],
                    'pit_side': 'right'
                })
    
    def _resample_path(self, points, spacing=1.0):
        """Resample a path to have evenly spaced points"""
        if len(points) < 2:
            return points
        
        resampled = [points[0]]
        accumulated_distance = 0.0
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            segment_dx = p2['x'] - p1['x']
            segment_dy = p2['y'] - p1['y']
            segment_length = math.sqrt(segment_dx**2 + segment_dy**2)
            
            if segment_length == 0:
                continue
            
            # How many points should we add in this segment?
            num_points = int(segment_length / spacing)
            
            for j in range(1, num_points + 1):
                t = (j * spacing) / segment_length
                if t <= 1.0:
                    # Interpolate x, y, and z coordinates
                    z1 = p1.get('z', 0)
                    z2 = p2.get('z', 0)
                    resampled.append({
                        'x': p1['x'] + t * segment_dx,
                        'y': p1['y'] + t * segment_dy,
                        'z': z1 + t * (z2 - z1)
                    })
        
        # Always include the last point
        if resampled[-1] != points[-1]:
            resampled.append(points[-1])
        
        return resampled

    def _draw_legend(self):
        # Simple overlay for legend
        # Since QGraphicsView doesn't support fixed overlay easily without subclassing drawForeground,
        # we'll just add items to the scene at a fixed position relative to the view?
        # No, that moves with zoom.
        # We can use a separate widget overlay or just drawForeground.
        pass

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.finalized:
            painter.setTransform(QTransform()) # Reset transform to draw in window coordinates
            
            # Draw Legend
            painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(10, 10, 160, 110, 5, 5)
            
            painter.setPen(Qt.GlobalColor.white)
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(20, 30, "Legend")
            
            font.setBold(False)
            painter.setFont(font)
            
            def draw_item(y, color, text):
                painter.setBrush(QBrush(QColor(color)))
                painter.drawRect(20, y, 15, 15)
                painter.drawText(45, y + 12, text)
                
            draw_item(40, "#3b82f6", "Sector 1")
            draw_item(60, "#eab308", "Sector 2")
            draw_item(80, "#ef4444", "Sector 3")
            draw_item(100, "#f97316", "Pit Lane")

    def _draw_from_spline_points(self, data):
        points = data['splinePoints']
        sectors = data.get('splineData', {}).get('sectors', {})
        s1_end = sectors.get('s1EndIndex', -1)
        s2_end = sectors.get('s2EndIndex', -1)
        
        # Only draw up to original track nodes, not pit lane nodes
        track_points = points[:self.original_track_node_count] if self.original_track_node_count > 0 else points
        pit_points = points[self.original_track_node_count:] if self.original_track_node_count > 0 else []
        
        def get_color(idx):
            if s1_end > 0 and idx <= s1_end: return "#3b82f6" # Blue
            if s2_end > 0 and idx <= s2_end: return "#eab308" # Yellow
            if s1_end > 0 and s2_end > 0: return "#ef4444" # Red
            return "#10b981" # Green

        # Draw main track
        groups = []
        if not track_points: return
        
        current_group = {'points': [], 'color': get_color(0)}
        
        for i, p in enumerate(track_points):
            color = get_color(i)
            if color != current_group['color']:
                current_group['points'].append(p) # Connect to next
                groups.append(current_group)
                current_group = {'points': [p], 'color': color}
            else:
                current_group['points'].append(p)
        groups.append(current_group)
        
        for g in groups:
            self._draw_poly_edges(g['points'], g['color'])
        
        # Draw pit lanes as separate segments grouped by pit_lane ID
        if self.fine_tune_mode and pit_points:
            pit_groups = {}
            for p in pit_points:
                pit_id = p.get('pit_lane')
                if pit_id:
                    if pit_id not in pit_groups:
                        pit_groups[pit_id] = []
                    pit_groups[pit_id].append(p)
            
            # Draw each pit lane segment
            for pit_id, pit_pts in pit_groups.items():
                if len(pit_pts) >= 2:
                    self._draw_poly_edges(pit_pts, "#ff6600")  # Orange for pit lanes

    def _draw_poly_edges(self, points, color_hex):
        if len(points) < 2: return
        
        left_edge = QPainterPath()
        right_edge = QPainterPath()
        
        for i in range(len(points)):
            p = points[i]
            w = p.get('width', 12) / 2
            
            if i < len(points) - 1:
                p_next = points[i+1]
                dx = p_next['x'] - p['x']
                dy = p_next['y'] - p['y']
            else:
                p_prev = points[i-1]
                dx = p['x'] - p_prev['x']
                dy = p['y'] - p_prev['y']
            
            l = math.sqrt(dx*dx + dy*dy)
            if l == 0: continue
            nx = -dy / l
            ny = dx / l
            
            lx = p['x'] + nx * w
            ly = p['y'] + ny * w
            rx = p['x'] - nx * w
            ry = p['y'] - ny * w
            
            if i == 0:
                left_edge.moveTo(lx, ly)
                right_edge.moveTo(rx, ry)
            else:
                left_edge.lineTo(lx, ly)
                right_edge.lineTo(rx, ry)
                
        pen = QPen(QColor(color_hex), 2)
        l_item = QGraphicsPathItem(left_edge)
        l_item.setPen(pen)
        self.scene.addItem(l_item)
        
        r_item = QGraphicsPathItem(right_edge)
        r_item.setPen(pen)
        self.scene.addItem(r_item)

    def _add_finalized_path(self, path_data):
        # Legacy/Pit path drawer using constant width
        d_str = path_data['d']
        parts = d_str.split(' ')
        points = []
        if len(parts) >= 3:
            points.append({'x': float(parts[1]), 'y': float(parts[2])})
            i = 3
            while i < len(parts):
                if parts[i] == 'L':
                    points.append({'x': float(parts[i+1]), 'y': float(parts[i+2])})
                    i += 3
                else:
                    i += 1
        
        if len(points) < 2: return
        
        # Draw edges (same logic as _draw_poly_edges but with constant width)
        self._draw_poly_edges([{'x': p['x'], 'y': p['y'], 'width': path_data['width']} for p in points], path_data['color'])

    def _update_between_nodes_markers(self):
        """Update markers for nodes between Node A and Node B"""
        # Clear existing markers
        self._clear_between_nodes_markers()
        
        if self.node_a_index < 0 or self.node_b_index < 0 or not self.finalized_data:
            return
        
        points = self.finalized_data['splinePoints']
        N = len(points)
        
        start = min(self.node_a_index, self.node_b_index)
        end = max(self.node_a_index, self.node_b_index)
        
        # Check if both nodes are in pit lane (beyond original track)
        both_in_pit = (start >= self.original_track_node_count and 
                       end >= self.original_track_node_count)
        
        # Check if both nodes are in the SAME pit lane
        same_pit_lane = False
        if both_in_pit:
            pit_a = points[start].get('pit_lane') if start < len(points) else None
            pit_b = points[end].get('pit_lane') if end < len(points) else None
            same_pit_lane = (pit_a and pit_b and pit_a == pit_b)
        
        if same_pit_lane:
            # Pit lanes are linear - just select all nodes between start and end
            self.between_nodes_indices = list(range(start + 1, end))
        elif both_in_pit and not same_pit_lane:
            # Different pit lanes - no in-between nodes
            self.between_nodes_indices = []
        else:
            # At least one node is on the main track - use circular logic
            # Constrain the circular logic to only the original track nodes
            track_N = self.original_track_node_count if self.original_track_node_count > 0 else N
            
            # If one node is in pit and one is on track, no sensible path
            if (start >= track_N) != (end >= track_N):
                self.between_nodes_indices = []
            else:
                # Both on main track - use circular path logic
                direct_distance = end - start
                
                if not self.selection_inverted:
                    # Default: use shorter path
                    if direct_distance <= track_N / 2:
                        self.between_nodes_indices = list(range(start + 1, end))
                    else:
                        # Wrap-around path is shorter
                        self.between_nodes_indices = list(range(end + 1, track_N)) + list(range(0, start))
                else:
                    # Inverted: use longer path
                    if direct_distance <= track_N / 2:
                        self.between_nodes_indices = list(range(end + 1, track_N)) + list(range(0, start))
                    else:
                        self.between_nodes_indices = list(range(start + 1, end))
        
        # Create markers for in-between nodes
        for idx in self.between_nodes_indices:
            if 0 <= idx < len(points):
                p = points[idx]
                marker = QGraphicsEllipseItem(p['x'] - 2.5, p['y'] - 2.5, 5, 5)
                marker.setBrush(QBrush(QColor("#10b981")))  # Green for in-between
                marker.setPen(QPen(QColor("#059669"), 1))
                marker.setZValue(99)
                self.scene.addItem(marker)
                self.between_nodes_markers.append(marker)
        
        # Start pulse animation if there are in-between nodes
        if self.between_nodes_markers:
            self.pulse_timer.start(50)  # Update every 50ms
        else:
            self.pulse_timer.stop()
    
    def _clear_between_nodes_markers(self):
        """Clear all in-between node markers"""
        for marker in self.between_nodes_markers:
            self.scene.removeItem(marker)
        self.between_nodes_markers = []
        self.between_nodes_indices = []
        self.pulse_timer.stop()
    
    def _pulse_animation_step(self):
        """Animate the pulse effect for in-between nodes"""
        if not self.between_nodes_markers:
            return
        
        # Update pulse value (0 to 100)
        self.pulse_value += self.pulse_direction * 5
        if self.pulse_value >= 100:
            self.pulse_value = 100
            self.pulse_direction = -1
        elif self.pulse_value <= 0:
            self.pulse_value = 0
            self.pulse_direction = 1
        
        # Calculate color with pulse
        base_color = QColor("#10b981")  # Green
        highlight_color = QColor("#34d399")  # Lighter green
        
        # Interpolate between base and highlight
        r = int(base_color.red() + (highlight_color.red() - base_color.red()) * self.pulse_value / 100)
        g = int(base_color.green() + (highlight_color.green() - base_color.green()) * self.pulse_value / 100)
        b = int(base_color.blue() + (highlight_color.blue() - base_color.blue()) * self.pulse_value / 100)
        pulse_color = QColor(r, g, b)
        
        # Apply to all markers
        for marker in self.between_nodes_markers:
            marker.setBrush(QBrush(pulse_color))
    
    def invert_between_nodes_selection(self):
        """
        Invert the selection between nodes.
        Toggle between selecting the shortest path and the longest path around the track.
        """
        if self.node_a_index < 0 or self.node_b_index < 0 or not self.finalized_data:
            return
        
        # Simply toggle the inversion flag
        self.selection_inverted = not self.selection_inverted
        
        # Update markers to show the new selection
        self._update_between_nodes_markers()
        
        # Emit signal to update FineTuner (node positions don't change)
        self.twoNodesPlaced.emit(self.node_a_index, self.node_b_index)
    
    def create_turn(self, turn_num, indices):
        """Create a turn marker with clickable label"""
        if not self.finalized_data or 'splinePoints' not in self.finalized_data:
            return
        
        points = self.finalized_data['splinePoints']
        
        # Calculate center position of the turn (average of all selected nodes)
        avg_x = sum(points[i]['x'] for i in indices if i < len(points)) / len(indices)
        avg_y = sum(points[i]['y'] for i in indices if i < len(points)) / len(indices)
        
        # Create clickable text label
        from PyQt6.QtWidgets import QGraphicsTextItem, QGraphicsEllipseItem
        from PyQt6.QtGui import QFont
        
        # Create background circle
        circle = QGraphicsEllipseItem(avg_x - 15, avg_y - 15, 30, 30)
        circle.setBrush(QBrush(QColor("#f59e0b")))  # Orange background
        circle.setPen(QPen(QColor("#d97706"), 2))
        circle.setZValue(150)
        circle.setData(0, f"turn_{turn_num}")  # Tag for identification
        circle.setData(1, {'turn_num': turn_num, 'indices': indices})
        circle.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.scene.addItem(circle)
        
        # Create text label
        label = QGraphicsTextItem(str(turn_num))
        label.setPos(avg_x - 8, avg_y - 12)
        font = QFont("Arial", 14, QFont.Weight.Bold)
        label.setFont(font)
        label.setDefaultTextColor(QColor("#ffffff"))  # White text
        label.setZValue(151)
        label.setData(0, f"turn_{turn_num}")
        label.setData(1, {'turn_num': turn_num, 'indices': indices})
        self.scene.addItem(label)
        
        # Store turn data
        self.turns[turn_num] = {
            'indices': indices,
            'circle': circle,
            'label': label
        }
        self.turn_labels.append(label)
    
    def _handle_turn_label_click(self, turn_num, indices):
        """Highlight nodes when turn label is clicked"""
        # Clear current selection
        self.clear_node_markers()
        
        # Emit signal to notify FineTuner that a turn was selected
        self.turnSelected.emit(turn_num)
        
        # Highlight all nodes in this turn
        if not self.finalized_data or 'splinePoints' not in self.finalized_data:
            return
        
        points = self.finalized_data['splinePoints']
        
        # Create markers for all turn nodes
        for idx in indices:
            if 0 <= idx < len(points):
                p = points[idx]
                marker = QGraphicsEllipseItem(p['x'] - 2.5, p['y'] - 2.5, 5, 5)
                marker.setBrush(QBrush(QColor("#f59e0b")))  # Orange for turn
                marker.setPen(QPen(QColor("#d97706"), 1))
                marker.setZValue(99)
                self.scene.addItem(marker)
                self.between_nodes_markers.append(marker)
    
    def delete_turn(self, turn_num):
        """Delete a turn marker and its label"""
        if turn_num not in self.turns:
            return
        
        turn_data = self.turns[turn_num]
        
        # Remove visual elements
        if turn_data['circle']:
            self.scene.removeItem(turn_data['circle'])
        if turn_data['label']:
            self.scene.removeItem(turn_data['label'])
            if turn_data['label'] in self.turn_labels:
                self.turn_labels.remove(turn_data['label'])
        
        # Remove from turns dictionary
        del self.turns[turn_num]

    def set_fine_tune_mode(self, enabled):
        self.fine_tune_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)  # Allow normal clicking
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.clear_node_selection()

    def clear_node_selection(self):
        if self.selected_node_marker:
            self.scene.removeItem(self.selected_node_marker)
            self.selected_node_marker = None
        # Clear multi-selection markers if any (not implemented yet)

    def mouseReleaseEvent(self, event):
        # Reset drag mode and cursor after middle mouse button release
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_middle_dragging = False
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            # Restore previous drag mode
            if self.fine_tune_mode:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            else:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Simulate left button release
            from PyQt6.QtGui import QMouseEvent
            fake_event = QMouseEvent(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mouseReleaseEvent(fake_event)
            return
        
        super().mouseReleaseEvent(event)
        if self.fine_tune_mode and self.finalized_data:
            # Handle rubber band selection
            selection = self.scene.selectionArea().boundingRect()
            if not selection.isEmpty():
                self._select_nodes_in_rect(selection)

    def _select_nodes_in_rect(self, rect):
        if not self.finalized_data or 'splinePoints' not in self.finalized_data:
            return
            
        points = self.finalized_data['splinePoints']
        selected_indices = []
        
        for i, p in enumerate(points):
            if rect.contains(p['x'], p['y']):
                selected_indices.append(i)
        
        if selected_indices:
            # Emit signal with list of indices
            # We need to update the signal signature or emit multiple times?
            # Let's emit a new signal for multi-selection
            self.nodesSelected.emit(selected_indices)
    
    def _extract_pit_boundary_points(self, path_data):
        """Extract left and right boundary points from a pit lane path for selection"""
        # Get centerline points from path
        if 'points' in path_data and path_data['points']:
            centerline = path_data['points']
        else:
            # Parse from 'd' string
            d_str = path_data['d']
            parts = d_str.split(' ')
            centerline = []
            if len(parts) >= 3:
                centerline.append({'x': float(parts[1]), 'y': float(parts[2])})
                i = 3
                while i < len(parts):
                    if parts[i] == 'L':
                        centerline.append({'x': float(parts[i+1]), 'y': float(parts[i+2])})
                        i += 3
                    else:
                        i += 1
        
        if len(centerline) < 2:
            return
        
        width = path_data.get('widthValue', path_data.get('width', 12)) / 2
        
        # Calculate left and right boundary points
        for i in range(len(centerline)):
            p = centerline[i]
            
            # Calculate perpendicular direction
            if i < len(centerline) - 1:
                # Use vector to next point
                dx = centerline[i+1]['x'] - p['x']
                dy = centerline[i+1]['y'] - p['y']
            elif i > 0:
                # Use vector from previous point
                dx = p['x'] - centerline[i-1]['x']
                dy = p['y'] - centerline[i-1]['y']
            else:
                continue
            
            # Normalize
            length = math.sqrt(dx*dx + dy*dy)
            if length == 0:
                continue
            dx /= length
            dy /= length
            
            # Perpendicular (rotate 90 degrees)
            perp_x = -dy
            perp_y = dx
            
            # Left and right points
            left_pt = {
                'x': p['x'] + perp_x * width,
                'y': p['y'] + perp_y * width,
                'side': 'left',
                'path_id': path_data['id']
            }
            right_pt = {
                'x': p['x'] - perp_x * width,
                'y': p['y'] - perp_y * width,
                'side': 'right',
                'path_id': path_data['id']
            }
            
            self.pit_boundary_points.append(left_pt)
            self.pit_boundary_points.append(right_pt)
            
            # Add visual markers to see pit boundary points
            if True:  # Enable to visualize pit boundary points
                for pt in [left_pt, right_pt]:
                    marker = QGraphicsEllipseItem(pt['x'] - 2, pt['y'] - 2, 4, 4)
                    marker.setBrush(QBrush(QColor("#ff00ff")))  # Magenta for pit boundaries
                    marker.setPen(QPen(QColor("#ff00ff"), 1))
                    marker.setZValue(98)
                    self.scene.addItem(marker)
                    self.pit_boundary_markers.append(marker)
