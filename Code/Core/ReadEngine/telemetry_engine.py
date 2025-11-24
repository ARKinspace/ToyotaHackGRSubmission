import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VehicleState:
    """Current state of the vehicle"""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [roll, pitch, yaw] in radians
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    speed: float
    rpm: float
    gear: int
    steering_angle: float
    throttle: float
    brake: float
    timestamp: float
    lap: int


class TelemetryEngine:
    """
    High-performance telemetry processing engine for large CSV files.
    Handles vehicle simulation with position, rotation, and state tracking.
    """
    
    def __init__(self, chunk_size: int = 100000):
        """
        Initialize the telemetry engine.
        
        Args:
            chunk_size: Number of rows to process at once (for memory efficiency)
        """
        self.chunk_size = chunk_size
        self.current_state = None
        self.state_history = []
        self.vehicle_dimensions = np.array([2.0, 4.5, 1.5])  # [width, length, height] in meters
        
        # Configuration
        self.wheelbase = 2.7  # meters
        self.mass = 1500  # kg
        self.dt = 0.016  # default timestep (60Hz)
        
    def load_telemetry_chunked(self, filepath: str):
        """
        Generator that yields chunks of telemetry data for memory-efficient processing.
        
        Args:
            filepath: Path to the CSV file
            
        Yields:
            DataFrame chunks
        """
        return pd.read_csv(filepath, chunksize=self.chunk_size, low_memory=False)
    
    def initialize_from_data(self, first_row: pd.Series, start_position: np.ndarray = None):
        """
        Initialize vehicle state from first telemetry row.
        
        Args:
            first_row: First row of telemetry data
            start_position: Optional starting position [x, y, z], defaults to origin
        """
        if start_position is None:
            start_position = np.array([0.0, 0.0, 0.0])
        
        self.current_state = VehicleState(
            position=start_position.copy(),
            rotation=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            acceleration=np.array([0.0, 0.0, 0.0]),
            speed=self._get_value(first_row, 'speed', 0.0),
            rpm=self._get_value(first_row, 'nmot', 0.0),
            gear=int(self._get_value(first_row, 'gear', 0)),
            steering_angle=self._get_value(first_row, 'Steering_Angle', 0.0),
            throttle=self._get_value(first_row, 'aps', 0.0) / 100.0,
            brake=max(self._get_value(first_row, 'pbrake_f', 0.0), 
                     self._get_value(first_row, 'pbrake_r', 0.0)),
            timestamp=0.0,
            lap=int(self._get_value(first_row, 'lap', 1))
        )
    
    def _get_value(self, row: pd.Series, key: str, default=0.0):
        """Safely get value from row with fallback."""
        try:
            val = row.get(key, default)
            return default if pd.isna(val) else val
        except:
            return default
    
    def process_telemetry_row(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> VehicleState:
        """
        Process a single telemetry row and update vehicle state.
        
        Args:
            row: Current telemetry row
            prev_row: Previous telemetry row for calculating deltas
            
        Returns:
            Updated VehicleState
        """
        if self.current_state is None:
            self.initialize_from_data(row)
            return self.current_state
        
        # Extract telemetry values
        speed = self._get_value(row, 'speed', 0.0)  # m/s
        steering_angle = self._get_value(row, 'Steering_Angle', 0.0)
        accel_x = self._get_value(row, 'accx_can', 0.0)
        accel_y = self._get_value(row, 'accy_can', 0.0)
        rpm = self._get_value(row, 'nmot', 0.0)
        gear = int(self._get_value(row, 'gear', 0))
        throttle = self._get_value(row, 'aps', 0.0) / 100.0
        brake_f = self._get_value(row, 'pbrake_f', 0.0)
        brake_r = self._get_value(row, 'pbrake_r', 0.0)
        
        # Calculate time delta
        if prev_row is not None:
            prev_time = self._get_value(prev_row, 'timestamp', 0.0)
            curr_time = self._get_value(row, 'timestamp', 0.0)
            dt = max(curr_time - prev_time, 0.001) if curr_time > prev_time else self.dt
        else:
            dt = self.dt
        
        # Update rotation (yaw) based on steering and speed
        # Using bicycle model approximation
        if abs(speed) > 0.1:  # Only update rotation if moving
            # Convert steering angle to radians (assuming degrees)
            steering_rad = np.radians(steering_angle)
            
            # Calculate yaw rate using bicycle model
            yaw_rate = (speed / self.wheelbase) * np.tan(steering_rad)
            self.current_state.rotation[2] += yaw_rate * dt  # Update yaw
            
            # Normalize yaw to [-pi, pi]
            self.current_state.rotation[2] = np.arctan2(
                np.sin(self.current_state.rotation[2]),
                np.cos(self.current_state.rotation[2])
            )
        
        # Update position based on velocity and rotation
        yaw = self.current_state.rotation[2]
        
        # Calculate velocity in vehicle frame
        vx_local = speed * np.cos(yaw)
        vy_local = speed * np.sin(yaw)
        
        # Update position
        self.current_state.position[0] += vx_local * dt
        self.current_state.position[1] += vy_local * dt
        # Z position (height) remains constant unless terrain data available
        
        # Update velocity
        self.current_state.velocity = np.array([vx_local, vy_local, 0.0])
        
        # Update acceleration (from CAN data)
        self.current_state.acceleration = np.array([accel_x, accel_y, 0.0])
        
        # Update other states
        self.current_state.speed = speed
        self.current_state.rpm = rpm
        self.current_state.gear = gear
        self.current_state.steering_angle = steering_angle
        self.current_state.throttle = throttle
        self.current_state.brake = max(brake_f, brake_r)
        self.current_state.timestamp += dt
        self.current_state.lap = int(self._get_value(row, 'lap', 1))
        
        return self.current_state
    
    def get_vehicle_corners(self) -> np.ndarray:
        """
        Calculate the 8 corners of the vehicle bounding box in world space.
        
        Returns:
            Array of shape (8, 3) with corner positions
        """
        if self.current_state is None:
            return np.zeros((8, 3))
        
        # Half dimensions
        hw, hl, hh = self.vehicle_dimensions / 2
        
        # Local corners (vehicle frame)
        corners_local = np.array([
            [-hw, -hl, -hh], [hw, -hl, -hh], [hw, hl, -hh], [-hw, hl, -hh],  # Bottom
            [-hw, -hl, hh],  [hw, -hl, hh],  [hw, hl, hh],  [-hw, hl, hh]    # Top
        ])
        
        # Rotation matrix (yaw only for now)
        yaw = self.current_state.rotation[2]
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Transform to world space
        corners_world = (R @ corners_local.T).T + self.current_state.position
        
        return corners_world
    
    def process_file(self, filepath: str, start_position: np.ndarray = None, 
                    save_history: bool = True, history_stride: int = 10) -> Dict:
        """
        Process entire telemetry file.
        
        Args:
            filepath: Path to CSV file
            start_position: Starting position [x, y, z]
            save_history: Whether to save state history
            history_stride: Save every Nth state to reduce memory
            
        Returns:
            Dictionary with processing summary
        """
        print(f"Processing telemetry file: {filepath}")
        
        row_count = 0
        chunk_count = 0
        prev_row = None
        
        self.state_history = []
        
        for chunk in self.load_telemetry_chunked(filepath):
            chunk_count += 1
            
            for idx, row in chunk.iterrows():
                if row_count == 0:
                    self.initialize_from_data(row, start_position)
                else:
                    self.process_telemetry_row(row, prev_row)
                
                # Save history periodically
                if save_history and row_count % history_stride == 0:
                    self.state_history.append(self._state_to_dict())
                
                prev_row = row
                row_count += 1
                
                if row_count % 10000 == 0:
                    print(f"Processed {row_count} rows...")
        
        print(f"Processing complete: {row_count} rows, {chunk_count} chunks")
        
        return {
            'total_rows': row_count,
            'final_position': self.current_state.position.tolist(),
            'final_rotation': self.current_state.rotation.tolist(),
            'total_distance': self._calculate_total_distance(),
            'max_speed': self._get_max_from_history('speed'),
            'max_rpm': self._get_max_from_history('rpm'),
            'gear_changes': self._count_gear_changes()
        }
    
    def _state_to_dict(self) -> Dict:
        """Convert current state to dictionary."""
        return {
            'position': self.current_state.position.copy(),
            'rotation': self.current_state.rotation.copy(),
            'velocity': self.current_state.velocity.copy(),
            'speed': self.current_state.speed,
            'rpm': self.current_state.rpm,
            'gear': self.current_state.gear,
            'steering_angle': self.current_state.steering_angle,
            'throttle': self.current_state.throttle,
            'brake': self.current_state.brake,
            'timestamp': self.current_state.timestamp,
            'lap': self.current_state.lap
        }
    
    def _calculate_total_distance(self) -> float:
        """Calculate total distance traveled from history."""
        if len(self.state_history) < 2:
            return 0.0
        
        distance = 0.0
        for i in range(1, len(self.state_history)):
            pos1 = self.state_history[i-1]['position']
            pos2 = self.state_history[i]['position']
            distance += np.linalg.norm(pos2 - pos1)
        
        return distance
    
    def _get_max_from_history(self, key: str) -> float:
        """Get maximum value from history."""
        if not self.state_history:
            return 0.0
        return max(state[key] for state in self.state_history)
    
    def _count_gear_changes(self) -> int:
        """Count number of gear changes."""
        if len(self.state_history) < 2:
            return 0
        
        changes = 0
        for i in range(1, len(self.state_history)):
            if self.state_history[i]['gear'] != self.state_history[i-1]['gear']:
                changes += 1
        
        return changes
    
    def get_state_at_time(self, timestamp: float) -> Optional[Dict]:
        """
        Get interpolated state at specific timestamp.
        
        Args:
            timestamp: Time to query
            
        Returns:
            State dictionary or None
        """
        if not self.state_history:
            return None
        
        # Find bracketing states
        for i in range(len(self.state_history) - 1):
            if (self.state_history[i]['timestamp'] <= timestamp <= 
                self.state_history[i+1]['timestamp']):
                # Linear interpolation
                t1 = self.state_history[i]['timestamp']
                t2 = self.state_history[i+1]['timestamp']
                alpha = (timestamp - t1) / (t2 - t1) if t2 != t1 else 0
                
                state = {}
                for key in self.state_history[i].keys():
                    if isinstance(self.state_history[i][key], np.ndarray):
                        state[key] = (1 - alpha) * self.state_history[i][key] + \
                                    alpha * self.state_history[i+1][key]
                    elif isinstance(self.state_history[i][key], (int, float)):
                        state[key] = (1 - alpha) * self.state_history[i][key] + \
                                    alpha * self.state_history[i+1][key]
                    else:
                        state[key] = self.state_history[i][key]
                
                return state
        
        return None
    
    def export_trajectory(self, output_path: str):
        """Export trajectory data to CSV."""
        if not self.state_history:
            print("No history to export")
            return
        
        data = []
        for state in self.state_history:
            data.append({
                'timestamp': state['timestamp'],
                'x': state['position'][0],
                'y': state['position'][1],
                'z': state['position'][2],
                'yaw': state['rotation'][2],
                'speed': state['speed'],
                'rpm': state['rpm'],
                'gear': state['gear'],
                'steering': state['steering_angle'],
                'throttle': state['throttle'],
                'brake': state['brake']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Trajectory exported to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = TelemetryEngine(chunk_size=50000)
    
    # Process telemetry file
    summary = engine.process_file(
        "telemetry_data.csv",
        start_position=np.array([0, 0, 0]),
        save_history=True,
        history_stride=5  # Save every 5th row
    )
    
    print("\nProcessing Summary:")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Final position: {summary['final_position']}")
    print(f"Total distance: {summary['total_distance']:.2f} m")
    print(f"Max speed: {summary['max_speed']:.2f} m/s")
    print(f"Max RPM: {summary['max_rpm']:.0f}")
    print(f"Gear changes: {summary['gear_changes']}")
    
    # Get vehicle corners at current position
    corners = engine.get_vehicle_corners()
    print(f"\nVehicle corners:\n{corners}")
    
    # Export trajectory
    engine.export_trajectory("vehicle_trajectory.csv")