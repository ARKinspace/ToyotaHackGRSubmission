"""
State processor for converting telemetry data to vehicle states
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import math


@dataclass
class VehicleState:
    """Current state of the vehicle"""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [roll, pitch, yaw] in radians
    velocity: np.ndarray  # [vx, vy, vz]
    speed: float
    rpm: float
    gear: int
    steering_angle: float
    throttle: float
    brake: float
    timestamp: float
    lap: int


class KalmanFilter:
    """
    Simple Kalman Filter for 2D vehicle tracking (Constant Acceleration Model).
    State: [x, y, vx, vy, ax, ay]
    Measurement: [gps_x, gps_y, accel_x, accel_y]
    """
    def __init__(self, dt=0.016, process_noise=0.01, measure_noise_gps=2.0, measure_noise_accel=0.5):
        self.dt = dt
        
        # State vector [x, y, vx, vy, ax, ay]
        self.x = np.zeros(6)
        
        # State Transition Matrix (F)
        # x = x + vx*dt + 0.5*ax*dt^2
        # v = v + ax*dt
        # a = a
        self.F = np.eye(6)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = 0.5 * dt**2
        self.F[1, 5] = 0.5 * dt**2
        self.F[2, 4] = dt
        self.F[3, 5] = dt
        
        # Measurement Matrix (H)
        # We measure x, y (GPS) and ax, ay (Accelerometer)
        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1  # Measure x
        self.H[1, 1] = 1  # Measure y
        self.H[2, 4] = 1  # Measure ax
        self.H[3, 5] = 1  # Measure ay
        
        # Covariance Matrix (P)
        self.P = np.eye(6) * 100.0
        
        # Process Noise Covariance (Q)
        self.Q = np.eye(6) * process_noise
        
        # Measurement Noise Covariance (R)
        self.R = np.diag([measure_noise_gps, measure_noise_gps, measure_noise_accel, measure_noise_accel])
        
        # Identity matrix
        self.I = np.eye(6)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        z: Measurement vector [gps_x, gps_y, accel_x, accel_y]
        If GPS is missing (NaN), we only update with accelerometer.
        """
        # Check for missing GPS
        mask = ~np.isnan(z)
        
        if not np.any(mask):
            # No measurements, just predict
            return self.x
            
        # Adaptive H and R based on available measurements
        # This is a simplified approach: zero out rows for missing measurements
        # Better approach: Construct H_k and R_k dynamically
        
        H_k = self.H[mask]
        R_k = self.R[np.ix_(mask, mask)]
        z_k = z[mask]
        
        # Standard Kalman Update
        y = z_k - H_k @ self.x  # Innovation
        S = H_k @ self.P @ H_k.T + R_k  # Innovation Covariance
        try:
            K = self.P @ H_k.T @ np.linalg.inv(S)  # Kalman Gain
        except np.linalg.LinAlgError:
            # Fallback if singular
            return self.x
            
        self.x = self.x + K @ y
        self.P = (self.I - K @ H_k) @ self.P
        
        return self.x


class StateProcessor:
    """
    Converts telemetry DataFrame to vehicle states for 3D visualization.
    """
    
    def __init__(self):
        self.wheelbase = 2.575  # GR86 wheelbase in meters
        self.steering_ratio = 13.5  # GR86 steering ratio (wheel turns to road wheel angle)
        self.geo_ref = None
        self.scaling_factor = 1.0
        self.R = 6378137  # Earth radius in meters

    def set_geo_reference(self, lat, lon, scaling_factor=1.0):
        """Set the geographic reference point (center) and scaling factor for projection."""
        self.geo_ref = {'lat': lat, 'lon': lon}
        self.scaling_factor = scaling_factor
        print(f"StateProcessor: Geo reference set to {lat}, {lon} (scale: {scaling_factor})")

    def process_telemetry(self, telemetry_df: pd.DataFrame, start_position: np.ndarray = None) -> List[Dict]:
        """
        Convert telemetry DataFrame to list of vehicle states.
        Uses vectorized operations for performance.
        """
        if start_position is None:
            start_position = np.array([0.0, 0.0, 0.0])
        
        if telemetry_df.empty:
            return []
        
        # Make a copy to avoid modifying original
        df = telemetry_df.copy()
        
        # Determine which timestamp column to use
        timestamp_col = None
        if 'elapsed_seconds' in df.columns:
            timestamp_col = 'elapsed_seconds'
        elif 'timestamp' in df.columns:
            timestamp_col = 'timestamp'
        
        # Ensure data is sorted by timestamp
        if timestamp_col:
            # Convert timestamp to numeric if it's a string
            if df[timestamp_col].dtype == 'object':
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df[timestamp_col] = (df[timestamp_col] - df[timestamp_col].min()).dt.total_seconds()
            df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Fill NaN values with defaults (vectorized)
        df['speed'] = df.get('speed', pd.Series([0] * len(df))).fillna(0).astype(float)
        df['nmot'] = df.get('nmot', pd.Series([0] * len(df))).fillna(0).astype(float)
        df['gear'] = df.get('gear', pd.Series([0] * len(df))).fillna(0).astype(int)
        df['Steering_Angle'] = df.get('Steering_Angle', pd.Series([0] * len(df))).fillna(0).astype(float)
        df['aps'] = df.get('aps', pd.Series([0] * len(df))).fillna(0).astype(float)
        df['pbrake_f'] = df.get('pbrake_f', pd.Series([0] * len(df))).fillna(0).astype(float)
        df['pbrake_r'] = df.get('pbrake_r', pd.Series([0] * len(df))).fillna(0).astype(float)
        df['lap'] = df.get('lap', pd.Series([1] * len(df))).fillna(1).astype(int)
        
        # Handle GPS coordinates
        lon_col = 'VBOX_Long_Minutes' if 'VBOX_Long_Minutes' in df.columns else 'Longitude'
        lat_col = 'VBOX_Lat_Min' if 'VBOX_Lat_Min' in df.columns else 'Latitude'
        
        df['lon'] = df.get(lon_col, pd.Series([0] * len(df))).fillna(0).astype(float)
        df['lat'] = df.get(lat_col, pd.Series([0] * len(df))).fillna(0).astype(float)
        
        # Filter out rows without valid GPS (but keep the indices continuous)
        # Instead of skipping, we'll interpolate missing GPS values
        valid_gps_mask = (df['lon'] != 0) & (df['lat'] != 0)
        
        if not valid_gps_mask.any():
            print("Warning: No valid GPS coordinates found in telemetry data")
            return []
        
        # Interpolate missing GPS coordinates
        df.loc[~valid_gps_mask, 'lon'] = np.nan
        df.loc[~valid_gps_mask, 'lat'] = np.nan
        df['lon'] = df['lon'].interpolate(method='linear', limit_direction='both')
        df['lat'] = df['lat'].interpolate(method='linear', limit_direction='both')
        
        # Detect if coordinates are in degrees or minutes
        is_degrees = (df['lat'].abs() < 90).all() and (df['lat'].abs() > 0.1).any()
        
        # Vectorized GPS projection
        if self.geo_ref:
            center_lat_rad = self.geo_ref['lat'] * math.pi / 180
            
            if is_degrees:
                lat_deg = df['lat'].values
                lon_deg = df['lon'].values
            else:
                lat_deg = df['lat'].values / 60.0
                lon_deg = df['lon'].values / 60.0
            
            dLat = (lat_deg - self.geo_ref['lat']) * math.pi / 180
            dLon = (lon_deg - self.geo_ref['lon']) * math.pi / 180
            
            x = self.R * dLon * math.cos(center_lat_rad)
            y = self.R * dLat
        else:
            # Fallback: use first point as reference
            if is_degrees:
                lat_deg = df['lat'].values
                lon_deg = df['lon'].values
            else:
                lat_deg = df['lat'].values / 60.0
                lon_deg = df['lon'].values / 60.0
            
            ref_lat = lat_deg[0]
            ref_lon = lon_deg[0]
            
            x = (lon_deg - ref_lon) * 92000
            y = (lat_deg - ref_lat) * 111000
        
        df['x'] = x
        df['y'] = y
        
        # Get timestamps
        if timestamp_col:
            timestamps = df[timestamp_col].values
        else:
            timestamps = np.arange(len(df)) * 0.016
        
        df['timestamp'] = timestamps
        
        # Calculate dt vectorized
        dt_array = np.diff(timestamps, prepend=timestamps[0])
        dt_array[dt_array <= 0] = 0.016
        df['dt'] = dt_array
        
        # Calculate yaw using bicycle model (vectorized where possible)
        steering_rad = np.radians(df['Steering_Angle'].values / self.steering_ratio)
        speed_array = df['speed'].values
        
        # Calculate yaw rate
        yaw_rate = np.zeros(len(df))
        moving_mask = speed_array > 0.1
        yaw_rate[moving_mask] = (speed_array[moving_mask] / self.wheelbase) * np.tan(steering_rad[moving_mask])
        
        # Cumulative sum for yaw
        cumulative_yaw = np.cumsum(yaw_rate * dt_array)
        df['yaw'] = cumulative_yaw
        
        # Calculate velocity components
        df['vx'] = df['speed'] * np.cos(cumulative_yaw)
        df['vy'] = df['speed'] * np.sin(cumulative_yaw)
        
        # Build states list efficiently
        states = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            state = {
                'position': np.array([row['x'], row['y'], 0.0]),
                'rotation': np.array([0.0, 0.0, row['yaw']]),
                'velocity': np.array([row['vx'], row['vy'], 0.0]),
                'speed': row['speed'],
                'rpm': row['nmot'],
                'gear': row['gear'],
                'steering_angle': row['Steering_Angle'],
                'throttle': row['aps'] / 100.0,
                'brake': max(row['pbrake_f'], row['pbrake_r']),
                'timestamp': row['timestamp'],
                'lap': row['lap'],
                'distance': 0
            }
            
            states.append(state)
        
        print(f"Processed {len(states)} vehicle states (vectorized)")
        if states:
            print(f"Timestamp range: {states[0]['timestamp']:.3f}s to {states[-1]['timestamp']:.3f}s")
            print(f"Total duration: {states[-1]['timestamp'] - states[0]['timestamp']:.3f}s")
        
        return states
    
    def _project_coords(self, lat, lon, is_degrees=False):
        if not is_degrees:
            lon_deg = lon / 60.0
            lat_deg = lat / 60.0
        else:
            lon_deg = lon
            lat_deg = lat
            
        if self.geo_ref:
            center_lat_rad = self.geo_ref['lat'] * math.pi / 180
            dLat = (lat_deg - self.geo_ref['lat']) * math.pi / 180
            dLon = (lon_deg - self.geo_ref['lon']) * math.pi / 180
            
            # Don't apply scaling_factor - track coordinates are already properly scaled
            # Both track and vehicle should use the same natural projection
            x = self.R * dLon * math.cos(center_lat_rad)
            y = self.R * dLat
            return x, y
        else:
            # Fallback
            if not hasattr(self, '_ref_lon'):
                self._ref_lon = lon_deg
                self._ref_lat = lat_deg
                return 0, 0
            else:
                x = (lon_deg - self._ref_lon) * 92000
                y = (lat_deg - self._ref_lat) * 111000
                return x, y

    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling NaN."""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=0):
        """Safely convert value to int, handling NaN."""
        try:
            if pd.isna(value):
                return default
            return int(float(value))
        except (ValueError, TypeError):
            return default
