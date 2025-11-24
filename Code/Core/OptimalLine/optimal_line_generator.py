"""
Weather-Adjusted Optimal Racing Line Generator
Adapted for RaceTrack Studio from reference implementation
"""

import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


class OptimalLineGenerator:
    """
    Generate optimal racing line with weather-adjusted physics.
    
    Calculates the fastest possible path around a track given:
    - Track geometry (centerline coordinates)
    - Vehicle parameters (power, mass, grip limits)
    - Weather conditions (temperature, rain, wind)
    """
    
    def __init__(self, track_centerline, track_width, vehicle_config=None, weather_config=None):
        """
        Initialize optimal line generator.
        
        Args:
            track_centerline (pd.DataFrame): Track centerline with columns: x, y, distance
            track_width (float): Track width in meters
            vehicle_config (dict, optional): Vehicle parameters
            weather_config (dict, optional): Weather conditions
        """
        self.centerline = track_centerline
        self.track_width = track_width
        
        # Default vehicle config (Toyota GR86 Cup Car)
        self.vehicle = vehicle_config or self._get_default_vehicle_config()
        
        # Default weather config (ideal dry conditions)
        self.weather = weather_config or self._get_default_weather_config()
        
        # Validate inputs
        self._validate_inputs()
        
        # Extract centerline coordinates
        self.center_x = track_centerline['x'].values
        self.center_y = track_centerline['y'].values
        self.distance = track_centerline['distance'].values
        self.track_length = self.distance[-1]
        
        # Calculate effective tire grip based on weather
        self.effective_grip = self._calculate_tire_grip()
        
        print(f"[OptimalLine] Initialized - Track: {self.track_length:.1f}m, Grip: {self.effective_grip:.2f}")
    
    def _get_default_vehicle_config(self):
        """Default vehicle configuration (Toyota GR86 Cup Car)"""
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
    
    def _get_default_weather_config(self):
        """Default weather configuration (ideal dry conditions)"""
        return {
            'track_temp': 85.0,  # Optimal temperature
            'air_temp': 25.0,
            'humidity': 50,
            'wind_speed': 0.0,
            'wind_direction': 0,
            'rainfall': 0.0
        }
    
    def _validate_inputs(self):
        """Validate input parameters."""
        required_track_cols = ['x', 'y', 'distance']
        for col in required_track_cols:
            if col not in self.centerline.columns:
                raise ValueError(f"Track centerline must have '{col}' column")
        
        required_vehicle_keys = ['tire_friction_dry', 'top_speed_ms', 'max_lateral_g', 
                                'max_brake_g', 'max_accel_g', 'corner_speed_factor']
        for key in required_vehicle_keys:
            if key not in self.vehicle:
                raise ValueError(f"Vehicle config must have '{key}' parameter")
        
        if 'track_temp' not in self.weather:
            raise ValueError("Weather config must have 'track_temp' parameter")
    
    def _calculate_tire_grip(self):
        """
        Calculate weather-adjusted tire grip coefficient.
        
        Factors:
        1. Track temperature (optimal at 85C, ±30% at ±100C deviation)
        2. Rainfall intensity (dry/damp/intermediate/wet)
        
        Returns:
            float: Effective grip coefficient
        """
        base_friction = self.vehicle['tire_friction_dry']
        
        # Temperature effect: optimal at 85C
        optimal_temp = self.vehicle.get('optimal_tire_temp', 85)
        temp_diff = abs(self.weather['track_temp'] - optimal_temp)
        temp_factor = 1.0 - (temp_diff / 100.0) * 0.3  # 30% reduction at 100C deviation
        temp_factor = max(0.4, min(1.0, temp_factor))  # Clamp to [0.4, 1.0]
        
        # Rain effect: determines tire compound
        rainfall = self.weather.get('rainfall', 0)
        if rainfall > 10.0:  # Heavy rain - wet tires
            effective_friction = self.vehicle.get('tire_friction_wet', 0.70)
        elif rainfall > 2.0:  # Moderate rain - intermediate tires
            effective_friction = self.vehicle.get('tire_friction_intermediate', 1.00)
        elif rainfall > 0.1:  # Light rain - damp conditions
            effective_friction = base_friction * 0.85
        else:  # Dry conditions
            effective_friction = base_friction
        
        # Apply temperature factor
        return effective_friction * temp_factor
    
    def generate_optimal_line(self, n_points=2000):
        """
        Generate optimal racing line for current weather conditions.
        
        Algorithm:
        1. Interpolate track centerline to desired resolution
        2. Calculate track geometry (tangents, normals, curvature)
        3. Compute geometric racing line adjusted for weather grip
        4. Calculate corner speed limits based on lateral acceleration
        5. Apply acceleration constraints (forward pass)
        6. Apply braking constraints (backward pass)
        7. Calculate lap time
        
        Args:
            n_points (int): Number of points for line discretization
        
        Returns:
            pd.DataFrame: Optimal line with columns:
                - x, y: Coordinates
                - distance: Cumulative distance
                - speed: Optimal speed at each point
                - curvature: Track curvature
                - grip_coefficient: Effective grip
                - lap_time: Total lap time (constant for all rows)
        """
        print(f"[OptimalLine] Generating line with {n_points} points...")
        
        # Step 1: Remove duplicate points and ensure minimum spacing
        # This prevents "Invalid inputs" error in splprep
        cleaned_x = [self.center_x[0]]
        cleaned_y = [self.center_y[0]]
        min_spacing = 0.1  # Minimum distance between points in meters
        
        for i in range(1, len(self.center_x)):
            dx = self.center_x[i] - cleaned_x[-1]
            dy = self.center_y[i] - cleaned_y[-1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist >= min_spacing:
                cleaned_x.append(self.center_x[i])
                cleaned_y.append(self.center_y[i])
        
        # Need at least k+1 points for cubic spline
        if len(cleaned_x) < 4:
            print(f"[OptimalLine] ERROR: Insufficient unique points ({len(cleaned_x)})")
            raise ValueError(f"Track has too few unique points ({len(cleaned_x)} < 4 required)")
        
        print(f"[OptimalLine] Using {len(cleaned_x)} unique points (from {len(self.center_x)} total)")
        
        # Interpolate centerline with cleaned data
        try:
            tck, u = splprep([cleaned_x, cleaned_y], s=0, per=True, k=3)
        except Exception as e:
            print(f"[OptimalLine] ERROR: Spline interpolation failed: {e}")
            # Try with smoothing if exact interpolation fails
            print(f"[OptimalLine] Retrying with smoothing...")
            tck, u = splprep([cleaned_x, cleaned_y], s=10, per=True, k=3)
        u_fine = np.linspace(0, 1, n_points, endpoint=False)
        center_x, center_y = splev(u_fine, tck)
        
        # Step 2: Calculate tangent and normal vectors
        dx, dy = splev(u_fine, tck, der=1)
        tangent_len = np.sqrt(dx**2 + dy**2)
        tangent_x = dx / tangent_len
        tangent_y = dy / tangent_len
        normal_x = -tangent_y
        normal_y = tangent_x
        
        # Calculate distance array
        distance = np.zeros(n_points)
        for i in range(1, n_points):
            ds = np.sqrt((center_x[i] - center_x[i-1])**2 + (center_y[i] - center_y[i-1])**2)
            distance[i] = distance[i-1] + ds
        
        # Step 3: Calculate centerline curvature
        d2x, d2y = splev(u_fine, tck, der=2)
        curvature_center = np.abs(dx * d2y - dy * d2x) / (tangent_len**3)
        curvature_center = gaussian_filter1d(curvature_center, sigma=15)
        
        # Step 4: Generate racing line using geometric principle
        max_offset = self.track_width / 2 * 0.9
        grip_factor = self.effective_grip / self.vehicle['tire_friction_dry']
        
        # Lateral offsets (negative = inside of corner)
        # Higher grip allows tighter lines
        lateral_offsets = -np.sign(curvature_center) * np.minimum(
            np.abs(curvature_center) * 800 * grip_factor,
            max_offset
        )
        lateral_offsets = gaussian_filter1d(lateral_offsets, sigma=20)
        
        # Calculate racing line coordinates
        race_x = center_x + lateral_offsets * normal_x
        race_y = center_y + lateral_offsets * normal_y
        
        # Step 5: Calculate racing line curvature
        race_tck, _ = splprep([race_x, race_y], s=10, per=True, k=3)
        race_dx, race_dy = splev(u_fine, race_tck, der=1)
        race_d2x, race_d2y = splev(u_fine, race_tck, der=2)
        race_tangent_len = np.sqrt(race_dx**2 + race_dy**2)
        curvature_race = np.abs(race_dx * race_d2y - race_dy * race_d2x) / (race_tangent_len**3)
        curvature_race = gaussian_filter1d(curvature_race, sigma=10)
        curvature_race = np.maximum(curvature_race, 1e-6)
        
        # Step 6: Calculate speeds using weather-adjusted physics
        g = 9.81
        corner_radius = 1.0 / curvature_race
        
        # Maximum cornering speed: v = sqrt(mu * g * r * factor)
        max_speed_lateral = np.sqrt(
            self.effective_grip * g * corner_radius * self.vehicle['corner_speed_factor']
        )
        max_speed_lateral = np.minimum(max_speed_lateral, self.vehicle['top_speed_ms'])
        
        # Forward pass (acceleration-limited)
        speeds = np.zeros(n_points)
        speeds[0] = max_speed_lateral[0]
        for i in range(1, n_points):
            ds = distance[i] - distance[i-1]
            # v_final^2 = v_initial^2 + 2*a*d
            v_max_accel = np.sqrt(speeds[i-1]**2 + 2 * self.vehicle['max_accel_g'] * g * ds)
            speeds[i] = min(max_speed_lateral[i], v_max_accel)
        
        # Backward pass (braking-limited)
        for i in range(n_points-2, -1, -1):
            ds = distance[i+1] - distance[i]
            v_max_brake = np.sqrt(speeds[i+1]**2 + 2 * self.vehicle['max_brake_g'] * g * ds)
            speeds[i] = min(speeds[i], v_max_brake)
        
        # Step 7: Calculate lap time
        segment_times = []
        for i in range(len(distance)-1):
            ds = distance[i+1] - distance[i]
            v_avg = (speeds[i] + speeds[i+1]) / 2
            segment_times.append(ds / max(v_avg, 1.0))
        lap_time = sum(segment_times)
        
        # Create result DataFrame
        optimal_line = pd.DataFrame({
            'x': race_x,
            'y': race_y,
            'distance': distance,
            'speed': speeds,
            'curvature': curvature_race,
            'grip_coefficient': self.effective_grip,
            'lap_time': lap_time
        })
        
        print(f"[OptimalLine] Complete - Lap time: {lap_time:.2f}s (grip: {self.effective_grip:.2f})")
        
        return optimal_line
    
    def update_weather(self, new_weather):
        """Update weather conditions and recalculate grip."""
        self.weather.update(new_weather)
        self.effective_grip = self._calculate_tire_grip()
        print(f"[OptimalLine] Weather updated - New grip: {self.effective_grip:.2f}")
