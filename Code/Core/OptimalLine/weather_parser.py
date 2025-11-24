"""
Weather Data Parser
Parses weather data from race telemetry datasets
"""

import pandas as pd
import os
from pathlib import Path


class WeatherParser:
    """
    Parse weather data from telemetry dataset folders.
    
    Looks for files matching pattern: 26_Weather_*.CSV
    Extracts temperature, humidity, rainfall, wind data.
    """
    
    def __init__(self):
        self.weather_data = None
        self.track_name = None
        self.race_name = None
    
    def parse_weather_file(self, dataset_folder, race_name=None):
        """
        Parse weather data from a dataset folder.
        
        Args:
            dataset_folder (str): Path to dataset folder (e.g., .../barber-motorsports-park/barber)
            race_name (str, optional): Specific race name (e.g., "Race 1"). 
                                      If None, uses first available weather file.
        
        Returns:
            dict: Weather conditions with keys:
                - track_temp (float): Track temperature in Celsius
                - air_temp (float): Air temperature in Celsius
                - humidity (float): Humidity percentage
                - rainfall (float): Rainfall in mm/hr
                - wind_speed (float): Wind speed in m/s
                - wind_direction (float): Wind direction in degrees
                - pressure (float): Atmospheric pressure in mbar
            None if no weather data found.
        """
        dataset_path = Path(dataset_folder)
        if not dataset_path.exists():
            print(f"[WeatherParser] Dataset folder not found: {dataset_folder}")
            return None
        
        # Find weather files
        if race_name:
            weather_pattern = f"26_Weather_{race_name}_*.CSV"
        else:
            weather_pattern = "26_Weather_*.CSV"
        
        weather_files = list(dataset_path.glob(weather_pattern))
        
        if not weather_files:
            print(f"[WeatherParser] No weather data found in {dataset_folder}")
            return self._get_default_weather()
        
        # Use first matching file
        weather_file = weather_files[0]
        print(f"[WeatherParser] Loading weather data from {weather_file.name}")
        
        try:
            # Read CSV - format appears to be:
            # timestamp;datetime;air_temp;?;humidity;pressure;wind_speed;wind_direction;rainfall
            # May have header row, so try to detect it
            df = pd.read_csv(weather_file, sep=';', header=None, dtype=str)
            
            # Check if first row contains headers (non-numeric values in temperature column)
            if df.shape[0] > 0:
                first_val = df.iloc[0, 2]  # Air temp column
                try:
                    float(first_val)
                    # First row is data, no header
                    start_row = 0
                except (ValueError, TypeError):
                    # First row is header, skip it
                    start_row = 1
                    df = df.iloc[1:]
            else:
                print(f"[WeatherParser] Weather file is empty")
                return self._get_default_weather()
            
            # Convert to numeric, coercing errors
            df[2] = pd.to_numeric(df[2], errors='coerce')  # air_temp
            df[4] = pd.to_numeric(df[4], errors='coerce')  # humidity
            df[5] = pd.to_numeric(df[5], errors='coerce')  # pressure
            df[6] = pd.to_numeric(df[6], errors='coerce')  # wind_speed
            df[7] = pd.to_numeric(df[7], errors='coerce')  # wind_direction
            df[8] = pd.to_numeric(df[8], errors='coerce')  # rainfall
            
            # Drop rows with NaN in critical columns
            df = df.dropna(subset=[2, 4, 5, 6, 7, 8])
            
            # Column mapping based on sample data:
            # 0: timestamp
            # 1: datetime string
            # 2: air_temp (Celsius)
            # 3: unknown (0 in sample)
            # 4: humidity (%)
            # 5: pressure (mbar)
            # 6: wind_speed (m/s)
            # 7: wind_direction (degrees)
            # 8: rainfall (mm/hr)
            
            if df.empty:
                print(f"[WeatherParser] No valid weather data rows")
                return self._get_default_weather()
            
            # Calculate average conditions (or use first row if you want start conditions)
            # Using median to avoid outliers
            air_temp = df[2].median()
            humidity = df[4].median()
            pressure = df[5].median()
            wind_speed = df[6].median()
            wind_direction = df[7].median()
            rainfall = df[8].median()
            
            # Estimate track temperature (typically 10-15C higher than air temp)
            # This is a simplified model - actual track temp depends on sun, track material, etc.
            track_temp = air_temp + 10.0
            
            weather_dict = {
                'track_temp': float(track_temp),
                'air_temp': float(air_temp),
                'humidity': float(humidity),
                'rainfall': float(rainfall),
                'wind_speed': float(wind_speed),
                'wind_direction': float(wind_direction),
                'pressure': float(pressure)
            }
            
            print(f"[WeatherParser] Conditions: Air={air_temp:.1f}C, Track={track_temp:.1f}C, "
                  f"Humidity={humidity:.0f}%, Rain={rainfall:.1f}mm/hr, Wind={wind_speed:.1f}m/s")
            
            self.weather_data = weather_dict
            return weather_dict
            
        except Exception as e:
            print(f"[WeatherParser] Error parsing weather file: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self):
        """Return default ideal weather conditions."""
        return {
            'track_temp': 85.0,  # Optimal tire temperature
            'air_temp': 25.0,
            'humidity': 50,
            'rainfall': 0.0,
            'wind_speed': 0.0,
            'wind_direction': 0,
            'pressure': 1013.25
        }
    
    def find_dataset_folder(self, track_name):
        """
        Find dataset folder for a given track name.
        
        Args:
            track_name (str): Track name to search for
        
        Returns:
            str: Path to dataset folder, or None if not found
        """
        # Look in Non_Code/dataSets/
        base_path = Path(__file__).parent.parent.parent.parent / "Non_Code" / "dataSets"
        
        if not base_path.exists():
            return None
        
        # Search for folders matching track name
        track_name_lower = track_name.lower().replace(' ', '-')
        
        for folder in base_path.iterdir():
            if folder.is_dir() and track_name_lower in folder.name.lower():
                # Look for subfolder with actual data
                subfolders = [f for f in folder.iterdir() if f.is_dir() and not f.name.startswith('__')]
                if subfolders:
                    return str(subfolders[0])
        
        return None
    
    def parse_weather_for_track(self, track_name, race_name=None):
        """
        Convenience method to find and parse weather data for a track.
        
        Args:
            track_name (str): Track name (e.g., "Barber Motorsports Park")
            race_name (str, optional): Race name (e.g., "Race 1")
        
        Returns:
            dict: Weather conditions, or default conditions if not found
        """
        dataset_folder = self.find_dataset_folder(track_name)
        
        if dataset_folder:
            return self.parse_weather_file(dataset_folder, race_name)
        else:
            print(f"[WeatherParser] No dataset found for track: {track_name}")
            return self._get_default_weather()


# Quick test
if __name__ == "__main__":
    parser = WeatherParser()
    
    # Test with Barber
    weather = parser.parse_weather_for_track("Barber Motorsports Park", "Race 1")
    if weather:
        print("\nParsed weather data:")
        for key, value in weather.items():
            print(f"  {key}: {value}")
