"""
Telemetry data loader for GR Cup racing data
Handles the long-format telemetry CSV files where each row is a single parameter
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class TelemetryLoader:
    """
    Loads and processes telemetry data from long-format CSV files.
    Each row in the CSV contains: timestamp, vehicle_id, telemetry_name, telemetry_value
    """
    
    def __init__(self):
        self.raw_data = None
        self.vehicles = []
        self.mode = 'raw'  # 'raw' or 'parsed'
        self.parsed_data_cache = {}  # Cache for parsed dataframes
        self.parsed_folder = None
        self.parsed_sessions = {} # { 'Race 1': ['Vehicle A', ...], ... }

    def load_telemetry_file(self, filepath: str) -> pd.DataFrame:
        """
        Load telemetry CSV file (raw format).
        """
        print(f"Loading telemetry from: {filepath}")
        self.mode = 'raw'
        self.raw_data = pd.read_csv(filepath, low_memory=False)
        
        # Detect vehicle column
        if 'original_vehicle_id' in self.raw_data.columns:
            self.vehicles = sorted(self.raw_data['original_vehicle_id'].unique())
        elif 'vehicle_id' in self.raw_data.columns:
            self.vehicles = sorted(self.raw_data['vehicle_id'].unique())
        else:
            self.vehicles = sorted(self.raw_data['vehicle_number'].unique())
            
        print(f"Found {len(self.vehicles)} vehicles in raw file")
        return self.raw_data

    def load_from_parsed_folder(self, folder_path: str):
        """
        Load vehicle list from a folder containing parsed Race/Vehicle/telemetry.csv structure.
        """
        self.mode = 'parsed'
        self.parsed_folder = Path(folder_path)
        self.parsed_data_cache = {}
        self.parsed_sessions = {}
        self.vehicles = []
        
        if not self.parsed_folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        print(f"Scanning parsed folder: {folder_path}")
        
        # Scan for Race directories
        for race_dir in self.parsed_folder.iterdir():
            if race_dir.is_dir():
                race_name = race_dir.name
                self.parsed_sessions[race_name] = []
                
                # Scan for Vehicle directories
                for vehicle_dir in race_dir.iterdir():
                    if vehicle_dir.is_dir():
                        # Check if telemetry file exists
                        if (vehicle_dir / "telemetry.csv").exists() or (vehicle_dir / "telemetry.parquet").exists():
                            vid = vehicle_dir.name
                            self.parsed_sessions[race_name].append(vid)
                            if vid not in self.vehicles:
                                self.vehicles.append(vid)
                
                self.parsed_sessions[race_name].sort()
                print(f"  {race_name}: {len(self.parsed_sessions[race_name])} vehicles")
            
        self.vehicles.sort()
        print(f"Total parsed vehicles: {len(self.vehicles)}")

    def get_races(self) -> List[str]:
        """Get list of available races (only in parsed mode)."""
        if self.mode == 'parsed':
            return sorted(list(self.parsed_sessions.keys()))
        return []

    def get_vehicles(self, race_id: str = None) -> List:
        """Get list of unique vehicles. If race_id is provided, returns vehicles for that race."""
        if self.mode == 'parsed' and race_id:
            return self.parsed_sessions.get(race_id, [])
        return self.vehicles
    
    def get_vehicle_data(self, vehicle_id: str, lap: Optional[int] = None, race_id: str = None) -> pd.DataFrame:
        """
        Extract telemetry data for a specific vehicle.
        """
        if self.mode == 'parsed':
            return self._get_parsed_vehicle_data(vehicle_id, lap, race_id)
        else:
            return self._get_raw_vehicle_data(vehicle_id, lap)

    def _get_parsed_vehicle_data(self, vehicle_id: str, lap: Optional[int] = None, race_id: str = None) -> pd.DataFrame:
        # Construct filename
        # Structure: Output/Race X/Vehicle Y/telemetry.csv
        
        cache_key = f"{race_id}_{vehicle_id}" if race_id else vehicle_id
        
        if cache_key in self.parsed_data_cache:
            df = self.parsed_data_cache[cache_key]
        else:
            filepath = None
            
            if race_id:
                # Direct lookup
                filepath = self.parsed_folder / race_id / vehicle_id / "telemetry.csv"
            else:
                # Search for vehicle in all races
                for r_name, v_list in self.parsed_sessions.items():
                    if vehicle_id in v_list:
                        filepath = self.parsed_folder / r_name / vehicle_id / "telemetry.csv"
                        break
            
            if not filepath or not filepath.exists():
                print(f"File not found for vehicle {vehicle_id}")
                return pd.DataFrame()
                
            print(f"Loading parsed file: {filepath}")
            df = pd.read_csv(filepath)
            # Parse meta_time
            if 'meta_time' in df.columns:
                try:
                    df['meta_time'] = pd.to_datetime(df['meta_time'], format='ISO8601')
                except ValueError:
                    # Fallback for other formats
                    df['meta_time'] = pd.to_datetime(df['meta_time'])
                
            self.parsed_data_cache[cache_key] = df
            
        # Filter by lap if requested
        if lap is not None and 'lap' in df.columns:
            df = df[df['lap'] == lap].copy()
            
        return df

    def _get_raw_vehicle_data(self, vehicle_id: str, lap: Optional[int] = None) -> pd.DataFrame:
        if self.raw_data is None:
            raise ValueError("No telemetry data loaded.")
        
        # Filter by vehicle
        if 'original_vehicle_id' in self.raw_data.columns:
            vehicle_df = self.raw_data[self.raw_data['original_vehicle_id'] == vehicle_id].copy()
        elif 'vehicle_id' in self.raw_data.columns:
            vehicle_df = self.raw_data[self.raw_data['vehicle_id'] == vehicle_id].copy()
        else:
            # Fallback for when we might have a mismatch in ID format
            # Try to match loosely if exact match fails?
            # For now assume exact match
            vehicle_df = self.raw_data[self.raw_data['vehicle_number'] == vehicle_id].copy()
        
        if len(vehicle_df) == 0:
            # Try to find by partial match if it's a string
            print(f"Warning: No data found for vehicle {vehicle_id} in raw data")
            return pd.DataFrame()
        
        # Filter by lap if specified
        if lap is not None and 'lap' in vehicle_df.columns:
            vehicle_df = vehicle_df[vehicle_df['lap'] == lap].copy()
        
        # Pivot
        try:
            pivoted = vehicle_df.pivot_table(
                index='timestamp',
                columns='telemetry_name',
                values='telemetry_value',
                aggfunc='first'
            ).reset_index()
            
            pivoted = pivoted.sort_values('timestamp').reset_index(drop=True)
            return pivoted
            
        except Exception as e:
            print(f"Error pivoting data: {e}")
            raise
    
    def get_available_parameters(self, vehicle_id: str) -> List[str]:
        """
        Get list of available telemetry parameters for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            List of parameter names
        """
        if self.raw_data is None:
            return []
        
        if 'original_vehicle_id' in self.raw_data.columns:
            vehicle_df = self.raw_data[self.raw_data['original_vehicle_id'] == vehicle_id]
        elif 'vehicle_id' in self.raw_data.columns:
            vehicle_df = self.raw_data[self.raw_data['vehicle_id'] == vehicle_id]
        else:
            vehicle_df = self.raw_data[self.raw_data['vehicle_number'] == vehicle_id]
        
        if 'telemetry_name' in vehicle_df.columns:
            return sorted(vehicle_df['telemetry_name'].unique())
        else:
            return []
    
    def get_laps(self, vehicle_id: str) -> List[int]:
        """
        Get list of laps for a specific vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            
        Returns:
            List of lap numbers
        """
        if self.raw_data is None:
            return []
        
        if 'original_vehicle_id' in self.raw_data.columns:
            vehicle_df = self.raw_data[self.raw_data['original_vehicle_id'] == vehicle_id]
        elif 'vehicle_id' in self.raw_data.columns:
            vehicle_df = self.raw_data[self.raw_data['vehicle_id'] == vehicle_id]
        else:
            vehicle_df = self.raw_data[self.raw_data['vehicle_number'] == vehicle_id]
        
        if 'lap' in vehicle_df.columns:
            return sorted(vehicle_df['lap'].unique())
        else:
            return []


class SessionManager:
    """
    Manages race session data by automatically detecting and loading race files from a folder.
    """
    
    def __init__(self):
        self.folder_path = None
        self.sessions = {}
        
    def load_folder(self, folder_path: str) -> Dict[int, Dict[str, str]]:
        """
        Scan folder and detect race files.
        
        Args:
            folder_path: Path to folder containing race data
            
        Returns:
            Dictionary mapping race number to file paths
        """
        self.folder_path = Path(folder_path)
        
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        print(f"Scanning folder: {folder_path}")
        
        # Find telemetry files - support multiple naming patterns
        # Pattern 1: *telemetry_data.csv (e.g., R1_barber_telemetry_data.csv)
        # Pattern 2: *_telemetry.csv (e.g., R1_indianapolis_motor_speedway_telemetry.csv)
        telemetry_files = list(self.folder_path.glob("*telemetry_data.csv")) + \
                         list(self.folder_path.glob("*_telemetry.csv"))
        
        # Remove duplicates if any
        telemetry_files = list(set(telemetry_files))
        
        if not telemetry_files:
            print("No telemetry files found. Looking for patterns:")
            print(f"  *telemetry_data.csv: {list(self.folder_path.glob('*telemetry_data.csv'))}")
            print(f"  *_telemetry.csv: {list(self.folder_path.glob('*_telemetry.csv'))}")
            print(f"Available files in folder: {[f.name for f in self.folder_path.glob('*.csv')]}")
        else:
            print(f"Found {len(telemetry_files)} telemetry file(s): {[f.name for f in telemetry_files]}")
        
        for telem_file in telemetry_files:
            # Extract race number from filename (R1_ or R2_)
            if 'R1_' in telem_file.name or '_R1_' in telem_file.name:
                race_num = 1
            elif 'R2_' in telem_file.name or '_R2_' in telem_file.name:
                race_num = 2
            else:
                continue
            
            if race_num not in self.sessions:
                self.sessions[race_num] = {}
            
            self.sessions[race_num]['telemetry'] = str(telem_file)
            
            # Look for associated files
            # Handle both patterns: R1_track_telemetry_data.csv and R1_track_telemetry.csv
            if '_telemetry_data.csv' in telem_file.name:
                prefix = telem_file.name.split('_telemetry_data.csv')[0]
            else:
                prefix = telem_file.name.split('_telemetry.csv')[0]
            
            lap_time_file = self.folder_path / f"{prefix}_lap_time.csv"
            if lap_time_file.exists():
                self.sessions[race_num]['lap_time'] = str(lap_time_file)
            
            lap_start_file = self.folder_path / f"{prefix}_lap_start.csv"
            if lap_start_file.exists():
                self.sessions[race_num]['lap_start'] = str(lap_start_file)
            
            lap_end_file = self.folder_path / f"{prefix}_lap_end.csv"
            if lap_end_file.exists():
                self.sessions[race_num]['lap_end'] = str(lap_end_file)
        
        # Find analysis files
        analysis_files = list(self.folder_path.glob("*AnalysisEnduranceWithSections*.CSV"))
        for analysis_file in analysis_files:
            if 'Race 1' in analysis_file.name or 'Race_1' in analysis_file.name:
                race_num = 1
            elif 'Race 2' in analysis_file.name or 'Race_2' in analysis_file.name:
                race_num = 2
            else:
                continue
            
            if race_num not in self.sessions:
                self.sessions[race_num] = {}
            
            self.sessions[race_num]['analysis'] = str(analysis_file)
        
        print(f"Found {len(self.sessions)} race session(s)")
        for race_num, files in self.sessions.items():
            print(f"  Race {race_num}: {list(files.keys())}")
        
        return self.sessions
    
    def get_session_files(self, race_number: int) -> Optional[Dict[str, str]]:
        """Get file paths for a specific race session."""
        return self.sessions.get(race_number)
