"""
Telemetry Parsing Module
Encapsulates logic to parse raw telemetry CSVs into per-vehicle wide-format CSVs.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class TelemetryParser:
    """
    Parses raw telemetry CSV files into per-vehicle DataFrames and saves them.
    """

    def parse_csv_to_vehicle_dfs(self, csv_path: str, output_dir: str | None = None, save_format: str = "csv", iso_time_z: bool = False, race_name: str = "Unknown Race") -> Dict[str, pd.DataFrame]:
        """Parse the CSV and return a dict of vehicle_id -> DataFrame.

        Each DataFrame has columns: `meta_time`, `elapsed_seconds` and columns for each unique telemetry_name.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path, low_memory=False)

        # Ensure necessary columns exist
        if "vehicle_id" not in df.columns:
            if "original_vehicle_id" in df.columns:
                df["vehicle_id"] = df["original_vehicle_id"]
            elif "vehicle_number" in df.columns:
                df["vehicle_id"] = df["vehicle_number"]
            else:
                raise ValueError(f"CSV file missing vehicle identifier column. Columns: {df.columns}")

        required_cols = {"meta_time", "telemetry_name", "telemetry_value", "vehicle_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV file missing required columns: {missing}")

        # parse meta_time as datetime
        df["meta_time"] = pd.to_datetime(df["meta_time"], utc=True, errors="coerce")
        if df["meta_time"].isna().any():
            df = df[~df["meta_time"].isna()].copy()

        # Try to convert telemetry_value to numeric where possible
        df["telemetry_value_numeric"] = pd.to_numeric(df["telemetry_value"], errors="coerce")
        df["telemetry_value"] = df["telemetry_value_numeric"].combine_first(df["telemetry_value"])
        df.drop(columns=["telemetry_value_numeric"], inplace=True)

        # Group by vehicle
        vehicles = {}
        unique_vehicles = df["vehicle_id"].dropna().unique()
        
        print(f"Parsing {len(unique_vehicles)} vehicles from {csv_path.name}...")

        for vid in unique_vehicles:
            subset = df[df["vehicle_id"] == vid].copy()
            subset.sort_values("meta_time", inplace=True)
            if subset.empty:
                continue

            # Pivot telemetry rows
            pivot = subset.pivot_table(index="meta_time", columns="telemetry_name", values="telemetry_value", aggfunc="last")
            pivot = pivot.reset_index()

            # Compute elapsed seconds
            pivot = pivot.sort_values("meta_time")
            pivot["elapsed_seconds"] = (pivot["meta_time"] - pivot["meta_time"].iloc[0]).dt.total_seconds()

            # bring lap column if present
            if "lap" in subset.columns:
                lap_per_time = subset[["meta_time", "lap"]].drop_duplicates().groupby("meta_time").last()
                lap_per_time = lap_per_time.reset_index()
                pivot = pivot.merge(lap_per_time, on="meta_time", how="left")

            # Sort columns
            cols = list(pivot.columns)
            telemetry_cols = [c for c in cols if c not in {"meta_time", "elapsed_seconds", "lap"}]
            ordered_cols = ["meta_time", "elapsed_seconds"] + (["lap"] if "lap" in pivot.columns else []) + sorted(telemetry_cols)
            pivot = pivot.reindex(columns=ordered_cols)

            if iso_time_z:
                pivot["meta_time"] = pivot["meta_time"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if pd.notna(x) else x)

            vehicles[vid] = pivot
            
            # Save to nested structure: Output/Race X/Vehicle Y/telemetry.csv
            if output_dir:
                safe_vid = str(vid).replace("/", "_")
                
                # Create race directory
                race_dir = Path(output_dir) / race_name
                race_dir.mkdir(parents=True, exist_ok=True)
                
                # Create vehicle directory
                vehicle_dir = race_dir / safe_vid
                vehicle_dir.mkdir(parents=True, exist_ok=True)
                
                if save_format == "csv":
                    out_path = vehicle_dir / "telemetry.csv"
                    pivot.to_csv(out_path, index=False)
                elif save_format == "parquet":
                    out_path = vehicle_dir / "telemetry.parquet"
                    pivot.to_parquet(out_path, index=False)

        return vehicles

    def parse_folder(self, input_folder: str, output_folder: str) -> int:
        """
        Scans input_folder for telemetry files and parses them to output_folder.
        Returns number of vehicles processed.
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        telemetry_files = list(input_path.glob("*telemetry_data.csv"))
        if not telemetry_files:
            print("No *telemetry_data.csv files found.")
            return 0
            
        total_vehicles = 0
        for telem_file in telemetry_files:
            print(f"Processing {telem_file.name}...")
            
            # Detect race name
            race_name = "Unknown Race"
            if "R1_" in telem_file.name or "_R1_" in telem_file.name:
                race_name = "Race 1"
            elif "R2_" in telem_file.name or "_R2_" in telem_file.name:
                race_name = "Race 2"
            
            try:
                vehicles = self.parse_csv_to_vehicle_dfs(str(telem_file), output_dir=output_folder, race_name=race_name)
                total_vehicles += len(vehicles)
            except Exception as e:
                print(f"Error processing {telem_file.name}: {e}")
                
        return total_vehicles
