"""
Track Data Fetcher Module
Fetches race track data from OpenStreetMap Overpass API
"""

import requests
import time
import math


class TrackFetcher:
    """Fetches track data from OpenStreetMap"""
    
    def __init__(self):
        self.overpass_urls = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
        ]
    
    def get_distance_from_lat_lon_km(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two lat/lon points in kilometers"""
        R = 6371  # Earth radius in km
        dLat = self._deg2rad(lat2 - lat1)
        dLon = self._deg2rad(lon2 - lon1)
        a = (math.sin(dLat/2) * math.sin(dLat/2) +
             math.cos(self._deg2rad(lat1)) * math.cos(self._deg2rad(lat2)) *
             math.sin(dLon/2) * math.sin(dLon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _deg2rad(self, deg):
        """Convert degrees to radians"""
        return deg * (math.pi / 180)
    
    def fetch_by_coords(self, lat, lon, max_retries=3):
        """Fetch track data by coordinates"""
        query = f"""
        [out:json][timeout:90];
        (
          way(around:2000,{lat},{lon})["highway"="raceway"];
          way(around:2000,{lat},{lon})["leisure"="track"];
          way(around:2000,{lat},{lon})["sport"="motor_racing"];
          way(around:1000,{lat},{lon})["highway"="service"]["service"~"driveway|alley|pit_lane"];
        )->.seeds;
        (.seeds;rel(bw.seeds)["type"="circuit"];);
        (._;>;);
        out body qt;
        """
        return self._execute_query(query, max_retries)
    
    def fetch_by_name(self, name, max_retries=3):
        """Fetch track data by name"""
        clean_query = name.strip().replace('"', '\\"')
        query = f"""
        [out:json][timeout:25];
        (
          nwr["name"~"{clean_query}",i]["sport"="motor_racing"];
          nwr["name"~"{clean_query}",i]["highway"="raceway"];
          relation["type"="circuit"]["name"~"{clean_query}",i];
        );
        (._;>;);
        out body qt;
        """
        return self._execute_query(query, max_retries)
    
    def _execute_query(self, query, max_retries):
        """Execute Overpass API query with retries"""
        for url in self.overpass_urls:
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url,
                        data=query,
                        timeout=90,
                        headers={'User-Agent': 'RaceTrackScanner/1.0'}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if data and 'elements' in data and len(data['elements']) > 0:
                        return data
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                except Exception as e:
                    print(f"Error: {e}")
                    break
        
        return None
    
    def fetch_elevation(self, locations):
        """Fetch elevation data for a list of lat/lon coordinates
        
        Args:
            locations: List of dicts with 'lat' and 'lon' keys
            
        Returns:
            Dict mapping (rounded lat, rounded lon) tuples to elevation in meters
        """
        if not locations:
            return {}
        
        # Open-Elevation API endpoint
        url = "https://api.open-elevation.com/api/v1/lookup"
        
        # Prepare locations in the required format
        location_data = {
            "locations": [{"latitude": loc['lat'], "longitude": loc['lon']} for loc in locations]
        }
        
        try:
            print(f"  Calling API with {len(locations)} points (timeout: 60s)...")
            response = requests.post(
                url,
                json=location_data,
                timeout=60,  # Increased timeout
                headers={'User-Agent': 'RaceTrackScanner/1.0'}
            )
            response.raise_for_status()
            data = response.json()
            
            # Build elevation map with rounded keys
            elevation_map = {}
            if 'results' in data:
                for i, result in enumerate(data['results']):
                    if i < len(locations):
                        # Round to 6 decimals for consistent key matching
                        key = (round(locations[i]['lat'], 6), round(locations[i]['lon'], 6))
                        elevation = result.get('elevation', 0)
                        elevation_map[key] = elevation
                
                # Debug: show sample elevations
                sample_vals = list(elevation_map.values())[:5]
                print(f"  API returned {len(elevation_map)} elevations. Sample: {sample_vals}")
            
            return elevation_map
            
        except Exception as e:
            print(f"  Elevation fetch failed: {e}. Using flat elevation (z=0).")
            # Return all zeros as fallback with rounded keys
            return {(round(loc['lat'], 6), round(loc['lon'], 6)): 0 for loc in locations}

