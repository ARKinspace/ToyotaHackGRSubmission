"""
Track Processor Module
Processes OSM data and generates track splines with sector information
"""

import numpy as np
import math
from scipy.interpolate import splprep, splev


class TrackProcessor:
    """Processes track data and generates splines with Local Flat-Earth Cartesian Projection"""
    
    def __init__(self):
        self.R = 6378137  # Earth radius in meters (exact specification)
    
    def project_to_local_meters(self, lat, lon, center_lat, center_lon):
        """Project lat/lon to Local Flat-Earth Cartesian system
        Origin: Center of bounding box = (0, 0)
        Unit: 1.0 = exactly 1.0 meter
        Uses spherical trigonometry for precision
        """
        dLat = (lat - center_lat) * math.pi / 180
        dLon = (lon - center_lon) * math.pi / 180
        latRad = center_lat * math.pi / 180
        
        # Precise projection: 1 unit = 1 meter
        x = self.R * dLon * math.cos(latRad)
        y = self.R * dLat
        return {'x': x, 'y': y}
    
    def get_distance_meters(self, p1, p2):
        """Calculate distance between two points in meters"""
        return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)
    
    def process_osm_data(self, osm_data, name, pit_anchor=None):
        """Process OSM data into track geometry"""
        if not osm_data or 'elements' not in osm_data:
            return None
        
        nodes = {}
        ways = []
        seen_ways = set()
        
        min_lat = float('inf')
        max_lat = float('-inf')
        min_lon = float('inf')
        max_lon = float('-inf')
        
        # Parse nodes and ways
        for el in osm_data['elements']:
            if el['type'] == 'node':
                nodes[el['id']] = {'lat': el['lat'], 'lon': el['lon']}
                if el['lat'] < min_lat:
                    min_lat = el['lat']
                if el['lat'] > max_lat:
                    max_lat = el['lat']
                if el['lon'] < min_lon:
                    min_lon = el['lon']
                if el['lon'] > max_lon:
                    max_lon = el['lon']
            elif el['type'] == 'way':
                if el['id'] not in seen_ways:
                    ways.append(el)
                    seen_ways.add(el['id'])
        
        if not ways:
            return None
        
        # Center point for projection
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Find pit lane if anchor provided
        pit_lane_id = None
        if pit_anchor:
            min_dist = float('inf')
            for way in ways:
                if way['nodes']:
                    first_node = nodes.get(way['nodes'][0])
                    if first_node:
                        d_lat = (pit_anchor['lat'] - first_node['lat']) * 111000
                        d_lon = (pit_anchor['lon'] - first_node['lon']) * 111000
                        d = math.sqrt(d_lat**2 + d_lon**2)
                        if d < min_dist:
                            min_dist = d
                            pit_lane_id = way['id']
        
        # Filter and process ways
        all_paths = []
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')
        
        for way in ways:
            # Skip if not enough nodes
            if 'nodes' not in way or len(way['nodes']) < 2:
                continue
            
            # Get node coordinates
            points = []
            for node_id in way['nodes']:
                if node_id in nodes:
                    points.append(nodes[node_id])
            
            if len(points) < 2:
                continue
            
            # Check if should filter out
            tags = way.get('tags', {})
            if pit_lane_id and way['id'] != pit_lane_id:
                if tags.get('highway') == 'service':
                    continue
            
            if tags.get('barrier') or tags.get('wall') or tags.get('building'):
                continue
            
            # Project to meters
            projected_points = []
            for p in points:
                m = self.project_to_local_meters(p['lat'], p['lon'], center_lat, center_lon)
                projected_points.append(m)
                
                if m['x'] > max_x:
                    max_x = m['x']
                if m['x'] < min_x:
                    min_x = m['x']
                if m['y'] > max_y:
                    max_y = m['y']
                if m['y'] < min_y:
                    min_y = m['y']
            
            # Determine width and type
            is_pit = False
            width_value = 12
            
            if way['id'] == pit_lane_id:
                is_pit = True
                width_value = 5
            elif tags.get('service') == 'pit_lane' or 'pit' in tags.get('name', '').lower():
                is_pit = True
                width_value = 5
            
            if 'width' in tags:
                try:
                    width_str = tags['width']
                    import re
                    match = re.match(r'([0-9.]+)', width_str)
                    if match:
                        width_value = float(match.group(1))
                except:
                    pass
            
            width_label = f"{width_value}m (est)" if 'width' not in tags else tags['width']
            
            all_paths.append({
                'id': way['id'],
                'points': projected_points,
                'rawPoints': points,
                'type': 'pit' if is_pit else 'track',
                'widthValue': width_value,
                'widthLabel': width_label
            })
        
        w = max_x - min_x
        h = max_y - min_y
        
        return {
            'name': name,
            'paths': all_paths,
            'center': {'lat': center_lat, 'lon': center_lon},
            'boundsMeters': {
                'minX': min_x,
                'maxX': max_x,
                'minY': min_y,
                'maxY': max_y,
                'width': w,
                'height': h
            }
        }
    
    def finalize_track(self, track_data, sf_lat, sf_lon, sector1_inches, sector2_inches, sector3_inches, circuit_miles, fetcher=None):
        """Generate high-res spline with sector analysis and elevation data
        
        Args:
            fetcher: TrackFetcher instance for elevation data (optional)
        """
        if not track_data or not track_data['paths']:
            return None
        
        center_lat = track_data['center']['lat']
        center_lon = track_data['center']['lon']
        
        # Anchor point
        if sf_lat is None or sf_lon is None:
            if track_data['paths'] and track_data['paths'][0]['rawPoints']:
                p = track_data['paths'][0]['rawPoints'][0]
                sf_lat = p['lat']
                sf_lon = p['lon']
            else:
                return None
        
        sf_meters = self.project_to_local_meters(sf_lat, sf_lon, center_lat, center_lon)
        
        # Get track segments only
        track_segs = [p for p in track_data['paths'] if p['type'] == 'track']
        pit_segs = [p for p in track_data['paths'] if p['type'] == 'pit']
        
        if not track_segs:
            return None

        
        # Find starting segment closest to SF
        best_start_idx = -1
        min_start_dist = float('inf')
        start_reverse = False
        
        for i, seg in enumerate(track_segs):
            head = seg['points'][0]
            tail = seg['points'][-1]
            
            d_head = self.get_distance_meters(head, sf_meters)
            d_tail = self.get_distance_meters(tail, sf_meters)
            
            if d_head < min_start_dist:
                min_start_dist = d_head
                best_start_idx = i
                start_reverse = False
            
            if d_tail < min_start_dist:
                min_start_dist = d_tail
                best_start_idx = i
                start_reverse = True
        
        if best_start_idx == -1:
            return None
        
        # Order segments
        ordered_segments = []
        used_indices = set()
        current_idx = best_start_idx
        reverse = start_reverse
        
        for _ in range(len(track_segs)):
            used_indices.add(current_idx)
            seg = track_segs[current_idx]
            pts = list(reversed(seg['points'])) if reverse else seg['points']
            raw_pts = list(reversed(seg['rawPoints'])) if reverse else seg['rawPoints']
            ordered_segments.append({
                'points': pts,
                'rawPoints': raw_pts,
                'width': seg['widthValue'],
                'id': seg['id']
            })
            
            if len(pts) == 0:
                break
            
            tail = pts[-1]
            next_idx = -1
            next_reverse = False
            min_gap = float('inf')
            
            for ni, n_seg in enumerate(track_segs):
                if ni in used_indices:
                    continue
                
                n_head = n_seg['points'][0]
                n_tail = n_seg['points'][-1]
                
                d_head = self.get_distance_meters(tail, n_head)
                d_tail = self.get_distance_meters(tail, n_tail)
                
                if d_head < min_gap:
                    min_gap = d_head
                    next_idx = ni
                    next_reverse = False
                
                if d_tail < min_gap:
                    min_gap = d_tail
                    next_idx = ni
                    next_reverse = True
            
            if next_idx != -1 and min_gap < 50:
                current_idx = next_idx
                reverse = next_reverse
            else:
                break
        
        # Resample to high-res spline (without elevation first)
        spline_points = []
        total_distance = 0
        
        for seg in ordered_segments:
            for i, p in enumerate(seg['points']):
                raw_p = seg['rawPoints'][i]
                
                if spline_points:
                    last_p = spline_points[-1]
                    d = self.get_distance_meters(last_p, p)
                    if d > 1.0:
                        steps = int(d)
                        for s in range(1, steps + 1):
                            t = s / d
                            # Interpolate lat/lon for elevation fetching
                            interp_lat = raw_p['lat'] if i == 0 else seg['rawPoints'][i-1]['lat'] + (raw_p['lat'] - seg['rawPoints'][i-1]['lat']) * t
                            interp_lon = raw_p['lon'] if i == 0 else seg['rawPoints'][i-1]['lon'] + (raw_p['lon'] - seg['rawPoints'][i-1]['lon']) * t
                            spline_points.append({
                                'x': last_p['x'] + (p['x'] - last_p['x']) * t,
                                'y': last_p['y'] + (p['y'] - last_p['y']) * t,
                                'lat': interp_lat,
                                'lon': interp_lon,
                                'dist': total_distance + s,
                                'width': seg['width']
                            })
                    total_distance += d
                
                spline_points.append({
                    'x': p['x'],
                    'y': p['y'],
                    'lat': raw_p['lat'],
                    'lon': raw_p['lon'],
                    'dist': total_distance,
                    'width': seg['width']
                })
        
        # Fetch elevation from OSM segment nodes BEFORE spline generation
        node_elevations = {}
        if fetcher and ordered_segments:
            print(f"Fetching elevation from OSM segment nodes...")
            
            # Collect all unique nodes from segments
            unique_nodes = {}
            for seg in ordered_segments:
                for node in seg['rawPoints']:
                    node_key = (round(node['lat'], 6), round(node['lon'], 6))
                    if node_key not in unique_nodes:
                        unique_nodes[node_key] = {'lat': node['lat'], 'lon': node['lon']}
            
            print(f"  Found {len(unique_nodes)} unique OSM nodes")
            
            # Fetch elevation for nodes in batches
            batch_size = 200
            node_list = list(unique_nodes.values())
            
            for i in range(0, len(node_list), batch_size):
                batch = node_list[i:i+batch_size]
                print(f"  Batch {i//batch_size + 1}/{(len(node_list)-1)//batch_size + 1}...")
                batch_elevations = fetcher.fetch_elevation(batch)
                node_elevations.update(batch_elevations)
            
            print(f"Total received: {len(node_elevations)} node elevations")
            
            # Apply elevations to raw points
            for seg in ordered_segments:
                for node in seg['rawPoints']:
                    node_key = (round(node['lat'], 6), round(node['lon'], 6))
                    node['z'] = node_elevations.get(node_key, 0)
        
        # Resample to high-res spline WITH elevation interpolation
        spline_points = []
        total_distance = 0
        
        for seg in ordered_segments:
            for i, p in enumerate(seg['points']):
                raw_p = seg['rawPoints'][i]
                
                if spline_points:
                    last_p = spline_points[-1]
                    last_raw = seg['rawPoints'][i-1] if i > 0 else raw_p
                    
                    d = self.get_distance_meters(last_p, p)
                    if d > 1.0:
                        steps = int(d)
                        for s in range(1, steps + 1):
                            t = s / d
                            # Interpolate lat/lon AND elevation
                            interp_lat = last_raw['lat'] + (raw_p['lat'] - last_raw['lat']) * t
                            interp_lon = last_raw['lon'] + (raw_p['lon'] - last_raw['lon']) * t
                            interp_z = last_raw.get('z', 0) + (raw_p.get('z', 0) - last_raw.get('z', 0)) * t
                            
                            spline_points.append({
                                'x': last_p['x'] + (p['x'] - last_p['x']) * t,
                                'y': last_p['y'] + (p['y'] - last_p['y']) * t,
                                'z': interp_z,
                                'lat': interp_lat,
                                'lon': interp_lon,
                                'dist': total_distance + s,
                                'width': seg['width']
                            })
                    total_distance += d
                
                spline_points.append({
                    'x': p['x'],
                    'y': p['y'],
                    'z': raw_p.get('z', 0),
                    'lat': raw_p['lat'],
                    'lon': raw_p['lon'],
                    'dist': total_distance,
                    'width': seg['width']
                })
        
        # Apply Gaussian smoothing to elevation for natural terrain
        if spline_points and len(spline_points) > 10:
            import numpy as np
            from scipy.ndimage import gaussian_filter1d
            
            z_array = np.array([pt['z'] for pt in spline_points])
            smoothed_z = gaussian_filter1d(z_array, sigma=5.0, mode='wrap')  # Wrap for circular track
            
            for i, pt in enumerate(spline_points):
                pt['z'] = smoothed_z[i]
            
            print(f"Applied Gaussian smoothing (sigma=5.0) to {len(spline_points)} spline points")
        
        # Fetch elevation for pit lane segments too
        if fetcher and pit_segs:
            print(f"Fetching elevation for {len(pit_segs)} pit lane segments...")
            for seg in pit_segs:
                # Add lat/lon to pit points if not already there
                if 'rawPoints' in seg:
                    pit_coords = [{'lat': rp['lat'], 'lon': rp['lon']} for rp in seg['rawPoints']]
                    print(f"  Pit segment {seg['id']}: {len(pit_coords)} points...")
                    pit_elevations = fetcher.fetch_elevation(pit_coords)
                    
                    # Apply elevations to pit points
                    for i, pt in enumerate(seg['points']):
                        if i < len(seg['rawPoints']):
                            rp = seg['rawPoints'][i]
                            key = (round(rp['lat'], 6), round(rp['lon'], 6))
                            pt['z'] = pit_elevations.get(key, 0)
        
        # Normalize elevations: set minimum to 0
        if spline_points:
            z_values = [pt.get('z', 0) for pt in spline_points]
            
            # Include pit lane elevations in min calculation
            for seg in pit_segs:
                for pt in seg.get('points', []):
                    if 'z' in pt:
                        z_values.append(pt['z'])
            
            if z_values:
                min_z = min(z_values)
                max_z = max(z_values)
                
                print(f"Raw elevation range: {min_z:.2f}m to {max_z:.2f}m")
                
                # Normalize all elevations
                for pt in spline_points:
                    pt['z'] = pt.get('z', 0) - min_z
                
                for seg in pit_segs:
                    for pt in seg.get('points', []):
                        if 'z' in pt:
                            pt['z'] -= min_z
                
                # Recalculate range
                z_values_norm = [pt.get('z', 0) for pt in spline_points]
                for seg in pit_segs:
                    for pt in seg.get('points', []):
                        if 'z' in pt:
                            z_values_norm.append(pt['z'])
                
                min_z_norm = min(z_values_norm)
                max_z_norm = max(z_values_norm)
                print(f"Normalized elevation range: {min_z_norm:.2f}m to {max_z_norm:.2f}m (delta: {max_z_norm - min_z_norm:.2f}m)")


        
        # Scaling
        scaling_factor = 1.0
        if circuit_miles and circuit_miles > 0:
            target_meters = circuit_miles * 1609.34
            scaling_factor = target_meters / total_distance
            
            for p in spline_points:
                p['x'] *= scaling_factor
                p['y'] *= scaling_factor
                p['dist'] *= scaling_factor
                # Don't scale z (elevation)
            
            total_distance *= scaling_factor
        
        # Sector logic
        s1m = (sector1_inches if sector1_inches else 0) * 0.0254
        s2m = (sector2_inches if sector2_inches else 0) * 0.0254
        
        # Find SF index
        unscaled_sf = self.project_to_local_meters(sf_lat, sf_lon, center_lat, center_lon)
        scaled_sf = {'x': unscaled_sf['x'] * scaling_factor, 'y': unscaled_sf['y'] * scaling_factor}
        
        sf_index = 0
        min_sf_dist = float('inf')
        for i in range(min(2000, len(spline_points))):
            d = self.get_distance_meters(spline_points[i], scaled_sf)
            if d < min_sf_dist:
                min_sf_dist = d
                sf_index = i
        
        # Rotate to start at SF
        if sf_index > 0:
            before = spline_points[:sf_index]
            after = spline_points[sf_index:]
            spline_points = after + before
            
            total_distance = 0
            spline_points[0]['dist'] = 0
            for i in range(1, len(spline_points)):
                total_distance += self.get_distance_meters(spline_points[i-1], spline_points[i])
                spline_points[i]['dist'] = total_distance
        
        # Find sector indices
        s1_idx = -1
        s2_idx = -1
        
        if s1m > 0:
            for i, p in enumerate(spline_points):
                if p['dist'] >= s1m:
                    s1_idx = i
                    break
        
        if s2m > 0:
            for i, p in enumerate(spline_points):
                if p['dist'] >= (s1m + s2m):
                    s2_idx = i
                    break
        
        # Create visual paths
        def create_path_d(pts):
            if not pts:
                return ""
            path_parts = []
            for i, p in enumerate(pts):
                cmd = 'M' if i == 0 else 'L'
                path_parts.append(f"{cmd} {p['x']:.2f} {p['y']:.2f}")
            return ' '.join(path_parts)
        
        visual_paths = []
        
        if s1_idx > 0 and s2_idx > 0:
            p1 = spline_points[0:s1_idx + 1]
            if p1:
                visual_paths.append({
                    'd': create_path_d(p1),
                    'color': '#3b82f6',
                    'width': p1[0]['width'],
                    'id': 's1'
                })
            
            p2 = spline_points[s1_idx:s2_idx + 1]
            if p2:
                visual_paths.append({
                    'd': create_path_d(p2),
                    'color': '#eab308',
                    'width': p2[0]['width'],
                    'id': 's2'
                })
            
            p3 = spline_points[s2_idx:]
            if p3:
                visual_paths.append({
                    'd': create_path_d(p3),
                    'color': '#ef4444',
                    'width': p3[0]['width'],
                    'id': 's3'
                })
        else:
            visual_paths.append({
                'd': create_path_d(spline_points),
                'color': '#10b981',
                'width': 12,
                'id': 'track_full'
            })
        
        # Add pit segments with elevation
        for seg in pit_segs:
            scaled_pit_points = [{'x': p['x'] * scaling_factor, 'y': p['y'] * scaling_factor, 'z': p.get('z', 0)} for p in seg['points']]
            visual_paths.append({
                'd': create_path_d(scaled_pit_points),
                'color': '#f97316',
                'width': seg['widthValue'],
                'id': seg['id'],
                'points': scaled_pit_points  # Include full points with elevation
            })

        
        # Enhanced telemetry with node samples
        node_sample = []
        if spline_points:
            # First 3 nodes
            node_sample.extend([{'x': p['x'], 'y': p['y'], 'dist': p['dist'], 'index': i} 
                               for i, p in enumerate(spline_points[:3])])
            # Middle 3 nodes
            mid_idx = len(spline_points) // 2
            node_sample.extend([{'x': p['x'], 'y': p['y'], 'dist': p['dist'], 'index': mid_idx + i} 
                               for i, p in enumerate(spline_points[mid_idx:mid_idx+3])])
            # Last 3 nodes
            node_sample.extend([{'x': p['x'], 'y': p['y'], 'dist': p['dist'], 'index': len(spline_points) - 3 + i} 
                               for i, p in enumerate(spline_points[-3:])])
        
        return {
            'visualPaths': visual_paths,
            'sfMarker': spline_points[0] if spline_points else None,
            'totalLen': f"{total_distance:.3f}",
            'sectors': s1_idx > 0,
            'splinePoints': spline_points,  # Full spline for advanced usage
            'splineData': {
                'meta': {
                    'totalLengthMeters': f"{total_distance:.4f}",
                    'scalingFactorApplied': f"{scaling_factor:.6f}",
                    'nodesCount': len(spline_points)
                },
                'sectors': {
                    'sfIndex': 0,
                    's1EndIndex': s1_idx,
                    's1LengthMeters': f"{s1m:.4f}",
                    's2EndIndex': s2_idx,
                    's2LengthMeters': f"{s2m:.4f}",
                    's3EndIndex': len(spline_points) - 1 if spline_points else -1
                },
                'inputs': {
                    'sector1_inches': sector1_inches if sector1_inches else 0,
                    'sector2_inches': sector2_inches if sector2_inches else 0,
                    'sector3_inches': sector3_inches if sector3_inches else 0,
                    'target_miles': circuit_miles if circuit_miles else 0
                },
                'nodeSample': node_sample
            },
            'center': {'lat': center_lat, 'lon': center_lon}
        }
