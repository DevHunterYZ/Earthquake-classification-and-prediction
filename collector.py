"""
Earthquake Data Collector
Supports USGS (global) and AFAD (Turkey-specific) APIs
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import json


class USGSCollector:
    """USGS Earthquake API client"""
    
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def fetch_earthquakes(
        self,
        start_time: datetime,
        end_time: datetime,
        min_magnitude: float = 3.0,
        min_latitude: float = None,
        max_latitude: float = None,
        min_longitude: float = None,
        max_longitude: float = None,
        limit: int = 20000
    ) -> pd.DataFrame:
        """Fetch earthquake catalog from USGS"""
        
        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%d"),
            "endtime": end_time.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "limit": limit,
            "orderby": "time-asc"
        }
        
        # Add optional spatial filters
        if min_latitude: params["minlatitude"] = min_latitude
        if max_latitude: params["maxlatitude"] = max_latitude
        if min_longitude: params["minlongitude"] = min_longitude
        if max_longitude: params["maxlongitude"] = max_longitude
        
        response = requests.get(self.BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse features
        records = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [None, None, None])
            
            records.append({
                "id": feature.get("id"),
                "time": pd.to_datetime(props.get("time"), unit='ms'),
                "latitude": coords[1],
                "longitude": coords[0],
                "depth": coords[2],
                "magnitude": props.get("mag"),
                "mag_type": props.get("magType"),
                "place": props.get("place"),
                "type": props.get("type"),
                "url": props.get("url"),
                "status": props.get("status"),
                "rms": props.get("rms"),
                "gap": props.get("gap"),
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.dropna(subset=['magnitude', 'time'])
            df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def fetch_aftershock_sequence(
        self,
        mainshock_lat: float,
        mainshock_lon: float,
        mainshock_time: datetime,
        mainshock_mag: float,
        radius_km: float = 100,
        days: int = 30
    ) -> pd.DataFrame:
        """Fetch aftershocks following a mainshock"""
        
        # Convert km to approximate degrees
        km_per_deg = 111.0
        lat_offset = radius_km / km_per_deg
        lon_offset = radius_km / (km_per_deg * abs(np.cos(np.radians(mainshock_lat))))
        
        start_time = mainshock_time
        end_time = mainshock_time + timedelta(days=days)
        
        df = self.fetch_earthquakes(
            start_time=start_time,
            end_time=end_time,
            min_magnitude=max(2.0, mainshock_mag - 3),  # Mc = Mm - 3
            min_latitude=mainshock_lat - lat_offset,
            max_latitude=mainshock_lat + lat_offset,
            min_longitude=mainshock_lon - lon_offset,
            max_longitude=mainshock_lon + lon_offset
        )
        
        # Calculate time since mainshock (days)
        df['days_since_mainshock'] = (df['time'] - mainshock_time).dt.total_seconds() / 86400
        
        # Calculate distance from mainshock
        df['distance_km'] = self._haversine_distance(
            mainshock_lat, mainshock_lon,
            df['latitude'], df['longitude']
        )
        
        return df
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """Calculate great circle distance in km"""
        import numpy as np
        
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


class AFADCollector:
    """AFAD (Turkey) Earthquake API client - using Kandilli/AFAD data"""
    
    # AFAD uses BOGAZICI University Kandilli Observatory data
    BASE_URL = "http://www.koeri.boun.edu.tr/scripts/lst7.asp"
    
    def fetch_recent_earthquakes(self, days: int = 7) -> pd.DataFrame:
        """Fetch recent Turkey earthquakes"""
        # AFAD doesn't have a modern REST API
        # This is a placeholder for web scraping or alternative data source
        
        # For now, use USGS with Turkey region filter
        usgs = USGSCollector()
        
        # Turkey bounding box
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        return usgs.fetch_earthquakes(
            start_time=start_time,
            end_time=end_time,
            min_magnitude=2.5,
            min_latitude=35.8,
            max_latitude=42.1,
            min_longitude=25.6,
            max_longitude=44.8
        )


def fetch_historical_mainshocks(
    min_magnitude: float = 6.0,
    years: int = 10,
    region: str = "global"
) -> pd.DataFrame:
    """Fetch historical mainshocks for training data"""
    
    collector = USGSCollector()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=years*365)
    
    # Region filters
    region_filters = {
        "global": {},
        "turkey": {
            "min_latitude": 35.8, "max_latitude": 42.1,
            "min_longitude": 25.6, "max_longitude": 44.8
        },
        "california": {
            "min_latitude": 32.0, "max_latitude": 42.0,
            "min_longitude": -125.0, "max_longitude": -114.0
        },
        "japan": {
            "min_latitude": 24.0, "max_latitude": 46.0,
            "min_longitude": 122.0, "max_longitude": 146.0
        }
    }
    
    filters = region_filters.get(region, {})
    
    df = collector.fetch_earthquakes(
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_magnitude,
        **filters
    )
    
    # Filter for mainshocks (not aftershocks of larger events)
    df = label_mainshocks_and_aftershocks(df)
    
    return df


def label_mainshocks_and_aftershocks(
    df: pd.DataFrame,
    space_window_km: float = 100,
    time_window_days: float = 3
) -> pd.DataFrame:
    """
    Label earthquakes as mainshock or aftershock
    Uses simple window-based declustering
    """
    import numpy as np
    
    df = df.copy()
    df['is_mainshock'] = True
    df['is_aftershock'] = False
    df['parent_id'] = None
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    for i, row in df.iterrows():
        if not df.loc[i, 'is_mainshock']:
            continue  # Already labeled as aftershock
            
        # Look for smaller preceding events within spatiotemporal window
        time_mask = (
            (df['time'] >= row['time'] - pd.Timedelta(days=time_window_days)) &
            (df['time'] < row['time'])
        )
        
        # For each potential parent, calculate distance
        candidates = df[time_mask & (df['magnitude'] >= row['magnitude'])]
        
        for j, candidate in candidates.iterrows():
            # Calculate distance
            dist = USGSCollector._haversine_distance(
                row['latitude'], row['longitude'],
                pd.Series([candidate['latitude']]),
                pd.Series([candidate['longitude']])
            ).iloc[0]
            
            if dist <= space_window_km:
                # This is an aftershock
                df.loc[i, 'is_mainshock'] = False
                df.loc[i, 'is_aftershock'] = True
                df.loc[i, 'parent_id'] = candidate['id']
                break
    
    return df


# Example usage
if __name__ == "__main__":
    collector = USGSCollector()
    
    # Fetch Turkey earthquakes from last 30 days
    df = collector.fetch_earthquakes(
        start_time=datetime.utcnow() - timedelta(days=30),
        end_time=datetime.utcnow(),
        min_magnitude=3.0,
        min_latitude=35.8,
        max_latitude=42.1,
        min_longitude=25.6,
        max_longitude=44.8
    )
    
    print(f"Fetched {len(df)} earthquakes")
    print(df.head())
