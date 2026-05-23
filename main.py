"""
Earthquake Prediction API
FastAPI service for real-time aftershock forecasting
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import json

# Import our modules
from data.collector import USGSCollector, fetch_historical_mainshocks, label_mainshocks_and_aftershocks
from data.features import extract_all_features, OmoriUtsuFeatures, GutenbergRichterFeatures
from models.neural_hawkes import AftershockPredictor, SequenceEvent

app = FastAPI(
    title="Earthquake Aftershock Prediction API",
    description="Neural Hawkes-based aftershock forecasting system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_model_cache = {}


class EarthquakeEvent(BaseModel):
    time: datetime
    magnitude: float = Field(..., ge=0, le=10)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    depth: Optional[float] = Field(default=10.0, ge=0, le=700)


class PredictionRequest(BaseModel):
    mainshock: EarthquakeEvent
    aftershocks_so_far: List[EarthquakeEvent] = []
    horizon_days: int = Field(default=30, ge=1, le=365)
    model_type: str = Field(default="statistical", pattern="^(statistical|neural|ensemble)$")


class PredictionResponse(BaseModel):
    mainshock_id: str
    predicted_aftershocks_7d: float
    predicted_aftershocks_30d: float
    predicted_aftershocks_90d: float
    prob_mag5_plus: float
    prob_mag6_plus: float
    current_omori_params: Dict
    current_gr_params: Dict
    intensity_forecast: List[Dict]
    magnitude_bins: Dict = {}
    time_windows: Dict = {}
    largest_aftershock: Dict = {}
    risk_score: int = 0
    risk_level: str = "MEDIUM"
    model_type: str
    generated_at: datetime


# Frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve dashboard"""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html><body>
    <h1>Earthquake Aftershock Prediction API</h1>
    <p>Dashboard not found. API endpoints available at /predict, /fetch-and-predict</p>
    </body></html>
    """

@app.get("/api-info")
def api_info():
    return {
        "service": "Earthquake Aftershock Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/fetch-and-predict",
            "/historical-analysis",
            "/health"
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "torch_available": torch.cuda.is_available()}


@app.post("/predict", response_model=PredictionResponse)
def predict_aftershocks(request: PredictionRequest):
    """
    Predict aftershocks based on mainshock and observed aftershocks
    """
    try:
        # Convert to pandas
        mainshock_data = {
            'time': pd.Timestamp(request.mainshock.time),
            'magnitude': request.mainshock.magnitude,
            'latitude': request.mainshock.latitude,
            'longitude': request.mainshock.longitude,
            'depth': request.mainshock.depth
        }
        
        aftershock_data = []
        for aft in request.aftershocks_so_far:
            aftershock_data.append({
                'time': pd.Timestamp(aft.time),
                'magnitude': aft.magnitude,
                'latitude': aft.latitude,
                'longitude': aft.longitude,
                'depth': aft.depth or 10.0
            })
        
        aftershock_df = pd.DataFrame(aftershock_data)
        if aftershock_df.empty:
            aftershock_df = pd.DataFrame(columns=['time', 'magnitude', 'latitude', 'longitude', 'depth'])
        
        # Extract features
        features = extract_all_features(mainshock_data, aftershock_df)
        
        # Generate time-series forecast
        intensity_forecast = []
        
        if features['omori'].get('fitted'):
            omori = OmoriUtsuFeatures(
                mainshock_data['time'],
                mainshock_data['magnitude']
            )
            omori.params = features['omori']
            
            # Forecast for next 30 days (daily)
            for day in range(1, min(request.horizon_days + 1, 31)):
                rate = omori.predict_rate(day)
                cumulative = omori.predict_cumulative(day)
                intensity_forecast.append({
                    'day': day,
                    'daily_rate': float(rate),
                    'cumulative_count': float(cumulative)
                })
        
        # Build response
        response = PredictionResponse(
            mainshock_id=f"{request.mainshock.time.isoformat()}_{request.mainshock.magnitude}",
            predicted_aftershocks_7d=features.get('predicted_aftershocks_7d', 0),
            predicted_aftershocks_30d=features.get('predicted_aftershocks_30d', 0),
            predicted_aftershocks_90d=features.get('predicted_aftershocks_90d', 0),
            prob_mag5_plus=features.get('prob_mag5_aftershock', 0),
            prob_mag6_plus=features.get('prob_mag5_aftershock', 0) * 0.1,
            current_omori_params=features['omori'],
            current_gr_params=features['gutenberg_richter'],
            intensity_forecast=intensity_forecast,
            magnitude_bins=features.get('magnitude_bins', {}),
            time_windows=features.get('time_windows', {}),
            largest_aftershock=features.get('largest_aftershock', {}),
            risk_score=features.get('risk_score', 0),
            risk_level=features.get('risk_level', 'MEDIUM'),
            model_type=request.model_type,
            generated_at=datetime.utcnow()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch-and-predict")
def fetch_and_predict(
    lat: float,
    lon: float,
    radius_km: float = 100,
    days: int = 30,
    min_magnitude: float = 2.5
):
    """
    Automatically fetch recent earthquakes and predict aftershocks
    """
    try:
        collector = USGSCollector()
        
        # Fetch earthquakes
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Bounding box
        km_per_deg = 111.0
        lat_offset = radius_km / km_per_deg
        lon_offset = radius_km / (km_per_deg * abs(np.cos(np.radians(lat))))
        
        catalog = collector.fetch_earthquakes(
            start_time=start_time,
            end_time=end_time,
            min_magnitude=min_magnitude,
            min_latitude=lat - lat_offset,
            max_latitude=lat + lat_offset,
            min_longitude=lon - lon_offset,
            max_longitude=lon + lon_offset
        )
        
        if len(catalog) == 0:
            return {"error": "No earthquakes found in specified region and time window"}
        
        # Identify mainshock (largest magnitude in recent window)
        recent_time = end_time - timedelta(hours=48)
        recent_catalog = catalog[catalog['time'] >= recent_time]
        
        if len(recent_catalog) > 0:
            mainshock_idx = recent_catalog['magnitude'].idxmax()
            mainshock = catalog.loc[mainshock_idx]
        else:
            mainshock = catalog.loc[catalog['magnitude'].idxmax()]
        
        # Get aftershocks (within 100km and after mainshock)
        catalog['days_since_ms'] = (catalog['time'] - mainshock['time']).dt.total_seconds() / 86400
        catalog['dist_km'] = collector._haversine_distance(
            mainshock['latitude'], mainshock['longitude'],
            catalog['latitude'], catalog['longitude']
        )
        
        aftershocks = catalog[
            (catalog['days_since_ms'] > 0) &
            (catalog['days_since_ms'] <= 30) &
            (catalog['dist_km'] <= 100)
        ]
        
        # Build prediction request
        mainshock_event = EarthquakeEvent(
            time=mainshock['time'].to_pydatetime(),
            magnitude=float(mainshock['magnitude']),
            latitude=float(mainshock['latitude']),
            longitude=float(mainshock['longitude']),
            depth=float(mainshock.get('depth', 10.0)) if pd.notna(mainshock.get('depth')) else 10.0
        )
        
        aftershock_events = []
        for _, row in aftershocks.iterrows():
            aftershock_events.append(EarthquakeEvent(
                time=row['time'].to_pydatetime(),
                magnitude=float(row['magnitude']),
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                depth=float(row.get('depth', 10.0)) if pd.notna(row.get('depth')) else 10.0
            ))
        
        request = PredictionRequest(
            mainshock=mainshock_event,
            aftershocks_so_far=aftershock_events,
            horizon_days=30,
            model_type="statistical"
        )
        
        # Get prediction
        prediction = predict_aftershocks(request)
        
        return {
            "auto_fetched": True,
            "mainshock": {
                "time": mainshock['time'].isoformat(),
                "magnitude": float(mainshock['magnitude']),
                "latitude": float(mainshock['latitude']),
                "longitude": float(mainshock['longitude']),
                "location": mainshock.get('place', 'Unknown')
            },
            "observed_aftershocks": len(aftershocks),
            "prediction": prediction.dict()
        }
        
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.get("/historical-analysis")
def historical_analysis(
    region: str = "turkey",
    min_magnitude: float = 6.0,
    years: int = 10
):
    """
    Analyze historical mainshock-aftershock sequences
    """
    try:
        mainshocks = fetch_historical_mainshocks(
            min_magnitude=min_magnitude,
            years=years,
            region=region
        )
        
        if len(mainshocks) == 0:
            return {"error": "No mainshocks found"}
        
        # Aggregate statistics
        stats = {
            "n_mainshocks": len(mainshocks),
            "avg_magnitude": mainshocks['magnitude'].mean(),
            "magnitude_range": [mainshocks['magnitude'].min(), mainshocks['magnitude'].max()],
            "time_span": {
                "start": mainshocks['time'].min().isoformat(),
                "end": mainshocks['time'].max().isoformat()
            },
            "by_year": mainshocks.groupby(mainshocks['time'].dt.year).size().to_dict()
        }
        
        return {
            "region": region,
            "statistics": stats,
            "mainshocks": mainshocks[['time', 'magnitude', 'latitude', 'longitude', 'place']].to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/active-regions")
def active_regions():
    """Get currently active seismic regions with recent activity"""
    try:
        collector = USGSCollector()
        
        # Turkey
        turkey = collector.fetch_earthquakes(
            datetime.utcnow() - timedelta(days=7),
            datetime.utcnow(),
            min_magnitude=3.0,
            min_latitude=35.8, max_latitude=42.1,
            min_longitude=25.6, max_longitude=44.8
        )
        
        # Japan
        japan = collector.fetch_earthquakes(
            datetime.utcnow() - timedelta(days=7),
            datetime.utcnow(),
            min_magnitude=3.0,
            min_latitude=24.0, max_latitude=46.0,
            min_longitude=122.0, max_longitude=146.0
        )
        
        # California
        california = collector.fetch_earthquakes(
            datetime.utcnow() - timedelta(days=7),
            datetime.utcnow(),
            min_magnitude=2.5,
            min_latitude=32.0, max_latitude=42.0,
            min_longitude=-125.0, max_longitude=-114.0
        )
        
        return {
            "turkey": {
                "n_events_7d": len(turkey),
                "max_magnitude": turkey['magnitude'].max() if len(turkey) > 0 else None,
                "latest": turkey.iloc[-1].to_dict() if len(turkey) > 0 else None
            },
            "japan": {
                "n_events_7d": len(japan),
                "max_magnitude": japan['magnitude'].max() if len(japan) > 0 else None,
                "latest": japan.iloc[-1].to_dict() if len(japan) > 0 else None
            },
            "california": {
                "n_events_7d": len(california),
                "max_magnitude": california['magnitude'].max() if len(california) > 0 else None,
                "latest": california.iloc[-1].to_dict() if len(california) > 0 else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recent-mainshocks")
def recent_mainshocks(min_magnitude: float = 5.0, days: int = 7):
    """
    Get recent significant earthquakes worldwide and in Turkey
    to use as mainshocks for prediction
    """
    try:
        collector = USGSCollector()
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Worldwide M>=min_magnitude
        world = collector.fetch_earthquakes(
            start_time, end_time, min_magnitude=min_magnitude
        )
        
        # Turkey region M>=4
        turkey = collector.fetch_earthquakes(
            start_time, end_time, min_magnitude=4.0,
            min_latitude=35.8, max_latitude=42.1,
            min_longitude=25.6, max_longitude=44.8
        )
        
        def format_events(df, max_n=20):
            events = []
            df = df.sort_values('magnitude', ascending=False).head(max_n)
            for _, row in df.iterrows():
                events.append({
                    'id': row.get('id', ''),
                    'time': row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                    'magnitude': float(row['magnitude']),
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'depth': float(row.get('depth', 10)) if pd.notna(row.get('depth')) else 10.0,
                    'place': row.get('place', 'Unknown'),
                    'url': row.get('url', ''),
                })
            return events
        
        return {
            'world': format_events(world),
            'turkey': format_events(turkey),
            'updated_at': datetime.utcnow().isoformat(),
            'parameters': {'min_magnitude': min_magnitude, 'days': days}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
