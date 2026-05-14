"""
Seismic Feature Engineering
Omori-Utsu law, Gutenberg-Richter relationship, ETAS features
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from typing import Dict, Tuple, Optional
import warnings


class OmoriUtsuFeatures:
    """
    Modified Omori Law: n(t) = K / (t + c)^p
    where n(t) is the aftershock rate at time t
    """
    
    def __init__(self, mainshock_time: pd.Timestamp, mainshock_mag: float):
        self.mainshock_time = mainshock_time
        self.mainshock_mag = mainshock_mag
        self.params = None
        
    def fit(self, aftershock_times: pd.Series, aftershock_mags: pd.Series) -> Dict:
        """
        Fit Modified Omori Law parameters (K, c, p)
        using maximum likelihood estimation
        """
        # Time since mainshock in days
        t = (aftershock_times - self.mainshock_time).dt.total_seconds() / 86400
        t = t[t > 0].values  # Only positive times
        
        if len(t) < 5:
            # Not enough data, use typical values
            self.params = {'K': 10.0, 'c': 0.05, 'p': 1.1, 'fitted': False}
            return self.params
        
        # Negative log-likelihood for Omori-Utsu
        def neg_log_likelihood(params):
            K, c, p = params
            if K <= 0 or c <= 0 or p <= 0:
                return 1e10
            
            # Omori rate
            rate = K / ((t + c) ** p)
            
            # Log-likelihood of observed times under Poisson process
            # LL = sum(log(lambda(t_i))) - integral(lambda(t))dt
            log_likelihood = np.sum(np.log(K) - p * np.log(t + c))
            
            # Integral from 0 to max(t) + 7 days
            t_max = np.max(t) + 7
            integral = (K / (1 - p)) * ((t_max + c)**(1-p) - c**(1-p)) if p != 1 else K * np.log((t_max + c) / c)
            
            return -(log_likelihood - integral)
        
        # Initial guess
        x0 = [10.0, 0.05, 1.1]
        bounds = [(0.01, 1000), (0.001, 1.0), (0.5, 2.5)]
        
        try:
            result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
            K, c, p = result.x
            self.params = {'K': K, 'c': c, 'p': p, 'fitted': True, 'log_likelihood': -result.fun}
        except:
            # Fallback to typical values
            self.params = {'K': 10.0, 'c': 0.05, 'p': 1.1, 'fitted': False}
        
        return self.params
    
    def predict_rate(self, times: np.ndarray) -> np.ndarray:
        """Predict aftershock rate at given times (days since mainshock)"""
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        K, c, p = self.params['K'], self.params['c'], self.params['p']
        return K / ((times + c) ** p)
    
    def predict_cumulative(self, t_end: float) -> float:
        """Predict cumulative number of aftershocks up to t_end days"""
        if self.params is None:
            return np.nan
        
        K, c, p = self.params['K'], self.params['c'], self.params['p']
        
        # Integral of Omori law
        if abs(p - 1.0) < 0.01:
            return K * np.log((t_end + c) / c)
        else:
            return (K / (1 - p)) * ((t_end + c)**(1-p) - c**(1-p))


class GutenbergRichterFeatures:
    """
    Gutenberg-Richter: log10(N) = a - b*M
    where N is number of earthquakes with magnitude >= M
    """
    
    def __init__(self, completeness_mag: float = None):
        self.completeness_mag = completeness_mag
        self.a = None
        self.b = None
        self.sigma_b = None
        
    def fit(self, magnitudes: pd.Series) -> Dict:
        """
        Fit Gutenberg-Richter parameters using maximum likelihood
        """
        mags = magnitudes.dropna().values
        
        # Estimate Mc (completeness magnitude) if not provided
        if self.completeness_mag is None:
            self.completeness_mag = self._estimate_mc(mags)
        
        # Filter above completeness
        mags_above_mc = mags[mags >= self.completeness_mag]
        
        if len(mags_above_mc) < 5:
            self.a = 3.0
            self.b = 1.0
            return {'a': self.a, 'b': self.b, 'Mc': self.completeness_mag, 'fitted': False}
        
        # Maximum likelihood estimate of b-value
        # b = 1 / (mean(M - Mc) * ln(10))
        mean_excess = np.mean(mags_above_mc - self.completeness_mag)
        self.b = 1.0 / (mean_excess * np.log(10))
        
        # Estimate sigma_b (uncertainty)
        n = len(mags_above_mc)
        self.sigma_b = self.b / np.sqrt(n)
        
        # Calculate a: log10(N_total) = a - b*Mc
        total_count = len(mags)
        self.a = np.log10(total_count) + self.b * self.completeness_mag
        
        return {
            'a': self.a,
            'b': self.b,
            'sigma_b': self.sigma_b,
            'Mc': self.completeness_mag,
            'n_above_mc': n,
            'fitted': True
        }
    
    def _estimate_mc(self, magnitudes: np.ndarray) -> float:
        """
        Estimate completeness magnitude using maximum curvature method
        """
        if len(magnitudes) < 10:
            return np.min(magnitudes) if len(magnitudes) > 0 else 2.0
        
        # Histogram of magnitudes
        hist, bin_edges = np.histogram(magnitudes, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find maximum curvature (change from convex to concave)
        # Simplified: find where histogram starts decreasing consistently
        max_idx = np.argmax(hist)
        mc = bin_centers[max_idx]
        
        return max(mc, np.percentile(magnitudes, 5))
    
    def predict_count_above_mag(self, mag_threshold: float) -> float:
        """Predict number of earthquakes with magnitude >= threshold"""
        if self.a is None or self.b is None:
            return np.nan
        return 10 ** (self.a - self.b * mag_threshold)
    
    def predict_prob_mag_above(self, mag: float, mag_threshold: float) -> float:
        """Probability that an earthquake is >= mag_threshold given it's >= mag"""
        if self.b is None:
            return 0.5
        # P(M >= mag_threshold | M >= mag) = 10^(-b*(mag_threshold - mag))
        if mag_threshold < mag:
            return 1.0
        return 10 ** (-self.b * (mag_threshold - mag))


class ETASFeatures:
    """
    Epidemic-Type Aftershock Sequence model features
    Combines background rate with triggered aftershocks
    """
    
    def __init__(self):
        self.mu = None  # Background rate
        self.K = None   # Aftershock productivity
        self.c = None   # Omori c parameter
        self.p = None   # Omori p parameter
        self.alpha = None  # Magnitude triggering factor
        self.D = None   # Spatial kernel parameter
        self.gamma = None  # Spatial fractal dimension
        
    def extract_features(self, earthquake_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ETAS-style features for each earthquake
        """
        df = earthquake_catalog.copy().sort_values('time').reset_index(drop=True)
        n = len(df)
        
        features = pd.DataFrame({
            'time': df['time'],
            'magnitude': df['magnitude'],
            'latitude': df['latitude'],
            'longitude': df['longitude'],
            'depth': df.get('depth', np.nan),
        })
        
        # Initialize feature columns
        features['background_rate'] = 0.0
        features['triggered_rate'] = 0.0
        features['total_rate'] = 0.0
        features['n_past_events'] = 0
        features['mean_dist_to_past'] = 0.0
        features['max_mag_in_window'] = 0.0
        features['time_since_last'] = np.inf
        features['avg_mag_recent'] = 0.0
        
        # Compute features for each event (using past events only)
        for i in range(n):
            current = df.iloc[i]
            
            # Past events within 30 days and 100km
            past_mask = (
                (df['time'] < current['time']) &
                (df['time'] >= current['time'] - pd.Timedelta(days=30))
            )
            past_events = df[past_mask]
            
            if len(past_events) > 0:
                # Temporal distance in days
                dt_days = (current['time'] - past_events['time']).dt.total_seconds() / 86400
                
                # Spatial distance (approximate)
                dlat = current['latitude'] - past_events['latitude']
                dlon = current['longitude'] - past_events['longitude']
                dist_km = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(current['latitude'])))**2) * 111
                
                # Space-time window
                close_mask = dist_km <= 100
                
                features.loc[i, 'n_past_events'] = len(past_events)
                features.loc[i, 'time_since_last'] = dt_days.min()
                
                if close_mask.any():
                    close_events = past_events[close_mask]
                    features.loc[i, 'mean_dist_to_past'] = dist_km[close_mask].mean()
                    features.loc[i, 'max_mag_in_window'] = close_events['magnitude'].max()
                    features.loc[i, 'avg_mag_recent'] = close_events['magnitude'].mean()
                
                # ETAS triggering contribution (simplified)
                # Contribution proportional to magnitude and 1/(t+c)^p / (r+D)^gamma
                if len(past_events) > 0:
                    triggering = np.sum(
                        (10 ** (0.5 * past_events['magnitude'])) /
                        ((dt_days + 0.01)**1.1)
                    )
                    features.loc[i, 'triggered_rate'] = triggering
        
        return features


class CoulombStressFeatures:
    """
    Simplified Coulomb Failure Stress change features
    For now: geometric stress change estimation
    """
    
    @staticmethod
    def estimate_stress_change(
        mainshock_lat: float,
        mainshock_lon: float,
        mainshock_depth: float,
        mainshock_mag: float,
        strike: float = 0,  # Fault strike in degrees
        dip: float = 90,    # Fault dip in degrees
        rake: float = 0,    # Fault rake in degrees
        target_lats: np.ndarray = None,
        target_lons: np.ndarray = None,
        target_depths: np.ndarray = None
    ) -> np.ndarray:
        """
        Estimate Coulomb stress change at target locations
        Simplified model based on Okada 1992
        """
        # This is a placeholder for full Coulomb 3D calculation
        # Real implementation requires PyCSEP or Coulomb 3
        
        if target_lats is None:
            return np.array([])
        
        # Distance-based decay (simplified)
        dlat = target_lats - mainshock_lat
        dlon = target_lons - mainshock_lon
        
        # Convert to km
        dist_km = np.sqrt(
            (dlat * 111)**2 + 
            (dlon * 111 * np.cos(np.radians(mainshock_lat)))**2 +
            ((target_depths - mainshock_depth) * 0.1)**2  # depth in km
        )
        
        # Stress change proportional to seismic moment and 1/r^3
        # Seismic moment M0 ~ 10^(1.5*Mw + 9.1)
        M0 = 10 ** (1.5 * mainshock_mag + 9.1)
        
        # Simplified stress change (positive = promotes failure)
        delta_CFS = M0 / (dist_km**3 + 1) * 1e-6  # Scale appropriately
        
        return delta_CFS


def extract_all_features(
    mainshock: pd.Series,
    aftershock_catalog: pd.DataFrame,
    historical_catalog: pd.DataFrame = None
) -> Dict:
    """
    Extract comprehensive feature set for aftershock prediction
    """
    features = {}
    
    # 1. Omori-Utsu parameters
    omori = OmoriUtsuFeatures(mainshock['time'], mainshock['magnitude'])
    omori_params = omori.fit(aftershock_catalog['time'], aftershock_catalog['magnitude'])
    features['omori'] = omori_params
    
    # Predict aftershocks in next 7, 30, 90 days
    for days in [7, 30, 90]:
        features[f'predicted_aftershocks_{days}d'] = omori.predict_cumulative(days)
    
    # 2. Gutenberg-Richter
    gr = GutenbergRichterFeatures()
    gr_params = gr.fit(aftershock_catalog['magnitude'])
    features['gutenberg_richter'] = gr_params
    
    # Probability of M>=5 aftershock
    features['prob_mag5_aftershock'] = gr.predict_prob_mag_above(
        mainshock['magnitude'] - 1, 5.0
    )
    
    # 3. ETAS-style features
    etas = ETASFeatures()
    etas_features = etas.extract_features(aftershock_catalog)
    features['etas_latest'] = etas_features.iloc[-1].to_dict() if len(etas_features) > 0 else {}
    
    # 4. Mainshock features
    features['mainshock'] = {
        'magnitude': mainshock['magnitude'],
        'depth': mainshock.get('depth', np.nan),
        'latitude': mainshock['latitude'],
        'longitude': mainshock['longitude'],
        'time': mainshock['time'].isoformat() if hasattr(mainshock['time'], 'isoformat') else str(mainshock['time']),
    }
    
    # 5. Catalog statistics
    features['catalog_stats'] = {
        'n_aftershocks_24h': len(aftershock_catalog[
            (aftershock_catalog['time'] - mainshock['time']).dt.total_seconds() <= 86400
        ]),
        'n_aftershocks_7d': len(aftershock_catalog[
            (aftershock_catalog['time'] - mainshock['time']).dt.total_seconds() <= 7*86400
        ]),
        'max_aftershock_mag': aftershock_catalog['magnitude'].max() if len(aftershock_catalog) > 0 else np.nan,
        'mean_aftershock_depth': aftershock_catalog.get('depth', pd.Series([np.nan])).mean(),
    }
    
    return features


# Example
if __name__ == "__main__":
    # Simulate data
    times = pd.date_range('2024-01-01', periods=100, freq='H')
    mags = np.random.normal(3.0, 0.5, 100)
    mags[0] = 6.5  # Mainshock
    
    catalog = pd.DataFrame({
        'time': times,
        'magnitude': mags,
        'latitude': np.random.normal(39.0, 0.5, 100),
        'longitude': np.random.normal(35.0, 0.5, 100),
        'depth': np.random.exponential(10, 100),
    })
    
    mainshock = catalog.iloc[0]
    aftershocks = catalog.iloc[1:]
    
    features = extract_all_features(mainshock, aftershocks)
    print(json.dumps(features, indent=2, default=str))
