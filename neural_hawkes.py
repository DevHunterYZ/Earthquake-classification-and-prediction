"""
Neural Hawkes Process for Earthquake Prediction
Combines temporal point process with deep learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SequenceEvent:
    time: float  # Time since sequence start (hours or days)
    magnitude: float
    latitude: float
    longitude: float
    depth: float
    is_aftershock: bool = True


class TemporalEncoding(nn.Module):
    """Sinusoidal temporal encoding like Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        times: (batch, seq_len) - normalized times
        """
        # Interpolate to get continuous encoding
        max_time = times.max().item()
        if max_time == 0:
            max_time = 1.0
        
        indices = (times / max_time * (self.pe.size(0) - 1)).long()
        indices = torch.clamp(indices, 0, self.pe.size(0) - 1)
        
        return self.pe[indices]


class EarthquakeTransformer(nn.Module):
    """
    Transformer-based model for earthquake sequence prediction
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Event embedding: [magnitude, log_time_delta, lat, lon, depth, aftershock_flag]
        self.event_embedding = nn.Linear(6, d_model)
        
        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.time_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
        self.mag_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )
        
        self.intensity_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Softplus()  # Ensure positive intensity
        )
        
    def forward(self, events: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        events: (batch, seq_len, 6) - [mag, time_delta, lat, lon, depth, is_af]
        mask: (batch, seq_len) - padding mask
        
        Returns: dict with time_pred, mag_pred, intensity
        """
        batch_size, seq_len, _ = events.shape
        
        # Embed events
        x = self.event_embedding(events)  # (batch, seq_len, d_model)
        
        # Add temporal encoding based on cumulative time
        times = events[:, :, 1].cumsum(dim=1)  # Cumulative time
        temp_enc = self.temporal_encoding(times)
        x = x + temp_enc
        
        # Create causal mask for autoregressive prediction
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(events.device)
        
        # Transformer encoding
        if mask is not None:
            # Combine causal and padding mask
            key_mask = mask.bool()
        else:
            key_mask = None
            
        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_mask)
        
        # Predictions (last position for next event prediction)
        last_hidden = h[:, -1, :]  # (batch, d_model)
        
        time_pred = self.time_predictor(last_hidden).squeeze(-1)  # Time until next event
        mag_pred = self.mag_predictor(last_hidden).squeeze(-1)   # Magnitude of next event
        intensity = self.intensity_predictor(last_hidden).squeeze(-1)  # Current intensity
        
        return {
            'time_until_next': time_pred,
            'next_magnitude': mag_pred,
            'current_intensity': intensity
        }


class NeuralHawkesProcess(nn.Module):
    """
    Neural Hawkes Process with LSTM-based intensity function
    More efficient for long sequences than Transformer
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        event_dim: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for history encoding
        self.lstm = nn.LSTM(
            input_size=event_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Intensity function: lambda(t) = f(history, t - t_last)
        self.intensity_net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()  # lambda > 0
        )
        
        # Time prediction (survival function based)
        self.time_dist_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)  # mu, log_sigma for log-normal
        )
        
        # Magnitude prediction (Gutenberg-Richter style)
        self.mag_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # a, b parameters
        )
        
    def forward(self, events: torch.Tensor, time_since_last: torch.Tensor) -> Dict:
        """
        events: (batch, seq_len, event_dim)
        time_since_last: (batch,) - time since last event for intensity evaluation
        
        Returns: dict with intensity, time_dist_params, mag_params
        """
        # Encode history with LSTM
        lstm_out, (h_n, c_n) = self.lstm(events)
        
        # Last hidden state represents current history
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Current intensity at time_since_last
        time_feat = time_since_last.unsqueeze(-1)  # (batch, 1)
        intensity_input = torch.cat([last_hidden, time_feat], dim=-1)
        intensity = self.intensity_net(intensity_input).squeeze(-1)
        
        # Time distribution parameters (log-normal)
        time_params = self.time_dist_net(last_hidden)
        mu, log_sigma = time_params[:, 0], time_params[:, 1]
        
        # Magnitude distribution parameters
        mag_params = self.mag_net(last_hidden)
        a_param, b_param = mag_params[:, 0], torch.abs(mag_params[:, 1])  # b > 0
        
        return {
            'intensity': intensity,  # Current conditional intensity
            'time_mu': mu,  # Log-normal mean for time to next event
            'time_log_sigma': log_sigma,
            'gr_a': a_param,  # Gutenberg-Richter a
            'gr_b': b_param,  # Gutenberg-Richter b (positive)
        }
    
    def sample_next_event(self, history: torch.Tensor, device: str = 'cpu') -> Tuple[float, float]:
        """
        Sample next event time and magnitude given history
        Uses thinning algorithm for temporal sampling
        """
        self.eval()
        with torch.no_grad():
            # Get current intensity upper bound
            time_candidates = torch.linspace(0, 100, 1000).to(device)
            intensities = []
            
            for dt in time_candidates:
                pred = self.forward(history.unsqueeze(0), dt.unsqueeze(0))
                intensities.append(pred['intensity'].item())
            
            lambda_max = max(intensities) * 1.5  # Upper bound
            
            # Thinning algorithm
            t = 0.0
            while True:
                # Sample candidate time from exponential with rate lambda_max
                u = np.random.exponential(1.0 / lambda_max)
                t += u
                
                # Evaluate intensity at t
                pred = self.forward(history.unsqueeze(0), torch.tensor([t]).to(device))
                lambda_t = pred['intensity'].item()
                
                # Accept with probability lambda_t / lambda_max
                if np.random.random() < (lambda_t / lambda_max):
                    # Sample magnitude
                    a = pred['gr_a'].item()
                    b = pred['gr_b'].item()
                    
                    # Sample from Gutenberg-Richter (exponential in magnitude)
                    # P(M >= m) = 10^(a - b*m) => P(M < m) = 1 - 10^(a-b*m)
                    # For M >= Mc, use truncated exponential
                    mc = 2.5  # Completeness magnitude
                    u_mag = np.random.random()
                    m = mc - (1/b) * np.log10(1 - u_mag * (1 - 10**(-b*mc)))
                    
                    return t, m
                
                if t > 365:  # Max 1 year
                    return None, None


class AftershockPredictor:
    """
    High-level wrapper for earthquake prediction
    """
    
    def __init__(self, model_type: str = 'transformer', device: str = 'cpu'):
        self.model_type = model_type
        self.device = device
        
        if model_type == 'transformer':
            self.model = EarthquakeTransformer().to(device)
        elif model_type == 'hawkes':
            self.model = NeuralHawkesProcess().to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
    def prepare_sequence(self, events: List[SequenceEvent]) -> torch.Tensor:
        """Convert event list to tensor"""
        data = []
        prev_time = 0.0
        
        for i, ev in enumerate(events):
            time_delta = ev.time - prev_time if i > 0 else 0.0
            prev_time = ev.time
            
            data.append([
                ev.magnitude,
                np.log1p(time_delta),  # Log time delta
                ev.latitude,
                ev.longitude,
                ev.depth,
                1.0 if ev.is_aftershock else 0.0
            ])
        
        return torch.FloatTensor(data).to(self.device)
    
    def fit(self, sequences: List[List[SequenceEvent]], epochs: int = 100):
        """Train on multiple sequences"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for seq in sequences:
                if len(seq) < 2:
                    continue
                    
                x = self.prepare_sequence(seq[:-1]).unsqueeze(0)  # All but last
                target = seq[-1]  # Last event is target
                
                self.optimizer.zero_grad()
                
                if self.model_type == 'transformer':
                    pred = self.model(x)
                    
                    # Loss: time prediction + magnitude prediction
                    time_loss = F.mse_loss(pred['time_until_next'], 
                                          torch.tensor([target.time - seq[-2].time]).to(self.device))
                    mag_loss = F.mse_loss(pred['next_magnitude'], 
                                        torch.tensor([target.magnitude]).to(self.device))
                    
                    loss = time_loss + mag_loss
                    
                elif self.model_type == 'hawkes':
                    # Hawkes process negative log-likelihood
                    loss = self._hawkes_loss(seq)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def _hawkes_loss(self, seq: List[SequenceEvent]) -> torch.Tensor:
        """Negative log-likelihood for Hawkes process"""
        # Simplified: use intensity-based loss
        history = self.prepare_sequence(seq[:-1])
        
        total_loss = 0.0
        for i in range(1, len(seq)):
            hist = history[:i].unsqueeze(0)
            dt = torch.tensor([seq[i].time - seq[i-1].time]).to(self.device)
            
            pred = self.model(hist, dt)
            
            # Intensity-based loss
            lambda_t = pred['intensity']
            
            # Log-likelihood contribution
            # LL = log(lambda(t_i)) - integral_0^T lambda(t) dt
            # Approximate integral
            ll = torch.log(lambda_t + 1e-10) - lambda_t * dt
            total_loss -= ll
        
        return total_loss / (len(seq) - 1)
    
    def predict(self, history: List[SequenceEvent], horizon_days: int = 30) -> Dict:
        """Predict next events and intensity"""
        self.model.eval()
        
        with torch.no_grad():
            x = self.prepare_sequence(history).unsqueeze(0)
            
            if self.model_type == 'transformer':
                pred = self.model(x)
                
                return {
                    'expected_time_next': pred['time_until_next'].item(),
                    'expected_magnitude_next': pred['next_magnitude'].item(),
                    'current_intensity': pred['current_intensity'].item(),
                }
            
            elif self.model_type == 'hawkes':
                # Sample multiple future events
                future_events = []
                current_history = history.copy()
                
                for _ in range(10):  # Sample up to 10 future events
                    hist_tensor = self.prepare_sequence(current_history)
                    dt, mag = self.model.sample_next_event(hist_tensor, self.device)
                    
                    if dt is None or sum([e.time for e in future_events]) + dt > horizon_days * 24:
                        break
                    
                    future_events.append((dt, mag))
                    
                    # Add to history for next prediction
                    last = current_history[-1]
                    current_history.append(SequenceEvent(
                        time=last.time + dt,
                        magnitude=mag,
                        latitude=last.latitude,
                        longitude=last.longitude,
                        depth=last.depth,
                        is_aftershock=True
                    ))
                
                return {
                    'n_predicted_events': len(future_events),
                    'expected_total_magnitude': sum([m for _, m in future_events]) if future_events else 0,
                    'predicted_sequence': future_events
                }


# Example usage
if __name__ == "__main__":
    # Create synthetic training data
    np.random.seed(42)
    
    sequences = []
    for _ in range(100):  # 100 mainshock-aftershock sequences
        mainshock = SequenceEvent(
            time=0,
            magnitude=np.random.uniform(5.5, 7.5),
            latitude=39.0,
            longitude=35.0,
            depth=10.0,
            is_mainshock=True
        )
        
        # Generate synthetic aftershocks (Omori law)
        seq = [mainshock]
        t = 0
        for i in range(20):
            dt = np.random.exponential(2.0)  # Mean 2 hours
            t += dt
            mag = mainshock.magnitude - 1 - np.random.exponential(0.5)
            
            seq.append(SequenceEvent(
                time=t,
                magnitude=max(mag, 2.0),
                latitude=39.0 + np.random.normal(0, 0.1),
                longitude=35.0 + np.random.normal(0, 0.1),
                depth=10.0 + np.random.normal(0, 3),
                is_aftershock=True
            ))
        
        sequences.append(seq)
    
    # Train model
    predictor = AftershockPredictor(model_type='transformer', device='cpu')
    predictor.fit(sequences, epochs=50)
    
    # Test prediction
    test_history = sequences[0][:5]
    prediction = predictor.predict(test_history, horizon_days=7)
    print("Prediction:", prediction)
