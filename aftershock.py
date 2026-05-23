"""
Artçı Deprem Tahmin Modelleri
==============================
- Omori-Utsu Yasası   : N(t) = K / (t + c)^p
- Gutenberg-Richter   : log10(N) = a - b*M
- Bath Kanunu         : ΔM = Mmain - Mlargest_aftershock ≈ 1.2
- Risk Skoru          : Büyüklük + Derinlik + Konum bazlı
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict


# ─── SABITLER ────────────────────────────────────────────────────────────────
OMORI_K_DEFAULT = 10.0   # Omori yoğunluk sabiti
OMORI_C_DEFAULT = 0.1    # Erken dönem sabiti (gün)
OMORI_P_DEFAULT = 1.1    # Zaman azalma üssü (tipik: 0.9–1.5)
GR_B_DEFAULT    = 1.0    # Gutenberg-Richter b değeri (global ortalama)
GR_A_DEFAULT    = 5.0    # a değeri (bölgeye göre değişir)
BATH_DELTA      = 1.2    # Bath kanunu delta büyüklük farkı


# ─── VERİ SINIFLARI ──────────────────────────────────────────────────────────
@dataclass
class Earthquake:
    magnitude: float
    depth_km: float
    latitude: float
    longitude: float
    place: str
    time_str: str
    usgs_id: str = ""


@dataclass
class AftershockForecast:
    main: Earthquake
    # Omori parametreleri
    omori_k: float
    omori_c: float
    omori_p: float
    # G-R parametreleri
    gr_a: float
    gr_b: float
    # Bath tahmini
    bath_largest_mag: float
    # Zaman dilimlerine göre beklenen sayılar
    count_6h: float
    count_24h: float
    count_1_3d: float
    count_3_7d: float
    count_7_30d: float
    # Büyüklük aralıklarına göre tahmin (30 gün)
    by_magnitude: Dict[str, float]
    # Risk skoru (0-100)
    risk_score: int
    risk_level: str   # "Düşük" | "Orta" | "Yüksek" | "Kritik"


# ─── YARDIMCI FONKSİYONLAR ───────────────────────────────────────────────────

def omori_rate(t_days: float, K: float, c: float, p: float) -> float:
    """Omori-Utsu anlık hızı: λ(t) = K / (t+c)^p"""
    if t_days < 0:
        return 0.0
    return K / ((t_days + c) ** p)


def omori_cumulative(t_start: float, t_end: float,
                     K: float, c: float, p: float) -> float:
    """[t_start, t_end] aralığında kümülatif artçı sayısı."""
    if abs(p - 1.0) < 1e-6:
        val = K * (math.log(t_end + c) - math.log(t_start + c))
    else:
        val = K / (1 - p) * ((t_end + c) ** (1 - p) - (t_start + c) ** (1 - p))
    return max(0.0, val)


def gr_count_above(M_threshold: float, a: float, b: float,
                   t_start: float, t_end: float,
                   K: float, c: float, p: float) -> float:
    """
    M >= M_threshold büyüklüğündeki artçı sayısı tahmini.
    Omori toplam * G-R oranı birleşimi.
    """
    total_aftershocks = omori_cumulative(t_start, t_end, K, c, p)
    # G-R oranı: M0'ın üzerinde kalanlar (normalize)
    # N(M>=M_threshold) / N(M>=M_min) = 10^(-b*(M_threshold - M_min))
    M_min = 2.0
    ratio = 10 ** (-b * (M_threshold - M_min))
    return total_aftershocks * ratio


def calibrate_omori(magnitude: float) -> tuple:
    """
    Ana deprem büyüklüğüne göre Omori K parametresini kalibre et.
    Ampirik ilişki: log10(K) ≈ magnitude - 3 (kabaca)
    """
    K = 10 ** (magnitude - 3.0)
    c = 0.1
    p = 1.05 + (magnitude - 6.0) * 0.02  # büyük depremlerde p biraz artar
    p = max(0.8, min(p, 1.4))
    return K, c, p


def calibrate_gr(magnitude: float, depth_km: float) -> tuple:
    """
    Bölge/derinlik bazlı G-R kalibrasyonu.
    Sığ depremler (< 70 km) daha fazla artçı üretir.
    """
    b = 1.0
    if depth_km < 30:
        b = 0.85    # sığ → daha fazla büyük artçı
    elif depth_km < 70:
        b = 0.95
    else:
        b = 1.1     # derin → daha az büyük artçı

    a = 1.5 * magnitude - 3.0  # ampirik
    return a, b


def compute_risk_score(magnitude: float, depth_km: float) -> tuple:
    """
    Risk skoru (0-100) ve seviyesi hesapla.
    Büyüklük + Derinlik faktörü.
    """
    # Büyüklük katkısı (0-60 puan)
    mag_score = max(0, min(60, (magnitude - 3.0) / 6.0 * 60))

    # Derinlik katkısı: sığ = daha tehlikeli (0-30 puan)
    if depth_km < 15:
        depth_score = 30
    elif depth_km < 35:
        depth_score = 22
    elif depth_km < 70:
        depth_score = 14
    elif depth_km < 150:
        depth_score = 7
    else:
        depth_score = 3

    # Büyük artçı olasılığı katkısı (0-10 puan)
    bath_score = max(0, min(10, (magnitude - 4.0) * 2.5))

    score = int(mag_score + depth_score + bath_score)
    score = max(0, min(100, score))

    if score < 25:
        level = "Düşük"
    elif score < 50:
        level = "Orta"
    elif score < 75:
        level = "Yüksek"
    else:
        level = "Kritik"

    return score, level


# ─── ANA TAHMİN FONKSİYONU ───────────────────────────────────────────────────

def forecast(eq: Earthquake) -> AftershockForecast:
    """Ana deprem için tam artçı tahmini üret."""
    M  = eq.magnitude
    D  = eq.depth_km

    # Parametre kalibrasyonu
    K, c, p = calibrate_omori(M)
    a, b    = calibrate_gr(M, D)

    # Bath kanunu: en büyük beklenen artçı
    bath_largest = M - BATH_DELTA

    # Zaman dilimlerine göre kümülatif sayılar (M>=2.0)
    def _count(t0, t1):
        return omori_cumulative(t0, t1, K, c, p)

    count_6h    = _count(0,    0.25)
    count_24h   = _count(0,    1.0)
    count_1_3d  = _count(1.0,  3.0)
    count_3_7d  = _count(3.0,  7.0)
    count_7_30d = _count(7.0, 30.0)

    # 30 günlük büyüklük aralığı dağılımı
    total_30d = _count(0, 30.0)
    by_mag: Dict[str, float] = {}
    ranges = [
        ("M2.0-M3.0", 2.0, 3.0),
        ("M3.0-M4.0", 3.0, 4.0),
        ("M4.0-M5.0", 4.0, 5.0),
        ("M5.0-M6.0", 5.0, 6.0),
        ("M6.0+",     6.0, 9.0),
    ]
    for label, m_lo, m_hi in ranges:
        n_above_lo = total_30d * (10 ** (-b * (m_lo - 2.0)))
        n_above_hi = total_30d * (10 ** (-b * (m_hi - 2.0)))
        by_mag[label] = max(0.0, round(n_above_lo - n_above_hi, 1))

    risk_score, risk_level = compute_risk_score(M, D)

    return AftershockForecast(
        main          = eq,
        omori_k       = round(K, 3),
        omori_c       = round(c, 3),
        omori_p       = round(p, 3),
        gr_a          = round(a, 3),
        gr_b          = round(b, 3),
        bath_largest_mag = round(bath_largest, 1),
        count_6h      = round(count_6h,    1),
        count_24h     = round(count_24h,   1),
        count_1_3d    = round(count_1_3d,  1),
        count_3_7d    = round(count_3_7d,  1),
        count_7_30d   = round(count_7_30d, 1),
        by_magnitude  = by_mag,
        risk_score    = risk_score,
        risk_level    = risk_level,
    )
