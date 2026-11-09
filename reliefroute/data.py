"""Synthetic chronological series for ReliefRoute-Resources — Disaster Response Demand Forecasting; not sourced from operational data."""
from __future__ import annotations
import numpy as np
FEATURES=["hour_sin","hour_cos","week_sin","week_cos","trend","temperature","rainfall","event_flag","lag_1","lag_24","lag_168","rolling_mean_24"]
BASELINE_FEATURE="lag_24"
UNIT='relative synthetic demand-index units'
def make_dataset(n=1200,seed=42):
    if n<240:raise ValueError("n must be at least 240")
    rng=np.random.default_rng(seed)
    t=np.arange(168,168+n);full=np.arange(n+168);hour=full%24;dow=(full//24)%7;temperature=21+8*np.sin(2*np.pi*(full/24-5)/365)+rng.normal(0,3,n+168);rain=np.clip(rng.gamma(1.1,2,n+168),0,22);event=(rng.random(n+168)<.018).astype(float);impact=np.zeros(n+168);for_index=range(1,n+168)
    for k in for_index: impact[k]=max(0,.985*impact[k-1]+(rng.uniform(8,22) if event[k] else 0))
    road_closure=(rng.random(n+168)<.08).astype(float);noise=rng.normal(0,5,n+168);series=np.maximum(0,22+1.3*impact+1.8*rain+18*road_closure+6*(hour<7)+5*np.sin(2*np.pi*dow/7)+noise);X=np.column_stack([np.sin(2*np.pi*hour[t]/24),np.cos(2*np.pi*hour[t]/24),np.sin(2*np.pi*dow[t]/7),np.cos(2*np.pi*dow[t]/7),t/(n+168),temperature[t],rain[t],event[t],series[t-1],series[t-24],series[t-168],np.array([series[k-24:k].mean() for k in t])]);y=series[t]
    if X.shape!=(n,len(FEATURES)) or y.shape!=(n,) or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError("invalid generated time series")
    return X.astype(float),y.astype(float)
