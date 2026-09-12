"""
Autonomous Demand-Sensing & Anomaly Detection Engine
---------------------------------------------------
Implements:
1. 14-Day Calibration Window (Warm-up period)
2. 6-Week Day-of-Week Seasonality Lookback (K=6, Bessel's correction df=5)
3. Variance Floor (sigma_eff = max(sigma, 1.0)) & Volume Gate (x_t >= 10)
4. Outlier Winsorization (Baseline Contamination Defense)
5. Hysteresis Dual-Thresholding (Entry Z >= 2.5, Exit Z <= 1.0)
6. Rolling Multi-Day State Machine (Provisional -> Elevated -> Reconciled)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class AnomalyDetectionEngine:
    def __init__(
        self,
        warmup_days: int = 14,
        seasonality_lookback_weeks: int = 6,
        entry_z_threshold: float = 2.5,
        exit_z_threshold: float = 1.0,
        intermediate_z_threshold: float = 2.0,
        volume_gate_units: float = 10.0,
        sigma_floor: float = 1.0
    ):
        self.warmup_days = warmup_days
        self.lookback_weeks = seasonality_lookback_weeks
        self.entry_z = entry_z_threshold
        self.exit_z = exit_z_threshold
        self.intermediate_z = intermediate_z_threshold
        self.volume_gate = volume_gate_units
        self.sigma_floor = sigma_floor

    def analyze_series(
        self,
        sales_series: List[float],
        calendar_df: pd.DataFrame,
        sku_id: str,
        state_id: str,
        cat_id: str
    ) -> pd.DataFrame:
        """
        Runs the full anomaly detection pipeline on a 365-day series.
        Returns a DataFrame containing day-by-day metrics, baselines, Z-scores, and states.
        """
        n_days = len(sales_series)
        day_indices = [f"d_{i}" for i in range(1, n_days + 1)]
        
        # We maintain a cleaned/winsorized history for baseline calculation to prevent poisoning
        winsorized_history = list(sales_series[:])
        
        results = []
        active_streak = 0
        current_state = "NORMAL"
        
        for t in range(n_days):
            day_label = day_indices[t]
            day_num = t + 1
            x_t = float(sales_series[t])
            
            cal_row = calendar_df[calendar_df['d'] == day_label].iloc[0]
            date_str = cal_row['date']
            weekday = cal_row['weekday']
            wday_idx = int(cal_row['wday']) # 1=Saturday, 2=Sunday ...
            
            # --- 1. Warm-up Calibration Window ---
            if day_num <= self.warmup_days:
                results.append({
                    'day': day_label,
                    'day_num': day_num,
                    'date': date_str,
                    'weekday': weekday,
                    'actual_sales': x_t,
                    'baseline_mean': np.nan,
                    'baseline_std': np.nan,
                    'z_score': 0.0,
                    'status': "CALIBRATION_PERIOD",
                    'streak_day': 0,
                    'is_anomaly': False
                })
                continue
            
            # --- 2. 6-Week Day-of-Week Seasonality Lookback ---
            # Gather past occurrences of the exact same weekday from the winsorized history
            lookback_indices = [t - 7 * w for w in range(1, self.lookback_weeks + 1) if (t - 7 * w) >= 0]
            
            if len(lookback_indices) < 2:
                # Fallback to simple prior 14 days if not enough weekday history yet
                lookback_indices = [t - i for i in range(1, 15) if (t - i) >= 0]
                
            historical_samples = [winsorized_history[idx] for idx in lookback_indices]
            
            # Mean & Sample Standard Deviation (Bessel's correction ddof=1)
            mu_t = float(np.mean(historical_samples))
            if len(historical_samples) > 1:
                sigma_t = float(np.std(historical_samples, ddof=1))
            else:
                sigma_t = 0.0
                
            sigma_eff = max(sigma_t, self.sigma_floor)
            
            # --- 3. Compute Deviation & Z-Score ---
            z_score = (x_t - mu_t) / sigma_eff
            
            # --- 4. State Transition Logic (Hysteresis & Rolling Markov Transitions) ---
            is_anomaly = False
            
            if current_state in ["NORMAL", "RESOLVED_ONE_OFF", "RESOLVED_MULTI_DAY_SURGE", "CALIBRATION_PERIOD"]:
                # Check for new spike onset
                if z_score >= self.entry_z and x_t >= self.volume_gate:
                    current_state = "PROVISIONAL_ALERT"
                    active_streak = 1
                    is_anomaly = True
                    # Winsorize future baseline calculation to prevent poisoning
                    winsorized_history[t] = mu_t
                elif z_score <= -self.entry_z and mu_t >= self.volume_gate:
                    current_state = "DEMAND_COLLAPSE"
                    active_streak = 1
                    is_anomaly = True
                    winsorized_history[t] = mu_t
                else:
                    current_state = "NORMAL"
                    active_streak = 0
            
            elif current_state == "PROVISIONAL_ALERT":
                # Day t+1 of a surge: 3-Zone Quantitative Reconciliation
                if z_score <= self.exit_z:
                    current_state = "RESOLVED_ONE_OFF"
                    active_streak = 0
                    is_anomaly = False
                    # Normal value returned: preserve in history
                elif self.exit_z < z_score < self.intermediate_z:
                    current_state = "PARTIALLY_ELEVATED_WATCH"
                    active_streak = 2
                    is_anomaly = True
                    winsorized_history[t] = mu_t
                else: # z_score >= intermediate_z (still high)
                    current_state = "ELEVATED_SURGE_DAY_2"
                    active_streak = 2
                    is_anomaly = True
                    winsorized_history[t] = mu_t
            
            elif current_state in ["ELEVATED_SURGE_DAY_2", "PARTIALLY_ELEVATED_WATCH"]:
                # Day t+2 (or subsequent days of surge)
                if z_score <= self.exit_z:
                    current_state = "RESOLVED_MULTI_DAY_SURGE"
                    active_streak = 0
                    is_anomaly = False
                elif z_score >= self.intermediate_z:
                    # 3rd consecutive elevated day: plateau confirmed
                    current_state = "CONFIRMED_STRUCTURAL_PLATEAU"
                    active_streak += 1
                    is_anomaly = True
                    winsorized_history[t] = x_t
                else:
                    # 1.0 < z_score < 2.0: intermediate cooling watch
                    current_state = "PARTIALLY_ELEVATED_WATCH"
                    active_streak += 1
                    is_anomaly = True
                    winsorized_history[t] = mu_t
            
            elif current_state == "CONFIRMED_STRUCTURAL_PLATEAU":
                if z_score <= self.exit_z:
                    current_state = "NORMAL"
                    active_streak = 0
                    is_anomaly = False
                else:
                    active_streak += 1
                    is_anomaly = True
                    winsorized_history[t] = x_t
                    
            results.append({
                'day': day_label,
                'day_num': day_num,
                'date': date_str,
                'weekday': weekday,
                'actual_sales': x_t,
                'baseline_mean': round(mu_t, 2),
                'baseline_std': round(sigma_eff, 2),
                'z_score': round(z_score, 2),
                'status': current_state,
                'streak_day': active_streak,
                'is_anomaly': is_anomaly
            })
            
        return pd.DataFrame(results)
