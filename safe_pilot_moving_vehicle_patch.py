"""
safe_pilot_moving_vehicle_patch.py
===================================
Patch for safe_pilot_moving_vehicle_20260415.ipynb

BUG FIXED
---------
Scenario: 70 mph highway + Tornado Warning (or heavy hail) on predicted path
Expected: RED alert
Actual:   YELLOW alert (before fix)

ROOT CAUSE
----------
Two compounding issues:

1. Training data gap in _build_alert_model():
   All RED training examples required p_int >= 0.72. A real highway scenario
   where the narrow corridor passes through a large tornado polygon produces
   p_int ~ 0.35-0.55 (overlap_frac x avg_confidence). That range had no RED
   training examples, so the Random Forest slid it into YELLOW.

2. No severity-based safety override in decide_alert():
   Even when the RF returned ~50-60% RED probability, the 0.70 RED threshold
   was not met, so the alert fell through to YELLOW.

CHANGES
-------
Replace Phase 4A (_build_alert_model) and Phase 4B (decide_alert) cells
with the versions below.  All other cells remain unchanged.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────────
# PHASE 4A — RANDOM FOREST ALERT DECISION MODEL  (FIXED)
# ─────────────────────────────────────────────────────────────────

def _build_alert_model() -> RandomForestClassifier:
    """
    Build and train a Random Forest classifier on synthetic labeled examples.

    FIX: Added three RED training examples covering high-severity events
    (tornado / heavy hail) at moderate intersection probability (0.35-0.50)
    and highway speeds (~70 mph).  Previously these scenarios fell into a
    training-data gap and were misclassified as YELLOW.

    Feature columns (order must match _build_decision_features):
      [severity_score, time_to_impact_norm, p_intersect, road_score,
       is_braking, current_severe, speed_norm]

    Labels: 0 = No Alert, 1 = Yellow Advisory, 2 = Red Alert
    """
    X_train = np.array([
        # severity  time_norm  p_int  road  brake  curr_sev  spd_norm  -> label
        [4.0,       0.1,       0.90,  0.9,  0,     0,        0.9],     # Red: tornado, imminent, highway
        [4.0,       0.3,       0.80,  0.9,  0,     0,        0.8],     # Red: tornado, 6 min, highway
        [4.0,       0.5,       0.50,  0.95, 0,     0,        0.93],    # Red: tornado 10min, 70mph highway   <- ADDED
        [4.0,       0.5,       0.35,  0.90, 0,     0,        0.85],    # Red: tornado 10min, moderate prob  <- ADDED
        [3.0,       0.2,       0.85,  0.8,  0,     0,        0.7],     # Red: hail, near, trunk road
        [3.0,       0.5,       0.75,  0.7,  1,     0,        0.5],     # Red: hail, driver braking
        [3.0,       0.4,       0.45,  0.90, 0,     0,        0.80],    # Red: hail/storm, 8min, highway     <- ADDED
        [4.0,       0.8,       0.72,  0.5,  0,     0,        0.3],     # Red: high severity, far but high prob
        [2.0,       0.2,       0.60,  0.7,  0,     1,        0.6],     # Red: severe now + incoming
        [3.0,       0.4,       0.65,  0.6,  0,     0,        0.5],     # Yellow: hail, moderate prob
        [2.0,       0.5,       0.55,  0.7,  0,     0,        0.6],     # Yellow: thunderstorm, moderate
        [2.0,       0.6,       0.45,  0.8,  0,     0,        0.7],     # Yellow: rain, 12 min, low prob
        [1.0,       0.3,       0.50,  0.5,  0,     0,        0.4],     # Yellow: fog, residential
        [3.0,       0.9,       0.35,  0.9,  0,     0,        0.8],     # Yellow: severe but far and uncertain
        [1.0,       0.8,       0.20,  0.6,  0,     0,        0.5],     # None: low prob, far, mild
        [0.0,       1.0,       0.05,  0.9,  0,     0,        0.8],     # None: clear sky on highway
        [1.0,       1.0,       0.10,  0.5,  0,     0,        0.3],     # None: mild weather, far
        [0.0,       0.5,       0.00,  0.8,  0,     0,        0.6],     # None: no intersection at all
        [2.0,       0.7,       0.28,  0.3,  0,     0,        0.2],     # None: residential, low prob
    ])

    # 9 RED (2), 5 YELLOW (1), 5 NONE (0)  -- was 6 RED before the fix
    y_train = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────
# PHASE 4B — ALERT DECISION  (FIXED)
# ─────────────────────────────────────────────────────────────────

ALERT_THRESHOLDS = {
    'RED':    0.70,   # >= 70% RF probability -> Red
    'YELLOW': 0.30,   # 30-70%               -> Yellow
}


def decide_alert(pings, corridor, intersection, weather_response,
                 ALERT_MODEL, SEVERITY_NUMERIC, ALERT_THRESHOLD_CODES,
                 ALERT_SEVERITY, SEVERE_WEATHER_CODES, MOVING_VEHICLE_ADVICE,
                 ROAD_TYPES, MovingVehicleAlert, _build_decision_features):
    """
    Run the Random Forest decision model and assemble the final alert.

    FIX: Added a severity-based hard override before the RF threshold check.
    If a tornado or heavy hail event (severity >= 4, i.e. WMO codes 96/99)
    is on the predicted path within 15 minutes with any meaningful intersection
    probability (>= 0.20), the alert is forced to RED regardless of what the
    RF outputs.  This prevents life-safety events from being downgraded to
    YELLOW due to the previously-existing training-data gap.
    """
    current_code  = 0
    wind_speed    = 0.0
    temperature_f = 70.0
    if weather_response:
        curr          = weather_response.get('current', {})
        current_code  = curr.get('weather_code', 0)
        wind_speed    = curr.get('wind_speed_10m', 0.0)
        temperature_f = curr.get('temperature_2m', 70.0)

    current_condition = SEVERE_WEATHER_CODES.get(current_code, 'Unknown')

    # Random Forest prediction
    features = _build_decision_features(pings, corridor, intersection, current_code)
    proba    = ALERT_MODEL.predict_proba(features.reshape(1, -1))[0]
    p_red    = proba[2] if len(proba) > 2 else 0.0
    p_yellow = proba[1] if len(proba) > 1 else 0.0
    p_impact = intersection.intersection_prob if intersection else 0.0

    # Extract individual feature values needed for the override check
    severity_num  = features[0]   # 0-4; 4 = tornado / heavy hail (codes 96, 99)
    time_norm_val = features[1]   # 0 = now, 1 = 20 min away
    p_int_val     = features[2]   # geometric intersection probability

    # ── FIX: Hard override for extreme-severity events ──────────────────────
    # Tornado or heavy hail on path within 15 min (time_norm <= 0.75) with
    # any meaningful intersection probability (>= 0.20) -> always RED.
    # This guards against the training-data gap in the moderate p_int range.
    if severity_num >= 4 and time_norm_val <= 0.75 and p_int_val >= 0.20:
        alert_level = 'RED'
    elif p_red >= ALERT_THRESHOLDS['RED']:
        alert_level = 'RED'
    elif (p_red + p_yellow) >= ALERT_THRESHOLDS['YELLOW']:
        alert_level = 'YELLOW'
    elif current_code in ALERT_THRESHOLD_CODES:
        alert_level = 'RED' if ALERT_SEVERITY.get(current_code) in ('HIGH', 'CRITICAL') else 'YELLOW'
    else:
        alert_level = 'NONE'
    # ────────────────────────────────────────────────────────────────────────

    advice_code    = intersection.severity_code if intersection else current_code
    vehicle_advice = MOVING_VEHICLE_ADVICE.get(advice_code, '')
    exit_tip       = intersection.exit_before_impact if intersection else None

    if alert_level == 'RED':
        wx_label = intersection.weather_type if intersection else current_condition
        tti_str  = (f'in ~{int(intersection.minutes_to_impact)} min'
                    if intersection and intersection.minutes_to_impact > 0
                    else 'at your location NOW')
        alert_message = (
            f'RED ALERT [{ALERT_SEVERITY.get(advice_code, "HIGH")}]: '
            f'{wx_label} on your predicted path {tti_str}. '
            f'Impact probability: {p_impact*100:.0f}%. '
            f'Speed: {pings[-1].speed_mph:.0f} mph | '
            f'Heading: {pings[-1].heading:.0f} deg'
        )
    elif alert_level == 'YELLOW':
        wx_label = intersection.weather_type if intersection else current_condition
        tti_str  = (f'in ~{int(intersection.minutes_to_impact)} min'
                    if intersection and intersection.minutes_to_impact > 0
                    else 'near your location')
        alert_message = (
            f'YELLOW ADVISORY: {wx_label} possible on your path {tti_str}. '
            f'Impact probability: {p_impact*100:.0f}%. Monitor conditions.'
        )
    else:
        alert_message = 'No alert needed. Your predicted path is clear of severe weather.'

    return MovingVehicleAlert(
        current_lat     = pings[-1].lat,
        current_lon     = pings[-1].lon,
        speed_mph       = round(pings[-1].speed_mph, 1),
        heading_degrees = round(pings[-1].heading, 1),
        road_type       = corridor.road_type,
        corridor        = corridor,
        intersection    = intersection,
        current_weather = current_condition,
        current_code    = current_code,
        wind_speed_mph  = wind_speed,
        temperature_f   = temperature_f,
        alert_level     = alert_level,
        alert_message   = alert_message,
        vehicle_advice  = vehicle_advice,
        exit_tip        = exit_tip,
        confidence_pct  = round(p_impact * 100, 1),
    )


# ─────────────────────────────────────────────────────────────────
# HOW TO APPLY IN YOUR NOTEBOOK
# ─────────────────────────────────────────────────────────────────
# 1. In Phase 4A cell: replace the _build_alert_model() function body
#    (X_train, y_train, and the model.fit() call) with the version above.
#
# 2. In Phase 4B cell: replace the decide_alert() threshold block
#    (the "Apply thresholds" section) with the FIX block above.
#    Also add the three lines that unpack severity_num, time_norm_val,
#    and p_int_val from the features array before the threshold check.
#
# 3. Re-run all Phase 4 cells.  Scenario 1 (70 mph highway + Tornado Warning)
#    will now correctly output RED instead of YELLOW.
