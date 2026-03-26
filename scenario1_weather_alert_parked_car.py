"""
Safe Pilot App Enhancement – Scenario 1
=======================================
Weather Alert for Parked / Static Vehicles

Logic:
  - If the car is parked (speed == 0 or status == "parked") AND
    severe weather is approaching within the alert radius,
    generate a contextual alert message for the driver.

Runnable in Google Colab – no external dependencies beyond the standard library.
"""

from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Data Structures (simulating what the Safe Pilot app would receive)
# ---------------------------------------------------------------------------

def make_vehicle(vehicle_id: str, speed_kmh: float, status: str, location: dict) -> dict:
    """
    speed_kmh : current speed in km/h (0 = parked/static)
    status    : "parked" | "moving" | "idle"
    location  : {"lat": float, "lon": float, "address": str}
    """
    return {
        "vehicle_id": vehicle_id,
        "speed_kmh": speed_kmh,
        "status": status,
        "location": location,
        "timestamp": datetime.now().isoformat(),
    }


def make_weather_event(
    event_type: str,
    severity: str,
    distance_km: float,
    eta_minutes: int,
    description: str,
    wind_speed_kmh: Optional[float] = None,
    rainfall_mm_hr: Optional[float] = None,
    hail_size_cm: Optional[float] = None,
) -> dict:
    """
    event_type    : e.g. "hail", "thunderstorm", "tornado", "flash_flood",
                         "blizzard", "high_winds", "clear"
    severity      : "none" | "moderate" | "severe" | "extreme"
    distance_km   : how far the weather event is from the vehicle (km)
    eta_minutes   : estimated minutes before the event reaches the vehicle
    description   : human-readable description of the weather
    """
    return {
        "event_type": event_type,
        "severity": severity,
        "distance_km": distance_km,
        "eta_minutes": eta_minutes,
        "description": description,
        "wind_speed_kmh": wind_speed_kmh,
        "rainfall_mm_hr": rainfall_mm_hr,
        "hail_size_cm": hail_size_cm,
        "observed_at": datetime.now().isoformat(),
        "projected_arrival": (
            datetime.now() + timedelta(minutes=eta_minutes)
        ).strftime("%H:%M"),
    }


# ---------------------------------------------------------------------------
# Severity Configuration
# ---------------------------------------------------------------------------

# Severity levels ranked lowest → highest
SEVERITY_RANK = {"none": 0, "moderate": 1, "severe": 2, "extreme": 3}

# Minimum severity that triggers an alert for a parked car
ALERT_SEVERITY_THRESHOLD = "moderate"

# Maximum distance (km) within which a weather event triggers an alert
ALERT_RADIUS_KM = 30

# Alert messages per weather event type
ALERT_TEMPLATES = {
    "hail": (
        "Hail storm approaching! Hailstones {hail_detail}are expected to reach your area "
        "around {arrival}. Move your vehicle to a covered shelter (garage, carport, or "
        "underpass) to prevent damage."
    ),
    "thunderstorm": (
        "Severe thunderstorm approaching your area around {arrival}. "
        "Ensure your vehicle is parked away from tall trees and power lines. "
        "Stay indoors until the storm passes."
    ),
    "tornado": (
        "TORNADO WARNING! A tornado is {distance:.1f} km away and may reach you around {arrival}. "
        "Do NOT shelter in your vehicle. Move to a sturdy building or underground shelter immediately!"
    ),
    "flash_flood": (
        "Flash flood warning in effect. Heavy rainfall {rain_detail}could cause flooding near your "
        "parked location around {arrival}. Move your vehicle to higher ground immediately."
    ),
    "blizzard": (
        "Blizzard conditions approaching around {arrival}. Expect heavy snowfall and whiteout "
        "conditions. Move your vehicle to a covered area and avoid travel until conditions improve."
    ),
    "high_winds": (
        "Severe wind advisory: Gusts up to {wind_detail}expected around {arrival}. "
        "Park your vehicle away from trees, billboards, and unstable structures to avoid damage."
    ),
    "dust_storm": (
        "Dust storm (haboob) approaching around {arrival}. Visibility may drop to near zero. "
        "Keep your vehicle parked and stay inside until the storm clears."
    ),
    "default": (
        "Severe weather ({event_type}) is approaching your parked vehicle and expected around {arrival}. "
        "Take precautions to ensure your vehicle is in a safe location."
    ),
}


# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def is_vehicle_parked(vehicle: dict) -> bool:
    """Return True if the vehicle is considered parked or static."""
    return vehicle["status"] == "parked" or vehicle["speed_kmh"] == 0


def is_alert_required(weather: dict) -> bool:
    """Return True if the weather event meets the threshold to trigger an alert."""
    severity_ok = (
        SEVERITY_RANK.get(weather["severity"], 0)
        >= SEVERITY_RANK[ALERT_SEVERITY_THRESHOLD]
    )
    distance_ok = weather["distance_km"] <= ALERT_RADIUS_KM
    not_clear = weather["event_type"] != "clear"
    return severity_ok and distance_ok and not_clear


def build_alert_message(vehicle: dict, weather: dict) -> str:
    """Compose a human-readable alert message."""
    template = ALERT_TEMPLATES.get(weather["event_type"], ALERT_TEMPLATES["default"])

    # Build optional detail strings
    hail_detail = (
        f"(up to {weather['hail_size_cm']} cm) " if weather.get("hail_size_cm") else ""
    )
    rain_detail = (
        f"({weather['rainfall_mm_hr']} mm/hr) " if weather.get("rainfall_mm_hr") else ""
    )
    wind_detail = (
        f"{weather['wind_speed_kmh']} km/h " if weather.get("wind_speed_kmh") else ""
    )

    message = template.format(
        arrival=weather["projected_arrival"],
        distance=weather["distance_km"],
        hail_detail=hail_detail,
        rain_detail=rain_detail,
        wind_detail=wind_detail,
        event_type=weather["event_type"].replace("_", " ").title(),
    )
    return message


def evaluate_and_alert(vehicle: dict, weather: dict) -> dict:
    """
    Main entry point.
    Returns a result dict with:
      - alert_triggered (bool)
      - alert_message   (str | None)
      - reason          (str)  – explains why an alert was or wasn't sent
    """
    parked = is_vehicle_parked(vehicle)
    alert_needed = is_alert_required(weather)

    if not parked:
        return {
            "alert_triggered": False,
            "alert_message": None,
            "reason": (
                f"Vehicle is moving ({vehicle['speed_kmh']} km/h). "
                "Scenario 1 alerts only apply to parked/static vehicles."
            ),
        }

    if not alert_needed:
        sev = weather["severity"]
        dist = weather["distance_km"]
        etype = weather["event_type"]
        if etype == "clear":
            reason = "Weather conditions are clear. No alert needed."
        elif SEVERITY_RANK.get(sev, 0) < SEVERITY_RANK[ALERT_SEVERITY_THRESHOLD]:
            reason = (
                f"Weather event '{etype}' has severity '{sev}', which is below "
                f"the alert threshold ('{ALERT_SEVERITY_THRESHOLD}'). No alert sent."
            )
        else:
            reason = (
                f"Weather event '{etype}' is {dist:.1f} km away, which is beyond "
                f"the alert radius ({ALERT_RADIUS_KM} km). No alert sent."
            )
        return {"alert_triggered": False, "alert_message": None, "reason": reason}

    message = build_alert_message(vehicle, weather)
    return {
        "alert_triggered": True,
        "alert_message": message,
        "reason": (
            f"Vehicle is parked. '{weather['event_type']}' ({weather['severity']}) "
            f"is {weather['distance_km']:.1f} km away, ETA {weather['eta_minutes']} min."
        ),
    }


# ---------------------------------------------------------------------------
# Display Helper
# ---------------------------------------------------------------------------

def print_result(scenario_name: str, vehicle: dict, weather: dict, result: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SCENARIO: {scenario_name}")
    print(sep)
    print(f"  Vehicle  : {vehicle['vehicle_id']}  |  "
          f"Status: {vehicle['status']}  |  Speed: {vehicle['speed_kmh']} km/h")
    print(f"  Location : {vehicle['location']['address']}")
    print(f"  Weather  : {weather['event_type'].replace('_',' ').title()}  |  "
          f"Severity: {weather['severity'].upper()}  |  "
          f"Distance: {weather['distance_km']} km  |  ETA: {weather['eta_minutes']} min")
    print("-" * 70)
    triggered = result["alert_triggered"]
    status_label = "ALERT TRIGGERED" if triggered else "NO ALERT"
    print(f"  [{status_label}]")
    print(f"  Reason : {result['reason']}")
    if triggered:
        print(f"\n  MESSAGE:\n  {result['alert_message']}")
    print(sep)


# ---------------------------------------------------------------------------
# Sample Scenarios
# ---------------------------------------------------------------------------

def run_all_scenarios():
    scenarios = [

        # ------------------------------------------------------------------
        # POSITIVE SCENARIOS – alert should fire
        # ------------------------------------------------------------------
        (
            "Positive 1 – Parked car + Severe Hail Storm approaching",
            make_vehicle(
                vehicle_id="TN-01-AB-1234",
                speed_kmh=0,
                status="parked",
                location={"lat": 13.0827, "lon": 80.2707, "address": "Anna Nagar, Chennai"},
            ),
            make_weather_event(
                event_type="hail",
                severity="severe",
                distance_km=15,
                eta_minutes=25,
                description="Large hailstones expected",
                hail_size_cm=3.5,
            ),
        ),
        (
            "Positive 2 – Static car (speed=0, status=idle) + Tornado Warning",
            make_vehicle(
                vehicle_id="KA-05-CD-5678",
                speed_kmh=0,
                status="idle",
                location={"lat": 12.9716, "lon": 77.5946, "address": "Indiranagar, Bengaluru"},
            ),
            make_weather_event(
                event_type="tornado",
                severity="extreme",
                distance_km=8,
                eta_minutes=12,
                description="EF3 tornado on ground",
            ),
        ),
        (
            "Positive 3 – Parked car + Flash Flood (heavy rainfall)",
            make_vehicle(
                vehicle_id="MH-12-EF-9012",
                speed_kmh=0,
                status="parked",
                location={"lat": 18.5204, "lon": 73.8567, "address": "Koregaon Park, Pune"},
            ),
            make_weather_event(
                event_type="flash_flood",
                severity="severe",
                distance_km=5,
                eta_minutes=20,
                description="Extreme rainfall causing rapid flooding",
                rainfall_mm_hr=120,
            ),
        ),
        (
            "Positive 4 – Parked car + Blizzard approaching",
            make_vehicle(
                vehicle_id="HP-65-GH-3456",
                speed_kmh=0,
                status="parked",
                location={"lat": 32.2190, "lon": 77.1910, "address": "Mall Road, Manali"},
            ),
            make_weather_event(
                event_type="blizzard",
                severity="extreme",
                distance_km=20,
                eta_minutes=45,
                description="Heavy snowfall with near-zero visibility",
            ),
        ),
        (
            "Positive 5 – Parked car + Severe High Winds",
            make_vehicle(
                vehicle_id="GJ-01-IJ-7890",
                speed_kmh=0,
                status="parked",
                location={"lat": 23.0225, "lon": 72.5714, "address": "Satellite, Ahmedabad"},
            ),
            make_weather_event(
                event_type="high_winds",
                severity="moderate",
                distance_km=10,
                eta_minutes=30,
                description="Damaging wind gusts",
                wind_speed_kmh=95,
            ),
        ),

        # ------------------------------------------------------------------
        # NEGATIVE SCENARIOS – no alert should fire
        # ------------------------------------------------------------------
        (
            "Negative 1 – Car is MOVING (not parked), severe weather present",
            make_vehicle(
                vehicle_id="DL-08-KL-2345",
                speed_kmh=60,
                status="moving",
                location={"lat": 28.6139, "lon": 77.2090, "address": "Connaught Place, Delhi"},
            ),
            make_weather_event(
                event_type="hail",
                severity="severe",
                distance_km=10,
                eta_minutes=15,
                description="Large hail approaching",
                hail_size_cm=2.0,
            ),
        ),
        (
            "Negative 2 – Parked car + Weather severity BELOW threshold (minor drizzle)",
            make_vehicle(
                vehicle_id="TN-10-MN-6789",
                speed_kmh=0,
                status="parked",
                location={"lat": 11.0168, "lon": 76.9558, "address": "RS Puram, Coimbatore"},
            ),
            make_weather_event(
                event_type="thunderstorm",
                severity="none",       # Below "moderate" threshold → no alert
                distance_km=5,
                eta_minutes=40,
                description="Light drizzle with distant thunder",
            ),
        ),
        (
            "Negative 3 – Parked car + Weather event is FAR AWAY (beyond radius)",
            make_vehicle(
                vehicle_id="AP-28-OP-1357",
                speed_kmh=0,
                status="parked",
                location={"lat": 17.3850, "lon": 78.4867, "address": "Jubilee Hills, Hyderabad"},
            ),
            make_weather_event(
                event_type="hail",
                severity="severe",
                distance_km=80,        # Beyond 30 km alert radius → no alert
                eta_minutes=120,
                description="Hail storm far from current location",
                hail_size_cm=1.5,
            ),
        ),
        (
            "Negative 4 – Parked car + Clear weather (no event)",
            make_vehicle(
                vehicle_id="KL-07-QR-2468",
                speed_kmh=0,
                status="parked",
                location={"lat": 9.9312, "lon": 76.2673, "address": "Marine Drive, Kochi"},
            ),
            make_weather_event(
                event_type="clear",
                severity="none",
                distance_km=0,
                eta_minutes=0,
                description="Sunny skies, no weather hazards",
            ),
        ),
        (
            "Negative 5 – Moving car + Clear weather (double negative)",
            make_vehicle(
                vehicle_id="RJ-14-ST-3579",
                speed_kmh=85,
                status="moving",
                location={"lat": 26.9124, "lon": 75.7873, "address": "C-Scheme, Jaipur"},
            ),
            make_weather_event(
                event_type="clear",
                severity="none",
                distance_km=0,
                eta_minutes=0,
                description="Clear conditions throughout the region",
            ),
        ),
    ]

    print("\n" + "#" * 70)
    print("#   SAFE PILOT APP – Scenario 1: Weather Alerts for Parked Cars   #")
    print("#" * 70)

    positive_count = 0
    negative_count = 0

    for name, vehicle, weather in scenarios:
        result = evaluate_and_alert(vehicle, weather)
        print_result(name, vehicle, weather, result)
        if result["alert_triggered"]:
            positive_count += 1
        else:
            negative_count += 1

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {positive_count} alert(s) triggered | {negative_count} suppressed")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_scenarios()
